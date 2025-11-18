import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import ase.io
from ase.neighborlist import primitive_neighbor_list
from pathlib import Path
from functools import partial
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import trange
from torch_geometric.data import Data, Dataset
from graphite.nn.basis import bessel
from graphite.transforms import RattleParticles, DownselectEdges
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Global constants
CUTOFF = 5.0 

class InitialEmbedding(nn.Module):
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        # Embed node
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data


# Compute the ase_graph and move to GPU
def ase_graph_gpu(data, cutoff):
    """GPU-optimized graph creation with proper shape handling"""
    # Compute neighbor list (still needs CPU as it's ASE-dependent)
    pos_cpu = data.pos.cpu().numpy()
    cell_cpu = data.cell.cpu().numpy() if torch.is_tensor(data.cell) else data.cell
    
    i, j, D = primitive_neighbor_list('ijD', 
                                    cutoff=cutoff, 
                                    pbc=data.pbc, 
                                    cell=cell_cpu, 
                                    positions=pos_cpu, 
                                    numbers=data.numbers)
    
    # Move to GPU directly without concatenation
    device = data.pos.device
    data.edge_index = torch.from_numpy(np.stack((i, j))).to(device).long()
    data.edge_attr = torch.from_numpy(D).to(device).float()
    
    return data

def set_gpu(gpu_id):
    """Set PyTorch to use a specific GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


@torch.no_grad()
def denoise_snapshot_with_noise_gpu(atoms, model, cooling_rate, scale=1.0, steps=8, max_sigma_for_denoising=0.1):
    device = next(model.parameters()).device
    print(f"Running denoising on device: {device}")
    print(f"Target cooling rate: {cooling_rate} K/ps")
    
    # Pre-compute sigma values for all steps
    sigmas = torch.linspace(max_sigma_for_denoising, 0.001, steps).to(device)
    
    # Initialize data on GPU in one go
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    pos = torch.tensor(atoms.positions, device=device).float() * scale
    cell = torch.tensor(atoms.cell, device=device).float() * scale
    
    # Log transform the cooling rate
    log_cooling_rate = torch.tensor([np.log10(cooling_rate)], device=device).float()
    
    data = Data(
        x=x,
        pos=pos,
        cell=cell,
        pbc=atoms.pbc,
        numbers=atoms.numbers,
        cooling_rate=log_cooling_rate,
        batch=torch.zeros(len(x), dtype=torch.long, device=device)
    )
    data.num_graphs = 1
    
    # Pre-allocate memory for trajectory - store only final positions on CPU
    pos_traj = [atoms.positions]
    sigma_min = 0.001

    # denoising loop
    for i, sigma in enumerate(sigmas, 1):
        data = ase_graph_gpu(data, cutoff=CUTOFF)
        # Apply noise and compute displacement
        noise_scale = torch.empty_like(data.pos).uniform_(sigma_min, sigma.item())
        noise = torch.randn_like(data.pos) * noise_scale
        data.dx = noise
        disp = model(data, data.cooling_rate) + data.dx
        data.pos.sub_(disp) # Update positions (in-place to save memory)
        
        # Transfer only when needed - store in CPU
        if i % 50 == 0 or i == steps:  
            pos_traj.append((data.pos.cpu() / scale).numpy())
            
        # Print progress
        print(f"Progress: {i}/{steps} steps (sigma = {sigma.item():.6f})", flush=True)
    
    return pos_traj

@torch.no_grad()
def denoise_snapshot_gpu(atoms, model, cooling_rate, scale=1.0, steps=8):
    # Get the device from the model
    device = next(model.parameters()).device
    print(f"Running denoising on device: {device}")
    print(f"Target cooling rate: {cooling_rate} K/ps")
    
    # Log transform the cooling rate
    log_cooling_rate = torch.tensor([np.log10(cooling_rate)], device=device).float()
    
    # Convert to PyG format and move to GPU immediately
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    data = Data(
        x=x,  # Avoid duplicate tensor creation
        pos=torch.tensor(atoms.positions, device=device).float(),
        cell=torch.tensor(atoms.cell, device=device).float(),
        pbc=atoms.pbc,
        numbers=atoms.numbers,
        cooling_rate=log_cooling_rate,
        batch=torch.zeros(len(x), dtype=torch.long, device=device)
    )
    data.num_graphs = 1
    
    # Scale
    data.pos *= scale
    data.cell *= scale
    
    # Denoising trajectory - store only initial position
    pos_traj = [atoms.positions]
    
    # Process each step
    for i in range(1, steps + 1):
        # Use the GPU-optimized graph creation
        data = ase_graph_gpu(data, cutoff=CUTOFF)
        gpu_disp = model(data, data.cooling_rate)
        data.pos.sub_(gpu_disp)
        
        # Transfer only periodically to save memory
        if i % 50 == 0 or i == steps:
            pos_traj.append((data.pos.cpu() / scale).numpy())
        
        # Print progress
        print(f"Progress: {i}/{steps} steps", flush=True)
    
    return pos_traj

def main():
    start_time = time.time() 

    # === Change here ===#
    gpu_id = 0
    set_gpu(gpu_id)
    model = torch.load('../model/gen-a-sio2-cond-v1.pt')
    CUTOFF = 5
    target_cooling_rate = 0.1  # Ideal range: 10^-2 to 10^3. Max range: 10^-4 to 10^4 
    model = model.to('cuda')
    TEST_FNAME = './inital_data/random_sio2_size_300_demo.dat'
    OUTPUT_FNAME = './gen_data/denoised_random_sio2_300_demo.extxyz'
    # ===================#

    noisy_atoms = ase.io.read(TEST_FNAME, format='lammps-data')

    pos_traj = denoise_snapshot_with_noise_gpu(
        noisy_atoms, model,
        cooling_rate=target_cooling_rate,
        scale=1.0, steps=2900, max_sigma_for_denoising=1.0
    )
    a = ase.Atoms(
        symbols=noisy_atoms.get_chemical_symbols(),
        positions=pos_traj[-1],
        cell=noisy_atoms.cell,
        pbc=True
    )
    pos_traj_2 = denoise_snapshot_gpu(a, model, cooling_rate=target_cooling_rate, scale=1.0, steps=100)
    pos_traj.extend(pos_traj_2)
    denoising_traj = [
        ase.Atoms(
            symbols=noisy_atoms.get_chemical_symbols(),
            positions=pos,
            cell=noisy_atoms.cell,
            pbc=True
        )
        for pos in pos_traj
    ]
    for atoms in denoising_traj:
        atoms.wrap()
    ase.io.write(OUTPUT_FNAME, denoising_traj)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\n✅ Total runtime: {elapsed_minutes:.2f} minutes ({elapsed_minutes/60:.2f} hours)")


if __name__ == "__main__":
    main()