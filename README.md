## DM2: diffusion models for disordered materials

This repo is mainly adopted from [LLNL/graphite](https://github.com/LLNL/graphite/) (version Dec 12, 2023).

**📂 Directory Overview**
- **demo/** — Example scripts for generation and training  
  - `gen_data/` — Input data generation utilities  
  - `inital_data/` — Initial random structures (e.g., SiO₂)  
    - `random_sio2_size_300_data/`  
    - `random_sio2_size_3000_data/`  
  - `denoise_generate_demo.py` — Unconditional generation example  
  - `denoise_train_conditional.py` — Conditional training example  
  - `denoiser_train.py` — Unconditional training example  

- **model/** — Pretrained diffusion models  
  - `gen-a-sio2-cond-v1.pt` — Conditional SiO₂ generator  
  - `gen-a-sio2-v1` — Unconditional SiO₂ generator  
  - `gen-cu50zr50-v1.pt` — Cu–Zr metallic glass generator  

- **src/** — Core source code and utilities  
                    

## 🧪 Demo: Generating Amorphous SiO₂

A demo script is provided at
```~/demo/denoise_generate_demo.py```
for generating amorphous SiO₂ structures using the trained  model.

Before running the demo, you may need to make minor adjustments (changing GPU ID or updating relevant file paths). Once configured, simply execute the script to reproduce the sample generation results.

The demo generating 300-atom a-SiO₂ took about 1.5 mins on NVIDIA RTX A6000.

## ⚙️ Installation

Common packages:
- `numpy`
- `scikit-learn`
- `pandas`
- `ase`

```bash
pip install numpy scikit-learn pandas ase
```
Install pytorch and other packages for graph data.
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install torch_geometric

pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

Install e3nn with specific version.  
The latest version of e3nn may cause some error.
```bash
pip install "e3nn==0.4.4"
```

Then, clone the repo and install ```dm2```  
```bash
pip install -e /path/to/the/repo
```

To uninstall: 
```bash
pip uninstall graphite
```