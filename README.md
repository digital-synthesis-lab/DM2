## <img src="media/DM2_logo.png" alt="alt text" width="60"> DM<sup>2</sup>: Diffusion Models for Disordered Materials

This repo is mainly adapted from [LLNL/graphite](https://github.com/LLNL/graphite/) (version Dec 12, 2023).
Modifications to the original code from [Tim Hsu](https://github.com/tim-hsu) were made to include an embedding for the processing conditions of the glassy materials, and integrate the generation with the simulation of amorphous structures.
This code is provided as a separate snapshot to ensure reproducibility of [our manuscript](https://arxiv.org/abs/2507.05024), but considering that the credits for the original `graphite` code belong to Hsu.

# 📂 Gallery

### Generation of cubic a-SiO<sub>2</sub> 
<p align="center">
  <img src="./media/gifs/a-sio2-cube.gif" width="30%">
</p>

### Generation of cubic a-SiO<sub>2</sub> (slice view)
<p align="center">
  <img src="./media/gifs/a-sio2-denoiser_200ppi.gif" width="30%">
</p>

### Generation of amorphous mesoporous SiO<sub>2</sub> (slice view)
<p align="center">
  <img src="./media/gifs/a-sio2-pore.gif" width="30%">
</p>

### Generation of cubic Cu<sub>50</sub>Zr<sub>50</sub> (slice view)
<p align="center">
  <img src="./media/gifs/cuzr_gen.gif" width="30%">
</p>


# 📂 Directory Overview
- **demo/** — Example scripts for training and generating
  - `demo_training/` 
    - `simu_data/` — Simulated SiO<sub>2</sub> structures for training
    - `denoiser_train_unconditional.py` -- unconditional training script
    - `denoiser_train_conditional.py` -- conditional training script

  - `demo_generating/` 
    - `inital_data/` —- Initial random structures (e.g., SiO<sub>2</sub>)
    - `gen_data/` -- Empty folder to store generated trajectory
    - `denoise_generate_unconditional.py` -- unconditional generation script
    - `denoise_generate_conditional.py`   -- conditional generation script

  - **model/** — Pretrained diffusion models
    - `gen-a-sio2-cond-v1.pt` — Conditional SiO<sub>2</sub> generator
    - `gen-a-sio2-uncond-v1.pt` — Unconditional SiO<sub>2</sub> generator
    - `gen-cu50zr50-v1.pt` — Cu–Zr metallic glass generator

- **src/** — Core source code and utilities


# 🧪 Demo: Training and generating

Demo scripts and related files are provided at ```~/demo/demo_training``` for training and ```~/demo/demo_generating``` for generating amorphous SiO<sub>2</sub> structures using the pre-trained model.

Before running the demo, you may need to make minor adjustments (changing GPU ID or updating relevant file paths). Once configured, simply execute the script by ```python DEMO.py``` to reproduce the our results.

The demo generating 300-atom a-SiO<sub>2</sub> took about 2 mins on NVIDIA RTX A6000.

The unconditional model training takes about 20 hours and the conditional model training takes about 40 hours on NVIDIA RTX A6000

# ⚙️ Installation

Create new environment

```bash
conda create -n dm2 python=3.10 -y
conda activate dm2
```

Install pytorch and other packages for graph data.
```bash
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install torch-geometric
```
Install e3nn with specific version.
The latest version of e3nn may cause some error.
```bash
pip install "e3nn==0.4.4"
```

Common packages:
```bash
pip install numpy==1.26.4

pip install ase scipy pandas scikit-learn matplotlib tqdm
```

Then, clone the repo and install ```dm2```
```bash
pip install -e /path/to/the/repo
```

Installation of different verison of packages may cause errors.

# Citing

If using this code, please cite the following papers:

```
@article{yang2025generative,
  title={A generative diffusion model for amorphous materials},
  author={Yang, Kai and Schwalbe-Koda, Daniel},
  journal={arXiv:2507.05024},
  year={2025}
}

@article{hsu2024score,
  title={Score-based denoising for atomic structure identification},
  author={Hsu, Tim and Sadigh, Babak and Bertin, Nicolas and Park, Cheol Woo and Chapman, James and Bulatov, Vasily and Zhou, Fei},
  journal={npj Computational Materials},
  volume={10},
  number={1},
  pages={155},
  year={2024},
}
```
We thank the support from TRI for this project.

<p align="center">
  <img src="./media/logos/Toyota_Research_Institute_Logo_Square.png" width="50%">
</p>