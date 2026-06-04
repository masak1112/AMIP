# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a climate/weather deep learning research project for training autoencoders and diffusion models on atmospheric state data (AMIP dataset). Models learn to compress/reconstruct or predict atmospheric fields across multiple pressure levels and variable types.

## Environment Setup

```bash
conda create -n "my_env" python=3.13
conda install pip
pip install torch torchvision
pip install lightning matplotlib wandb h5py timm einops h5pickle
# For SFNO models only:
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```

Running with PyTorch 2.10 and CUDA 12.8.

## Training

```bash
python train.py --config=configs/<config_name>.yaml [--seed SEED] [--devices GPU_IDS] [--model_name NAME] [--wandb_mode online/offline]
```

Data paths are defined in each YAML config and point to `/glade/derecho/scratch/` (NCAR HPC system). HPC jobs are submitted via PBS scripts (`main.sh`, `develop.sh`, etc.).

## Evaluation

```bash
python eval_bias.py   # atmospheric bias metrics
python eval_crps.py   # CRPS (Continuous Ranked Probability Score)
```

## Architecture Overview

### Training Entry Points
- **`train.py`** — dispatches to either `ae_module.py` (autoencoders) or `train_module.py` (diffusion/SFNO) based on config `model_name`
- **`modules/ae_module.py`** — `AutoencoderModule` (PyTorch Lightning): handles AE training, EMA, loss
- **`modules/train_module.py`** — `TrainModule` (PyTorch Lightning): handles diffusion/flow matching/SFNO training

### Model Zoo (`modules/models/`)
| File | Description |
|------|-------------|
| `AE.py` | Base convolutional autoencoder (DCAE) |
| `AE_simple.py` | Simplified AE variant |
| `AE_attn.py` | AE with attention |
| `AE_dit.py` | Diffusion Transformer AE |
| `SI_DiT.py` | Scale-Invariant Diffusion Transformer |
| `DiT.py` | Core Diffusion Transformer |
| `SFNO.py` | Spherical Fourier Neural Operator |
| `superres_dit.py` | Super-resolution Transformer |

### Layers (`modules/layers/`)
Key custom components: `dc_layers.py` (data-dependent conditioning), `arches_layers.py` (Arches framework layers), `s2convolutions.py` + `spherical_harmonics.py` (spherical operations for SFNO), `factorized_attention.py`, `axial_attention.py`.

### Diffusion/Flow Matching (`modules/diffusion/`)
- `flow_matching.py` — Conditional Flow Matching
- `data_dependent_interpolant.py` — Data-dependent interpolation (DDC)
- `interpolant.py` — Base interpolation

### Data (`data/`)
- `amip.py` — AMIP dataset: 249 total channels = 6 surface + 9×26 multi-level + 9 diagnostic + 3 forcing + 2 invariants
- `datamodule.py` — PyTorch Lightning DataModule
- `normalizer.py` — normalization utilities

### Common Utilities (`common/`)
- `loss.py` — `WeightedLoss` (latitude-weighted MSE), `latitude_weighted_rmse`, `SpectralBaseLoss`
- `utils.py` — YAML config loading, tensor assembly/disassembly for input/output formatting
- `plotting.py` — visualization helpers

## Config System

All hyperparameters live in `configs/*.yaml`. Config structure:
```yaml
model:
    model_name: DCAE  # selects which model class to instantiate
    lr: 5.0e-5
    DCAE: { ... }     # model-specific kwargs
data:
    train_data_path: /glade/...
    val_data_path: /glade/...
    norm_stats_path: /glade/...
training:
    seed: 42
    devices: 4
    accelerator: gpu
    strategy: ddp
    max_epochs: 50
    ema_decay: 0.99
```

The `model_name` in config determines which model class and training module are used.
