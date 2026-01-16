## Requirements

To install requirements:
```setup
conda create -n "my_env" 
conda install pip 
pip install torch torchvision
pip install lightning matplotlib wandb h5py timm einops
```
Currently running with pytorch 2.9.1 and CUDA 12.8

To run SFNO:
```
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```