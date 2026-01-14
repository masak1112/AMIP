## Requirements

To install requirements:
```setup
conda create -n "my_env" 
conda install pytorch=2.2.0 pytorch-cuda=12.1
conda install lightning matplotlib wandb h5py timm einops
```

On NCAR Derecho, setup pytorch according to https://github.com/NCAR/aiml_gpu_ncar_envs/tree/main/pytorch. 
Disable mpi options for hdf5 and h5py, and downgrade numpy to 1.26.2. 

To run SFNO:
```
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```