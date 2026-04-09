## Requirements

To install requirements:
```setup
conda create -n "my_env" python=3.13
conda install pip 
pip install torch torchvision
pip install lightning matplotlib wandb h5py timm einops h5pickle
```
Currently running with pytorch 2.10 and CUDA 12.8

To run SFNO:
```
conda install torch-harmonics
pip install -U tensorly tensorly-torch
```

    # model:
    #     img_resolution: [180, 360]
    #     in_channels: 308 # 2*151 + 2 + 4
    #     out_channels: 151 # 6 + 15 + 26*5 = 151
    #     window_size: [5, 10]
    #     shift_size: [3, 5]
    #     patch_size: [4, 4]
    #     depth: 16
    #     head_depth: 4
    #     dim: 1024
    #     heads: 16