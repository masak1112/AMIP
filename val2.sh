#!/bin/bash -l
#PBS -N SI_Latent_DiT_VAL_0.25
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=01:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_subpixel_0.25.yaml

python val.py --config=$CONFIG