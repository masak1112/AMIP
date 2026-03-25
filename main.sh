#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_subpixel.yaml

python train.py --config=$CONFIG