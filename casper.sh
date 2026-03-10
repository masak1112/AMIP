#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=1:ncpus=4:ngpus=4:mem=120G:gpu_type=h100
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -A URIC009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_med.yaml

python train.py --config=$CONFIG