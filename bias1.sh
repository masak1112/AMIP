#!/bin/bash -l
#PBS -N bias_history
#PBS -l select=1:ncpus=4:ngpus=1:mem=120G
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_subpixel_history.yaml

python bias.py --config=$CONFIG