#!/bin/bash -l
#PBS -N bias_vanilla
#PBS -l select=1:ncpus=4:ngpus=1:mem=120G
#PBS -q develop
#PBS -l walltime=01:30:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_VAE.yaml

python bias.py --config=$CONFIG