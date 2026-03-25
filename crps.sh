#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=1:ncpus=8:ngpus=1:mem=64G
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT.yaml

python crps_ssr.py --config=$CONFIG