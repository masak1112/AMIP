#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=2:ncpus=32:ngpus=8:mem=240G
#PBS -q develop
#PBS -l walltime=00:05:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT.yaml

python train.py --config=$CONFIG