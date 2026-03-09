#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=1:ncpus=4:ngpus=1:mem=30G
#PBS -q develop
#PBS -l walltime=00:02:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT.yaml

python train.py --config=$CONFIG