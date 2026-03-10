#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=2:ncpus=8:ngpus=8:mem=120G
#PBS -q preempt
#PBS -l walltime=00:05:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT.yaml

python train.py --config=$CONFIG

#qinteractive -A UCHI0014 -l walltime=00:10:00 -l select=2:ncpus=8:ngpus=8:mem=120G