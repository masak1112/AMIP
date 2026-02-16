#!/bin/bash -l
#PBS -N AE_CRPS
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=00:30:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_stochastic.yaml

python train.py --config=$CONFIG