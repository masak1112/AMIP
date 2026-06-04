#!/bin/bash -l
#PBS -N SI
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_NCAR.yaml

python train.py --config=$CONFIG