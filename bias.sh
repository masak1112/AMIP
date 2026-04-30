#!/bin/bash -l
#PBS -N bias_SI
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=03:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_NCAR.yaml

python bias.py --config=$CONFIG