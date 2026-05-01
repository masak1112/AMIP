#!/bin/bash -l
#PBS -N bias_combined
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=05:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/combined_NCAR_2.yaml

python bias.py --config=$CONFIG