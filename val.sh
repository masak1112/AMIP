#!/bin/bash -l
#PBS -N DDC_val
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=02:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/DDC_NCAR.yaml

python val.py --config=$CONFIG