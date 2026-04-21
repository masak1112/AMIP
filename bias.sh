#!/bin/bash -l
#PBS -N bias_SI
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=06:30:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/SI_NCAR.yaml

python bias.py --config=$CONFIG