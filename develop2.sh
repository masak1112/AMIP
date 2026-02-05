#!/bin/bash -l
#PBS -N ae_DC_diag
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=00:10:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_simple_DC_diag.yaml

python train.py --config=$CONFIG