#!/bin/bash -l
#PBS -N AE_FA
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_fa.yaml

python train.py --config=$CONFIG