#!/bin/bash -l
#PBS -N AE_3D_simple
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=0:10:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch2

CONFIG=configs/ae_3D.yaml

python train.py --config=$CONFIG