#!/bin/bash -l
#PBS -N ae_develop
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=0:30:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch2

CONFIG=configs/ae_simple_DS.yaml

python train.py --config=$CONFIG