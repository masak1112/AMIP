#!/bin/bash -l
#PBS -N debug
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=0:10:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch2

CONFIG=configs/flow.yaml

python train.py --config=$CONFIG