#!/bin/bash -l
#PBS -N AE_main
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch2

CONFIG=configs/ae.yaml

python train.py --config=$CONFIG