#!/bin/bash -l
#PBS -N AE_SI_Multi
#PBS -l select=4:ncpus=64:mpiprocs=4:ngpus=4:mem=384GB
#PBS -q develop
#PBS -l walltime=00:10:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_SI_multi.yaml

python train.py --config=$CONFIG