#!/bin/bash -l
#PBS -N AE_HFS_FT
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=00:10:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_decoder_hfs_ft.yaml

python train.py --config=$CONFIG