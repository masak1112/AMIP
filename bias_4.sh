#!/bin/bash -l
#PBS -N bias_combined
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -A UCHI0018
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/combined_NCAR.yaml

python bias.py --config=$CONFIG --checkpoint="/glade/derecho/scratch/ayz/AMIP_logs/SI_X_forcings_42_2026-05-01T09-46-57/model_epoch=02.ckpt" --description="ep_2"