#!/bin/bash -l
#PBS -N Decoder_CNN
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=00:05:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/ae_decoder.yaml

python train.py --config=$CONFIG