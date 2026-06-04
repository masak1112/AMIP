#!/bin/bash -l
#PBS -N Decoder_CNN
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=00:05:00
#PBS -A UCHI0014
#PBS -j oe

module load conda
conda activate torch
source /glade/work/bgong/.venv/bin/activate

CONFIG=configs/ae_decoder.yamlvgmail

python trainAE.py --config=$CONFIG