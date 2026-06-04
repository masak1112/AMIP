#!/bin/bash -l
#PBS -N finetune
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -A UCHI0018

source /glade/work/bgong/.venv/bin/activate

#CONFIG=configs/DDC_NCAR.yaml
CONFIG=configs/SI_NCAR_AIMIP_finetune.yaml

python train.py --config=$CONFIG