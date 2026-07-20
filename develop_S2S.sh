#!/bin/bash -l
#PBS -N forecast
#PBS -l select=1:ncpus=4:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -A UCHI0018

source /glade/work/bgong/.venv/bin/activate

#CONFIG=configs/DDC_NCAR.yaml
#CONFIG=configs/SI_NCAR_S2S.yaml
CONFIG=configs/SI_NCAR_S2S_crps.yaml

# Finetunes from /glade/work/bgong/amip/SI_checkpoints/last.ckpt (set via
# training.partial_checkpoint in $CONFIG) with CRPS loss added on top of the
# base interpolant loss. New checkpoints are saved under SI_checkpoints_crps.
python train.py --config=$CONFIG --checkpoint=/glade/work/bgong/amip/SI_checkpoints_crps/last.ckpt