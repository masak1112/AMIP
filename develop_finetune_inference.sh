#!/bin/bash -l
#PBS -N inference
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q develop
#PBS -l walltime=06:00:00
#PBS -A UCHI0018


source /glade/work/bgong/.venv/bin/activate

#CONFIG=configs/DDC_NCAR.yaml
CONFIG=configs/combined_NCAR_finetune.yaml
SEED=${SEED:-1}
YEAR=${YEAR:-2019}

echo "Running inference for seed=$SEED, year=$YEAR"
python rollout_single_S2S.py --config="$CONFIG" --seed="$SEED" --year="$YEAR"