#!/bin/bash -l
#PBS -N lagged_member_19790104
#PBS -l select=1:ncpus=8:ngpus=1:mem=192G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0018
#PBS -j oe

module load conda
conda activate torch

python eval_lagged.py \
    --config=configs/combined_NCAR.yaml \
    --start_date=1979-01-04 \
    --end_date=2025-01-01 \
    --output_root=/glade/campaign/univ/uchi0014/ayz/ \
    --member_name=member_19790104
