#!/bin/bash -l
#PBS -N aimip_rollout
#PBS -l select=1:ncpus=8:ngpus=1:mem=192G
#PBS -q main
#PBS -l walltime=24:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=configs/combined_NCAR.yaml
CHECKPOINT_PATH=""           # forecaster checkpoint (Combined) or full state dict (TrainModule)
DOWNSCALER_CHECKPOINT=""     # only needed if model_name=Combined and you want to override the YAML
OUTPUT_DIR=/glade/derecho/scratch/ayz/AIMIP_submission

python eval_AIMIP.py \
    --config=$CONFIG \
    --checkpoint=$CHECKPOINT_PATH \
    --downscaler_checkpoint=$DOWNSCALER_CHECKPOINT \
    --output_dir=$OUTPUT_DIR \
    --institute=CMU \
    --aimip_model_name=CMU-AMIP \
    --ensemble_size=5 \
    --start_year=1979 \
    --end_year=2025
