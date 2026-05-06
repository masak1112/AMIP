#!/bin/bash -l
#PBS -N lagged_rollout
#PBS -l select=1:ncpus=8:ngpus=1:mem=192G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0018
#PBS -j oe

# Single-member lagged-ensemble rollout. Override START_DATE per submission to
# stagger ensemble members across GPUs. End date matches eval_AIMIP.py.

module load conda
conda activate torch

CONFIG=configs/combined_NCAR.yaml
START_DATE=${START_DATE:-1979-01-01}
END_DATE=${END_DATE:-2025-01-01}
OUTPUT_ROOT=${OUTPUT_ROOT:-/glade/campaign/univ/uchi0014}
MEMBER_NAME=${MEMBER_NAME:-}                # auto-derived from START_DATE if empty
CHECKPOINT=${CHECKPOINT:-}                  # optional forecaster checkpoint override
DOWNSCALER_CHECKPOINT=${DOWNSCALER_CHECKPOINT:-}
SEED=${SEED:-}

ARGS=(--config="$CONFIG"
      --start_date="$START_DATE"
      --end_date="$END_DATE"
      --output_root="$OUTPUT_ROOT")
[[ -n "$MEMBER_NAME" ]] && ARGS+=(--member_name="$MEMBER_NAME")
[[ -n "$CHECKPOINT" ]] && ARGS+=(--checkpoint="$CHECKPOINT")
[[ -n "$DOWNSCALER_CHECKPOINT" ]] && ARGS+=(--downscaler_checkpoint="$DOWNSCALER_CHECKPOINT")
[[ -n "$SEED" ]] && ARGS+=(--seed="$SEED")

python eval_lagged.py "${ARGS[@]}"
