#!/bin/bash
#SBATCH --job-name="seed_multi_19800101"
#SBATCH --output="slurm-%j.out"
#SBATCH --error="slurm-%j.err"
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export OMP_NUM_THREADS=1

conda activate torch

CONFIG=configs/combined_stampede.yaml
START_DATE=1980-01-01
END_DATE=${END_DATE:-2022-01-01}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/10512/azhou4/rollouts/}

python eval_lagged_batch.py --config=${CONFIG} \
    --start_date=${START_DATE} \
    --end_date=${END_DATE} \
    --seeds 60 64 68 72 \
    --devices 0 1 2 3 \
    --batch_size 4 \
    --output_root=${OUTPUT_ROOT} \
    --spectral_monitor \
    --spectral_lag_days=10 \
    --spectral_threshold=0.5