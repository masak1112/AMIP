#!/bin/bash
#SBATCH --job-name="seed_42_19800101"
#SBATCH --output="slurm-%j.out"
#SBATCH --error="slurm-%j.err"
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdiu-delta-gpu
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:02:00

# Seed-sweep ensemble member: fixed initialization date, varying seed.
# Override SEED per submission to stagger ensemble members across GPUs.

export OMP_NUM_THREADS=1

module purge
ml cudatoolkit
module load pytorch-conda/2.8
conda activate torch

CONFIG=configs/combined_Delta.yaml
START_DATE=1980-01-01
END_DATE=${END_DATE:-2022-01-01}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/hdd/bdiu/ayz}
SEED=42
MEMBER_NAME=seed_${SEED}

srun python eval_lagged.py \
    --config=${CONFIG} \
    --start_date=${START_DATE} \
    --end_date=${END_DATE} \
    --output_root=${OUTPUT_ROOT} \
    --member_name=${MEMBER_NAME} \
    --seed=${SEED} \
    --spectral_monitor \
    --spectral_lag_days=10 \
    --spectral_threshold=0.5
