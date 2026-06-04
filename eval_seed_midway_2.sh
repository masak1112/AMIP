#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p pedramh-gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o eval_4_%x_%j.out
#SBATCH -e eval_4_%x_%j.err

ml python
conda activate /project/pedramh/ayz/envs/torch2

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CONFIG=configs/combined_midway.yaml
START_DATE=1980-01-01
END_DATE=${END_DATE:-2025-01-01}
OUTPUT_ROOT=${OUTPUT_ROOT:-/project/pedramh/ayz/rollouts}
SEED=59
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
