#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH -p pedramh-gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o bias_%x_%j.out
#SBATCH -e bias_%x_%j.err

ml python
conda activate /project/pedramh/ayz/envs/torch2

# sinteractive --account=pi-pedramh --partition=pedramh-gpu --nodes=1 --time=48:00:00 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 --mem=500G

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

config_file=configs/SI_midway.yaml

srun python bias.py --config=$config_file --checkpoint="/project/pedramh/ayz/AMIP_logs/SI_X_forcings_42_2026-04-24T15-01-46/model_epoch=20.ckpt" --description="ep_20_forcing"