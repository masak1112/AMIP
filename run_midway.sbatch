#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=24:00:00
#SBATCH -p pedramh-gpu 
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 #16 
#SBATCH -o midway_ddp_%x_%j.out
#SBATCH -e midway_ddp_%x_%j.err

export MPICH_GPU_SUPPORT_ENABLED=1

ulimit -l unlimited
ml python
conda activate /project/pedramh/ayz/envs/torch2

# sinteractive --account=pi-pedramh --partition=pedramh-gpu --nodes=1 --time=48:00:00 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 --mem=500G

echo nvidia-smi

export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

config_file=configs/ae_SI_DDC.yaml

python train.py --config=$config_file