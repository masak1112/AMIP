#!/bin/bash
#SBATCH --account=pi-pedramh
#SBATCH --time=00:10:00
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

echo nvidia-smi

export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

config_file=configs/ae_SI_DDC.yaml

python train.py --config=$CONFIG