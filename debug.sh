#!/bin/bash -l
#PBS -N debug
#PBS -l select=1:ncpus=16:ngpus=1:mem=120G
#PBS -q develop
#PBS -l walltime=0:05:00
#PBS -A UCHI0014
#PBS -j oe

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

ml conda
conda activate pt220gpu_conda

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

CONFIG=configs/flow.yaml 

# Launch your script using torch.distributed.launch
python train.py --config=$CONFIG
# --enable_amp if needed