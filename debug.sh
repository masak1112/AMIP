#!/bin/bash -l
#PBS -N debug
#PBS -l select=1:ncpus=16:ngpus=4:mem=120G
#PBS -q develop
#PBS -l walltime=0:10:00
#PBS -A UCHI0014
#PBS -j oe

# Enable GPU-MPI (if supported by application)
export NCCL_DEBUG=INFO

ml gcc/12.4.0
ml ncarenv/24.12
ml craype/2.7.31
ml ncarcompilers/1.0.0
ml libfabric/1.15.2.0
ml cuda/12.3.2
ml cray-mpich/8.1.29
ml hdf5/1.12.3
ml netcdf/4.9.2
ml conda/latest
ml intel/2024.2.1
ml mkl/2024.2.2

conda activate credit

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"

CONFIG=configs/flow.yaml 

# Launch your script using torch.distributed.launch
python train.py --config=$CONFIG
# --enable_amp if needed