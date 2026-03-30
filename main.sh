#!/bin/bash -l
#PBS -N SI_Latent_DiT
#PBS -l select=1:ncpus=32:ngpus=4:mem=256G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0014
#PBS -j oe

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=1800
export NCCL_P2P_DISABLE=1                                                                                                                                                                                                                                                       
export NCCL_SHM_DISABLE=0 

module load conda
conda activate torch

CONFIG=configs/SI_Latent_DiT_subpixel.yaml

python train.py --config=$CONFIG