#!/bin/bash
#PBS -A UCHI0018
#PBS -N SI-DiT-PlaSim
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4:gpu_type=a100
#PBS -e logs/si_climadit_plasim_err.txt
#PBS -o logs/si_climadit_plasim_out.txt

# Use scratch for tmp.
export TMPDIR=/glade/derecho/scratch/$USER/tmp
mkdir -p $TMPDIR

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP}"

WORK_DIR="/glade/work/weidong/code/benchmarkGM/interpolant_baseline"
cd $WORK_DIR
mkdir -p logs checkpoints

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

# Multi-node setup (kept generic so this scales to >1 node by editing select=N).
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
NNODES=$(< $PBS_NODEFILE wc -l)
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES GPUS_PER_NODE=$NUM_TASKS_PER_NODE WORLD_SIZE=$WORLD_SIZE"

export MASTER_ADDR=$MASTER_ADDR
# Distinct port from diffusion baseline (29510) and ocean run (29500).
export MASTER_PORT=19521

PRELOAD="module load conda && "
PRELOAD+="conda activate /glade/work/weidong/conda-envs/myenv && "
PRELOAD+="export MASTER_ADDR=$MASTER_ADDR && "
PRELOAD+="export MASTER_PORT=$MASTER_PORT && "
PRELOAD+="export MPICH_GPU_SUPPORT_ENABLED=1 && "
PRELOAD+="export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1 && "
# DataLoader / IO tuning. HDF5_USE_FILE_LOCKING=FALSE is required on Lustre
# (campaign + scratch) — without it concurrent h5 reads hit file-lock stalls.
# OMP/MKL caps keep each DataLoader worker from oversubscribing the host CPUs.
PRELOAD+="export HDF5_USE_FILE_LOCKING=FALSE && "
PRELOAD+="export OMP_NUM_THREADS=2 && "
PRELOAD+="export MKL_NUM_THREADS=2 && "

# Leave ~2 min headroom inside the 12h wallclock so checkpoints flush.
TIMER="timeout 718m "

CMD="$WORK_DIR/train_si_plasim.py"

# Default training hyper-parameters baked into the launcher. Any args passed
# on the qsub line ("$@") are appended after these and therefore override them
# (argparse uses the last occurrence of each flag).
TRAIN_ARGS="--grad_accum 1 --lr 5e-5 --warmup_epochs 5 --epochs 200 --num_workers 8"

# --- Model-size preset (uncomment one) -----------------------------------
# Switches the ClimaDiT backbone width/depth. Numbers below are the total
# trainable params; the diffusion baseline next door is ~175M for reference.
#
# SIZE=base    # dim=768,  12 SA / 6 CA, heads=12, batch=16   -> ~178 M  (default)
# SIZE=large   # dim=1024, 12 SA / 6 CA, heads=16, batch=8    -> ~317 M
SIZE=small   # dim=512,  20 SA / 10 CA, heads=8, batch=16   -> ~130 M
SIZE=${SIZE:-base}

case "$SIZE" in
    base)
        SIZE_ARGS=""  # python defaults already match
        ;;
    large)
        SIZE_ARGS="--dit_dim 1024 --dit_cond_dim 1024 --dit_heads 16 \
                   --dit_proj_bottleneck_dim 1024 --batch_size 8"
        ;;
    small)
        SIZE_ARGS="--dit_dim 512 --dit_cond_dim 512 --dit_heads 8 \
                   --dit_proj_bottleneck_dim 512 \
                   --dit_sa_blocks 20 --dit_ca_blocks 10"
        ;;
    *)
        echo "Unknown SIZE='$SIZE' (expected: base | large | small)" >&2
        exit 2
        ;;
esac
echo "Model size preset: SIZE=$SIZE  -> $SIZE_ARGS"

# Order: defaults, then size preset, then $EXTRA env var, then qsub positional
# args ("$@"). argparse uses the last occurrence of each flag, so later wins.
# Pass overrides via either:
#   qsub -v SIZE=large train_si_plasim.sh
#   qsub -v EXTRA="--lr 1e-4 --epochs 400" train_si_plasim.sh
#   qsub train_si_plasim.sh -- --lr 1e-4        (positional, after `--`)
EXTRA_ARGS="$TRAIN_ARGS $SIZE_ARGS ${EXTRA:-} $@"

NODE_RANK=0
for NODE in $(cat $PBS_NODEFILE | uniq); do
    LAUNCHER="python -m torch.distributed.run "
    LAUNCHER+="--nnodes=$NNODES "
    LAUNCHER+="--nproc_per_node=$NUM_TASKS_PER_NODE "
    LAUNCHER+="--node_rank=$NODE_RANK "
    LAUNCHER+="--master_addr=$MASTER_ADDR "
    LAUNCHER+="--master_port=$MASTER_PORT "
    LAUNCHER+="--max_restarts=0 "

    FULL_CMD="$PRELOAD $TIMER $LAUNCHER $CMD $EXTRA_ARGS"

    if [[ "$NODE" == "$(hostname)" ]]; then
        echo "Launching node_rank $NODE_RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching node_rank $NODE_RANK on remote node $NODE"
        ssh $NODE "cd $WORK_DIR; $FULL_CMD" &
    fi
    NODE_RANK=$((NODE_RANK + 1))
done

wait
echo "Job finished at: $(date '+%Y-%m-%d-%H%M%S')"
