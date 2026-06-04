#!/bin/bash
# Generate per-seed Slurm scripts for a seed-sweep ensemble on Delta and
# sbatch them. Each job runs eval_lagged.py with a fixed start date
# (1980-01-01) and a unique seed (40..55) in short (4h) walltime chunks
# that daisy-chain themselves: each job sbatches a successor when it
# hits walltime or exits non-zero, up to MAX_CHAIN_DEPTH. eval_lagged.py
# auto-resumes from rollout_checkpoint.pt, so the chain picks up cleanly.
#
# Slurm picks whichever of the four listed partitions has availability.
#
# Steady-state queue usage: 16 jobs running (one per seed); each finishing
# job spawns exactly one successor up to MAX_CHAIN_DEPTH, so total wallclock
# budget per seed is ~MAX_CHAIN_DEPTH * 4h.
#
# Usage:
#   ./submit_seed_sweep_delta.sh             # generate + sbatch
#   ./submit_seed_sweep_delta.sh --dry-run   # generate scripts only

set -euo pipefail

CONFIG="configs/combined_Delta.yaml"
START_DATE="1980-01-01"
END_DATE="2025-01-01"
OUTPUT_ROOT="/work/hdd/bdiu/ayz"
ACCOUNT="bdiu-delta-gpu"
WALLTIME="04:00:00"
OUT_DIR="seed_sweep"
PARTITIONS="gpuA100x4,gpuA100x8,gpuH200x8,gpuA40x4"

MAX_CHAIN_DEPTH=9

SEEDS=(40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    member="seed_${seed}"
    script="${OUT_DIR}/eval_${member}.sh"
    abs_script="${REPO_ROOT}/${script}"
    member_dir="${OUTPUT_ROOT%/}/${member}"

    cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name="${member}_19800101"
#SBATCH --output="slurm-%j.out"
#SBATCH --error="slurm-%j.err"
#SBATCH --partition=${PARTITIONS}
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=${ACCOUNT}
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH -t ${WALLTIME}

set -u

MEMBER_DIR="${member_dir}"
DONE_FLAG="\${MEMBER_DIR}/.seed_done"
THIS_SCRIPT="${abs_script}"
MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH}
CHAIN_DEPTH="\${CHAIN_DEPTH:-1}"
RESUBMITTED=0

resubmit_chain() {
    if (( RESUBMITTED == 1 )); then return 0; fi
    RESUBMITTED=1
    if [[ -f "\$DONE_FLAG" ]]; then
        echo "[chain] \$DONE_FLAG present -- rollout complete, not resubmitting."
        return 0
    fi
    if (( CHAIN_DEPTH >= MAX_CHAIN_DEPTH )); then
        echo "[chain] hit MAX_CHAIN_DEPTH=\$MAX_CHAIN_DEPTH -- not resubmitting. Bump the cap if more chunks are needed."
        return 0
    fi
    local next=\$((CHAIN_DEPTH + 1))
    echo "[chain] resubmitting (chain depth \$next / \$MAX_CHAIN_DEPTH) ..."
    if ! sbatch --export=ALL,CHAIN_DEPTH=\$next "\$THIS_SCRIPT"; then
        echo "[chain] sbatch failed -- chain broken for ${member}." >&2
    fi
}

# Slurm sends SIGTERM at walltime (and 60s before, via --signal=B:SIGTERM@60),
# then SIGKILL after KillWait. Resubmit before bash exits; do NOT exit from
# the trap so the post-srun path also runs (resubmit_chain is idempotent
# via RESUBMITTED).
trap 'echo "[chain] caught SIGTERM"; resubmit_chain' SIGTERM
trap 'echo "[chain] caught SIGINT";  resubmit_chain' SIGINT

# Bail fast if a previous chunk already finished the rollout.
if [[ -f "\$DONE_FLAG" ]]; then
    echo "[chain] \$DONE_FLAG already present -- rollout complete. Exiting."
    exit 0
fi

export OMP_NUM_THREADS=1

# Conda's deactivate hooks reference unset vars (e.g. CONDA_MKL_INTERFACE_LAYER_BACKUP);
# relax -u around module/conda so they don't trip set -u.
set +u
module purge
ml cudatoolkit
module load pytorch-conda/2.8
conda activate torch
set -u

cd "${REPO_ROOT}"

echo "[chain] starting eval_lagged.py for seed ${seed} (chain depth \$CHAIN_DEPTH / \$MAX_CHAIN_DEPTH)"

srun python eval_lagged.py \\
    --config=${CONFIG} \\
    --start_date=${START_DATE} \\
    --end_date=${END_DATE} \\
    --output_root=${OUTPUT_ROOT} \\
    --member_name=${member} \\
    --seed=${seed} \\
    --spectral_monitor \\
    --spectral_lag_days=10 \\
    --spectral_threshold=0.5
PYRC=\$?
echo "[chain] eval_lagged.py exited with code \$PYRC"

if [[ \$PYRC -eq 0 ]]; then
    mkdir -p "\$MEMBER_DIR"
    touch "\$DONE_FLAG"
    echo "[chain] rollout complete -- wrote \$DONE_FLAG"
else
    resubmit_chain
fi
EOF

    chmod +x "$script"
    echo "Generated $script  (seed=${seed})"

    if [[ $DRY_RUN -eq 0 ]]; then
        sbatch "$script"
    fi
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. Inspect ${OUT_DIR}/ then re-run without --dry-run to submit."
fi
