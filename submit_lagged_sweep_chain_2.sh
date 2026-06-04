#!/bin/bash
# Generate per-start-date PBS scripts for a lagged ensemble on the *develop*
# queue (6h walltime, 8-job concurrent cap) and daisy-chain them: each job
# resubmits itself when eval_lagged.py exits non-zero (crash, OOM, or PBS
# walltime SIGTERM), so a multi-day rollout completes across several 6h
# chunks. eval_lagged.py auto-resumes from rollout_checkpoint.pt, so chained
# jobs pick up where the previous chunk left off.
#
# Steady-state queue usage: 8 running, 0 queued (each finishing job spawns
# exactly one successor), which stays inside the develop-queue 8-job cap.
#
# Usage:
#   ./submit_lagged_sweep_chain.sh             # generate + submit
#   ./submit_lagged_sweep_chain.sh --dry-run   # generate scripts, do not qsub

set -euo pipefail

CONFIG="configs/combined_NCAR.yaml"
END_DATE="2025-01-01"
OUTPUT_ROOT="/glade/campaign/univ/uchi0014/ayz/rollouts_new/"
OUT_DIR="lagged_sweep"

MAX_CHAIN_DEPTH=7

START_DATES=(
    "1980-03-22"
    "1980-04-01"
    "1980-04-11"
    "1980-04-21"
    "1980-05-01"
    "1980-05-11"
    "1980-05-21"
    "1980-05-31"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"

for sd in "${START_DATES[@]}"; do
    member="member_${sd//-/}"
    script="${OUT_DIR}/lagged_${member}.sh"
    abs_script="${REPO_ROOT}/${script}"
    member_dir="${OUTPUT_ROOT%/}/${member}"

    cat > "$script" <<EOF
#!/bin/bash -l
#PBS -N lagged_${member}
#PBS -l select=1:ncpus=8:ngpus=1:mem=192G
#PBS -q main
#PBS -l walltime=6:00:00
#PBS -A UCHI0018
#PBS -j oe

set -u
cd "${REPO_ROOT}"

MEMBER_DIR="${member_dir}"
DONE_FLAG="\${MEMBER_DIR}/.lagged_done"
THIS_SCRIPT="${abs_script}"
MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH}
CHAIN_DEPTH="\${CHAIN_DEPTH:-1}"
RESUBMITTED=0

resubmit_chain() {
    if (( RESUBMITTED == 1 )); then return 0; fi
    RESUBMITTED=1
    if [[ -f "\$DONE_FLAG" ]]; then
        echo "[chain] \$DONE_FLAG present — rollout complete, not resubmitting."
        return 0
    fi
    if (( CHAIN_DEPTH >= MAX_CHAIN_DEPTH )); then
        echo "[chain] hit MAX_CHAIN_DEPTH=\$MAX_CHAIN_DEPTH — not resubmitting. Bump the cap if more chunks are needed."
        return 0
    fi
    local next=\$((CHAIN_DEPTH + 1))
    echo "[chain] resubmitting (chain depth \$next / \$MAX_CHAIN_DEPTH) ..."
    if ! qsub -v "CHAIN_DEPTH=\$next" "\$THIS_SCRIPT"; then
        echo "[chain] qsub failed — chain broken for ${member}." >&2
    fi
}

# PBS walltime expiry sends SIGTERM (kill_delay seconds before SIGKILL).
# Resubmit before bash exits; do NOT exit from the trap so the post-python
# path also runs (resubmit_chain is idempotent via RESUBMITTED).
trap 'echo "[chain] caught SIGTERM"; resubmit_chain' SIGTERM
trap 'echo "[chain] caught SIGINT";  resubmit_chain' SIGINT

# Bail fast if a previous chunk already finished the rollout.
if [[ -f "\$DONE_FLAG" ]]; then
    echo "[chain] \$DONE_FLAG already present — rollout complete. Exiting."
    exit 0
fi

module load conda
conda activate torch

echo "[chain] starting eval_lagged.py (chain depth \$CHAIN_DEPTH / \$MAX_CHAIN_DEPTH)"

python eval_lagged.py \\
    --config=${CONFIG} \\
    --start_date=${sd} \\
    --end_date=${END_DATE} \\
    --output_root=${OUTPUT_ROOT} \\
    --member_name=${member} \\
    --spectral_monitor \\
    --spectral_lag_days=10 \\
    --spectral_threshold=0.4
PYRC=\$?
echo "[chain] eval_lagged.py exited with code \$PYRC"

if [[ \$PYRC -eq 0 ]]; then
    mkdir -p "\$MEMBER_DIR"
    touch "\$DONE_FLAG"
    echo "[chain] rollout complete — wrote \$DONE_FLAG"
else
    resubmit_chain
fi
EOF

    chmod +x "$script"
    echo "Generated $script"

    if [[ $DRY_RUN -eq 0 ]]; then
        qsub "$script"
    fi
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. Inspect ${OUT_DIR}/ then re-run without --dry-run to submit."
fi
