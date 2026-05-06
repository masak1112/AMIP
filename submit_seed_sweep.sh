#!/bin/bash
# Generate per-seed bias.sh scripts for N seeds and submit each via qsub.
# The model checkpoint is taken from configs/combined_NCAR.yaml.
#
# Usage:
#   ./submit_seed_sweep.sh                  # default: seeds 1..5, submit
#   ./submit_seed_sweep.sh 10               # seeds 1..10, submit
#   ./submit_seed_sweep.sh 10 --dry-run     # generate only, do not qsub
#   ./submit_seed_sweep.sh --dry-run        # default N, generate only

set -euo pipefail

CONFIG="configs/combined_NCAR.yaml"
OUT_DIR="bias_sweep"
N=10

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        ''|*[!0-9]*) ;;
        *) N=$arg ;;
    esac
done

mkdir -p "$OUT_DIR"

for seed in $(seq 1 "$N"); do
    desc="seed_${seed}"
    script="${OUT_DIR}/bias_seed_${seed}.sh"

    cat > "$script" <<EOF
#!/bin/bash -l
#PBS -N bias_seed_${seed}
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=${CONFIG}

python bias.py --config=\$CONFIG --seed=${seed} --description="${desc}"
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
