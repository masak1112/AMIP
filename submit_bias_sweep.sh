#!/bin/bash
# Generate per-epoch bias.sh scripts for checkpoints ep04..ep19 of
# SI_X_forcings_42_2026-05-01T09-46-57 and submit each via qsub.
#
# Usage:
#   ./submit_bias_sweep.sh           # generate + submit
#   ./submit_bias_sweep.sh --dry-run # generate only, do not qsub

set -euo pipefail

CKPT_DIR="/glade/derecho/scratch/ayz/AMIP_logs/SI_X_forcings_42_2026-05-03T19-47-56/"
CONFIG="configs/combined_NCAR.yaml"
OUT_DIR="bias_sweep"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$OUT_DIR"

for ep in $(seq -w 20 24); do
    ckpt="${CKPT_DIR}/model_epoch=${ep}.ckpt"
    desc="ep_${ep}_forcing"
    script="${OUT_DIR}/bias_ep_${ep}.sh"

    cat > "$script" <<EOF
#!/bin/bash -l
#PBS -N bias_ep_${ep}
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -A URIC0009
#PBS -j oe

module load conda
conda activate torch

CONFIG=${CONFIG}

python bias.py --config=\$CONFIG --checkpoint="${ckpt}" --description="${desc}"
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
