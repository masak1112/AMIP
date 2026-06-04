#!/bin/bash
# Generate per-epoch bias.sh scripts for checkpoints ep04..ep19 of
# SI_X_forcings_42_2026-05-01T09-46-57 and submit each via qsub.
#
# Usage:
#   ./submit_bias_sweep.sh           # generate + submit
#   ./submit_bias_sweep.sh --dry-run # generate only, do not qsub

set -euo pipefail

CKPT_DIR="/glade/campaign/univ/uchi0014/ayz/SI_X_forcings_smooth_42_2026-05-06T16-46-36"
CONFIG="configs/combined_NCAR_2.yaml"
OUT_DIR="bias_sweep_heun"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$OUT_DIR"

for ep in 32 34 36 38; do
    ckpt="${CKPT_DIR}/model_epoch=${ep}.ckpt"
    desc="ep_${ep}_forcing_smooth_5_heun"
    script="${OUT_DIR}/bias_ep_${ep}.sh"

    cat > "$script" <<EOF
#!/bin/bash -l
#PBS -N bias_ep_${ep}_heun
#PBS -l select=1:ncpus=8:ngpus=1:mem=128G
#PBS -q develop
#PBS -l walltime=04:00:00
#PBS -A UCHI0018
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
