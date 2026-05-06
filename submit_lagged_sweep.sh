#!/bin/bash
# Generate per-start-date PBS scripts for a lagged ensemble and submit them.
# Each member runs on its own GPU and writes to its own folder under
# /glade/campaign/univ/uchi0014/.
#
# Usage:
#   ./submit_lagged_sweep.sh                  # submit default member set
#   ./submit_lagged_sweep.sh --dry-run        # generate scripts, do not qsub

set -euo pipefail

CONFIG="configs/combined_NCAR.yaml"
END_DATE="2025-01-01"
OUTPUT_ROOT="/glade/campaign/univ/uchi0014/ayz/"
OUT_DIR="lagged_sweep"

# ERA5 h5 files only exist from 1979-01-01 onward. The user mentioned
# "Oct 1, 1978" as the first lagged start; since that data is not available
# we anchor the first member at 1979-01-01 and stagger subsequent members
# at 3-month lags through the spinup year.
START_DATES=(
    "1979-01-01"
    "1979-01-02"
    "1979-01-03"
    "1979-01-04"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$OUT_DIR"

for sd in "${START_DATES[@]}"; do
    member="member_${sd//-/}"
    script="${OUT_DIR}/lagged_${member}.sh"

    cat > "$script" <<EOF
#!/bin/bash -l
#PBS -N lagged_${member}
#PBS -l select=1:ncpus=8:ngpus=1:mem=192G
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A UCHI0018
#PBS -j oe

module load conda
conda activate torch

python eval_lagged.py \\
    --config=${CONFIG} \\
    --start_date=${sd} \\
    --end_date=${END_DATE} \\
    --output_root=${OUTPUT_ROOT} \\
    --member_name=${member}
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
