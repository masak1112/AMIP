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

# 37-member lagged ensemble: anchored at 1979-01-01 with 10-day spacing,
# spanning the full spinup year through 1979-12-27.
START_DATES=(
    "1979-01-01"
    "1979-01-11"
    "1979-01-21"
    "1979-01-31"
    "1979-02-10"
    "1979-02-20"
    "1979-03-02"
    "1979-03-12"
    "1979-03-22"
    "1979-04-01"
    "1979-04-11"
    "1979-04-21"
    "1979-05-01"
    "1979-05-11"
    "1979-05-21"
    "1979-05-31"
    "1979-06-10"
    "1979-06-20"
    "1979-06-30"
    "1979-07-10"
    "1979-07-20"
    "1979-07-30"
    "1979-08-09"
    "1979-08-19"
    "1979-08-29"
    "1979-09-08"
    "1979-09-18"
    "1979-09-28"
    "1979-10-08"
    "1979-10-18"
    "1979-10-28"
    "1979-11-07"
    "1979-11-17"
    "1979-11-27"
    "1979-12-07"
    "1979-12-17"
    "1979-12-27"
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
