#!/bin/bash -l

SCRIPT=/glade/work/bgong/amip/develop_finetune_inference.sh

for seed in 0 1 2; do
	for year in $(seq 2019 2024); do
		qsub -v "SEED=${seed},YEAR=${year}" "$SCRIPT"
	done
done


