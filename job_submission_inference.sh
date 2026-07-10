#!/bin/bash -l

SCRIPT=/glade/work/bgong/amip/develop_finetune_inference.sh

for seed in 10 11 12 13 ; do
	for year in $(seq 2019 2022); do
		qsub -v "SEED=${seed},YEAR=${year}" "$SCRIPT"
	done
done


