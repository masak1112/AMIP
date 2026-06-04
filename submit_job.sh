#!/bin/bash

# Resolve this wrapper's directory and submit SLURM scripts using absolute paths

# Submit the first job and capture its job ID
JOB1=$(qsub  -W depend=afterany:6345814 "/glade/work/bgong/amip/develop_S2S.sh")
echo "Submitted job 1 with Job ID:$JOB1"

cd /glade/work/bgong/amip/

# Submit the second job with dependency on the first
JOB2=$(qsub -W depend=afterany:$JOB1 "/glade/work/bgong/amip/develop_S2S.sh")
echo "Submitted job 2 with Job ID: $JOB2, dependent on Job ID:$JOB1"


# cd /glade/work/bgong/ReForecastNet/AIFS_ENS

# Submit the third job with dependency on the second
JOB3=$(qsub -W depend=afterany:$JOB2 "/glade/work/bgong/amip/develop_S2S.sh")
echo "Submitted job 3 with Job ID: $JOB3, dependent on Job ID:$JOB2"


JOB4=$(qsub -W depend=afterany:$JOB3 "/glade/work/bgong/amip/develop_S2S.sh")

echo "Submitted job 4 with Job ID: $JOB4, dependent on Job ID:$JOB3"
# JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 "aifs_ens_job_2021.sh")
# echo "Submitted job 4 with Job ID: $JOB4, dependent on Job ID:$JOB3"

# JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 "aifs_ens_job_2020.sh")
# echo "Submitted job 5 with Job ID: $JOB5, dependent on Job ID:$JOB4"
JOB5=$(qsub -W depend=afterany:$JOB4 "/glade/work/bgong/amip/develop.sh")
echo "Submitted job 5 with Job ID: $JOB5, dependent on Job ID:$JOB4"

JOB6=$(qsub -W depend=afterany:$JOB5 "/glade/work/bgong/amip/develop.sh")
echo "Submitted job 6 with Job ID: $JOB6, dependent on Job ID:$JOB5"