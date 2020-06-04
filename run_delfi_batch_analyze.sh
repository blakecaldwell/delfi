#!/bin/bash
exitfn(){
  trap SIGINT
  scancel $1
}


[[ $EXP_FNAME ]] || export EXP_FNAME="$HOME/delfi/beta_event43_scaled.txt"
[[ $NUM_SIMS ]] || export NUM_SIMS=50000
jobno=$(sbatch delfi-analyze.sbatch --export=EXP_FNAME=$EXP_FNAME --export=NUM_SIMS=$NUM_SIMS|cut -d' ' -f4)

trap "exitfn $jobno" INT; 

until [ -e slurm-${jobno}.out ]; do sleep 1; done

tail -f slurm-${jobno}.out;

trap SIGINT
