#!/bin/bash
exitfn(){
  trap SIGINT
  scancel $1
}


[[ $EXP_FNAME ]] || export EXP_FNAME="$HOME/delfi/beta_event23.txt"
[[ $NUM_SIMS ]] || export NUM_SIMS=1000000
jobno=$(sbatch delfi-round.sbatch --export=EXP_FNAME=$EXP_FNAME --export=NUM_SIMS=$NUM_SIMS|cut -d' ' -f4)

trap "exitfn $jobno" INT; 

until [ -e slurm-${jobno}.out ]; do sleep 1; done

tail -f slurm-${jobno}.out;

trap SIGINT
