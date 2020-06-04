#!/bin/bash
exitfn(){
  trap SIGINT
  scancel $1
  scancel $2
  scancel $3
  scancel $4
}

[[ $NUM_SIMS ]] || export NUM_SIMS=20000
jobno1=$(sbatch delfi.sbatch --export=NUM_SIMS=$NUM_SIMS |cut -d' ' -f4)
jobno2=$(sbatch -d afterok:$jobno1 delfi-gpu.sbatch --export=NUM_SIMS=$NUM_SIMS |cut -d' ' -f4)
export EXP_FNAME="$HOME/delfi/beta_event23.txt"
jobno3=$(sbatch -d afterok:$jobno2 delfi-analyze.sbatch --export=EXP_FNAME=$EXP_FNAME --export=NUM_SIMS=$NUM_SIMS |cut -d' ' -f4)
export EXP_FNAME="$HOME/delfi/beta_event43_scaled.txt"
jobno4=$(sbatch -d afterok:$jobno2 delfi-analyze.sbatch --export=EXP_FNAME=$EXP_FNAME --export=NUM_SIMS=$NUM_SIMS |cut -d' ' -f4)

trap "exitfn $jobno1 $jobno2 $jobno3 $jobno4" INT; 

until [ -e slurm-${jobno1}.out ]; do sleep 1; done

( until [ -e slurm-${jobno2}.out ]; do sleep 1; done ;tail -f slurm-${jobno2}.out ) &

tail -f slurm-${jobno1}.out;

trap SIGINT
