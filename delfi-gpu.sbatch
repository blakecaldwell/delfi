#!/bin/bash
#SBATCH --mem=63g -p gpu --gres=gpu:1 --ntasks-per-node=16 -N 1
#SBATCH -t 24:00:00
#SBATCH -A carney-sjones-condo

#source activate mne3
module load cuda/10.1.105
source /users/bcaldwe2/anaconda/mne3/lib/python3.7/venv/scripts/common/activate
cd $HOME/delfi

#export OMPI_MCA_btl_openib_allow_ib=1
#export OMPI_MCA_mpi_warn_on_fork=0
#export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_rmaps_base_mapping_policy="node"
#export OMPI_MCA_btl_base_verbose=100
#export OMPI_MCA_rmaps_base_schedule_local=0
#export OMPI_MCA_rmaps_base_n_pernode=1
#export OMPI_MCA_mpi_yield_when_idle=0
#export OMPI_MCA_rmaps_base_oversubscribe=1
#export OMPI_MCA_rmaps_base_inherit=0
#export OMPI_MCA_btl="tcp,vader"
#export OMPI_MCA_btl="tcp"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/bcaldwe2/anaconda/mne3/lib
export PATH=/users/bcaldwe2/anaconda/mne3/bin:$PATH:$HOME/nrn/build/x86_64/bin
export PYTHONPATH=$HOME/nrn/build/lib/python:/users/bcaldwe2/anaconda/mne3/lib/python3.7/site-packages

# for openmpi-3.x
#export PATH=$PATH:$HOME/ompi-3/build/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ompi-3/build/lib
#export OMPI_MCA_routed="direct"

# for openmpi-4.x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ompi/build/lib
export PATH=$PATH:$HOME/ompi/build/bin
#export UCX_LOG_LEVEL=DEBUG
#export UCX_NET_DEVICES=mlx4_0:1
#export OMPI_MCA_pml="^ucx"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^tcp,vader,openib"
#export OMPI_MCA_btl="^uct,openib"
#export OMPI_MCA_btl="^vader,openib,uct,self"
export OMPI_MCA_routed="direct"

export CFLAGS=-I$HOME/cuda/include
export THEANO_FLAGS='device=cuda'

#export PARAMS_FNAME="051220_peribeta_basedonpostbeta16_6_opt_0.05_timestep_opt2_10ms_smoothing_opt.json"
#export EXP_FNAME="$HOME/mne-neuron/yes_trial_S1_ERP_all_avg.txt"
#export EXP_FNAME="$HOME/delfi/beta_event43_long.txt"
#export PARAMS_FNAME="$HOME/delfi/beta_event_220ms_2_inputs_30ms_smoothing_50trials_opt3_23.json"
export PARAMS_FNAME="$HOME/delfi/beta_event_220ms_2_inputs_15ms_smothing_3trials_multiple_spikes_23_scaling_200k.json"
[[ $EXP_FNAME ]] || EXP_FNAME="$HOME/delfi/beta_event43.txt"
export EXP_FNAME
export INPUT_NAME_1="evprox_1"
export INPUT_NAME_2="evdist_1"
#export INPUT_NAME_3="evprox_2"
export INCLUDE_WEIGHTS="timing_and_weights"
#mpiexec -np 1 --oversubscribe python3 train_delfi_hnn.py $NUM_SIMS
mpiexec -np 1 --oversubscribe python3 train_delfi_round_hnn.py $NUM_SIMS
