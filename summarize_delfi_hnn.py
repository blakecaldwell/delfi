"""
=========================================
Delfi on dipole simulation
=========================================

This example performs parameter inference on dipole simulations
using the Delfi package

MPI master/worker task scheme used from:
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Run with:
mpiexec -np 4 python examples/run_delfi_hnn_mpi.py
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from mpi4py import MPI
from os import environ
import os.path as op
import numpy as np
import theano
import argparse
parser = argparse.ArgumentParser(description='Run delfi on HNN simulations.')
parser.add_argument('num_sims', metavar='NUM', type=int, default=20000,
                   help='number of first round simulations to train on')
args = parser.parse_args()
pilot_samples = args.num_sims

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
name = MPI.Get_processor_name()

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'SLEEP', 'EXIT', 'START')

import matplotlib.pyplot as plt
from datetime import date

task_index = 0
import delfi.distribution as dd
from priors import set_prior_direct, fitted_values1, param_dict


from json import load

# Parse command-line arguments
if environ['PARAMS_FNAME'] and op.exists(environ['PARAMS_FNAME']):
    params_fname = environ['PARAMS_FNAME']
else:
    import mne_neuron
    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    print("using default param file:", params_fname)

if environ['EXP_FNAME'] and op.exists(environ['EXP_FNAME']):
    exp_fname = environ['EXP_FNAME']
else:
    import mne_neuron
    mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
    exp_fname = op.join(mne_neuron_root, 'yes_trial_S1_ERP_all_avg.txt')
    print("using default experimental data:", exp_fname)

exp_data_prefix = op.splitext(op.basename(exp_fname))[0]

print("Master loading file data")
# Read the dipole data and params files once
exp_data = np.loadtxt(exp_fname)
with open(params_fname) as json_data:
    params_input = load(json_data)

input_names = []
input_name = ''
if 'INPUT_NAME_1' in environ:
    input_names.append(environ['INPUT_NAME_1'])
    input_name = input_name + '_' + environ['INPUT_NAME_1']
if 'INPUT_NAME_2' in environ:
    input_names.append(environ['INPUT_NAME_2'])
    input_name = input_name + '_' + environ['INPUT_NAME_2']
if 'INPUT_NAME_3' in environ:
    input_names.append(environ['INPUT_NAME_3'])
    input_name = input_name + '_' + environ['INPUT_NAME_3']

include_weights = environ['INCLUDE_WEIGHTS']
prior_min, prior_max = set_prior_direct()
prior = dd.Uniform(lower=np.asarray(prior_min), upper=np.asarray(prior_max), seed=2)
params_input['sim_prefix'] = "%s%s_%s" % (op.basename(params_fname).split('.json')[0], input_name, include_weights)

simdata = (exp_data, params_input)

# broadcast simdata to all of the workers
#comm.bcast(simdata, root=0)

if 'SLURM_NNODES' in environ:
    n_nodes = max(1, size - 1)
else:
    n_nodes = 1

def HNNsimulator(params):
    # number of processes to run nrniv with
    if 'SLURM_CPUS_ON_NODE' in environ:
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = 1

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('host', name.split('.')[0])
    mpiinfo.Set('ompi_param', 'rmaps_base_inherit=0')
    mpiinfo.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpiinfo.Set('ompi_param', 'rmaps_base_oversubscribe=1')
    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn('nrniv',
            args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                  '/users/bcaldwe2/mne-neuron/examples/calculate_dipole_err_delfi_beta.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input, list(param_dict.keys()))
    subcomm.bcast(simdata, root=MPI.ROOT)

    # send new_params to spawned nrniv procs
    temp_params = np.append(params, pilot_samples)
    subcomm.bcast(temp_params, root=MPI.ROOT)

    # wait to recevie results from child rank 0
    #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
    #                         np.zeros(2)])
    data = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    agg_dpl = data[0]
    sim_params = data[1]
    rmse = sim_params[0]
    result_index = sim_params[2]

    # send empty new_params to stop nrniv procs
    subcomm.bcast(None, root=MPI.ROOT)

    return agg_dpl.reshape(-1,1)

from delfi.simulator.BaseSimulator import BaseSimulator

class HNN(BaseSimulator):
    def __init__(self, seed=None):
        """ 

        Parameters
        ----------
        """
        #dim_param = 28
        dim_param = len(prior_min)

        super().__init__(dim_param=dim_param, seed=seed)
        self.HNNsimulator = HNNsimulator

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        from numpy import linspace
        from scipy import signal

        print(params.reshape(1, -1))
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'

        states = self.HNNsimulator(params.reshape(1, -1))

        return {'data': states.reshape(-1)}
                #'rmse': rmse}

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
import scipy.signal
class HNNStats(BaseSummaryStats):
    """
    Calculates summary statistics
    """
    def __init__(self, seed=None):
        """See SummaryStats.py for docstring"""
        super(HNNStats, self).__init__(seed=seed)

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """

        from math import ceil, floor
        stats = []
        for obs_index, r in enumerate(range(len(repetition_list))):
            x = repetition_list[r]
            N = x['data'].shape[0]
            numspikes = x['numspikes']
            troughs = []
            slope1=slope2=slope3=slope4=amplitude=sharpness=np.nan

            width = 5
            prominence = 50
            while len(troughs) == 0:
                troughs, _ = scipy.signal.find_peaks(-x['data'],width=width, prominence=prominence)
                if prominence < 2:
                    #print("failed to find trough")
                    troughs = []
                    break
                else:
                    prominence -= 1

            if len(troughs) > 1:
                # pick the lowest one
                index = np.where(troughs == troughs.max())
                temp = np.array(troughs[index])
                troughs = temp
            else:
                sum_stats_vec = np.concatenate((
                    np.array([x['numspikes']]),
                    np.array([slope1,slope2,slope3,slope4]),
                    np.array([amplitude,sharpness])
                ))
                stats.append(sum_stats_vec)
                continue


            # find peak 1
            width=4
            height=1
            lpeaks, lp_props = scipy.signal.find_peaks(x['data'][0:troughs[0]],height=height,width=width)
            while True:
                old_lpeaks = lpeaks
                old_lp_props = lp_props
                lpeaks, lp_props = scipy.signal.find_peaks(x['data'][0:troughs[0]],height=height,width=width)
                if len(lpeaks) == 0:
                    lpeaks = old_lpeaks
                    lp_props = old_lp_props
                    break
                else:
                    height += 1

            # in case we need to use a lot of smoothing
            if len(lpeaks) == 0:
              lpeaks = scipy.signal.find_peaks_cwt(x['data'][0:troughs[0]], np.arange(10,20))
              lp_props = {'left_ips': [0]}

            # find peak 2
            width=4
            height=1
            right_data = np.array(x['data'])
            right_data[0:troughs[0]+1] = 0
            rpeaks, rp_props = scipy.signal.find_peaks(right_data,height=height,width=width)
            while True:
                old_rpeaks = rpeaks
                old_rp_props = rp_props
                rpeaks, rp_props = scipy.signal.find_peaks(right_data,height=height,width=width)
                if len(rpeaks) == 0:
                    rpeaks = old_rpeaks
                    rp_props = old_rp_props
                    break
                else:
                    height += 1

            # in case we need to use a lot of smoothing
            if len(rpeaks) == 0:
              rpeaks = scipy.signal.find_peaks_cwt(right_data, np.arange(10,20))
              rp_props = {'right_ips': [len(x['data'])-1]}

##
#            if len(peaks) == 1:
#                if peaks[0] > len(x['data'])/2.0:
#                  peaks = np.insert(peaks, 0,np.where(x['data'] == x['data'][0:round(len(x['data'])/2.0)].max())[0][0])
#                  p_props['left_ips'] = np.insert(p_props['left_ips'], 0, 0)
#                  p_props['right_ips'] = np.insert(p_props['right_ips'], 0, peaks[0])
#                elif peaks[0] < len(x['data'])/2.0:
#                  peaks = peaks.append(np.where(x['data'] == x['data'][round(len(x['data'])/2.0):-1].max())[0][0])
#                  p_props['left_ips'] = np.append(p_props['left_ips'], peaks[1])
#                  p_props['right_ips'] = np.append(p_props['right_ips'], len(x['data']))



            if len(lpeaks) > 0 and len(rpeaks) > 0 and len(troughs) > 0:
                dt = 130/len(x['data'])
#                t = np.linspace(0, 130.0, num=len(x['data']))
#                plt.plot(t, x['data'], '--',lw=1, label='simulation %d'%obs_index)
#                plt.plot(t, y_obs, '--',lw=2, label='observation')
#                plt.xlabel('time (ms)')
#                plt.ylabel('Dipole (nAm)')
#                plt.vlines([lpeaks[0]*dt, floor(lp_props['left_ips'][0])*dt, rpeaks[0]*dt, troughs[0]*dt, ceil(rp_props['right_ips'][0])*dt],x['data'].min(),x['data'].max())
#                plt.savefig('/users/bcaldwe2/scratch/delfi/obs_%d_%s.png' % (obs_index, exp_data_prefix))
#                plt.clf()

                slope1=(x['data'][lpeaks[0]]-x['data'][floor(lp_props['left_ips'][0])])/(dt*(lpeaks[0] - floor(lp_props['left_ips'][0])))
                slope2=(x['data'][troughs[0]]-x['data'][lpeaks[0]])/(dt*(troughs[0]-lpeaks[0]))
                slope3=(x['data'][rpeaks[0]]-x['data'][troughs[0]])/(dt*(rpeaks[0]-troughs[0]))
                slope4=(x['data'][ceil(rp_props['right_ips'][0])]-x['data'][rpeaks[0]])/(dt*(ceil(rp_props['right_ips'][0])-rpeaks[0]))
                #print("Slope 1: %f" % (slope1))
                #print("Slope 2: %f" % (slope2))
                #print("Slope 3: %f" % (slope3))
                #print("Slope 4: %f" % (slope4))

                amplitude = max(x['data'][lpeaks[0]],x['data'][rpeaks[0]]) - x['data'][troughs[0]]
                sharpness = amplitude / (.5 * dt * (rpeaks[0]-lpeaks[0]))
            #t = x['time']
            #rmse = x['rmse']

            sum_stats_vec = np.concatenate((
                    np.array([x['numspikes']]),
                    np.array([slope1,slope2,slope3,slope4]),
                    np.array([amplitude,sharpness])
                ))
            #print(sum_stats_vec)
            stats.append(sum_stats_vec)

        return np.asarray(stats)


def stat_rejection_kernel(stat):
    if stat is None:
        return 0
    else:
        return 1

def dpl_rejection_kernel(dpl):
    if dpl.max() > 50 or dpl.min() < -100:
        return 0
    else:
        return 1

import delfi.summarystats as ds
import delfi.generator as dg
import delfi.inference as infer

n_processes = n_nodes

# seeds
seed_m = 1

m = []
s = HNNStats()
seeds_m = np.arange(1,n_processes+1,1)
for i in range(n_processes):
    m.append(HNN(seed=seeds_m[i]))
#g = dg.MPGenerator(models=m, prior=prior, summary=ds.Identity, rej=dpl_rejection_kernel)
g = dg.MPGenerator(models=m, prior=prior, summary=s, rej=stat_rejection_kernel)

# true parameters and respective labels
#fitted_params = np.array([26.61, 0.01525, 0, 0.08831, 0, 0.00865, 0, 0.19934, 0, 63.53, 0.000007, 0.004317, 0.006562, 0.019482, 0.1423, 0.080074, 137.12, 1.43884, 0, 0.000003, 0, 0.684013, 0, 0.008958, 0])
#fitted_params = np.array([35.134283, 0.000039, 0.00003, 0.000033, 0.000022, 6.818029, 0.000103, 0.000091, 0.000055 ])
fitted_params = np.array(list(fitted_values1.values()))
labels_params = list(fitted_values1.keys())
#labels_params = [ 't_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L2Pyr_nmda', 'gbar_evprox_1_L2Basket_ampa', 'gbar_evprox_1_L2Basket_nmda', 'gbar_evprox_1_L5Pyr_ampa', 'gbar_evprox_1_L5Pyr_nmda', 'gbar_evprox_1_L5Basket_ampa', 'gbar_evprox_1_L5Basket_nmda', 't_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L2Pyr_nmda', 'gbar_evdist_1_L2Basket_ampa', 'gbar_evdist_1_L2Basket_nmda', 'gbar_evdist_1_L5Pyr_ampa', 'gbar_evdist_1_L5Pyr_nmda', 't_evprox_2', 'gbar_evprox_2_L2Pyr_ampa', 'gbar_evprox_2_L2Pyr_nmda', 'gbar_evprox_2_L2Basket_ampa', 'gbar_evprox_2_L2Basket_nmda', 'gbar_evprox_2_L5Pyr_ampa', 'gbar_evprox_2_L5Pyr_nmda', 'gbar_evprox_2_L5Basket_ampa', 'gbar_evprox_2_L5Basket_nmda' ]

# observed data: simulation given true parameters
#obs = m[0].gen_single(fitted_params)
#obs_stats = s.calc([obs])
#obs_stats = [ exp_data[:,1] ]

tstart = 90.0
tstop = 220
exp_times = exp_data[:,0]
exp_start_index = (np.abs(exp_times - tstart)).argmin()
exp_end_index = (np.abs(exp_times - tstop)).argmin()
#obs_stats = [ np.concatenate((exp_data[:,1],[10])) ]
y_obs = exp_data[exp_start_index:exp_end_index,1]
y_obs = np.insert(y_obs, 0, [0.0])
obs = {'data': exp_data[exp_start_index:exp_end_index,1], 'numspikes':0}
obs_stats = s.calc([obs])
seed_inf = 1

#pilot_samples = None


# network setup
#n_hiddens = [200,400]
n_hiddens = [50,50]

# convenience
#prior_norm = True

dim_per_t = 1
n_steps = len(obs_stats[0])
n_params = n_steps * dim_per_t
density='maf'
inf_setup_opts = dict(density=density, maf_mode='random', n_mades=5,
                      #n_components=3,
                      n_rnn=5*n_params, input_shape=(n_steps, dim_per_t)) 

import pickle
inference_file = open('/users/bcaldwe2/scratch/delfi/%s_%d_inference.pickle'%(params_input['sim_prefix'], pilot_samples), 'rb')
res=pickle.load(inference_file)

res.obs = obs_stats

# transform raw data to summary stats
n_train = len(res.unused_pilot_samples[1][:,0])
print("Starting with %d samples"%n_train)
data = []
for sample_idx in range(n_train):
  data.append({'data': res.unused_pilot_samples[1][sample_idx,:], 'numspikes': 0})

stats = s.calc(data)
params = res.unused_pilot_samples[0]

# from BaseInference.pilot_run()
res.stats_mean = np.nanmean(stats, axis=0)
res.stats_std = np.nanstd(stats, axis=0)
res.stats_std[res.stats_std == 0.0] = 1.0
min_std=1e-4
res.stats_std = np.maximum(res.stats_std, min_std)
ok_sims = np.logical_not(np.logical_or(np.isnan(stats).any(axis=1),
                                       np.isnan(params).any(axis=1)))
res.unused_pilot_samples = (params[ok_sims, :], stats[ok_sims, :])
n_train = len(res.unused_pilot_samples[1][:,0])
print("Left with %d samples"%n_train)

# inference object
#res2 = infer.SNPEC(g,
#                obs=obs_stats,
#                n_hiddens=n_hiddens,
#                seed=seed_inf,
#                pilot_samples=res.unused_pilot_samples,
#                **inf_setup_opts)



# training schedule
n_rounds = 1
n_atoms = 100
minibatch = 100
proposal='prior'
# fitting setup
#epochs = 100
epochs = 1000
val_frac = 0.05
 
print("Master starting training")
# train
res.kwargs.update({'n_inputs': stats.shape[1],
                   'n_outputs': params.shape[1],
                   'input_shape': (n_steps, dim_per_t),
                   'n_hiddens': n_hiddens,
                   'n_rnn': 5*n_params})

res.reinit_network()
log, training_data = res.run_round(n_train=n_train,
                                   minibatch=minibatch,
                                   epochs=epochs,
                                   proposal=proposal,
                                   val_frac=val_frac,
                                   train_on_all=True,
                                   verbose=True)

fig = plt.figure(figsize=(15,5))
plt.plot(log['loss'],lw=2)
plt.xlabel('iteration')
plt.ylabel('loss');
plt.savefig('/users/bcaldwe2/scratch/delfi/%s_ntrain_%d-loss.png' % (params_input['sim_prefix'], n_train))

print("Master saving trained inference object")
with open('../scratch/delfi/%s_%d_trained_summary_inference.pickle'%(params_input['sim_prefix'], pilot_samples), 'wb') as inference_file:
    pickle.dump(res, inference_file)

print("Master done. Closing")
