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

# Initializations and preliminaries
comm = MPI.COMM_SELF   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
name = MPI.Get_processor_name()
import sys
sys.stderr.write("starting %s with rank %d" % (name, rank))

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

def set_prior(include_weights, input_names):

    import delfi.distribution as dd

    timing_weight_bound = 5.00
    prior_min = []
    prior_max = []

    for name in input_names:
        param_input_name = 't_%s' % name
        input_times = { param_input_name: float(params_input[param_input_name]) }

        for k,v in input_times.items():
            input_name = k.split('t_', 1)[1]

            if 'timing_only' in include_weights or 'timing_and_weights' in include_weights:
                sigma_name = 'sigma_%s' % param_input_name
                sigma = params_input[sigma_name]
                timing_min = max(0, v - 1*sigma)
                #timing_min = 0
                timing_max = min(float(params_input['tstop']), v + 1*sigma)
                #timing_max = float(params_input['tstop'])
                #print("Varying %s in range[%.4f-%.4f]" % (k, timing_min, timing_max))
                #prior_min.append(timing_min)
                #prior_max.append(timing_max)
                sigma_min = max(0, sigma - sigma * .5)
                sigma_max = sigma + sigma * .5
                #sigma_min = 0
                #sigma_max = 10
                print("Varying %s in range[%.4f-%.4f]" % (sigma_name, sigma_min, sigma_max))
                prior_min.append(sigma_min)
                prior_max.append(sigma_max)
            if 'weights_only' in include_weights or 'timing_and_weights' in include_weights:
                for weight in ['L2Pyr_ampa', 'L5Pyr_ampa']:
                #               'L2Pyr_nmda', 'L2Basket_nmda', 'L5Pyr_nmda', 'L5Basket_nmda']:
                    timing_weight_name = "gbar_%s_%s"%(input_name, weight)
                    try:
                        timing_weight_value = float(params_input[timing_weight_name])
                        if np.isclose(timing_weight_value, 0., atol=1e-5):
                            weight_min = 0.
                            weight_max = 1.
                        else:
                            weight_min = max(0, timing_weight_value - timing_weight_value * timing_weight_bound)
                            weight_max = timing_weight_value
                        #weight_min = 0.
                        #weight_max = 1.
                        print("Varying %s in range[%.4f-%.4f]" % (timing_weight_name, weight_min, weight_max))
                        prior_min.append(weight_min)
                        prior_max.append(weight_max)
                    except KeyError:
                        pass

                for weight in ['L2Basket_ampa', 'L5Basket_ampa']:
                #               'L2Pyr_nmda', 'L2Basket_nmda', 'L5Pyr_nmda', 'L5Basket_nmda']:
                    timing_weight_name = "gbar_%s_%s"%(input_name, weight)
                    try:
                        timing_weight_value = float(params_input[timing_weight_name])
                        if np.isclose(timing_weight_value, 0., atol=1e-5):
                            weight_min = 0.
                            weight_max = 1.
                        else:
                            weight_min = max(0, timing_weight_value - timing_weight_value * timing_weight_bound)
                            weight_max = timing_weight_value + timing_weight_value * timing_weight_bound
                        #weight_min = 0.
                        #weight_max = 1.
                        print("Varying %s in range[%.4f-%.4f]" % (timing_weight_name, weight_min, weight_max))
                        prior_min.append(weight_min)
                        prior_max.append(weight_max)
                    except KeyError:
                        pass

    prior = dd.Uniform(lower=np.asarray(prior_min), upper=np.asarray(prior_max), seed=2)
    return prior


print("Master starting on %s" % name)

from json import load
print("Master started")

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
prior =  set_prior(include_weights, input_names)
params_input['sim_prefix'] = "%s%s_%s" % (op.basename(params_fname).split('.json')[0], input_name, include_weights)

simdata = (exp_data, params_input)
print("Master has finished loading file data. Sending to the workers.")

# broadcast simdata to all of the workers
comm.bcast(simdata, root=0)

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
                  '../mne-neuron/examples/calculate_dipole_err_delfi_beta.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input)
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
        dim_param = 9 

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
from scipy import stats as spstats

class HNNStats(BaseSummaryStats):
    """
    Calculates summary statistics
    """
    def __init__(self, n_summary=104, seed=None):
        """See SummaryStats.py for docstring"""
        super(HNNStats, self).__init__(seed=seed)
        self.n_summary = n_summary

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
        stats = []
        for r in range(len(repetition_list)):
            x = repetition_list[r]

            N = x['data'].shape[0]
            #t = x['time']
            #rmse = x['rmse']

            sum_stats_vec = np.concatenate((
                    np.array(x['data']),
             #       rmse
                ))

            stats.append(sum_stats_vec)

        return np.asarray(stats)

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
seeds_m = np.arange(1,n_processes+1,1)
for i in range(n_processes):
    m.append(HNN(seed=seeds_m[i]))
g = dg.MPGenerator(models=m, prior=prior, summary=ds.Identity, rej=dpl_rejection_kernel)

# true parameters and respective labels
# fitted_params = np.array([26.61, 0.01525, 0, 0.08831, 0, 0.00865, 0, 0.19934, 0, 63.53, 0.000007, 0.004317, 0.006562, 0.019482, 0.1423, 0.080074, 137.12, 1.43884, 0, 0.000003, 0, 0.684013, 0, 0.008958, 0])
#fitted_params = np.array([35.134283, 0.000039, 0.00003, 0.000033, 0.000022, 6.818029, 0.000103, 0.000091, 0.000055 ])
fitted_params = np.array([60, 0.00045, 0.000086, 0.00045, 0.000259, 5.862318, 0.00047, 0.000088, 0.000778 ])
labels_params = [ 'sigma_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L2Basket_ampa', 'gbar_evprox_1_L5Pyr_ampa', 'gbar_evprox_1_L5Basket_ampa', 'sigma_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L2Basket_ampa', 'gbar_evdist_1_L5Pyr_ampa' ]
# labels_params = [ 't_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L2Pyr_nmda', 'gbar_evprox_1_L2Basket_ampa', 'gbar_evprox_1_L2Basket_nmda', 'gbar_evprox_1_L5Pyr_ampa', 'gbar_evprox_1_L5Pyr_nmda', 'gbar_evprox_1_L5Basket_ampa', 'gbar_evprox_1_L5Basket_nmda', 't_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L2Pyr_nmda', 'gbar_evdist_1_L2Basket_ampa', 'gbar_evdist_1_L2Basket_nmda', 'gbar_evdist_1_L5Pyr_ampa', 'gbar_evdist_1_L5Pyr_nmda', 't_evprox_2', 'gbar_evprox_2_L2Pyr_ampa', 'gbar_evprox_2_L2Pyr_nmda', 'gbar_evprox_2_L2Basket_ampa', 'gbar_evprox_2_L2Basket_nmda', 'gbar_evprox_2_L5Pyr_ampa', 'gbar_evprox_2_L5Pyr_nmda', 'gbar_evprox_2_L5Basket_ampa', 'gbar_evprox_2_L5Basket_nmda' ]

# observed data: simulation given true parameters
#obs = m[0].gen_single(fitted_params)
#obs_stats = s.calc([obs])
#obs_stats = [ exp_data[:,1] ]

tstart = 90.0
tstop = 220
exp_times = exp_data[:,0]
exp_start_index = (np.abs(exp_times - tstart)).argmin()
exp_end_index = (np.abs(exp_times - tstop)).argmin()
obs_stats = [ exp_data[exp_start_index:exp_end_index,1] ]
obs_stats[0] = np.insert(obs_stats[0], 0, [0.0])
#obs_stats = [ np.concatenate((exp_data[:,1],[10])) ]
seed_inf = 1

pilot_samples = 50000
#pilot_samples = None


# network setup
n_hiddens = [50,50]

# convenience
#prior_norm = True

dim_per_t = 1
n_steps = len(obs_stats[0])
n_params = n_steps * dim_per_t

inf_setup_opts = dict(density='mog', maf_mode='random', n_mades=5,
                      n_components=3,
                      n_rnn=5*n_params, input_shape=(n_steps, dim_per_t)) 

print("Master starting inference on %d nodes" % n_nodes)
# inference object
res = infer.SNPEC(g,
                obs=obs_stats,
                n_hiddens=n_hiddens,
                seed=seed_inf,
                pilot_samples=pilot_samples,
                **inf_setup_opts)

# training schedule
n_train = pilot_samples
n_rounds = 1
n_atoms = 100
minibatch = 100
proposal='prior'
# fitting setup
#epochs = 100
epochs = 100
val_frac = 0.05
 
print("Master starting training on %d cores" % n_nodes)
# train
log, training_data, posterior = res.run(
                    n_train=n_train,
                    n_rounds=n_rounds,
                    minibatch=minibatch,
                    epochs=epochs,
                    silent_fail=False,
                    proposal=proposal,
                    val_frac=val_frac,
                    train_on_all=True,
                    verbose=True,)

print("Master saving results of simulations")
dataset = 0
params = np.transpose(np.array(training_data[dataset][0]))
data = np.transpose(np.array(training_data[dataset][1]))

np.savez_compressed('../scratch/delfi/%s_training_params-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *params)
np.savez_compressed('../scratch/delfi/%s_training_data-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *data)
print("done saving.")

posterior_samples = posterior[0].gen(10000)

np.savez_compressed('../scratch/delfi/%s_posterior-samples-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *posterior_samples)
print("Closing workers")

num_workers = n_nodes
closed_workers = 0

while closed_workers < num_workers:
    data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    source = status.Get_source()
    tag = status.Get_tag()

    if tag == tags.READY:
        comm.send(None, dest=source, tag=tags.EXIT)
    elif tag == tags.EXIT:
        print("Worker %d exited (%d running)" % (source, closed_workers))
        closed_workers += 1

print("Master has finished simulations")

fig = plt.figure(figsize=(15,5))
plt.plot(log[0]['loss'],lw=2)
plt.xlabel('iteration')
plt.ylabel('loss');
plt.savefig('../scratch/delfi/%s_ntrain_%d-loss.png' % (params_input['sim_prefix'], n_train))

from delfi.utils.viz import samples_nd

prior_min = g.prior.lower
prior_max = g.prior.upper
prior_lims = np.concatenate((prior_min.reshape(-1,1),prior_max.reshape(-1,1)),axis=1)

###################
# colors
hex2rgb = lambda h: tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# RGB colors in [0, 255]
col = {}
col['GT']      = hex2rgb('30C05D')
col['SNPE']    = hex2rgb('2E7FE8')
col['SAMPLE1'] = hex2rgb('8D62BC')
col['SAMPLE2'] = hex2rgb('AF99EF')

# convert to RGB colors in [0, 1]
for k, v in col.items():
    col[k] = tuple([i/255 for i in v])

print("Master calculating posterior...") 
###################
# posterior
fig, axes = samples_nd(posterior_samples,
                       limits=prior_lims,
                       ticks=prior_lims,
                       labels=labels_params,
                       fig_size=(35,35),
                       diag='kde',
                       upper='kde',
                       hist_diag={'bins': 50},
                       hist_offdiag={'bins': 50},
                       kde_diag={'bins': 50, 'color': col['SNPE']},
                       kde_offdiag={'bins': 50},
                       points=[fitted_params],
                       points_offdiag={'markersize': 5},
                       points_colors=[col['GT']],
                       title='');

plt.savefig('../scratch/delfi/%s_ntrain_%d-posterior.png' % (params_input['sim_prefix'], n_train))

fig = plt.figure(figsize=(7,5))

y_obs = obs_stats[0]
t = np.linspace(0, 130.0, num=len(y_obs))
duration = np.max(t)

#num_samp = 2

# get mean from posterior
x_mean = posterior[0].mean

num_samp = 2

# sample from posterior
x_samp = posterior[0].gen(n_samples=num_samp)
print(x_samp)
# reject samples for which prior is zero
ind = (x_samp > prior_min) & (x_samp < prior_max)
params = x_samp[np.prod(ind,axis=1)==1]

num_samp = len(params[:,0])
print("left with %d samples"%num_samp) 

# simulate and plot samples
V = np.zeros((len(t),num_samp))
for i in range(num_samp):
    #x = m[0].gen_single(params[i,:])
    x = m[0].gen_single(x_samp[i,:])
    V[:,i] = x['data']
    plt.plot(t, V[:, i], color = col['SAMPLE'+str(i+1)], lw=2, label='sample '+str(num_samp-i))

x = m[0].gen_single(x_mean)
plt.plot(t, x['data'], color = col['SNPE'], lw=2, label='posterior mean')

# plot observation
plt.plot(t, y_obs, '--',lw=2, label='observation')
plt.xlabel('time (ms)')
plt.ylabel('Dipole (nAm)')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), loc='upper right')

plt.savefig('../scratch/delfi/%s_ntrain_%d-observation.png' % (params_input['sim_prefix'], n_train))

print("Master done. Closing")
