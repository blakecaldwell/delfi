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

from os import environ
import os.path as op
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Analyze HNN simulations.')
parser.add_argument('num_sims', metavar='NUM', type=int, default=20000,
                   help='number of first round simulations to train on')
args = parser.parse_args()
pilot_samples = args.num_sims

import matplotlib.pyplot as plt
from datetime import date

task_index = 0
import delfi.distribution as dd
from priors import set_prior_direct, fitted_values1, param_dict


from json import load

params_fname = environ['PARAMS_FNAME']
exp_fname = environ['EXP_FNAME']
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
            width = 5
            prominence = 50
            while len(troughs) == 0:
                troughs, _ = scipy.signal.find_peaks(-x['data'],width=width, prominence=prominence)
                if prominence < 2:
                    print("failed to find trough")
                    troughs = None
                    break
                else:
                    prominence -= 1

            if len(troughs) > 1:
                # pick the lowest one
                index = np.where(troughs == troughs.max())
                temp = np.array([ troughs[index] ])
                troughs = temp
                 
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



            dt = 130/len(x['data'])
            t = np.linspace(0, 130.0, num=len(x['data']))
            plt.plot(t, x['data'], '--',lw=1, label='observation')
            plt.xlabel('time (ms)')
            plt.ylabel('Dipole (nAm)')
            slope1=slope2=slope3=slope4=amplitude=sharpness=None
            if len(lpeaks) > 0 and len(rpeaks) > 0 and len(troughs) > 0:
                plt.vlines([lpeaks[0]*dt, floor(lp_props['left_ips'][0])*dt, rpeaks[0]*dt, troughs[0]*dt, ceil(rp_props['right_ips'][0])*dt],x['data'].min(),x['data'].max())
                plt.savefig('/users/bcaldwe2/scratch/delfi/obs_%d_%s.png' % (obs_index, exp_data_prefix))

                slope1=(x['data'][lpeaks[0]]-x['data'][floor(lp_props['left_ips'][0])])/(dt*(lpeaks[0] - floor(lp_props['left_ips'][0])))
                slope2=(x['data'][troughs[0]]-x['data'][lpeaks[0]])/(dt*(troughs[0]-lpeaks[0]))
                slope3=(x['data'][rpeaks[0]]-x['data'][troughs[0]])/(dt*(rpeaks[0]-troughs[0]))
                slope4=(x['data'][ceil(rp_props['right_ips'][0])]-x['data'][rpeaks[0]])/(dt*(ceil(rp_props['right_ips'][0])-rpeaks[0]))
                print("Slope 1: %f" % (slope1))
                print("Slope 2: %f" % (slope2))
                print("Slope 3: %f" % (slope3))
                print("Slope 4: %f" % (slope4))

                amplitude = max(x['data'][lpeaks[0]],x['data'][rpeaks[0]]) - x['data'][troughs[0]]
                sharpness = amplitude / (.5 * dt * (rpeaks[0]-lpeaks[0]))
            #t = x['time']
            #rmse = x['rmse']

            sum_stats_vec = np.concatenate((
                    np.array([x['numspikes']]),
                    np.array([slope1,slope2,slope3,slope4]),
                    np.array([amplitude,sharpness])
                ))
            print(sum_stats_vec)
            stats.append(sum_stats_vec)

        return np.asarray(stats)

def dpl_rejection_kernel(dpl):
    if dpl.max() > 50 or dpl.min() < -100:
        return 0
    else:
        return 1

# seeds
seed_m = 1

m = []
s = HNNStats()
#for i in range(n_processes):
#    m.append(HNN(seed=seeds_m[i]))
#g = dg.MPGenerator(models=m, prior=prior, summary=s, rej=dpl_rejection_kernel)

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
#obs_stats = [ exp_data[exp_start_index:exp_end_index,1] ]
obs = {'data': exp_data[exp_start_index:exp_end_index,1], 'numspikes':0}
obs_stats = s.calc([obs])

#obs_stats = [ np.concatenate((exp_data[:,1],[10])) ]
seed_inf = 1

#pilot_samples = None

dim_per_t = 1
n_steps = len(obs_stats[0])
n_params = n_steps * dim_per_t
density='maf'
proposal='prior'

print("Master loading trained inference object")

import pickle
inference_file = open('/users/bcaldwe2/scratch/delfi/%s_%d_trained_inference.pickle'%(params_input['sim_prefix'], pilot_samples), 'rb')
res=pickle.load(inference_file)

from delfi.utils.data import combine_trn_datasets
trn_data = combine_trn_datasets(res.trn_datasets)
n_train = trn_data[0].shape[0]

print("Master computing posterior")
res.obs = obs_stats
posterior = res.predict(res.obs)

#log, training_data, posterior = res.run(
#                    n_train=n_train,
#                    n_rounds=n_rounds,
#                    minibatch=minibatch,
#                    epochs=epochs,
#                    silent_fail=False,
#                    proposal=proposal,
#                    val_frac=val_frac,
#                    train_on_all=True,
#                    verbose=True,)

#dataset = 0
#params = np.transpose(np.array(training_data[dataset][0]))
#data = np.transpose(np.array(training_data[dataset][1]))

#print("Master saving results of simulations")
#np.savez_compressed('/users/bcaldwe2/scratch/delfi/%s_training_params-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *params)
#np.savez_compressed('/users/bcaldwe2/scratch/delfi/%s_training_data-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *data)
#print("done saving.")

posterior_samples = posterior.gen(1000)

#np.savez_compressed('/users/bcaldwe2/scratch/delfi/%s_posterior-samples-%d-%s'%(params_input['sim_prefix'], pilot_samples,date.today().isoformat()), *posterior_samples)

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

print("Master genarting posterior plots...")
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
#                       points=[fitted_params],
                       points_offdiag={'markersize': 5},
                       points_colors=[col['GT']],
                       title='');


plt.savefig('/users/bcaldwe2/scratch/delfi/%s_%s_%s_ntrain_%d_%s-posterior.eps' % (params_input['sim_prefix'], exp_data_prefix, density, n_train, proposal))

fig = plt.figure(figsize=(7,5))

y_obs = obs_stats[0]
t = np.linspace(0, 130.0, num=len(y_obs))
duration = np.max(t)

#num_samp = 2

# get mean from posterior
#x_mean = posterior.mean

num_samp = 2

# sample from posterior
x_samp = posterior.gen(n_samples=num_samp)

# simulate and plot samples
V = np.zeros((len(t),num_samp))
for i in range(num_samp):
    for idx, param in enumerate(x_samp[i,:]):
        if x_samp[i,idx] > prior_max[idx] or x_samp[i,idx] < prior_min[idx]:
            x_samp[i,idx] = (prior_min[idx] + prior_max[idx])/2
    print("Params: ", x_samp[i,:])
    print("master running simulation %d/%d" % (i+1,num_samp))
    x = m[0].gen_single(x_samp[i,:])
    V[:,i] = x['data']
    plt.plot(t, V[:, i], color = col['SAMPLE'+str(i+1)], lw=2, label='sample '+str(num_samp-i))

#x = m[0].gen_single(x_mean)
#plt.plot(t, x['data'], color = col['SNPE'], lw=2, label='posterior mean')

# plot observation
plt.plot(t, y_obs, '--',lw=2, label='observation')
plt.xlabel('time (ms)')
plt.ylabel('Dipole (nAm)')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), loc='upper right')

plt.savefig('/users/bcaldwe2/scratch/delfi/%s_%s_ntrain_%d-observation.png' % (params_input['sim_prefix'], exp_data_prefix, n_train))

print("Master done. Closing")
