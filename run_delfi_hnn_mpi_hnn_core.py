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
import argparse

from priors import set_prior_direct, param_dict

parser = argparse.ArgumentParser(description='Run delfi on HNN simulations.')
parser.add_argument('num_sims', metavar='NUM', type=int, default=20000,
                   help='number of first round simulations to run')
args = parser.parse_args()

pilot_samples = int(args.num_sims)
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

if rank != 0:
    from statistics import mean
    import numpy as np
    from datetime import date
    import time
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    # start the workers

    print("Worker started with rank %d on %s." % (rank, name))

    # receive experimental data
    (exp_data, params_input) = comm.bcast(comm.Get_rank(), root=0)

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
                  '../hnn-core/examples/mpi_simulate_dipole_delfi.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input, list(param_dict.keys()))
    subcomm.bcast(simdata, root=MPI.ROOT)

    avg_sim_times = []

    #subcomm.Barrier()
    print("Worker %d waiting on master to signal start" % rank)
    # tell rank 0 we are ready
    comm.send(None, dest=0, tag=tags.READY)
    sleeping = False

    while True:
        # Start clock
        #start = MPI.Wtime()

        # Receive updated params (blocking)
        new_params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        if tag == tags.EXIT:
            print('worker %d on %s has received exit signal'%(rank, name))
            break
        elif tag == tags.SLEEP:
            if not sleeping:
              comm.send(None, dest=0, tag=tags.SLEEP)
              sleeping = True
            print("Worker %d sleeping for master to signal start" % rank)
            time.sleep(5)
            # tell rank 0 we are ready
            comm.send(None, dest=0, tag=tags.READY)
            continue

        sleeping = False
        #assert(tag == tags.START)

        #finish = MPI.Wtime() - start
        #print('worker %s waited %.2fs for param set' % (name, finish))

        # Start clock
        start = MPI.Wtime()

        # send new_params to spawned nrniv procs
        subcomm.bcast(new_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(2)])
        temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (name, finish, mean(avg_sim_times)))
  
        agg_dpl = temp_results[0]
        times = np.linspace(0, 130.0, num=len(agg_dpl))
        sim_params = temp_results[1]
        spikedata = sim_params[0]
        result_index = sim_params[2]
        #np.savetxt('/users/bcaldwe2/scratch/delfi/sim_dipoles/%s_sim_%d.txt'%(params_input['sim_prefix'], result_index),(times,agg_dpl))

        #plt.plot(np.linspace(0, 130.0, num=len(agg_dpl)), agg_dpl)
        #plt.savefig('/users/bcaldwe2/scratch/delfi/sim_plots/%s_sim_%d_rmse_%.2f.png' % (params_input['sim_prefix'], result_index,rmse))
        #plt.close()
 
        # send results back without task_index
        data = (temp_results[0], [new_params[:-1]])
        comm.send(data, dest=0, tag=tags.DONE)

        # tell rank 0 we are ready (again)
        comm.send(None, dest=0, tag=tags.READY)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

    # send empty new_params to stop nrniv procs
    subcomm.bcast(None, root=MPI.ROOT)
    #subcomm.Barrier()

if rank == 0:
    import matplotlib.pyplot as plt
    from datetime import date

    task_index = 0

    import delfi.distribution as dd
    def set_prior(include_weights, input_names):


        timing_weight_bound = 10.00
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
                    sigma_min = max(0, sigma - sigma * 2.0)
                    sigma_max = sigma + sigma * 2.0
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
                                weight_max = timing_weight_value + timing_weight_value * timing_weight_bound
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
        import hnn_core 
        hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
        params_fname = op.join(hnn_core_root, 'param', 'default.json')
        print("using default param file:", params_fname)

    if environ['EXP_FNAME'] and op.exists(environ['EXP_FNAME']):
        exp_fname = environ['EXP_FNAME']
    else:
        import hnn_core
        hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')
        exp_fname = op.join(hnn_core_root, 'yes_trial_S1_ERP_all_avg.txt')
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
    prior_min, prior_max = set_prior_direct()
    prior = dd.Uniform(lower=np.asarray(prior_min), upper=np.asarray(prior_max), seed=2)
    # prior =  set_prior(include_weights, input_names)
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
                      '../hnn-core/examples/mpi_simulate_dipole_delfi.py'],
                info = mpiinfo, maxprocs=n_procs)
    
        # send params and exp_data to spawned nrniv procs
        simdata = (exp_data, params_input, list(param_dict.keys()))
        subcomm.bcast(simdata, root=MPI.ROOT)

        # send new_params to spawned nrniv procs
        temp_params = np.append(params, self.task_index)
        subcomm.bcast(temp_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(2)])
        data = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        agg_dpl = data[0]
        sim_params = data[1]
        spikedata = sim_params[0]
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
#            dim_param = 28
            dim_param = len(prior_min)
    
            super().__init__(dim_param=dim_param, seed=seed)
            self.HNNsimulator = HNNsimulator

            # begin task index for generator after all pilot_samples    
            self.task_index=pilot_samples

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
            params = np.append(np.asarray(params), self.task_index)
            assert params.ndim == 1, 'params.ndim must be 1'
    
            states = self.HNNsimulator(params.reshape(1, -1))

            return {'data': states.reshape(-1)}
#                    'rmse': rmse}
    
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
#    fitted_params = np.array([26.61, 0.01525, 0, 0.08831, 0, 0.00865, 0, 0.19934, 0, 63.53, 0.000007, 0.004317, 0.006562, 0.019482, 0.1423, 0.080074, 137.12, 1.43884, 0, 0.000003, 0, 0.684013, 0, 0.008958, 0])
    #fitted_params = np.array([35.134283, 0.000039, 0.00003, 0.000033, 0.000022, 6.818029, 0.000103, 0.000091, 0.000055 ])
    fitted_params = np.array([60, 0.00045, 0.000086, 0.00045, 0.000259, 5.862318, 0.00047, 0.000088, 0.000778 ])
    #fitted_params = np.array([36.604943, 0.000038, 0.000031, 0.000031, 0.000022, 6.988911, 0.000099, 0.000092, 0.00005])
    labels_params = [ 'sigma_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L2Basket_ampa', 'gbar_evprox_1_L5Pyr_ampa', 'gbar_evprox_1_L5Basket_ampa', 'sigma_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L2Basket_ampa', 'gbar_evdist_1_L5Pyr_ampa' ]
#    labels_params = [ 't_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L2Pyr_nmda', 'gbar_evprox_1_L2Basket_ampa', 'gbar_evprox_1_L2Basket_nmda', 'gbar_evprox_1_L5Pyr_ampa', 'gbar_evprox_1_L5Pyr_nmda', 'gbar_evprox_1_L5Basket_ampa', 'gbar_evprox_1_L5Basket_nmda', 't_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L2Pyr_nmda', 'gbar_evdist_1_L2Basket_ampa', 'gbar_evdist_1_L2Basket_nmda', 'gbar_evdist_1_L5Pyr_ampa', 'gbar_evdist_1_L5Pyr_nmda', 't_evprox_2', 'gbar_evprox_2_L2Pyr_ampa', 'gbar_evprox_2_L2Pyr_nmda', 'gbar_evprox_2_L2Basket_ampa', 'gbar_evprox_2_L2Basket_nmda', 'gbar_evprox_2_L5Pyr_ampa', 'gbar_evprox_2_L5Pyr_nmda', 'gbar_evprox_2_L5Basket_ampa', 'gbar_evprox_2_L5Basket_nmda' ]
    
    # observed data: simulation given true parameters
    #obs = m[0].gen_single(fitted_params)
    #obs_stats = s.calc([obs])
    #obs_stats = [ exp_data[:,1] ]

    tstart = 90.0
    tstop = params_input['tstop']
    exp_times = exp_data[:,0]
    exp_start_index = (np.abs(exp_times - tstart)).argmin()
    exp_end_index = (np.abs(exp_times - tstop)).argmin()
    obs_stats = [ exp_data[exp_start_index:exp_end_index,1] ]
    obs_stats[0] = np.insert(obs_stats[0], 0, [0.0])
    #obs_stats = [ np.concatenate((exp_data[:,1],[10])) ]
    seed_inf = 1
   
    #pilot_samples = None
    
   
    # network setup
    n_hiddens = [50,50]
    
    # convenience
    #prior_norm = True
 
    dim_per_t = 1
    n_steps = len(obs_stats[0])
    n_params = n_steps * dim_per_t
 
    inf_setup_opts = dict(density='maf', maf_mode='random', n_mades=5,
                          n_components=3,
                          n_rnn=5*n_params, input_shape=(n_steps, dim_per_t)) 
    
    print("Master starting inference on %d workers" % n_nodes)
    # inference object
    res = infer.SNPEC(g,
                    obs=obs_stats,
                    n_hiddens=n_hiddens,
                    seed=seed_inf,
                    pilot_samples=pilot_samples,
                    **inf_setup_opts)
   

    print("Master saving inference object")
    import pickle
    with open('../scratch/delfi/%s_%d_inference.pickle'%(params_input['sim_prefix'], pilot_samples), 'wb') as inference_file:
        pickle.dump(res, inference_file)

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

    print("Master done. Closing")
