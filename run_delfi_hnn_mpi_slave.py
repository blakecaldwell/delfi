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
print("starting %s with rank %d\n" % (name, rank))
exit(0)
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'SLEEP', 'EXIT', 'START')

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
              '../mne-neuron/examples/calculate_dipole_err_delfi_beta.py'],
        info = mpiinfo, maxprocs=n_procs)

# send params and exp_data to spawned nrniv procs
simdata = (exp_data, params_input)
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
    rmse = sim_params[0]
    result_index = sim_params[2]
    print("rmse for sim %d is %f"%(result_index, rmse))
    np.savetxt('/users/bcaldwe2/scratch/delfi/sim_dipoles/%s_sim_%d.txt'%(params_input['sim_prefix'], result_index),(times,agg_dpl))

    plt.plot(np.linspace(0, 130.0, num=len(agg_dpl)), agg_dpl)
    plt.savefig('/users/bcaldwe2/scratch/delfi/sim_plots/%s_sim_%d_rmse_%.2f.png' % (params_input['sim_prefix'], result_index,rmse))
    plt.close()

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
