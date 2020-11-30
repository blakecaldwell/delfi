#!/bin/python

from os import environ
import os.path as op
import numpy as np

fitted_values1 = {
    "dipole_scalefctr": 200000,
    't_evprox_1': 141.0,
    'sigma_t_evprox_1': 17,
    'numspikes_evprox_1': 10,
    'gbar_evprox_1_L2Pyr_ampa': 0.0001,
    'gbar_evprox_1_L5Pyr_ampa': 0.0001,
    't_evdist_1': 149.0,
    'sigma_t_evdist_1': 12,
    'numspikes_evdist_1': 10,
    'gbar_evdist_1_L2Pyr_ampa': 0.0001,
    'gbar_evdist_1_L5Pyr_ampa': 0.0001
}

param_dict = {
    "dipole_scalefctr": (60000, 200000),
    't_evprox_1': (125,155),
    'sigma_t_evprox_1': (10,50),
    'numspikes_evprox_1': (1,20),
    'gbar_evprox_1_L2Pyr_ampa': (0.000001, 0.0005),
    'gbar_evprox_1_L5Pyr_ampa': (0.000001, 0.0005),
    't_evdist_1': (135,155),
    'sigma_t_evdist_1': (5,30),
    'numspikes_evdist_1': (1, 20),
    'gbar_evdist_1_L2Pyr_ampa': (0.000001, 0.0005),
    'gbar_evdist_1_L5Pyr_ampa': (0.000001, 0.0005)
}

def set_prior_direct():

    prior_min = []
    prior_max = []

    for name,value in param_dict.items():
        (min_value, max_value) = value
        prior_min.append(min_value)
        prior_max.append(max_value)

    return prior_min, prior_max
