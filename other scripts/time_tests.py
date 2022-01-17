# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:57:54 2022

@author: dleon
"""
import time
import numpy as np
from simple_ambit_field import simple_ambit_field
import pickle 

x   = 0.15
tau = 0.2
max_lag_space = 7
max_lag_time = 6
total_nr_points =   10**7
ambit_function  = lambda x :np.exp(x) * (x<=0)


#sa.simulate()
pythonic_times_njit = []
for simulation_nr in range(10):

    
    
    sa = simple_ambit_field(x=x, tau=tau, k_s= 35, k_t=35, nr_simulations=20,
                 ambit_function=ambit_function, decorrelation_time=-np.inf,
                 gaussian_part_params=(1,1), jump_part_name='gamma', jump_part_params=(2,3),
                 batch_size=10**5, total_nr_samples=10**6, values=None)
    start = time.time()
    sa.simulate()
    end = time.time()
    pythonic_times_njit.append(end-start)

with open('pytohnic_times_list_with_jit.pkl', 'wb') as f:
   pickle.dump(pythonic_times_njit, f)
    
    