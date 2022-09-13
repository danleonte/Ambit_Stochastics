# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:24:15 2022

@author: dleon
"""
import numpy as np
x = 0.25
tau = 1
k_s = 50
k_t = 25 
nr_simulations = 2 
trawl_function= lambda t : np.exp(t/3) * (t <=0)
decorrelation_time=-np.inf
gaussian_part_params= (2,3)
jump_part_name= 'gamma'
jump_part_params= (2,3)
batch_size=5 * 10**3
total_nr_samples= 10**5

self=simple_ambit_field(x, tau, k_s, k_t, nr_simulations, trawl_function, decorrelation_time,\
                   gaussian_part_params, jump_part_name, jump_part_params,\
                   batch_size, total_nr_samples)
    
self.simulate()