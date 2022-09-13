# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:39:38 2022

@author: dleon
"""
from scipy.stats import norm,gamma,nbinom
from ambit_stochastics.trawl import trawl
import numpy as np
import pickle
import time

def set_truncation_grid(delta):
    truncation_grid = -7

    return truncation_grid

def simulate_and_infer_with_the_grid_method(tau, nr_trawls, nr_simulations,jump_part_params,jump_part_name, delta, truncation_grid):
    
    trawl_function_grid = lambda x :  np.exp(x) * (x<=0) * (x>=truncation_grid)
    times_grid =  tau * np.arange(1,nr_trawls+1,1)#doesn't have to be equally spaced, but have to be strictly increasing

    trawl_grid  = trawl(tau = tau, nr_simulations = nr_simulations,trawl_function = trawl_function_grid,
                   times_grid=times_grid,mesh_size = delta,truncation_grid = truncation_grid,
                   gaussian_part_params = (0,0), jump_part_name =  jump_part_name,
                   jump_part_params = jump_part_params )

    trawl_grid.simulate(method='grid')
    trawl_grid.fit_gmm(input_values = trawl_grid.values, envelope ='exponential', levy_seed = 'gamma',
                       lags = (1,3,5),initial_guess=None)
    
    return trawl_grid.infered_parameters['params']
    
    
if __name__ == "__main__":
   
    tau = 0.15
    nr_trawls = 1000 #500
    nr_simulations = 1000 #250
    jump_part_params = (2,0.5)
    jump_part_name   = 'gamma'
    #delta_vector    =  [0.025, 0.05, 0.075, 0.1]
    delta_vector     =  [0.01,0.025,0.05,0.075,0.1] 
    np.random.seed(148239)

    
    
    
    
    for delta in delta_vector:
        
        start = time.time()
        truncation_grid = set_truncation_grid(delta)
        results = simulate_and_infer_with_the_grid_method(tau, nr_trawls, nr_simulations,jump_part_params,
                                                jump_part_name, delta, truncation_grid)
        
        with open(f'delta_{delta}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        
        end = time.time()
        print('delta = ',str(delta),'took ',(end - start)/60,'minutes')
        
        


#with open('delta_005.pickle', 'rb') as handle:
#    b = pickle.load(handle)

       

    
    