# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:54:18 2022

@author: dleon
"""

import numpy as np
from ambit_stochastics.trawl import trawl
from ambit_stochastics.helpers.sampler import jump_part_sampler

if __name__ == "__main__":

    nr_trawls = 300; tau = 0.25; nr_simulations = 2
    #jump_part_name = 'poisson';  jump_part_params = 1 #L^' ~ Poisson(poisson_intensity)
    jump_part_name = 'gamma'; jump_part_params = (2,3)
    lambda_ = 1
    trawl_function = lambda t: np.exp(lambda_ * t) * (t<=0) / lambda_ #trawl function
    t_kernel = lambda t: 1+ np.sin(t/5)
    
    #nr_simulations below has nothing to do with the number of simulations
    #of the kernel weighted process
    trawl_slice = trawl(nr_trawls = nr_trawls, nr_simulations = 2,  tau =  tau,
                        gaussian_part_params = (0,0), jump_part_name = jump_part_name,
                        jump_part_params = (jump_part_params,),
                        trawl_function = trawl_function)   
    
    trawl_slice.compute_slice_areas_infinite_decorrelation_time()
 
    #initialize result
    result = np.zeros([nr_simulations,nr_trawls]) 
    for i in range(nr_trawls):
        for j in range(nr_trawls):
            if i+j <= nr_trawls-1:
                
                trawl_indicator = np.arange(start = j, stop = i+j+1, step = 1)
                
                area =  trawl_slice.slice_areas_matrix[i,j]
                area =  area * t_kernel((trawl_indicator+1) * tau)
                area = np.repeat(area[np.newaxis,:],axis=0,repeats=nr_simulations)
                
                result[:,trawl_indicator] += jump_part_sampler(jump_part_params  =jump_part_params,
                                      areas = area, distr_name   = jump_part_name)
                
                
                
                
                
            
        
            
    