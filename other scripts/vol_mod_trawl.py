# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:04:55 2022

@author: dleon
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ambit_stochastics.trawl import trawl
from ambit_stochastics.helpers.alternative_convolution_implementation import cumulative_and_diagonal_sums


def approx_integral(values,time_index,low_x,high_x,low_t,high_t,trawl_time):
    
    values = values[(low_t <= time_index) & (time_index <= high_t)]
    time_index = time_index[(low_t <= time_index) & (time_index <= high_t)]
    
    heights = high_x(time_index) - low_x(time_index)
    return np.mean(heights * values)

if __name__ == '__main__':
    sns.set_theme()
    np.random.seed(524)
    
    #vol_params
    tau_vol = 0.05
    nr_trawls_vol = 2800  #this is k
    nr_simulations_vol = 2  
    inv_gauss_params_vol = (2,1)
    trawl_function_vol   = lambda t :   0.5 * (1-t)**(-1.5) * (t<=0) 
    
    trawl_vol = trawl(nr_trawls = nr_trawls_vol, nr_simulations = nr_simulations_vol,
                        trawl_function = trawl_function_vol  , tau =  tau_vol,
                        gaussian_part_params = (0,0), jump_part_name = 'invgauss',
                        jump_part_params = inv_gauss_params_vol)   
    trawl_vol.simulate(method='slice')
    
    vol_squared_values = trawl_vol.values[0]
    vol_values = vol_squared_values**0.5
    time_index = np.linspace(-20,125,2800)

    
    #trawl params
    tau_trawl = 0.5
    nr_trawls_trawl = 250
    #nr_simulations_trawl = 2
    gaussian_part_params_trawl = (1,1)
    lambda_ = 0.25
    trawl_function_trawl = lambda t : lambda_ * np.exp(t * lambda_) * (t<=0)
    
    
    #trawl_trawl = trawl(nr_trawls = nr_trawls_trawl, nr_simulations = nr_simulations_trawl,
    #                    trawl_function = trawl_function_trawl  , tau =  tau_trawl,
    #                    gaussian_part_params = gaussian_part_params_trawl,
    #                    jump_part_name = None, jump_part_params = (0,0))   
    #trawl_trawl.simulate(method='slice')
    
    slice_matrix = np.zeros([nr_trawls_trawl,nr_trawls_trawl])
    

    
    for j in range(nr_trawls_trawl):
        if j %50 == 0:
            print(j)
        for i in range(nr_trawls_trawl):

            if j==0: 
                a = -np.inf
            else: 
                a = j * tau_trawl
            b = (j+1) * tau_trawl    
        
            if i+j+1 == nr_trawls_trawl:
                #gfun = 0
                #hfun = lambda t_bar : trawl_function(t_bar - k * tau)
                l_bound = lambda t: 0
                h_bound = lambda t: trawl_function_trawl(t - nr_trawls_trawl * tau_trawl)
                         
                 
            else:
                #gfun  = lambda t_bar : trawl_function(t_bar - (i+j+2)*tau)
                #hfun = lambda t_bar : trawl_function(t_bar - (i+j+1)*tau)
                l_bound = lambda t_bar: trawl_function_trawl(t_bar - (i+j+2)*tau_trawl)
                h_bound = lambda t_bar: trawl_function_trawl(t_bar - (i+j+1)*tau_trawl)
                
            integral_squared_vol = approx_integral(values = vol_squared_values,time_index = time_index,
                                low_x = l_bound,high_x = h_bound,
                                low_t = a ,high_t = b, trawl_time = j * tau_trawl)
                
            int_vol = approx_integral(values = vol_values,time_index = time_index,
                                low_x = l_bound,high_x = h_bound,
                                low_t = a ,high_t = b, trawl_time = j * tau_trawl)
            
            slice_matrix[i,j] = np.random.normal(int_vol*gaussian_part_params_trawl[0],
                                                 integral_squared_vol**0.5* gaussian_part_params_trawl[1])
            #to double check this
    result = cumulative_and_diagonal_sums(slice_matrix)
    
    f,ax = plt.subplots()
    ax.plot(tau_trawl * np.arange(1,nr_trawls_trawl+1), result, linewidth=1)
    plt.savefig('vol_modulated.png', bbox_inches='tight')
    
    
    f,ax = plt.subplots()
    vals_to_plot = vol_squared_values[(tau_trawl <= time_index) & (time_index <= tau_trawl * nr_trawls_trawl)]
    time_index_to_plot = time_index[(tau_trawl <= time_index) & (time_index <= tau_trawl * nr_trawls_trawl)] 
    ax.plot(time_index_to_plot, vals_to_plot, linewidth=1)
    plt.savefig('vol_squared.png', bbox_inches='tight')


    
#plt.savefig('all_slice_sim_together.png', bbox_inches='tight')


