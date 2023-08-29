# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:59:43 2022

@author: dleon
"""

"""p
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats  import poisson
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()




def kernel(x_bar,t_bar):
    return np.floor(2 *  (1+x_bar) *  (t_bar - np.floor(t_bar)) )

def generate_cpp_points(min_x,max_x,min_t,max_t,cpp_intensity):
    
    area_times_intensity = (max_x-min_x)*(max_t-min_t) * cpp_intensity  
    nr_points = poisson.rvs(mu = area_times_intensity)
    points_x = np.random.uniform(low = min_x, high = max_x, size = nr_points)
    points_t = np.random.uniform(low = min_t, high = max_t, size = nr_points)
    
    return points_x,points_t

if __name__ == '__main__':
    np.random.seed(2342345)
    tau = 0.01; nr_trawls = 250*50; nr_simulations = 2;
    times = np.arange(tau, (nr_trawls+1) * tau, tau)
    #trawl_function
    T = -2; 
    trawl_function = lambda t : (1-t/T) * (t >= T) * (t<=0)
    jump_part_params = (2,)
    result = np.zeros([nr_simulations,nr_trawls])
    
    min_t = tau + T
    max_t = tau * nr_trawls
    min_x =  0
    max_x = trawl_function(0)
    
    cpp_intensity = jump_part_params[0]


    
    for simulation_nr in range(nr_simulations):
        
    
        points_x, points_t = generate_cpp_points(min_x = min_x, max_x = max_x, 
                    min_t = min_t, max_t = max_t, cpp_intensity = cpp_intensity)
        
        associated_values = np.zeros(len(points_x))
        
        for i in range(len(points_x)):
            associated_values[i] +=  kernel(points_x[i],points_t[i])
                                
        #(x_i,t_i) in A_t if t < t_i and x_i < phi(t_i-t)
        indicator_matrix = np.tile(points_x[:,np.newaxis],(1,nr_trawls)) < \
                        trawl_function(np.subtract.outer(points_t, times))
        result[simulation_nr,:] = associated_values @ indicator_matrix
        #5) add back the drift: integral 0f a *  exp(-b*x)) between jump_T and 1
        #drift = alpha/beta * ( 1 - np.exp(-beta) )
        f,ax = plt.subplots()
        ax.plot(tau * np.arange(1,nr_trawls+1), result[simulation_nr],linewidth=1)
        #ax.set_title()
        plt.savefig('seaborn_style_poisson_simulation_nr_' + str(simulation_nr) + '.png', bbox_inches='tight')
        
        

    
    
                        