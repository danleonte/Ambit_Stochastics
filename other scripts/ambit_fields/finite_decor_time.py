# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:30:09 2022

@author: dleon
"""

from ambit_stochastics.simple_ambit_field import simple_ambit_field
import matplotlib.pyplot as plt
import numpy as np

def plot_simple_ambit_field(x,t,values,simulation_to_use):
    x0 = np.linspace(x, k_s * x, k_s)
    t0 = np.linspace(t, k_t * t, k_t)
    T0, X0 = np.meshgrid(t0, x0)

    plt.contourf(T0, X0, values[simulation_to_use], 20, cmap='cividis')
    plt.colorbar();
    #plt.title('Simple ambit field')
    plt.savefig('simple_ambit_field_simulation_vertical' + str(simulation_to_use),bbox_inches='tight')
    

if __name__ == '__main__':
    #np.random.seed(24325) #for horizontal
    np.random.seed(4384359) #for vertical
    x   = 0.2
    tau = 0.2
    k_s = 100
    k_t = 100
    nr_simulations = 2
    decorrelation_time = -5
    #ambit_function = lambda t: (0>=t) * (t>=-5)*  (1+t/5)   #horizontal
    ambit_function = lambda t: 5 *(0>=t) * (t>=-1)*  (1+t)  #vertical
    
    simple_ambit_field_instance = simple_ambit_field(x = x, tau = tau, k_s = k_s, k_t = k_t,
                 nr_simulations = nr_simulations, ambit_function = ambit_function,
                 decorrelation_time = decorrelation_time, gaussian_part_params= (0,0),
                 jump_part_name = 'gamma', jump_part_params = (2,3),
                 batch_size= 10**5, total_nr_samples=  10**8)
    
    simple_ambit_field_instance.simulate()
    plot_simple_ambit_field(x,tau,simple_ambit_field_instance.values,0)
    #plot_simple_ambit_field(x,tau,simple_ambit_field_instance.values,1)


