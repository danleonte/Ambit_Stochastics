# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:56:44 2022

@author: dleon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,\
                            geninvgauss,poisson 
                            
from ambit_stochastics.trawl import trawl                            

#from fit_distr_for_tests import fit_trawl_distribution_aux 
#from ambit_stochastics import trawl
#trawl simulation tests

def check_values(trawl_,simulations_to_use=[1,2,-2,-1]):
        
        if trawl_.nr_trawls > 5000:
            values = trawl_.values[:,:500]
            times  = trawl_.tau * np.arange(1,500+1,1)

        else:
            values, times = trawl_.values, trawl_.tau * np.arange(1,trawl_.nr_trawls+1,1)

        
        f,ax = plt.subplots(2,2,figsize = (24,20),sharex= True)
        ax[0,0].plot(times, values[simulations_to_use[0]])
        ax[0,1].plot(times, values[simulations_to_use[1]])
        ax[1,0].plot(times, values[simulations_to_use[2]])
        ax[1,1].plot(times, values[simulations_to_use[3]])
        ax[1,0].set_xlabel('time')
        ax[1,1].set_xlabel('time')
        f.suptitle('Sample paths of the trawl process')

def check_acf(trawl_, simulation_to_use=0, lags=20):
        """plot_acf produces a horizontal line at y-0, can t figure out how to eliminate it from the plot"""
        values = trawl_.values[simulation_to_use]
        times  = trawl_.tau * np.arange(1,trawl_.nr_trawls+1,1)


        fig_acf, ax_acf = plt.subplots(1,1,figsize=(12,6))
        plot_acf(values, lags = lags-1, ax=ax_acf, color = 'blue', label='empirical')
        ax_acf.set_xlabel('lag')
        x = np.arange(1,lags,1)
        y = trawl_.theoretical_acf(np.arange(1,lags,1)*trawl_.tau)

        ax_acf.scatter(x,y.values(),marker = "*", color = 'r',s = 300,alpha = 0.5,label='theoretical')
        ax_acf.legend()
    
def check_trawl_slice(trawl_slice):
        check_values(trawl_slice)
        check_acf(trawl_slice,simulation_to_use = 1,lags=20)
        check_acf(trawl_slice,simulation_to_use = 7,lags=20)
        check_acf(trawl_slice,simulation_to_use = 12,lags=20)
        check_acf(trawl_slice,simulation_to_use = -5,lags=20)

def check_trawl_gaussian_part(trawl_):
    a = [norm.fit(data = trawl_.gaussian_values[simulation,:]) for simulation in range(trawl_.nr_simulations)]
    total_area = quad(trawl_.trawl_function,a=-np.inf,b=0)[0]
    a = np.array(a) / np.array([total_area, total_area ** 0.5])
    
    f,ax= plt.subplots(1,2,sharey=True, tight_layout=True)
    ax[0].hist(a[:,0],density=True)
    ax[0].set_title('infered means and true value')
    
    ax[1].hist(a[:,1],density=True)
    ax[1].set_title('infered sd and true value')
    
    ax[0].axvline(x=trawl_.gaussian_part_params[0],color='r')
    ax[1].axvline(x=trawl_.gaussian_part_params[1],color='r')
    
def check_trawl_jump_part_distribution(trawl_):
    total_area = quad(trawl_.trawl_function,a=-np.inf,b=0)[0]
    
    if trawl_.jump_part_name == 'gamma':
        a = [gamma.fit(data = simulation,floc=0) for simulation in trawl_.jump_values]
        a = np.array([[i[0],i[2]] for i in a])   #a, scale
        a = a / np.array([total_area,1])
        
        f,ax= plt.subplots(1,2,sharey=True, tight_layout=True)
        ax[0].hist(a[:,0],density=True)
        ax[0].set_title('infered means and true value')
        
        ax[1].hist(a[:,1],density=True)
        ax[1].set_title('infered scale and true value')
        
        ax[0].axvline(x=trawl_.jump_part_params[0],color='r')
        ax[1].axvline(x=trawl_.jump_part_params[1],color='r')
        
    else:
        raise ValueError('not yet implemented')
        
    

    
        

    
if __name__ == "__main__":
    tau = 0.15
    nr_trawls = 1000 
    nr_simulations = 50
    trawl_function = lambda x :   2*(1-x)**(-3) * (x<=0)
    #trawl_function= lambda x :  (x> -2) * (x<=0) * (2 - (-x) **2/2) 
    #decorrelation_time =-2
    gaussian_part_params = (-3,7)
    jump_part_params = (2,3)
    jump_part_name   = 'gamma'
    decorrelation_time = -np.inf
    #mesh_size = 0.05
    #truncation_grid = -2
    #times_grid =  tau * np.arange(1,nr_trawls+1,1) #important to keep it this way
    
    trawl_slice = trawl(nr_trawls = nr_trawls, nr_simulations = nr_simulations,
                   trawl_function = trawl_function,tau =  tau,decorrelation_time =  decorrelation_time, 
                   gaussian_part_params = gaussian_part_params,
                   jump_part_name =  jump_part_name,jump_part_params = jump_part_params )   
    

    

    #trawl_grid  = trawl(nr_trawls = nr_trawls, nr_simulations = nr_simulations,
    #               trawl_function = trawl_function,times_grid=times_grid,
    #               mesh_size = mesh_size,truncation_grid = truncation_grid,
    #               gaussian_part_params = gaussian_part_params,
    #               jump_part_name =  jump_part_name,jump_part_params = jump_part_params )
    
    print('started')
    trawl_slice.simulate(method='slice')
    print('finished')
    #trawl_grid.simulate(method='grid')
    
    check_trawl_slice(trawl_slice)
    #check_trawl(trawl_grid)
    check_trawl_gaussian_part(trawl_slice)
    check_trawl_jump_part_distribution(trawl_slice)
    #check_trawl_jump_part_distribution(trawl_grid)
    
    
    
