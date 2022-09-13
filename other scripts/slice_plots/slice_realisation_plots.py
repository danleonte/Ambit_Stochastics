# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:29:31 2022

@author: dleon
"""
import numpy as np
from ambit_stochastics.trawl import trawl
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
np.random.seed(40)

tau = 0.5
nr_trawls = 250  #this is k
nr_simulations = 2

sm_trawl_function = lambda t :   np.exp(t) * (t<=0)  
lm_trawl_function = lambda t :   0.5 * (1-t)**(-1.5) * (t<=0)

func_dict = {'Short memory, light tails':{'trawl_function':sm_trawl_function,'gaussian_part_params':(0,1),
                                          'jump_part_name' : None, 'jump_part_params': (0,0)}, 
             
             'Short memory, heavy tails':{'trawl_function':sm_trawl_function,'gaussian_part_params':(0,0),
                                                       'jump_part_name' : 'cauchy', 'jump_part_params': (1,)},
             
             'Long memory, light tails':{'trawl_function':lm_trawl_function,'gaussian_part_params':(0,1),
                                                       'jump_part_name' : None, 'jump_part_params': (0,0)},
             
             'Long memory, heavy tails':{'trawl_function':lm_trawl_function,'gaussian_part_params':(0,0),
                                                       'jump_part_name' : 'cauchy', 'jump_part_params': (1,)}
             }

#f,ax = plt.subplots(nrows =2 ,ncols=2,figsize = (6.4*1.5,4.8*1.5))
#index_list = [(0,0),(0,1),(1,0),(1,1)]
#count = 0

for key,value in func_dict.items():
    f,ax = plt.subplots()
    #index = index_list[count]
    
    trawl_slice = trawl(nr_trawls = nr_trawls, nr_simulations = nr_simulations,
                        trawl_function = value['trawl_function'], tau =  tau,
                        gaussian_part_params = value['gaussian_part_params'],
                        jump_part_name =  value['jump_part_name'],jump_part_params = value['jump_part_params'])   
    trawl_slice.simulate(method='slice')
    
    #ax[index].plot(tau * np.arange(1,nr_trawls+1), trawl_slice.values[0],linewidth=1)
    #ax[index].set_title(key)
    ax.plot(tau * np.arange(1,nr_trawls+1), trawl_slice.values[0],linewidth=1)
    ax.set_title(key)
    plt.savefig(key + '.png', bbox_inches='tight')
    #count += 1

    
#plt.savefig('all_slice_sim_together.png', bbox_inches='tight')


