# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:23:37 2022

@author: dleon
"""
import pandas as pd
import numpy as np
from ambit_stochastics.trawl import trawl
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf,acovf
from scipy.integrate import quad
import seaborn as sns

sns.set_theme()
sns.set_palette("colorblind")



sm_trawl_function = lambda t :   np.exp(t) * (t<=0)  
lm_trawl_function = lambda t :   0.5 * (1-t)**(-1.5) * (t<=0) 

func_dict = {'Short memory':{'trawl_function':sm_trawl_function, 'lags': list(range(11)),
                             'nr_trawls' : 1000, 'tau': 0.25, 'nr_simulations': 500}, 
              
             'Long memory':{'trawl_function':lm_trawl_function,'lags':[0,1,2,3,5,7,10,15,20,30,50],
                             'nr_trawls' : 5000, 'tau': 0.25, 'nr_simulations': 500}}


for key,value in func_dict.items():
    
    trawl_slice = trawl(nr_trawls = value['nr_trawls'], nr_simulations = value['nr_simulations'],
                        trawl_function = value['trawl_function'], tau =  value['tau'],
                        gaussian_part_params = (0,1), jump_part_name =  None, jump_part_params = (0,0))   
    trawl_slice.simulate(method='slice')
    print(key + ' simulation ready')
    
    empirical_acf   = np.apply_along_axis(lambda x: acf(x,nlags = max(value['lags'])),
                                          arr = trawl_slice.values,axis=1)[:,value['lags']]
    
    empirical_acov = np.apply_along_axis(lambda x: acovf(x,nlag = max(value['lags'])),
                                          arr = trawl_slice.values,axis=1)[:,value['lags']]
    
    df = pd.DataFrame(empirical_acf,index = list(range(value['nr_simulations'])), columns = range(len(value['lags'])))
    true_acf = lambda h: quad(value['trawl_function'],-np.inf,-h)[0] / quad(value['trawl_function'],-np.inf,0)[0]
    
    f,ax = plt.subplots() 
    sns.boxplot(data = df , ax= ax, showfliers = False,showmeans=True,
                meanprops={"markeredgecolor":"red","markerfacecolor":"red"})
    ax.scatter(range(len(value['lags'])), [true_acf(i * value['tau']) for i in value['lags']])
    
    ax.set_xticklabels(value['lags'])
    ax.set_title(key +' autocorrelation function')

    title_fig = key +' '+ f"tau = {value['tau']}, nr_trawls = {value['nr_trawls']}, nr_simulations = {value['nr_simulations']}"
    plt.savefig(title_fig+'.png',bbox_inches='tight')
    #plt.show()
    
    

    
    