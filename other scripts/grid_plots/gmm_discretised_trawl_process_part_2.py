# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:02:39 2022

@author: dleon
"""

import numpy
import pandas as pd
#from matplotlib import pyplot
import seaborn
import pickle
import numpy as np
#import matplotlib.pylab as plt

#seaborn.set(style="ggplot")
seaborn.set_palette("colorblind")


#read the data
delta_vector_to_use =  [0.025, 0.05, 0.075, 0.1]#, 0.25]
delta_vector_to_use_latex = ['$2.5 10^{-2}$','$5 10^{-2}$','$7.5 10^{-2}$','$10^{-1}$']#,
                             #'$2.5 10^{-1}$']
true_params = np.array([1,2,0.5])

result = []

for delta in delta_vector_to_use:
    
    with open(f'delta_{delta}.pickle', 'rb') as handle:
        b = pickle.load(handle)
        
    #envelope first    
    results_to_add = np.concatenate(list(b.values()),axis=1) -true_params[np.newaxis,:]
    results_to_add = 100 * np.abs(results_to_add / true_params[np.newaxis,:])
    #add delta
    results_to_add = np.concatenate([results_to_add,np.ones([len(results_to_add),1])*delta],axis=1)
    result.append(results_to_add)
    
    
df = pd.DataFrame(np.concatenate(result,axis=0), columns=list('λkθΔ'))


ax = (
    df.set_index('Δ', append=True)  # set E as part of the index
      .stack()                      # pull A - D into rows 
      .to_frame()                   # convert to a dataframe
      .reset_index()                # make the index into reg. columns
      .rename(columns={'level_2': 'parameters', 0: 'Relative error(%)'})  # rename columns
      .drop('level_0', axis='columns')   # drop junk columns
      .pipe((seaborn.boxplot, 'data'), x='Δ', y='Relative error(%)',
            hue='parameters', order= delta_vector_to_use,width=0.6,
            showfliers = False)  
)
seaborn.despine(trim=False)
seaborn.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)

fig = ax.get_figure()
fig.savefig("output.png")