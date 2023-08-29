# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:53:55 2022

@author: dleon
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

tau = 0.5 #check the two scripts agree
nr_trawls = 250  #check the two scripts agree

with open('gaussian_part_for_gauss_gamma.npy', 'rb') as f:
    result_gauss = np.load(f)

with open('gamma_part_for_gauss_gamma.npy', 'rb') as f:
    result_gamma = np.load(f)
    
result = result_gauss + result_gamma[1] #use  simulation

f,ax = plt.subplots()
ax.plot(tau * np.arange(1,nr_trawls+1), result, linewidth=1)
#ax.set_title()
plt.savefig('sns_theme_gaussian_gamma_simulation.png', bbox_inches='tight')
 
