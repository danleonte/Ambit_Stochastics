# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:03:47 2022

@author: dleon
"""
import numpy as np
import math
from scipy.integrate import quad
import matplotlib
from matplotlib import pyplot as plt
#import matplotlib.ticker as mticker
#plt.style.use('ggplot')
import seaborn as sns
sns.set_theme()
sns.set_palette("colorblind")



def set_nr_subdivisions(delta):
            
    if   delta <= 0.001:   nr_subdivisions = 1000
    elif delta <= 0.01:    nr_subdivisions = 500
    elif delta <= 0.05:    nr_subdivisions = 100
    else:                  nr_subdivisions = 20
    
    return nr_subdivisions
        
    
def generate_grid(min_t,max_t,max_x,delta):
    """outputs grid on [min_t,0] x [0,max_x] with stepsize delta starting from 0,
    not from the left endpoint"""
    
    coords = np.mgrid[0:max_x:delta, min_t:max_t:delta]
    x, t   = coords[0].flatten(), coords[1].flatten() 
    return t,x+delta #each cell is represented by its top left corner, not 
    #bottom left corner
    
    
def calcualte_nr_cells_at_h(delta,phi,h,nr_subdivisions=1):
    """only to be used with unbounde trawls
    if we use it with a bounded trawl, we have to account for 0 < x < phi(..)"""
    
    break_points = np.linspace(-5/delta**0.5 + h, 0, num = nr_subdivisions+1)
    break_points = np.floor(break_points/delta) * delta
    
    s = 0
    
    for i in range(nr_subdivisions):
        
        min_t = break_points[i]
        max_t = break_points[i+1]
        max_x = math.ceil(phi(max_t)/delta)*delta
        
        t,x = generate_grid(min_t,max_t,max_x,delta)
        s += np.sum(x < phi(t-h))
    
    return s

def calculate_nr_cells_function(delta,phi,h_vector,nr_subdivisions=1):
    l = []
    for h in h_vector:
        l.append(calcualte_nr_cells_at_h(delta,phi,h,nr_subdivisions))
    return np.array(l)
        
        
def calculate_nr_cells_in_trawl_set(delta,phi,nr_subdivisions=1):
    return calcualte_nr_cells_at_h(delta,phi,0,nr_subdivisions) 
        

#nr_cells = calculate_var(0.02,lambda t: (1-t)**(-0.5) * (t<=0),1)    
#print(nr_cells)

#phi = lambda t: (1-t)**(-1.5) * (t<=0)
#delta= 0.05
#nr_cells2 = calculate_var(delta,phi,100)    
#print(nr_cells2)
#area_estimate2 = nr_cells2 * delta **2
#print(area_estimate2)




if __name__ == "__main__":
    
    #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    #g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    #fmt = mticker.FuncFormatter(g)

    trawl_function = lambda t: 0.5 * (1-t)**(-1.5) * (t<=0) #np.exp(t) * (t<=0)
    trawl_set_area = quad(trawl_function,-np.inf,0)[0]
    print('trawl set area is ',trawl_set_area)
    theoretical_acf_func = lambda h: quad(trawl_function,-np.inf,-h)[0] / trawl_set_area

    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    #variance ratio plot = nr_of cells * cell area / trawl set area
    delta_vector_var  = [0.00005,0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01] #
    #delta_vector_var= [0.01,0.025,0.05,0.075,0.1]
    d_var             = dict()

    for delta in delta_vector_var:
        
        print('delta is',delta)
        nr_subdivisions = set_nr_subdivisions(delta)
        nr_cells = calculate_nr_cells_in_trawl_set(delta = delta,phi = trawl_function, 
                                                   nr_subdivisions = nr_subdivisions)
        
        d_var[delta] = nr_cells * delta**2 / trawl_set_area
        
    
    ax[1].plot(d_var.keys(),d_var.values())
    ax[1].hlines(y=1, color='r', linestyle='-',xmin=0,xmax=0.01)
    ax[1].set_xscale('log')      #ax[1].set_xticklabels(ax[1].get_xticks(), rotation = 45)     #ax[1].tick_params(axis='x', labelrotation = 45)

    #lim = ax.get_xlim()
    ax[1].set_title("Var($L_{T,\Delta}$(A)) / Var($L$(A))")
    #ax.set_xlim(lim)
    print('right figure finished') 
    
      
    #correlation plot
    #delta_vector_corr = [0.01,0.025,0.05,0.075,0.1]
    delta_vector_corr = [0.0001,0.001,0.01]
    delta_legend      = ['10^{-4}$','10^{-3}$','10^{-2}$']
    #h_vector  = np.concatenate([np.linspace(0,2.5,15),np.linspace(2.6,5,20),np.linspace(5.1,10,15),np.linspace(10.1,17.5,15)])
    h_vector = np.linspace(0,16,100)#np.linspace(0,6,100)
    #true correlation
    true_corr = np.array([theoretical_acf_func(h) for h in h_vector])
    ax[0].plot(h_vector,true_corr,label='theoretical acf')
    plt.tight_layout()
    plt.savefig('var_plot.png', bbox_inches='tight')
    
    #empirical correlations
    for i in range(len(delta_vector_corr)):
        delta = delta_vector_corr[i]
        print('delta is',delta)
        
        nr_subdivisions = set_nr_subdivisions(delta)
        l = calculate_nr_cells_function(delta = delta ,phi = trawl_function ,h_vector = h_vector,
                                    nr_subdivisions = nr_subdivisions)
        ax[0].plot(h_vector,l / d_var[delta] * delta**2,label = '$\Delta =' + delta_legend[i])
                   #$'+"{:,.1e}".format(delta))#.format(fmt(delta)))
    
    ax[0].ticklabel_format(style='sci')    
    ax[0].set_title("Autocorrelation function")
    ax[0].legend()
    plt.setp(ax[0].get_legend().get_texts(), fontsize='11.75') # for legend text
    #plt.setp(ax[0].get_legend().get_title(), fontsize='32') # for legend title
    plt.tight_layout()
    plt.savefig('acf_and_corr.png', bbox_inches='tight')
    plt.show()
    
    
    