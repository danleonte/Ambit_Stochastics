# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:00:51 2022

@author: dleon
"""
import numpy as np
from scipy.integrate import quad
from ambit_stochastics.helpers.alternative_convolution_implementation import cumulative_and_diagonal_sums

#def kernel(x_bar,t_bar):
#       return 1+ t_bar * 0.1
   
def kernel(t_bar):
       return 1+ t_bar * 0.1
   
def x_integral(a,b):
        #return (b+b**1.5/1.5) - (a + a **1.5/1.5) 
        return b-a
    
if __name__ == '__main__':
    np.random.seed(3248234)
    tau = 0.5; k = 250 
    nr_trawls = k
    times = np.arange(tau, (nr_trawls+1) * tau, tau)
    max_nr_rows = k
    #trawl_function
    lambda_ = 1;
    trawl_function = lambda t : lambda_ * np.exp(t * lambda_) * (t<=0)
    #gaussian params L'~ N(mu,sigma**2)
    mu,sigma = (2,2)
    slice_matrix = np.zeros([min(nr_trawls,max_nr_rows-1),nr_trawls])
    

    
    for j in range(nr_trawls):
        if j %50 == 0:
            print(j)
        for i in range(min(nr_trawls,max_nr_rows-1)):

            if j==0: 
                a = -np.inf
            else: 
                a = j * tau
            b = (j+1) * tau    
        
            if i+j+1 == nr_trawls:
                #gfun = 0
                #hfun = lambda t_bar : trawl_function(t_bar - k * tau)
                l_bound = 0
                h_bound = lambda t_bar: trawl_function(t_bar - k * tau)
                
                #func_to_integrate = lambda t_bar : x_integral(l_bound,h_bound(t_bar)) * (1+ np.sin(t_bar)) 
                func_to_integrate = lambda t_bar : x_integral(l_bound,h_bound(t_bar)) * kernel(t_bar) 

                
 
  
            else:
                #gfun  = lambda t_bar : trawl_function(t_bar - (i+j+2)*tau)
                #hfun = lambda t_bar : trawl_function(t_bar - (i+j+1)*tau)
                l_bound = lambda t_bar: trawl_function(t_bar - (i+j+2)*tau)
                h_bound = lambda t_bar: trawl_function(t_bar - (i+j+1)*tau)
                func_to_integrate = lambda t_bar : kernel(t_bar) * (x_integral(l_bound(t_bar),h_bound(t_bar)))
                
            int_f = quad(func_to_integrate,a,b)[0]
            int_f_squared = quad(lambda t_bar: (func_to_integrate(t_bar))**2,a,b)[0]
            
            slice_matrix[i,j] = np.random.normal(int_f*mu,int_f_squared**0.5* sigma)
            #to double check this
    result = cumulative_and_diagonal_sums(slice_matrix)
    
    with open('gaussian_part_for_gauss_gamma.npy', 'wb') as f:
        np.save(f, result)
        
    
    
    
    
