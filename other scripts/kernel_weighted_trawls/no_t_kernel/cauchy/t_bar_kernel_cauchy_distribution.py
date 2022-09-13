# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:02:36 2022

@author: dleon
"""
import numpy as np
from scipy.integrate import quad 

def indefinite_integral(lambda_, t_bar):
    """computes teh definite integral of exp(lambda_ * t_bar) |sin_t_bar| dt_bar"""
    
    if t_bar == -np.inf:
        return 0
    
    elif t_bar >0:
        raise ValueError('t_bar should be <= 0')
        
    else:
        r =  np.exp(lambda_ * t_bar) * np.sign(np.sin(t_bar)) * \
                    (lambda_ * np.sin(t_bar) - np.cos(t_bar)) / (lambda_**2 + 1)
        
        while (np.sin(t_bar) == 0 or r<=0):
            t_bar = t_bar - 10**(-10)
            r =  np.exp(lambda_ * t_bar) * np.sign(np.sin(t_bar)) * \
                        (lambda_ * np.sin(t_bar) - np.cos(t_bar)) / (lambda_**2 + 1)
        return r
        
        

                    
def definite_integral(lambda_,a,b):
    assert a < b
    return indefinite_integral(lambda_,b) - indefinite_integral(lambda_,a)

result_1 = quad(lambda t: np.exp(3*t) * np.abs(np.sin(t)), -np.inf,0)[0]
result_2 = definite_integral(3,-np.inf,0)
(result_2 - result_1) / result_2

if __name__ == '__main__':

    nr_trawls = 15; tau = 0.25; nr_simulations = 2; max_nr_rows = 10
    jump_part_name = 'cauchy'; jump_part_params = (0.5,) #L' ~ Cauchy(jump_part_params)
    lambda_ = 1
    trawl_function = lambda t: np.exp(lambda_ * t) * (t<=0) / lambda_ #trawl function
    #t_bar_kernel is |sin t_bar|
    slice_matrix = np.zeros([nr_simulations,min(nr_trawls,max_nr_rows-1),nr_trawls])
    
    for j in range(nr_trawls):
        for i in range(min(nr_trawls,max_nr_rows-1)):

            if j==0: 
                a = -np.inf
            else: 
                a = j * tau
            b = (j+1) * tau    
        
            if i+j+1 == nr_trawls:
                #gfun = 0
                #hfun = lambda t_bar : trawl_function(t_bar - k * tau)
                c_ij = np.exp(-nr_trawls  * tau) * definite_integral(lambda_, a, b)
                
    
            else:
                #gfun  = lambda t_bar : trawl_function(t_bar - (i+j+2)*tau)
                #hfun = lambda t_bar : trawl_function(t_bar - (i+j+1)*tau)
                slice_matrix[i,j] = (np.exp(- (i+j+1)*tau) - np.exp(-(i+j+2)*tau)) * definite_integral(lambda_, a, b)
                    

    
                
        

