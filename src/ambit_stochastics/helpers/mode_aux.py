"""Auxuiliary files to determine the mode of certain distributions."""

import numpy as np
from scipy.optimize import brent
from scipy.stats import norminvgauss
from .densities import get_new_params_and_sampler


def find_mode(distr_name, params, area):
    
    params,_ = get_new_params_and_sampler(params = params, area = area, distr_name = distr_name)
    
    if distr_name == 'gaussian':
        return params[0]
    
    elif distr_name == 'norminvgauss':
        return find_mode_numerically(distr_name, params)
    
    raise ValueError('density name not recognised')
    



def find_mode_numerically(distr_name, params):
    """ inspired from ghypMode in R"""
    
    if distr_name == 'norminvgauss':
        #x0 = ...(params)
        
        a,b, mu, delta =  params
        log_density = norminvgauss(a,b,mu,delta).logpdf
        
        assert a**2 - b**2 > 0,'a^2 has to be bigger than b^2'
        assert a > 0, 'a is posiive to ensure identifiability'
        assert delta > 0 , 'scale is positive to ensure identifiability'
        
        mu    = params[2]
        delta = min(delta, mu/2)     #same as scale
        
        x_high = mu + delta
        x_low  = max(mu/2, mu - delta)  
        
        while log_density(x_high) > log_density(mu):
            x_high += delta
            
        while log_density(x_low) > log_density(mu):
            x_low  = max(x_low - delta, x_low/2)

        #print(x_low, x_high)    
        mode   = brent(func = lambda x: -log_density(x),  brack = (0.99 * x_low, 1.01 * x_high))
        f_mode = norminvgauss(params[0], params[1], params[2], params[3]).pdf(mode)
        
        return mode, f_mode
    

#compare with R's ghypmode
#find_mode_numerically('norminvgauss',[2,1,2.5,3]) #a, b, mu, delta = a/ delta, b/ delta, mu, delta
#ghypMode(mu = 2.5, delta = 3, alpha = 2/3, beta=1/3, lambda = -0.5)

#find_mode_numerically('norminvgauss',[4, 0.5,7,0.1])
#ghypMode(mu = 7, delta = 0.1, alpha = 40, beta= 5, lambda = -0.5)