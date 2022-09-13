# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:25:31 2022

@author: dleon
"""
#from scipy.integrate import quad

from quadpy import quad
import numpy as np
import matplotlib.pyplot as plt
from .Laplace_transform_inversion import sample_from_laplace_transform
#from scipy.integrate import quad


############  Kernels ##############
def triangular_kernel(t_bar):
    floor = np.floor(t_bar)
    ceil  = floor + 1
    mid   = (floor + ceil) / 2
    return 2 * (t_bar - floor) * (t_bar <= mid) + (1 - 2 * (t_bar - mid)) * (t_bar >= mid)

def sine_kernel(t_bar):
    floor = np.floor(t_bar)
    ceil  = floor + 1
    return 1 + np.sin(2 * t_bar * np.pi)

def one_kernel(t_bar):
    return 1
####################################

def log_gamma_laplace_transform(theta,gamma_distr_params):
    alpha,scale = gamma_distr_params
    return - alpha * np.log(1 + theta*scale)

def func__(theta,t_bar,aux_params):
    kernel,alpha,scale = aux_params
    return log_gamma_laplace_transform(theta*kernel(t_bar),[alpha,scale])
     

#lapalce transform function
def ltpdf_helper(theta,aux_params,gfun,hfun,a,b):

    # func must be of the form f(y,x) in xy coordinates. in our case, this is (x,y)
    #return dblquad(func = lambda x,t : func_to_integrate(theta,t,aux_params), a=a, b=b, gfun = gfun,hfun= hfun)[0]
    log_value = quad(f = lambda t_bar: (hfun(t_bar)-gfun(t_bar))*func__(theta,t_bar,aux_params),
                a=a,b=b,epsabs=1.49e-5, epsrel=1.49e-5,limit=200)[0]
    
    return np.exp(log_value)


#ltpdf = lambda theta,aux_params: func__(theta,aux_params)
#aux_params= [2,1]
#sample_from_laplace_transform(100, ltpdf, aux_params)[0]
#sample_from_laplace_transform(5, lambda theta,aux_params: 0.5 * gamma_laplace_transform(theta,aux_params) , aux_params)
#sample_from_laplace_transform(1, lambda theta,aux_params: gamma_laplace_transform(theta,aux_params) , [0.2,1])


if __name__ == "__main__":


    alpha,scale = 2,1 # Levy seed ~ Gamma(alpha,k)
    lambda_ = 1
    trawl_function = lambda t: np.exp(lambda_ * t) * (t<=0) / lambda_ #trawl function
    k= 25; tau = 0.1; max_row_nr = 25 #starts at 0, check the -1 below
    aux_params = [triangular_kernel,alpha,scale]
    
    m = min(k,max_row_nr)
    slice_matrix = np.zeros((m,m))
    
    for j in range(k):
        print('j is ', j)
        if j==0: 
            a = -np.inf
        else: 
            a = j * tau
        b = (j+1) * tau
        
        for i in range(min(k-j,max_row_nr)):
            print('i is ,',i)
            if i+j+1 == k:
                gfun = lambda t_bar : 0
                hfun = lambda t_bar : trawl_function(t_bar - k * tau)
    
            else:
                gfun  = lambda t_bar : trawl_function(t_bar - (i+j+2)*tau)
                hfun = lambda t_bar : trawl_function(t_bar - (i+j+1)*tau)
                
            
            ltpdf = lambda theta,aux_params: ltpdf_helper(theta,aux_params,gfun,hfun,a,b)
            slice_matrix[i,j] = sample_from_laplace_transform(1, ltpdf, aux_params)[0]
                
            
            
            
            
    

    
    
    
   


#def func__(theta,t_bar,aux_params):
#    alpha,scale = aux_params
#    integral = quadpy.quad(lambda t:  np.exp(2*t) * (t<=0)*np.log(1+scale*theta*sine_kernel(t)),a=-5,b=0,
#                          epsabs=1e-06, epsrel=1e-06, limit=100)[0]
#    return np.exp(-alpha * integral) 
#    return (1+scale*theta)**(-alpha)
    #l(t)-h(t)