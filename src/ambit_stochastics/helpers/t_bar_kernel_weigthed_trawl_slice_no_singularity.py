# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:33:38 2022

@author: dleon
"""
import numpy as np
import matplotlib.pyplot as plt
from Laplace_transform_inversion import sample_from_laplace_transform
from scipy.integrate import quad


def triangular_kernel(t_bar):
    floor = np.floor(t_bar)
    ceil  = floor + 1
    mid   = (floor + ceil) / 2
    return 1 + 2 * (t_bar - floor) * (t_bar <= mid) + (1 - 2 * (t_bar - mid)) * (t_bar >= mid)

#def sine_kernel(t_bar):
#    #floor = np.floor(t)
#    #ceil  = floor + 1
#    #return np.sin(2 * t * np.pi)
#    return np.sin(t_bar)



    
#lapalce transform function
def ltpdf_helper(theta,aux_params,gfun,hfun,a,b):

    # func must be of the form f(y,x) in xy coordinates. in our case, this is (x,y)
    #return dblquad(func = lambda x,t : func_to_integrate(theta,t,aux_params), a=a, b=b, gfun = gfun,hfun= hfun)[0]
    return quad(func = lambda t_bar: (hfun(t_bar)-gfun(t_bar))*func_to_integrate(theta,t_bar,aux_params),a=a,b=b,epsabs=1.49e-6, epsrel=1.49e-6,limit=200)[0]  
   


if __name__ == "__main__":

#t = np.linspace(-1,3,1000)
#y_triangle = triangular_kernel(t)
#y_sine     = sine_kernel(t)
#plt.plot(t,y_triangle)
#plt.plot(t,y_sine)
    alpha,scale = 2,1 # Levy seed ~ Gamma(alpha,k)
    lambda_ = 1
    trawl_function = lambda t: np.exp(lambda_ * t) * (t<=0) / lambda_ #trawl function
    k= 2; tau = 0.1; max_row_nr = 10 #starts at 0
    aux_params = [triangular_kernel,alpha,scale]
    slice_matrix = np.zeros(k)


    
    for j in range(k):
        print(j)
        if j==0: 
            a = -np.inf
        else: 
            a = j * tau
        b = (j+1) * tau
        
        for i in range(min(k-j,max_row_nr)):
            print(i)
            if i+j+1 == k:
                gfun = 0
                hfun = lambda t_bar : trawl_function(t_bar - k * tau)
    
            else:
                gfun  = lambda t_bar : trawl_function(t_bar - (i+j+2)*tau)
                hfun = lambda t_bar : trawl_function(t_bar - (i+j+1)*tau)
                
            
            ltpdf = lambda theta,aux_params: ltpdf_helper(theta,aux_params,gfun,hfun,a,b)
            slice_matrix[i,j] = sample_from_laplace_transform(1, ltpdf, aux_params)[0]
                
            
            
            
            
    







