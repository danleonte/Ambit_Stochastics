# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:15:05 2023

@author: dleon
"""

import numpy as np
from math import ceil

def automated_rej_sampling(sampler_f_max, other_pdfs, f_modes, k, testing = 1):
    """
    sampler_f_max: function which samples from the pdf which has the highest
    value for max_x pdf(x) 
    other_pdfs: list of functions which contains all pdfs apart from the one 
    corresponding to sampler_f_max
    f_modes: 1D np.array of floats giving pdf(mode)
    k: normalizing constant s.t. the product of all k * pdfs integrates to 1       
    """
    
    alpha = np.prod(f_modes) / np.max(f_modes)
    acceptance_rate = 1/ (k * alpha)
    stop  = False
    count = 1
        
    while stop == False:
    
        nr_samples_to_generate = ceil(1/ (2*acceptance_rate)) *  testing 
        samples                = sampler_f_max(size = nr_samples_to_generate)
        #assert nr_samples_to_generate == len(numerator.shape)   
        acceptance_probability = np.prod([pdf(samples) for pdf in other_pdfs],axis=0) / alpha 
        
        u = np.random.uniform(low = 0, high = 1, size = nr_samples_to_generate)
        indicator = u <= acceptance_probability
        
        if np.sum(indicator) > 0:

            if testing == 1:
                return samples[indicator][0]
            
            else: 
                
                print(f'acceptance_rate is {acceptance_rate}')
                print(f'expected nr of samples is {acceptance_rate * nr_samples_to_generate}')
                print(f'number of samples is {np.sum(indicator)}')
            
                return samples[indicator]
        
        
        count +=1 
        
        if count > 20: #to increase to 75
            raise ValueError('the acceptance probability was computed incorrectly')
            
            
            
        
if __name__ == "__main__":
    
    import scipy
    from scipy.stats import norm
    from scipy.integrate import quad   
    import matplotlib.pyplot as plt     

    mu1, mu2 = 1.5, 2.5
    sigma1, sigma2 = 0.45, 0.15 
    
    d1 = scipy.stats.norm(loc = mu1, scale = sigma1).pdf
    d2 = scipy.stats.norm(loc = mu2, scale = sigma2).pdf
    
    assert sigma1 > sigma2 # bigger variance means smaller mode
    
    f_modes = [d1(mu1), d2(mu2)]
    
    sampler_f_max = scipy.stats.norm(loc = mu2, scale = sigma2).rvs
    other_pdfs    = [d1]
    
        
    k = 1 / quad(func = lambda x :d1(x) * d2(x),  a = -np.inf, b = np.inf)[0]
    
    samples  = automated_rej_sampling(sampler_f_max, other_pdfs, f_modes, k, 10**4)
    plt.hist(samples,density= True)
    x_values = np.linspace(np.min(samples), np.max(samples), 1000)
    plt.plot(x_values, d1(x_values) * d2(x_values) * k) 
    
    #usecase: automated_rej_sampling(sampler_f_max, other_pdfs, f_modes, k) , with the last variable = 1 or missing 
    
    

    
    
    
    
       
       
       
       
    
    
    