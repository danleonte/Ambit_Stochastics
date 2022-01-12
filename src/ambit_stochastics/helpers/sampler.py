"""Samplers for the Gaussian and jump parts of the Levy basis; we include the drift 
   term in the Gaussian part. The distributions are parametrised as in
           https://docs.scipy.org/doc/scipy/reference/stats.html"""
         
import numpy as np
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,\
                            geninvgauss,poisson             

def gaussian_part_sampler(gaussian_part_params,areas):
    """Simulates the Gaussian part (including drift) of the Levy basis over disjoint sets

    Args:
      gaussian_part_params: list or numpy array with the drift term and variance of the Gaussian part   
      areas: A number / numpy arrray containing the areas of the given sets

    Returns:
      A number / numpy array with law \(\mathcal{N}(drift \cdot areas, scale \cdot \sqrt{areas})\); we use the
      mean-scale parametrisation for consistency with scipy
    """
    
    drift,scale = gaussian_part_params
    gaussian_sample = norm.rvs(loc = drift * areas, scale = scale *(areas)**0.5)
    return gaussian_sample
    
def jump_part_sampler(jump_part_params,areas,distr_name):
    """Simulates the jump part of the Levy basis over disjoint sets; distributions are named 
    and parametrised as in https://docs.scipy.org/doc/scipy/reference/stats.html
    
    Args:
      distr_name: Name of the distribution of the jump part L_j
      jump_part_params: List or numpy array which contains the parameters of the 
      distribution of the jump part L_j
      areas: A number/numpy arrray containing the areas of the given sets
    
    Returns:
      A number / numpy array with law specified by params and distr_name
    """

    if distr_name == None:
        samples = np.zeros(shape= areas.shape)
    
    ###continuous distributions
    elif distr_name == 'gamma':
        a,scale = jump_part_params
        samples = gamma.rvs(a = a * areas, loc = 0, scale = scale)
    
    elif distr_name == 'cauchy':
        scale = jump_part_params[0]
        samples = cauchy.rvs(loc = 0, scale = scale * areas)
        
    elif distr_name == 'invgauss':
        #this is a different parametrisation
        #from the wikipedia page
        raise ValueError('not implemented')
        
    ###discrete distributions
    elif distr_name == 'poisson':
        lambda_poisson = jump_part_params[0]
        samples = poisson.rvs(mu = lambda_poisson * areas,loc=0)
    

    return samples       
