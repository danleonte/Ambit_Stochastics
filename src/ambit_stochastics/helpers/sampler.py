"""Samplers for the Gaussian, jump and cpp parts of the Levy basis. The distributions are parametrised as in https://docs.scipy.org/doc/scipy/reference/stats.html"""
         
import numpy as np
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,\
                            geninvgauss,bernoulli,binom,nbinom,poisson,logser           

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
    areas_copy = areas.copy()
    index      = areas_copy == 0
    if np.any(areas_copy < 0):
        raise ValueError('slice areas cant be negative')
    
    areas[index] = 100 #random number which will be removed
    
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

        
        mu, scale = jump_part_params
        samples   = invgauss.rvs(loc = 0, mu = mu / areas , scale = scale * areas**2)  
        
        #this is a different parametrisation
        #from the wikipedia pagey
        #scipy (mu,scale) -> wiki (mu= mu * scale, lambda = scale)
        #wiki (mu, lambda) -> scipy (mu = mu/lambda, scale = lambda)
        
        #        #wrong?
        #scipy scaling: L' ~ IG(mu,scale) ->  L(A) ~ IG(mu / (scale * Leb(A)) , scale * Leb(A)^2)
        #correct?
        #scipy scaling: L' ~ IG(mu,scale) ->  L(A) ~ IG(mu / Leb(A) , scale * Leb(A)^2)
        
        #scipy mean = mu * scale, scipy var: (mu * scale )^3 / scale = mu ^3 * scale ^2 
        #wiki scaling: L' ~ IG(mu,lambda) ->  L(A) ~ IG( mu * Leb(A), lamda * Leb(A)^2)
        #TO CHECK THIS AGAIN
        #mu, scale = jump_part_params
       # samples   = invgauss.rvs(loc = 0, mu = mu / areas , scale = scale * areas**2)         
            
    ###discrete distributions
    elif distr_name == 'poisson':
        lambda_poisson = jump_part_params[0]
        samples = poisson.rvs(mu = lambda_poisson * areas,loc=0)     
    
    samples[index] = 0
    return samples 

def generate_cpp_values_associated_to_points(nr_points,cpp_part_name,cpp_part_params,custom_sampler):  
    if cpp_part_name == 'custom':
         return custom_sampler(nr_points)
    
    elif cpp_part_name == 'bernoulli':
        
         return bernoulli.rvs(p = cpp_part_params[0], size = nr_points)
         
    elif cpp_part_name == 'poisson':
         
         return poisson.rvs(mu = cpp_part_params[0], size = nr_points)
     
    elif cpp_part_name == 'logser':
        return logser.rvs(p = cpp_part_params[0], size = nr_points)
    
    elif cpp_part_name == 'binom':
        
         return binom.rvs(n = cpp_part_params[0], p = cpp_part_params[1], size = nr_points)
        
    elif cpp_part_name == 'nbinom':
        
         return nbinom.rvs(n = cpp_part_params[0], p = cpp_part_params[1], size = nr_points)

        
        
        

def generate_cpp_points(min_x,max_x,min_t,max_t,cpp_part_name,cpp_part_params,cpp_intensity,custom_sampler):
    
    area_times_intensity = (max_x-min_x)*(max_t-min_t) * cpp_intensity  
    nr_points = poisson.rvs(mu = area_times_intensity)
    points_x = np.random.uniform(low = min_x, high = max_x, size = nr_points)
    points_t = np.random.uniform(low = min_t, high = max_t, size = nr_points)
    
    associated_values = generate_cpp_values_associated_to_points(nr_points,cpp_part_name,\
                                             cpp_part_params,custom_sampler)
    
    return points_x,points_t,associated_values
        
    
    
    
    