###################################################################################
#                            forecasting                                          #
# we  put these in the same script as pdf_or_log_pdf matches the functions above  #
###################################################################################

         
import numpy as np
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss

    
    
                            
def get_new_params_and_sampler(params, area, distr_name):
    
    ###continuous distributions
    if distr_name == 'norminvgauss':
        a, b, loc, scale = params 
        a, b, loc, scale =  (a *area, b * area, loc* area, scale * area)
        return (a, b, loc, scale), norminvgauss(a= a, b = b, loc = loc, scale = scale).rvs
    
    elif distr_name == 'gaussian':
        loc, scale = params
        loc, scale = loc * area, scale * area**0.5
        return (loc, scale), norminvgauss(loc = loc, scale = scale).rvs
    
    raise ValueError
    

def pdf_or_log_pdf(distr_name, params, areas, log):
    """closely follows the jump_part_sampler function in the same script. """
    
    if distr_name   == 'gaussian':
        
        mu, scale = params
        rv        = norm(loc  = mu * areas, scale = scale *(areas)**0.5)
        
    elif distr_name == 'gamma':
        
        a,scale = params
        rv      = gamma(a = a * areas, loc = 0, scale = [scale] * len(areas))
        
    elif distr_name == 'cauchy':

        scale = params[0]
        rv    = cauchy(loc = [0] * len(areas), scale = scale * areas)
        
    elif distr_name == 'invgauss':
    
        mu, scale = params
        rv        = invgauss(loc = [0] * len(areas) , mu = mu / areas , scale = scale * areas**2) 
        
    elif distr_name == 'norminvgauss':
        
        a, b, loc, scale = params 

        assert a**2 > b**2,'a**2 must be greater than b**2'
        assert a > 0 
        assert scale > 0     
        
        rv = norminvgauss(a = a *areas, b = b * areas, loc = loc* areas, scale = scale * areas)
        
        
    if log == True:
        
        return rv.logpdf
    
    elif log == False:
        
        return rv.pdf
        
    raise ValueError('something went wrong in the pdf_or_log_pdf function')
        

    
    
    

    
    