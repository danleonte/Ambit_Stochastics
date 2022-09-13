from .acf_functions import trawl_acf,corr_matrix_from_corr_vector
#import acf_functions
from scipy.stats import beta,gamma,norm
import numpy as np


def get_trawl_process_mean(levy_seed,levy_seed_params):
    #levy_seed_params is a tuple (not a tuple of tuples)
     if  levy_seed == 'gamma':
         alpha,theta = levy_seed_params
         return gamma(a = alpha, loc = 0, scale = theta).mean()
         
     elif levy_seed == 'gaussian':
         return levy_seed_params[0]
     
        
     else:
         raise ValueError('not yet implemented')

         

def conditional_gaussian_distribution(values,tau,nr_steps_ahead,max_gaussian_lag,levy_seed_params,envelope,envelope_params,envelope_function=None):
    
    acf_function_helper =  trawl_acf(envelope, envelope_function)
    acf_function        =  lambda t: acf_function_helper(t,envelope_params)
    mu_,scale_ = levy_seed_params
    
    joints = [np.array(values[i:i+max_gaussian_lag]) for i in range(0,len(values) - max_gaussian_lag +1 )]
    
    mu_1 = mu_                             #mean of X_{gaussian_lags + nr_steps_ahead}
    mu_2 = mu_ *  np.ones(max_gaussian_lag)   #mean of (X_1,...,X_{gaussian_lags})                     
    
    sigma_11 =  scale_**2
    sigma_22 =  scale_**2 * corr_matrix_from_corr_vector(acf_function(np.array([i*tau for i in range(max_gaussian_lag)])))
    sigma_21 =  scale_**2 * (acf_function(np.array([i*tau for i in range(nr_steps_ahead, max_gaussian_lag+nr_steps_ahead)])))[::-1]
    sigma_12 =  sigma_21
    
    sigma_22_inv = np.linalg.inv(sigma_22)
    
    conditional_mean = [mu_1   + sigma_12 @ sigma_22_inv @ (joint -mu_2)         for joint in joints]
    conditional_var  = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21
    
    return np.array(conditional_mean), conditional_var**0.5
         

def deterministic_forecasting_sanity_checks(values,tau,nr_steps_ahead,levy_seed,levy_seed_params,envelope_params):
    assert isinstance(values,np.ndarray)
    assert isinstance(tau,float) and tau > 0
    assert isinstance(nr_steps_ahead,int) and nr_steps_ahead > 0
    assert levy_seed in ['gaussian','gamma','custom']
    assert isinstance(levy_seed_params,np.ndarray)
    #assert envelope and envelope_function checked in trawl_acf
    assert isinstance(envelope_params,np.ndarray)
    
def probabilistic_forecasting_sanity_checks(values,tau,nr_steps_ahead,levy_seed,levy_seed_params,nr_samples,envelope_params):
    assert isinstance(values,np.ndarray)
    assert isinstance(tau,float) and tau > 0
    assert isinstance(nr_steps_ahead,int) and nr_steps_ahead > 0
    assert levy_seed in ['gaussian','gamma','custom']
    assert isinstance(levy_seed_params,np.ndarray)
    #assert envelope and envelope_function checked in trawl_acf
    assert isinstance(envelope_params,np.ndarray)    
    
    
    
def deterministic_forecasting(tau, nr_steps_ahead,values,levy_seed,levy_seed_params,envelope,
                              envelope_params, envelope_function = None,  max_gaussian_lag = None):
    
    deterministic_forecasting_sanity_checks(values, tau, nr_steps_ahead, levy_seed,levy_seed_params,envelope_params)
    
    if envelope == 'gaussian':
        assert isinstance(max_gaussian_lag,int) and   max_gaussian_lag > 0
        conditional_mean,_ = conditional_gaussian_distribution(values,tau,nr_steps_ahead,max_gaussian_lag,levy_seed_params,envelope,envelope_params,envelope_function=None)
        return conditional_mean
        
    else:
        acf_function_helper = trawl_acf(envelope, envelope_function)
        overlap_area = acf_function_helper(tau * nr_steps_ahead,envelope_params)
        #print (overlap_area,type(values),get_trawl_process_mean(levy_seed,levy_seed_params) )   
        return overlap_area * values + (1-overlap_area) * get_trawl_process_mean(levy_seed,levy_seed_params) 


def probabilistic_forecasting(tau,nr_steps_ahead,values,levy_seed,levy_seed_params,envelope,
                              envelope_params, nr_samples, envelope_function = None, max_gaussian_lag = None):
    """assumes the area of the lebesgue measure is 1
    values is a 1 dimensional array """
    
    probabilistic_forecasting_sanity_checks(values,tau,nr_steps_ahead,levy_seed,levy_seed_params,nr_samples,envelope_params)
    
    acf_function_helper = trawl_acf(envelope, envelope_function)
    acf_function = lambda x: acf_function_helper(x,envelope_params)
    overlap_area = acf_function(tau*nr_steps_ahead)
    

    if levy_seed == 'gaussian':
        
        assert isinstance(max_gaussian_lag,int) and max_gaussian_lag > 0 
        
        conditional_mean,conditional_scale = conditional_gaussian_distribution(values,tau,nr_steps_ahead,max_gaussian_lag,
                                                                               levy_seed_params,envelope,envelope_params,envelope_function)
        
        return np.array([norm.rvs(loc = i, scale = conditional_scale, size=nr_samples) for i in conditional_mean])
        
     
    elif levy_seed == 'gamma':
            
            alpha,theta = levy_seed_params
            
            
            
            alpha0 = alpha *  overlap_area
            alpha1 = alpha * (1-overlap_area)
            print('before overlap')
            overlap_samples     = values[:,np.newaxis] * beta.rvs(a = alpha0, b = alpha1, size = [len(values),nr_samples])
            print('before independent')
            independent_samples = gamma.rvs(a = alpha1, loc = 0, scale = theta, size = nr_samples)[np.newaxis,:]
            print('after independent')
            
    elif levy_seed in ['invgauss','gig','cauchy','student']:
        raise ValueError('not yet implemented')
        
    return overlap_samples + independent_samples
        
    
    
        
        
        
        
        
        
    
    