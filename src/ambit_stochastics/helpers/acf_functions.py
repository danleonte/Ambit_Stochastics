from statsmodels.tsa.stattools import acf
from scipy.integrate import quad
from scipy.optimize import minimize
import numpy as np

def corr_matrix_from_corr_vector(corr_vector):
    """inputs a vector = [corr(0),corr(s),...,corr((k-1)*s)] and outputs the arrray
    Sigma_tilde_ij = overlap area at lag |i-j| = corr(|i-j|*s)
    """
    if isinstance(corr_vector,np.ndarray):
        assert len(corr_vector.shape) == 1
        corr_vector = tuple(corr_vector)
    assert isinstance(corr_vector,(tuple,list))
    
    k = len(corr_vector)
    corr_matrix = [corr_vector[1:i+1][::-1] + corr_vector[:k-i] for i in range(k)]
    return np.array(corr_matrix)

#in the following correlation functions corr_exponnetial,corr_gama,corr_ig      h > 0
def corr_exponential_envelope(h,params):
     u = params[0]
     return np.exp(-u * h)

def corr_gamma_envelope(h,params):
    H,delta = params
    return (1+h/delta)**(-H)

def corr_ig_envelope(h,params):
    gamma,delta =  params
    return np.exp(delta * gamma *(1-np.sqrt(2*h/gamma**2+1)))

def trawl_acf(envelope, envelope_function=None):
    assert envelope in ['exponential','gamma','ig','custom'],'please check the value of envelope'
    
    if envelope == "custom":
        """describe how to specify envelope_function"""
        assert callable(envelope_function)
        
        def corr_other(h,params):
            return quad(envelope_function(params), a=-np.inf, b=-h)[0] / quad(envelope_function(params), a=-np.inf, b=0)[0]

        return corr_other
    
    else:
        
        assert envelope_function == None
        
        if envelope == "exponential":
            return corr_exponential_envelope

        if envelope == "gamma":
            return corr_gamma_envelope

        if envelope == "ig":
            return corr_ig_envelope


def bounds_and_initial_guess_for_acf_params(envelope):
    assert envelope in ['exponential','gamma','ig']
    
    if envelope == 'exponential':
        bounds = ((0,np.inf),)
        initial_guess = (1,)
        
    elif envelope  == 'gamma' or envelope == 'ig':
        bounds = ((0.0001,np.inf),(0.0001,np.inf))
        initial_guess = (1,1)
    
    
    return bounds,initial_guess


def fit_trawl_envelope_gmm(s,simulations,lags,envelope,initial_guess = None,
                           bounds = None, envelope_function = None):

    
    #parameter checks
    assert isinstance(s,(float,int))
    assert isinstance(lags,tuple)
    assert envelope in ['exponential','gamma','ig','custom']
    assert len(simulations.shape) == 2
    #assert isinstance(envelope_params,tuple)
    
    assert (isinstance(initial_guess,tuple) and all(isinstance(i,tuple) for i in initial_guess)) or initial_guess     == None
    assert isinstance(bounds,tuple)        or bounds            == None
    assert callable(envelope_function)     or envelope_function == None 

    
    theoretical_acf_func = trawl_acf(envelope, envelope_function)
    empirical_acf   = np.apply_along_axis(lambda x: acf(x,nlags = max(lags)),arr = simulations,axis=1)
    empirical_acf   = empirical_acf[:,lags]
    
    #this will look s up in the `fit_trawl_envelope_gmm` scope
    def criterion(params,empirical_acf_row):
        theoretical = np.array([theoretical_acf_func(s*i,params) for i in lags])
        return np.sum((empirical_acf_row - theoretical)**2)
        
    if envelope == 'custom':
        #must pass the envelope function and the initial guess
        assert isinstance(initial_guess,tuple) 
        assert callable(envelope_function)
    
    if envelope != 'custom':
        bounds,_ = bounds_and_initial_guess_for_acf_params(envelope)
        if initial_guess  == None:
            initial_guess = tuple([_ for index_ in range(len(empirical_acf))])
    
    #if the custom function has no bounds
    #if bounds == None:
    #    result = [minimize(criterion,x0 = initial_guess, args= (empirical_acf_row,),
    #                                 method='BFGS').x for empirical_acf_row in empirical_acf]
    #    
    #in all the other cases, we have bounds    
    #else:
    result = [minimize(criterion,x0 = initial_guess[j], args= (empirical_acf[j],),
                       method='L-BFGS-B', bounds = bounds).x for j in range(len(empirical_acf))]
        
    return np.array(result)