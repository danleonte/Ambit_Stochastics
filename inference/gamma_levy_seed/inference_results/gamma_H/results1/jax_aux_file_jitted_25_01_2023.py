# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:36:49 2023

@author: dleon
"""

from jax.config import config
#config.update("jax_enable_x64", True)
from tensorflow_probability.substrates import jax as tfp
from jax import jacfwd #grad, jacrev, value_and_grad
#from jax import jit, random, vmap, pmap
from functools import partial
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax
#--------------------------------------------------
#default_precision = jnp.float64
#print('Default precision is: ',default_precision)
#--------------------------------------------------
import scipy
import numpy as np


#special functions
from jax.scipy.special import gammaln
#distributions
Beta  = tfp.distributions.Beta

#import scipy.stats
#mgf = scipy.stats.beta.moment


def create_joints_jax(path,lag):
    assert isinstance(lag, int),'variable <lag> is not integer'
    path_length = len(path)
    pairs =  jnp.array([sorted([path[i],path[i+lag]]) for i in range(path_length-lag)])
    return pairs

def create_joints_jax_with_padding(path,lag):
    #print('LAG IS ')
    #print(lag)
    assert isinstance(lag, int),'variable <lag> is not integer'
    assert lag > 0,'variable <lag> should be positive'
    assert lag < len(path)
    path_length = len(path)
    pairs_unpadded = [sorted([path[i],path[i+lag]]) for i in range(path_length-lag)] 
    if lag > 1:
      padding = [pairs_unpadded[-1].copy() for j in range(lag-1)]
      return jnp.array(pairs_unpadded + padding)

    if lag == 1:
      return jnp.array(pairs_unpadded)

#@partial(jax.jit, static_argnames=['envelope'])
def corr_jax2(s,envelope,envelope_params):
    """return overlap area (i.e. correlation) Corr(X_t,X_{t+s}). these are equivalent because the area of the 
    ambit set is normalised to 1. s >0"""

    assert envelope in ['gamma','exponential','ig']
    
    if envelope == 'exponential':
        u = envelope_params[0]
        area = jnp.exp(- u * s)
        
    elif envelope == 'gamma':
        H,delta = envelope_params
        area = (1+s/delta)**(-H)
        
    elif envelope == 'ig':
        gamma,delta =  envelope_params
        area = jnp.exp(delta * gamma *(1-jnp.sqrt(2*s/gamma**2+1)))
    return area


#overide corr
#def corr_np_GMM_gamma_H(delta_):
#    
@partial(jax.jit, static_argnames=['envelope'])
def corr_jax(s,envelope,envelope_params):
        
    assert envelope == 'gamma_H'  
    return  jnp.array((1+s/1.)**(-envelope_params[0]))
    




def corr_np(s,envelope,envelope_params):
    """return overlap area (i.e. correlation) Corr(X_t,X_{t+s}). these are equivalent because the area of the 
    ambit set is normalised to 1. s >0"""

    assert envelope in ['gamma','exponential','ig']
    
    if envelope == 'exponential':
        u = envelope_params[0]
        area = np.exp(- u * s)
        
    elif envelope == 'gamma':
        H,delta = envelope_params
        area = (1+s/delta)**(-H)
        
    elif envelope == 'ig':
        gamma,delta =  envelope_params
        area = np.exp(delta * gamma *(1-np.sqrt(2*s/gamma**2+1)))
    return area


#______________________ likelihood approximation _________________#
@partial(jax.jit, static_argnames=['envelope'])
def get_beta_params(log_params_1,tau_,envelope):
    log_levy_params = log_params_1[:2]
    log_env_params  = log_params_1[2:]
        
    rho = corr_jax(tau_, envelope, jnp.exp(log_env_params))
        
    alpha, beta = jnp.exp(log_levy_params)
    alpha_0 = rho * alpha
    alpha_1 = (1 - rho) * alpha

    return alpha_0,alpha_1

@partial(jax.jit, static_argnames=['envelope','nr_samples'])
def sample_z(log_params_1_, tau_, l1, envelope, nr_samples, key_):
    log_levy_params = log_params_1_[:2]
    log_env_params  = log_params_1_[2:]
    
    rho = corr_jax(tau_, envelope, jnp.exp(log_env_params))

    alpha, beta = jnp.exp(log_levy_params)
    alpha_0 = rho * alpha
    alpha_1 = (1 - rho) * alpha

    sampler = Beta(concentration1 = alpha_0, concentration0 = alpha_1)
    z  = sampler.sample([len(l1), nr_samples] , seed = key_) * jnp.expand_dims(l1,axis=1) #don't forget to use subkey to generate new samples

    return z 

@partial(jax.jit, static_argnames=['envelope'])
def f(z, tau_, envelope, log_params_, l1, l2):
        
      log_levy_params = log_params_[:2]
      log_env_params  = log_params_[2:]
      alpha, beta = jnp.exp(log_levy_params)
      rho = corr_jax(tau_, envelope, jnp.exp(log_env_params))
      #alpha_0 = rho * alpha
      alpha_1 = (1 - rho) * alpha
      

      inside_exp = (jnp.expand_dims(l2,axis=1) - z)**(alpha_1-1) * jnp.exp(z * beta)
      numerator = jnp.exp(-(l1+l2)*beta) * l1**(alpha-1)
      denominator = (1/beta)**(alpha + alpha_1) * jnp.exp(gammaln(alpha_1)) * jnp.exp(gammaln(alpha))
      multiplying_constant = numerator/ denominator
      exp = jnp.expand_dims(multiplying_constant,axis=1) * inside_exp  
      return exp



#main function
@partial(jax.jit, static_argnames=['envelope','nr_samples','max_taylor_deg'])
def estimators_likelihood_CV_demo(log_params_, log_params_1_, pairs, envelope, tau, nr_samples, max_taylor_deg, key):

  l1,l2 = pairs[:,0], pairs[:,1]

  z           = sample_z(log_params_1_ = log_params_1_, tau_ = tau, l1 = l1, envelope = envelope, nr_samples = nr_samples, key_ = key)
  likelihood  = f(z = z, tau_ = tau, envelope = envelope, log_params_ = log_params_, l1 = l1, l2 = l2)

  return likelihood, z

likelihood_jacobians = jax.jit(jacfwd(estimators_likelihood_CV_demo, argnums = (0,1)), static_argnames=['envelope','nr_samples','max_taylor_deg'])
 #jax.jit(jacfwd(estimators_likelihood_CV_demo, argnums = (0,1)), static_argnames=['envelope','nr_samples','max_taylor_deg'])


#def jacobian_forward(func):
#    return jacfwd(func)


@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])  
def compute_taylor_coeff_new(tau_, envelope, log_params_, max_taylor_deg, pairs):
  """ the beta samples z are already scaled by l_1"""
  l1,l2 = pairs[:,0], pairs[:,1]
  f_ = f 
  g_ = jacfwd(f_, argnums = 3)


  l = [f_(0, tau_, envelope,  log_params_, l1, l2)[:,0]]
  #print('shape is', l[0].shape)

  nabla_theta_l = [g_(0, tau_, envelope,  log_params_, l1, l2)[:,0]]
  #print('shape is', nabla_theta_l[0].shape)


  for i in range(1,max_taylor_deg+1):
    f_ = jacfwd(f_, argnums = 0)  
    g_ = jacfwd(f_, argnums = 3)

    l.append(f_(0., tau_, envelope,  log_params_, l1, l2)[:,0]/ scipy.special.factorial(i)) #use some vmap
    nabla_theta_l.append(g_(0., tau_, envelope,  log_params_, l1, l2)[:,0]/ scipy.special.factorial(i))

    #print('shape is', l[i].shape)
    #print('shape is', nabla_theta_l[i].shape)


  return jnp.stack(l,axis=1), jnp.stack(nabla_theta_l,axis=1)

#@partial(jax.jit, static_argnames=['nr_samples','taylor_deg'])
#def compute_taylor_approx(coeff, nr_samples, z, taylor_deg):
#    taylor_approx =  jnp.repeat(jnp.expand_dims(coeff[:,0], axis = 1), repeats = nr_samples, axis = 1)
#
#    for i in range(1,taylor_deg + 1):
#      taylor_approx += jnp.expand_dims(coeff[:,i], axis=1) * z**i

#--------------------- moments ------------------
def get_differentiable_moments(a,b,max_moment):
  """X ~ Beta (a, b)"""
  numerators = a + jnp.arange(0,max_moment+1)
  denominators = numerators + b
  return jnp.concatenate([jnp.array([1]),jnp.cumprod(numerators/denominators)])

@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])
def differentiable_moments_func(log_params_1, tau_, envelope, pairs, max_taylor_deg):

  alpha_0,alpha_1 = get_beta_params(log_params_1 = log_params_1, tau_ = tau_, envelope = envelope)
  l1,l2 = pairs[:,0], pairs[:,1]
  moments_list = get_differentiable_moments(a = alpha_0, b = alpha_1, max_moment = max_taylor_deg )
  return jnp.transpose(jnp.array([moments_list[i_] *l1**i_ for i_ in range(0, max_taylor_deg+1)]))

differentiable_moments_gradients_func = jax.jit(jacfwd(differentiable_moments_func), static_argnames=['envelope','max_taylor_deg'])


#test
#differentiable_moments(0.5,0.75,3)
#a = 0.5
##b = 0.75
#m1 = a / (a+b)
#m2 = m1 * (a+1)/ (a+b+1)
#m3 = m2 * (a+2) / (a+b+2)
#m4 = m3 * (a+3) / (a+b+3)
#print(m1,m2,m3,m4)

@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])
def calculate_moments_to_add(log_params_1, pairs, envelope, tau_, max_taylor_deg, coeff, nabla_theta_coeff):

    moments_list    = differentiable_moments_func(log_params_1 = log_params_1, tau_ = tau_, envelope = envelope, pairs = pairs, max_taylor_deg = max_taylor_deg)
    pg_moments_list = differentiable_moments_gradients_func(log_params_1, tau_, envelope, pairs, max_taylor_deg)
    #shapes are [nr_samples,max_taylor_deg+1] and [nr_samples, max_taylor_deg+1, nr_params]
    
    
    moments_to_add_likelihood  = coeff[:,0] * moments_list[:,0]
    moments_to_add_nabla_theta = nabla_theta_coeff[:,0] * moments_list[:,[0]]
    moments_to_add_pg          = coeff[:,[0]] * pg_moments_list[:,0]  #first moment should be 0

    for i_ in range(1,max_taylor_deg+1):
        moments_to_add_likelihood  += coeff[:,i_] * moments_list[:,i_]
        moments_to_add_nabla_theta += nabla_theta_coeff[:,i_] * moments_list[:,[i_]]
        moments_to_add_pg          += coeff[:,[i_]] * pg_moments_list[:,i_] 

    return moments_to_add_likelihood, moments_to_add_nabla_theta, moments_to_add_pg   

#---------------cl functions for cluster ----------------

#@partial(jax.jit, static_argnames=['envelope','nr_samples','max_taylor_deg'])  
#the for loop might not be suitable for jitting, as it would unfold it?

def likelihood_and_grad(log_theta, log_theta_1, pairs, envelope, tau , nr_mc_samples_per_batch, nr_batches, max_taylor_deg, key ):
  """log_theta_1 goes into z and pathwise gradients
  his code works for lag 1. to make it work for higher lags, use tau = tau * lag
  don't forget to change the key between simulations"""

  #results initialization
  likelihood_mean  = 0#np.zeros(pairs.shape[0])
  nabla_theta_mean = 0#np.zeros(pairs.shape[0], len(log_theta))
  pg_mean          = 0#np.zeros(pairs.shape[0], len(log_theta))

  for batch_number in range(nr_batches): #_ is batch number

    #update the random numbers!!!
    key, subkey = jax.random.split(key)   
    likelihood, z = estimators_likelihood_CV_demo(log_params_ = log_theta, log_params_1_ = log_theta_1, pairs = pairs, envelope = envelope,
                                tau = tau, nr_samples = nr_mc_samples_per_batch, max_taylor_deg =  max_taylor_deg, 
                                key = subkey) #subkey not key here
    #get gradients
    jacobians_f, jacobians_z =  likelihood_jacobians(log_theta, log_theta_1, pairs, envelope, tau, nr_mc_samples_per_batch, max_taylor_deg, subkey)
    nabla_theta_f, pg_f = jacobians_f  
    zero_variable, pg_z = jacobians_z

    if batch_number == 0: #if it's the 1st batch

        #coefficients of taylor approximation of deg taylor_deg, which we use first to substract the control variate and then to add back its expectation
        #shapes are [nr_samples, max_taylor_deg+1] and [nr_samples, max_taylor_deg+1, nr_params]
        coeff, nabla_theta_coeff   = compute_taylor_coeff_new(tau_ = tau, envelope = envelope, log_params_ = log_theta,
                                                              max_taylor_deg = max_taylor_deg, pairs = pairs)
        
        #moments to add back (calculation of moments and initialization with the first moment)
        moments_to_add_likelihood, moments_to_add_nabla_theta, moments_to_add_pg = calculate_moments_to_add(log_params_1 = log_theta_1, pairs = pairs, 
                                                                                     envelope = envelope, tau_ = tau, max_taylor_deg = max_taylor_deg,         
                                                                                     coeff = coeff, nabla_theta_coeff = nabla_theta_coeff)

    #control variate (initialization)  
    #shapes are [nr_samples, nr_params] and [nr_samples, max_taylor_deg + 1, nr_params] and [TO ADD]
    T_f = jnp.repeat(jnp.expand_dims(coeff[:,0], axis = 1), repeats = nr_mc_samples_per_batch, axis = 1)
    T_nabla_theta_f = jnp.repeat(jnp.expand_dims(nabla_theta_coeff[:,0],axis=1), repeats = nr_mc_samples_per_batch, axis = 1)
    T_pg_f          = jnp.zeros(shape = T_nabla_theta_f.shape)
    
    #compute control variates
    for i_ in range(1,max_taylor_deg+1):
      T_f += jnp.expand_dims(coeff[:,i_], axis=1) * z**i_
      T_nabla_theta_f += jnp.expand_dims(nabla_theta_coeff[:,i_],axis=1) * jnp.expand_dims(z**i_,axis=2)
      T_pg_f          += i_ * jnp.expand_dims(coeff[:,[i_]] * z**(i_-1),axis=2) * pg_z

    if batch_number == 0:      #compute opimal gamma with first batch of samples

      #gamma_0 =   Cov(f,T_f) / Var(T_f)  
      demeaned_likelihood  = likelihood - jnp.nanmean(likelihood,axis=1,keepdims=True)
      demeaned_T_f = T_f - jnp.nanmean(T_f,axis=1,keepdims=True)
      gamma_0 = jnp.nanmean(demeaned_likelihood * demeaned_T_f,axis=1) / jnp.nanvar(T_f,axis=1)
      

      #gamma_1 = Cov(grad f,T_nabla_theta_f) / Var(T_nabla_theta_f)
      demeaned_nabla_theta_f   = nabla_theta_f - jnp.mean(nabla_theta_f, axis=1, keepdims=True)
      demeaned_T_nabla_theta_f = T_nabla_theta_f - jnp.mean(T_nabla_theta_f, axis=1, keepdims=True)
      gamma_1 = jnp.nanmean(demeaned_nabla_theta_f * demeaned_T_nabla_theta_f, axis = 1) / jnp.nanvar(T_nabla_theta_f, axis= 1)

      assert max_taylor_deg > 1, 'no cv used'
        
      #gamma_2 = Cov(df/dz * nabla_theta_z, T_df_dz * nabla_theta_z) / Var(T_df_dz * nabla_theta_z)
      demeaned_pg_f = pg_f - jnp.mean(pg_f, axis=1, keepdims = True) #this should be df_dz * nabla_theta_z)
      demeaned_T_pg_f = T_pg_f - jnp.mean(T_pg_f, axis =1 , keepdims = True)
      gamma_2 = jnp.nanmean(demeaned_pg_f * demeaned_T_pg_f, axis = 1 ) / jnp.nanvar(T_pg_f, axis = 1) #makes nan's into 0's
      gamma_2 = jnp.nan_to_num(gamma_2, nan=0.0)
         

    #likelihood and gradient calculations-------- 
    cv_likelihood = likelihood - jnp.expand_dims(gamma_0,axis=1) * T_f + jnp.expand_dims(gamma_0 * moments_to_add_likelihood,axis=1)
    nabla_theta_component_of_gradient = nabla_theta_f - jnp.expand_dims(gamma_1,axis=1)  * T_nabla_theta_f +\
                   jnp.expand_dims(gamma_1 * moments_to_add_nabla_theta, axis=1)

    pg_component_of_gradient    =  pg_f   - jnp.expand_dims(gamma_2,axis=1) * T_pg_f  + jnp.expand_dims(gamma_2 * moments_to_add_pg, axis=1)

    #append results without control variates
    likelihood_mean  += cv_likelihood.mean(axis=1)
    nabla_theta_mean += nabla_theta_component_of_gradient.mean(axis=1)
    assert max_taylor_deg > 1, 'no cv used'
    pg_mean += pg_component_of_gradient.mean(axis=1)
    
  #key, subkey = jax.random.split(key)   
  return likelihood_mean / nr_batches, (nabla_theta_mean + pg_mean)/ nr_batches, key


def composite_likelihood_and_grad_at_lags(log_theta, log_theta_1, trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  
  result_log_likelihood          = 0
  result_gradient_log_likelihood = 0

  for index in range(len(lags_list)):

    lag_to_use   = lags_list[index]
    tau_to_use   = tau * lag_to_use
    pairs_to_use = create_joints_jax_with_padding(trawl_path, lag_to_use)
    
    #make sure to pass the key, as to not use the same random numbers, and to update the tau_to_use and pairs_to_use accordingly
    #the key should be updated inside the likelihood_and_grad function
    likelihood, grads, key = likelihood_and_grad(log_theta = log_theta, log_theta_1 = log_theta_1, pairs = pairs_to_use, envelope = envelope,
                                                 tau = tau_to_use , nr_mc_samples_per_batch= nr_samples, nr_batches = nr_batches, max_taylor_deg = max_taylor_deg, key = key)
    
    total = len(likelihood)
    likelihood = likelihood[:total - lag_to_use+1] #to check if this takes out the duplicates
    grads      = grads[:total - lag_to_use+1,:]

    result_log_likelihood           += jnp.mean(jnp.log(likelihood)) #* total / (total - lag_to_use + 1)
    result_gradient_log_likelihood  += jnp.mean(grads / jnp.expand_dims(likelihood, axis=1), axis=0)  #* total / (total - lag_to_use + 1)
    #print(result_log_likelihood)
    #print(result_gradient_log_likelihood)
    return (result_log_likelihood, result_gradient_log_likelihood)


    
#r = likelihood_and_grad(log_theta = initial_log_tensor, log_theta_1 = initial_log_tensor_1, pairs = pairs, envelope = envelope, 
#                      tau = tau, nr_samples = nr_mc_samples_per_batch, nr_batches = nr_batches, max_taylor_deg = max_taylor_deg, key = key)

#r1, r2 = composite_likelihood_and_grad_at_lags(log_theta = initial_log_tensor, log_theta_1 = initial_log_tensor_1, trawl_path = all_values[0], envelope = envelope,
#                             tau = tau , nr_samples = nr_mc_samples_per_batch, nr_batches = nr_batches, max_taylor_deg = max_taylor_deg, key = key, lags_list = (1,2,3))
    

#r1,r2
def minus_lambda_function_for_cl_and_grad_at_lags(trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  """x is a 2d array, x[0] is leg_theta and x[1] is log_theta_1"""
  #print('used')
  def lambda_(x):
    #print(jnp.exp(x)) #doesn t actually print the value of the array, just its shape as a dynamic array
    tuple_ =  composite_likelihood_and_grad_at_lags(x, x.copy(), trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list)
    return (-tuple_[0], -tuple_[1])

  return lambda_


def numpy_minus_lambda_function_for_cl_and_grad_at_lags(trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  """x is a 2d array, x[0] is leg_theta and x[1] is log_theta_1"""
  #print('used')
  def lambda_(x):
    #print(x)
    tuple_ =  composite_likelihood_and_grad_at_lags(x, x.copy(), trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list)
    return (-np.array(tuple_[0]), -np.array(tuple_[1]))

  return lambda_  


#----------- old ------------
#@partial(jax.jit, static_argnames=['envelope','taylor_deg'])  #can't jit this as it has a function input. way around it?
def compute_taylor_coeff_old(f_, tau_, envelope, log_params_, max_taylor_deg, pairs):

  l1,l2 = pairs[:,0], pairs[:,1]

  l = [f_(0, tau_, envelope,  log_params_, l1, l2)[:,0]]
  #print('shape is', l[0].shape)

  for i in range(1,max_taylor_deg+1):
    f_ = jacfwd(f_)  
    l.append(stop_gradient(f_(0., tau_, envelope,  log_params_, l1, l2)[:,0])/ scipy.special.factorial(i)) #use some vmap
  return jnp.stack(l,axis=1)


#main function2
@partial(jax.jit, static_argnames=['envelope','nr_samples','max_taylor_deg'])
def just_mean_estimators_likelihood_CV_demo(log_params_, log_params_1_, pairs, envelope, tau, nr_samples, max_taylor_deg, key):

  l1,l2 = pairs[:,0], pairs[:,1]

  z           = sample_z(log_params_1_ = log_params_1_, tau_ = tau, l1 = l1, envelope = envelope, nr_samples = nr_samples, key_ = key)
  likelihood  = f(z = z, tau_ = tau, envelope = envelope, log_params_ = log_params_, l1 = l1, l2 = l2)
  

  return jnp.mean(likelihood,axis=1)

mean_likelihood_jacobians = jax.jit(jacfwd(just_mean_estimators_likelihood_CV_demo, argnums = (0,1)), static_argnames=['envelope','nr_samples','max_taylor_deg'])

