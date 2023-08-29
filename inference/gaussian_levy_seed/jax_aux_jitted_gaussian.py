from jax.config import config
config.update("jax_enable_x64", True)
from tensorflow_probability.substrates import jax as tfp
from jax import jacfwd #grad, jacrev, value_and_grad
#from jax import jit, random, vmap, pmap
from functools import partial
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax
#--------------------------------------------------
default_precision = jnp.float64
print('Default precision is in jax_aux file is: ',default_precision)
#--------------------------------------------------
import scipy
import numpy as np

#distributions
Normal  = tfp.distributions.Normal

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

@partial(jax.jit, static_argnames=['envelope'])
def corr_jax(s,envelope,envelope_params):
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
def get_sampling_params(transformed_params_1,tau_,envelope):
    transformed_levy_params = transformed_params_1[:2]
    log_env_params          = transformed_params_1[2:]
        
    rho = corr_jax(tau_, envelope, jnp.exp(log_env_params))
        
    mu, sigma = transformed_levy_params[0], jnp.exp(transformed_levy_params[1])
    
    mu_0    = mu    * rho
    sigma_0 = sigma * jnp.sqrt(rho)

    return mu_0,sigma_0


@partial(jax.jit, static_argnames=['envelope','nr_samples'])
def sample_z(transformed_params_1_, tau_, l1, envelope, nr_samples, key_):
    mu_0, sigma_0 = get_sampling_params(transformed_params_1_,tau_,envelope)

    sampler = Normal(loc = mu_0, scale = sigma_0)
    z  = sampler.sample([len(l1), nr_samples] , seed = key_)  #don't forget to use subkey to generate new samples

    return z 


@partial(jax.jit, static_argnames=['envelope']) #TO ADD BACK!!!
def f(z, tau_, envelope, transformed_params_, l1, l2):
        
    transformed_levy_params = transformed_params_[:2]
    log_env_params          = transformed_params_[2:]
        
    rho = corr_jax(tau_, envelope, jnp.exp(log_env_params))    
    mu, sigma = transformed_levy_params[0], jnp.exp(transformed_levy_params[1])
    
    mu_0, mu_1        = mu    * rho, mu * (1-rho)
    sigma_0, sigma_1  = sigma * jnp.sqrt(rho), sigma * jnp.sqrt(1 - rho)
          
    multiplying_constant  = 1 / (2 * jnp.pi * sigma_1**2)

    intermediary = ((jnp.expand_dims(l1,axis=1) - z) - mu_1)**2 + ((jnp.expand_dims(l2,axis=1) - z) - mu_1)**2
    inside_exp = jnp.exp(- intermediary / (2 * sigma_1**2))  
    exp = multiplying_constant * inside_exp  
    
    return exp

#main function
@partial(jax.jit, static_argnames=['envelope','nr_samples'])
def estimators_likelihood_CV_demo(transformed_params_, transformed_params_1_, pairs, envelope, tau, nr_samples, key):

  l1,l2 = pairs[:,0], pairs[:,1]

  z           = sample_z(transformed_params_1_ = transformed_params_1_, tau_ = tau, l1 = l1, envelope = envelope, nr_samples = nr_samples, key_ = key)
  likelihood  = f(z = z, tau_ = tau, envelope = envelope, transformed_params_ = transformed_params_, l1 = l1, l2 = l2)

  return likelihood, z

likelihood_jacobians = jax.jit(jacfwd(estimators_likelihood_CV_demo, argnums = (0,1)), static_argnames=['envelope','nr_samples'])
 #jax.jit(jacfwd(estimators_likelihood_CV_demo, argnums = (0,1)), static_argnames=['envelope','nr_samples','max_taylor_deg'])


@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])  
def compute_taylor_coeff_new(base_point,tau_, envelope, transformed_params_, max_taylor_deg, pairs):
  """ the beta samples z are already scaled by l_1"""
  l1,l2 = pairs[:,0], pairs[:,1]
  f_ = f 
  g_ = jacfwd(f_, argnums = 3)


  l = [f_(base_point, tau_, envelope,  transformed_params_, l1, l2)[:,0]]
  #print('shape is', l[0].shape)

  nabla_theta_l = [g_(base_point, tau_, envelope,  transformed_params_, l1, l2)[:,0]]
  #print('shape is', nabla_theta_l[0].shape)


  for i in range(1,max_taylor_deg+1):
    f_ = jacfwd(f_, argnums = 0)  
    g_ = jacfwd(f_, argnums = 3)

    l.append(f_(base_point, tau_, envelope,  transformed_params_, l1, l2)[:,0]/ scipy.special.factorial(i)) #use some vmap
    nabla_theta_l.append(g_(base_point, tau_, envelope,  transformed_params_, l1, l2)[:,0]/ scipy.special.factorial(i))

  return jnp.stack(l,axis=1), jnp.stack(nabla_theta_l,axis=1)


#--------------------- moments ------------------#
def get_differentiable_moments(sigma,max_moment):
  """X ~ N(mu, sigma^2)"""
  #https://math.stackexchange.com/questions/1945448/methods-for-finding-raw-moments-of-the-normal-distribution
  #length should be max_moment+2
  result = [1.,0.,sigma**2,0., 3. * sigma**4 , 0. , 15. * sigma**6 , 0., 105. * sigma**8]  

  return jnp.array(result[:max_moment+2])


@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])
def differentiable_moments_func(transformed_params_1, tau_, envelope, pairs, max_taylor_deg):

  ___,sigma = get_sampling_params(transformed_params_1 = transformed_params_1, tau_ = tau_, envelope = envelope)
  moments_list = get_differentiable_moments(sigma = sigma, max_moment = max_taylor_deg)
  return jnp.transpose(jnp.array([moments_list[i_] *jnp.ones(len(pairs[:,0])) for i_ in range(0, max_taylor_deg+1)]))

differentiable_moments_gradients_func = jax.jit(jacfwd(differentiable_moments_func), static_argnames=['envelope','max_taylor_deg'])

#-------------------- moments ------------------#

@partial(jax.jit, static_argnames=['envelope','max_taylor_deg'])  #TO ADD BACK
def calculate_moments_to_add(transformed_params_1, pairs, envelope, tau_, max_taylor_deg, coeff, nabla_theta_coeff):

    moments_list    = differentiable_moments_func(transformed_params_1 = transformed_params_1, tau_ = tau_, envelope = envelope, pairs = pairs, max_taylor_deg = max_taylor_deg)
    pg_moments_list = differentiable_moments_gradients_func(transformed_params_1, tau_, envelope, pairs, max_taylor_deg)
    #shapes are [nr_samples,max_taylor_deg+1] and [nr_samples, max_taylor_deg+1, nr_params]
    
    #print(moments_list.shape,'n')
    #print(coeff[:,0].shape)
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

def likelihood_and_grad(transformed_theta, transformed_theta_1, pairs, envelope, tau , nr_mc_samples_per_batch, nr_batches, max_taylor_deg, key ):
  """transformed_theta_1 goes into z and pathwise gradients
  his code works for lag 1. to make it work for higher lags, use tau = tau * lag
  don't forget to change the key between simulations"""

  #results initialization
  likelihood_mean  = 0#np.zeros(pairs.shape[0])
  nabla_theta_mean = 0#np.zeros(pairs.shape[0], len(transformed_theta))
  pg_mean          = 0#np.zeros(pairs.shape[0], len(transformed_theta))

  mu_taylor_base_point =  get_sampling_params(transformed_theta,tau,envelope)[0]


  for batch_number in range(nr_batches): #_ is batch number

    #update the random numbers!!!
    key, subkey = jax.random.split(key)   
    likelihood, z = estimators_likelihood_CV_demo(transformed_params_ = transformed_theta, transformed_params_1_ = transformed_theta_1, pairs = pairs, envelope = envelope,
                                tau = tau, nr_samples = nr_mc_samples_per_batch, key = subkey) #subkey not key here
    #get gradients
    jacobians_f, jacobians_z =  likelihood_jacobians(transformed_theta, transformed_theta_1, pairs, envelope, tau, nr_mc_samples_per_batch, subkey)
    nabla_theta_f, pg_f = jacobians_f  
    zero_variable, pg_z = jacobians_z

    if batch_number == 0: #if it's the 1st batch

        #coefficients of taylor approximation of deg taylor_deg, which we use first to substract the control variate and then to add back its expectation
        #shapes are [nr_samples, max_taylor_deg+1] and [nr_samples, max_taylor_deg+1, nr_params]
        coeff, nabla_theta_coeff   = compute_taylor_coeff_new(base_point = mu_taylor_base_point, tau_ = tau, envelope = envelope, transformed_params_ = transformed_theta,
                                                              max_taylor_deg = max_taylor_deg, pairs = pairs)
        
        #moments to add back (calculation of moments and initialization with the first moment)
        moments_to_add_likelihood, moments_to_add_nabla_theta, moments_to_add_pg = calculate_moments_to_add(transformed_params_1 = transformed_theta_1, pairs = pairs, 
                                                                                     envelope = envelope, tau_ = tau, max_taylor_deg = max_taylor_deg+1,         
                                                                                     coeff = coeff, nabla_theta_coeff = nabla_theta_coeff)

    #control variate (initialization)  
    #shapes are [nr_samples, nr_params] and [nr_samples, max_taylor_deg + 1, nr_params] and [TO ADD]
    T_f = jnp.repeat(jnp.expand_dims(coeff[:,0], axis = 1), repeats = nr_mc_samples_per_batch, axis = 1)
    T_nabla_theta_f = jnp.repeat(jnp.expand_dims(nabla_theta_coeff[:,0],axis=1), repeats = nr_mc_samples_per_batch, axis = 1)
    T_pg_f          = jnp.zeros(shape = T_nabla_theta_f.shape)
    
    #compute control variates
    for i_ in range(1,max_taylor_deg+1):
      T_f += jnp.expand_dims(coeff[:,i_], axis=1) * (z-mu_taylor_base_point)**i_
      T_nabla_theta_f += jnp.expand_dims(nabla_theta_coeff[:,i_],axis=1) * jnp.expand_dims((z-mu_taylor_base_point)**i_,axis=2)
      T_pg_f          += i_ * jnp.expand_dims(coeff[:,[i_]] * (z-mu_taylor_base_point)**(i_-1),axis=2) * pg_z

    if batch_number == 0:      #compute opimal gamma with first batch of samples

      #gamma_0 =   Cov(f,T_f) / Var(T_f)  
      demeaned_likelihood  = likelihood - jnp.nanmean(likelihood,axis=1,keepdims=True)
      demeaned_T_f = T_f - jnp.nanmean(T_f,axis=1,keepdims=True)
      gamma_0 = jnp.nanmean(demeaned_likelihood * demeaned_T_f,axis=1) / jnp.nanvar(T_f,axis=1)
      

      #gamma_1 = Cov(grad f,T_nabla_theta_f) / Var(T_nabla_theta_f)
      demeaned_nabla_theta_f   = nabla_theta_f - jnp.mean(nabla_theta_f, axis=1, keepdims=True)
      demeaned_T_nabla_theta_f = T_nabla_theta_f - jnp.mean(T_nabla_theta_f, axis=1, keepdims=True)
      gamma_1 = jnp.nanmean(demeaned_nabla_theta_f * demeaned_T_nabla_theta_f, axis = 1) / jnp.nanvar(T_nabla_theta_f, axis= 1)

      assert max_taylor_deg >= 1, 'no cv used'
        
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
    #return T_pg_f,moments_to_add_pg,gamma_2

    #append results without control variates
    likelihood_mean  += cv_likelihood.mean(axis=1)
    nabla_theta_mean += nabla_theta_component_of_gradient.mean(axis=1)
    assert max_taylor_deg >= 1, 'no cv used'
    pg_mean += pg_component_of_gradient.mean(axis=1)
    
  #key, subkey = jax.random.split(key)   
  return likelihood_mean / nr_batches, (nabla_theta_mean + pg_mean)/ nr_batches, key



def composite_likelihood_and_grad_at_lags(transformed_theta, transformed_theta_1, trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  
  result_log_likelihood          = 0
  result_gradient_log_likelihood = 0

  for index in range(len(lags_list)):

    lag_to_use   = lags_list[index]
    tau_to_use   = tau * lag_to_use
    pairs_to_use = create_joints_jax_with_padding(trawl_path, lag_to_use)
    
    #make sure to pass the key, as to not use the same random numbers, and to update the tau_to_use and pairs_to_use accordingly
    #the key should be updated inside the likelihood_and_grad function
    likelihood, grads, key = likelihood_and_grad(transformed_theta = transformed_theta, transformed_theta_1 = transformed_theta_1, pairs = pairs_to_use, envelope = envelope,
                                                 tau = tau_to_use , nr_mc_samples_per_batch= nr_samples, nr_batches = nr_batches, max_taylor_deg = max_taylor_deg, key = key)
    
    total = len(likelihood)
    likelihood = likelihood[:total - lag_to_use+1] #to check if this takes out the duplicates
    grads      = grads[:total - lag_to_use+1,:]

    result_log_likelihood           += jnp.mean(jnp.log(likelihood)) #* total / (total - lag_to_use + 1)
    result_gradient_log_likelihood  += jnp.mean(grads / jnp.expand_dims(likelihood, axis=1), axis=0)  #* total / (total - lag_to_use + 1)
    #print(result_log_likelihood)
    #print(result_gradient_log_likelihood)
    return (result_log_likelihood, result_gradient_log_likelihood)


def minus_lambda_function_for_cl_and_grad_at_lags(trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  """x is a 2d array, x[0] is leg_theta and x[1] is transformed_theta_1"""
  #print('used')
  def lambda_(x):
    #print(jnp.exp(x)) #doesn t actually print the value of the array, just its shape as a dynamic array
    tuple_ =  composite_likelihood_and_grad_at_lags(x, x.copy(), trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list)
    return (-tuple_[0], -tuple_[1])

  return lambda_


def numpy_minus_lambda_function_for_cl_and_grad_at_lags(trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list):
  """x is a 2d array, x[0] is leg_theta and x[1] is transformed_theta_1"""
  #print('used')
  def lambda_(x):
    #print(x)
    tuple_ =  composite_likelihood_and_grad_at_lags(x, x.copy(), trawl_path, envelope, tau , nr_samples, nr_batches, max_taylor_deg, key, lags_list)
    return (-np.array(tuple_[0]), -np.array(tuple_[1]))

  return lambda_  


#def get_differentiable_moments(mu,sigma,max_moment):
#  """X ~ N(mu, sigma^2)"""
#  #https://math.stackexchange.com/questions/1945448/methods-for-finding-raw-moments-of-the-normal-distribution
#  #length should be max_moment+2
#  result = [1,mu, mu**2 + sigma**2, mu**3 + 3 * mu * sigma**2,  mu**4 + 6 * mu**2 *sigma**2 + 3 * sigma**4,\
#            mu**5 + 10 *mu**3 * sigma**2 + 15* mu *sigma**4]  
##
#  return jnp.array(result[:max_moment+2])

#import numpy as np
#mu = -1.5
#sigma = 4.1
#z = np.random.normal(loc = mu, scale = sigma, size = 5 * 10**7)
#z2 = np.array([np.mean(z**i) for i in range(6)])
#get_differentiable_moments(mu,sigma,4) - z2,get_differentiable_moments(mu,sigma,4)
