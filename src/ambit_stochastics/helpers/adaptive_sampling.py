# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 13:18:38 2023

@author: dleon
"""
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp; tfp = tfp.substrates.jax; tfd = tfp.distributions
from functools import partial


norm         = tfd.Normal
gamma        = tfd.Gamma
invgauss     = tfd.InverseGaussian
norminvgauss = tfd.NormalInverseGaussian

def jax_log_pdf_for_sampling(distr_name):
    
    assert distr_name in ('gaussian','norminvgauss')
    
    if distr_name   == 'gaussian':
        
        def log_density(x, sum_, params, areas):
            mu, scale = params
            rv        = norm(loc  = mu * areas, scale = scale *(areas)**0.5)
            return jnp.sum(rv.log_prob(jnp.append(x,sum_ - jnp.sum(x))))
            #return jnp.sum(rv.log_prob(x))

            
    elif distr_name   == 'norminvgauss':
        
        def log_density(x, sum_, params, areas):


            a, b, loc, scale = params #scipy params
            alpha, beta, mu, delta =  a/scale, b/scale, loc, scale #tf params
            
            rv = norminvgauss(loc = mu * areas, scale = delta * areas, tailweight  = alpha, skewness = beta)
            return jnp.sum(rv.log_prob(jnp.append(x,sum_ - jnp.sum(x))))
            #return jnp.sum(rv.log_prob(x))
        
    return log_density        


@partial(jax.jit, static_argnames = 'distr_name')
def approx_sampling_from_log_linear_interpolation(x_values, sum_, distr_name, params, areas, key):
    """f_prime is a componentwise differentiation, e.g. constructed by vmap"""
    
    #
    y_values, y_prime   = log_density_sampling_dictionary[distr_name](x_values, sum_, params, areas)
    x1, x2              = x_values[:-1], x_values[1:]
    y1, y2              = y_values[:-1], y_values[1:]
    y_prime1, y_prime2  = y_prime[:-1],  y_prime[1:]
    convexity_indicator = jnp.where(y_prime2 - y_prime1 > 0, 1, 0 )    
    
    ########## construct convex envelopes ##########
    #y - y1 = (y2 - y1)/(x2 - x1) × (x - x1) <-> y = (y2 - y1)/(x2 - x1) * x + y1 - (y2 - y1)/(x2 - x1) * x1
    m_convex = (y2 - y1) / (x2 - x1)
    n_convex = y1 - m_convex * x1
    area_below_secants = (jnp.exp(m_convex * x2 + n_convex) - jnp.exp(m_convex * x1 + n_convex))/m_convex
    
    #take care of 0 slopes, which we can't divide by
    zero_slope_convex_correction = jnp.where(m_convex == 0,(x2-x1) * jnp.exp(y1),0)
    area_below_secants  = jnp.nan_to_num(area_below_secants)
    area_below_secants += zero_slope_convex_correction     #m_convex_equal_0 = jnp.where(m_convex == 0,1,0)    #area_below_secants = area_below_secants.at[m_convex_equal_0].set((x2-x1) * jnp.exp(y1)[m_convex_equal_0])
    
    ########## construct concave envelopes ##########
    n1                  = y1 - y_prime1 * x1
    n2                  = y2 - y_prime2 * x2
    intersection_points = (n2 - n1) / (y_prime1 - y_prime2)
    
    y_primes_equal_correction = jnp.where(y_prime1 == y_prime2, x1/2 + x2/2, 0)
    intersection_points       = jnp.nan_to_num(intersection_points)
    intersection_points      += y_primes_equal_correction
    
    _area_1 = (jnp.exp(y_prime1 * intersection_points + n1) - jnp.exp(y_prime1 * x1 + n1))/y_prime1
    _area_2 = (jnp.exp(y_prime2 * x2 + n2) - jnp.exp(y_prime2 * intersection_points + n2))/y_prime2
    
    #take care of 0 slopes, which we can't divide by
    y_prime1_correction = jnp.where(y_prime1==0,(intersection_points-x1) * jnp.exp(y1),0)
    _area_1  = jnp.nan_to_num(_area_1)
    _area_1 += y_prime1_correction

    #take care of 0 slopes, which we can't divide by
    y_prime2_correction = jnp.where(y_prime2==0,(x2 - intersection_points) * jnp.exp(y2),0)
    _area_2  = jnp.nan_to_num(_area_2)
    _area_2 += y_prime2_correction

    area_below_tangents = _area_1 + _area_2
    
    ###################################################
    #deal with tails
    
    left_tail_log_slope      = y_prime[0]
    right_tail_log_slope     = y_prime[-1]
    
    left_tail_area  =  jnp.exp(y_values[0])  /  left_tail_log_slope
    right_tail_area = -jnp.exp(y_values[-1]) /  right_tail_log_slope
    
    big_number = len(x_values) * 3 + 100
    x_values = jnp.insert(x_values, jnp.array([0,big_number]) ,jnp.array([-jnp.inf,jnp.inf]))
    area_below_tangents = jnp.insert(area_below_tangents, jnp.array([0,big_number]), jnp.array([left_tail_area, right_tail_area]))
    area_below_secants  = jnp.insert(area_below_secants, jnp.array([0,big_number]), jnp.array([left_tail_area, right_tail_area]))    
    convexity_indicator = jnp.insert(convexity_indicator, jnp.array([0,big_number]), jnp.array([1,1]))
    ####################################################
    
    area_upper_bound   = convexity_indicator * area_below_secants + (1 - convexity_indicator) * area_below_tangents
    area_lower_bound   = convexity_indicator * area_below_tangents + (1 - convexity_indicator) * area_below_secants

    #choose interval
    key, subkey = jax.random.split(key)
    position = jax.random.choice(a = jnp.arange(len(x_values)-1), key = subkey, p = area_below_secants / jnp.sum(area_below_secants))#jnp.searchsorted(x_values,u)
    
    #generate sample in the chosen inerval
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(key = subkey)
    
    #sample from exp(a *x + b) 
    a,b             = m_convex[position], n_convex[position]
    x_left, x_right = x_values[position], x_values[position + 1]
    
    sample = jnp.log( u * jnp.exp(a * x_right) + (1-u) * jnp.exp(a * x_left) ) / a
    
    key, subkey = jax.random.split(key)

    return sample, area_upper_bound / area_lower_bound



    
#________________________________ log densitiy for sampling ________________________________#

log_density_sampling_dictionary = dict()
for _ in ['gaussian','norminvgauss']:
    
    log_density_sampling_dictionary[_] = jax.vmap(jax.value_and_grad(jax_log_pdf_for_sampling(_)), in_axes = (0,None, None,None), out_axes=0)
    log_density_sampling_dictionary[_] = jax.jit(log_density_sampling_dictionary[_])
    
#___________________________________________________________________________________________#



vectorized_sampling = jax.vmap(approx_sampling_from_log_linear_interpolation, in_axes = (0,0, None,None, None, 0), out_axes=0)
vectorized_sampling = jax.jit(vectorized_sampling,static_argnames = 'distr_name')
#test
num_x = 25
x_values   = jnp.array([jnp.linspace(-12,12,num_x), jnp.linspace(-13,14,25)]).reshape(2,num_x)
sum_       = jnp.array([2.5,3.5]).reshape(2,1)
areas      = jnp.array([1.2,0.9]) 
params     = jnp.array([3.,2.,1.5,1.5])
distr_name = 'norminvgauss'
key        = jax.random.PRNGKey(seed = 4) 
key        = jax.random.split(key, 2)

sample, area_ratios = vectorized_sampling(x_values, sum_, distr_name, params, areas, key)


@jax.jit
def approx_sampling_from_log_linear_interpolation_testing(x_values, key):
    """f_prime is a componentwise differentiation, e.g. constructed by vmap"""
    
    #
    y_values            = f(x_values)
    y_prime             = f_prime(x_values)
    x1, x2              = x_values[:-1], x_values[1:]
    y1, y2              = y_values[:-1], y_values[1:]
    y_prime1, y_prime2  = y_prime[:-1],  y_prime[1:]
    convexity_indicator = jnp.where(y_prime2 - y_prime1 > 0, 1, 0 )    
    
    ########## construct convex envelopes ##########
    #y - y1 = (y2 - y1)/(x2 - x1) × (x - x1) <-> y = (y2 - y1)/(x2 - x1) * x + y1 - (y2 - y1)/(x2 - x1) * x1
    m_convex = (y2 - y1) / (x2 - x1)
    n_convex = y1 - m_convex * x1
    area_below_secants = (jnp.exp(m_convex * x2 + n_convex) - jnp.exp(m_convex * x1 + n_convex))/m_convex
    
    #take care of 0 slopes, which we can't divide by
    zero_slope_convex_correction = jnp.where(m_convex == 0,(x2-x1) * jnp.exp(y1),0)
    area_below_secants  = jnp.nan_to_num(area_below_secants)
    area_below_secants += zero_slope_convex_correction     #m_convex_equal_0 = jnp.where(m_convex == 0,1,0)    #area_below_secants = area_below_secants.at[m_convex_equal_0].set((x2-x1) * jnp.exp(y1)[m_convex_equal_0])
    
    ########## construct concave envelopes ##########
    n1                  = y1 - y_prime1 * x1
    n2                  = y2 - y_prime2 * x2
    intersection_points = (n2 - n1) / (y_prime1 - y_prime2)
    
    y_primes_equal_correction = jnp.where(y_prime1 == y_prime2, x1/2 + x2/2, 0)
    intersection_points       = jnp.nan_to_num(intersection_points)
    intersection_points      += y_primes_equal_correction
    
    _area_1 = (jnp.exp(y_prime1 * intersection_points + n1) - jnp.exp(y_prime1 * x1 + n1))/y_prime1
    _area_2 = (jnp.exp(y_prime2 * x2 + n2) - jnp.exp(y_prime2 * intersection_points + n2))/y_prime2
    
    #take care of 0 slopes, which we can't divide by
    y_prime1_correction = jnp.where(y_prime1==0,(intersection_points-x1) * jnp.exp(y1),0)
    _area_1  = jnp.nan_to_num(_area_1)
    _area_1 += y_prime1_correction

    #take care of 0 slopes, which we can't divide by
    y_prime2_correction = jnp.where(y_prime2==0,(x2 - intersection_points) * jnp.exp(y2),0)
    _area_2  = jnp.nan_to_num(_area_2)
    _area_2 += y_prime2_correction

    area_below_tangents = _area_1 + _area_2
    
    ###################################################
    #deal with tails
    
    left_tail_log_slope      = y_prime[0]
    right_tail_log_slope     = y_prime[-1]
    
    left_tail_area  =  jnp.exp(y_values[0])  /  left_tail_log_slope
    right_tail_area = -jnp.exp(y_values[-1]) /  right_tail_log_slope
    
    big_number = len(x_values) * 3 + 100
    x_values = jnp.insert(x_values, jnp.array([0,big_number]) ,jnp.array([-jnp.inf,jnp.inf]))
    area_below_tangents = jnp.insert(area_below_tangents, jnp.array([0,big_number]), jnp.array([left_tail_area, right_tail_area]))
    area_below_secants  = jnp.insert(area_below_secants, jnp.array([0,big_number]), jnp.array([left_tail_area, right_tail_area]))    
    convexity_indicator = jnp.insert(convexity_indicator, jnp.array([0,big_number]), jnp.array([1,1]))
    ####################################################
    
    area_upper_bound   = convexity_indicator * area_below_secants + (1 - convexity_indicator) * area_below_tangents
    area_lower_bound   = convexity_indicator * area_below_tangents + (1 - convexity_indicator) * area_below_secants

    #choose interval
    key, subkey = jax.random.split(key)
    position = jax.random.choice(a = jnp.arange(len(x_values)-1), key = subkey, p = area_below_secants / jnp.sum(area_below_secants))#jnp.searchsorted(x_values,u)
    
    #generate sample in the chosen inerval
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(key = subkey)
    
    #sample from exp(a *x + b) 
    a,b             = m_convex[position], n_convex[position]
    x_left, x_right = x_values[position], x_values[position + 1]
    
    sample = jnp.log( u * jnp.exp(a * x_right) + (1-u) * jnp.exp(a * x_left) ) / a
    
    key, subkey = jax.random.split(key)


    #jax.lax.cond(position > 0 and position < len(x_values)-1)

    return sample
    #return (m_convex, n_convex), (intersection_points, (y_prime1, n1) , (y_prime2, n2)),(area_lower_bound, area_upper_bound),sample, key 




if __name__ == '__main__':
    
    def g(x):
        return jnp.exp(-0.05 - abs(x-0.15)**1.1)/jnp.sqrt(5 + abs(x-0.15)**1.1)
    
    g_prime = jax.grad(g)
    
    f = jax.jit(jax.vmap(lambda x: jnp.log(g(x))))
    f_prime = jax.jit(jax.vmap(jax.grad(lambda x: jnp.log(g(x)))))
    x_values = jnp.array([-15,-14,-13,-12,-11,-10,-9, -6,-4,-2,-0.5,-0.25,0,0.15,0.25, 0.5,2,4,5,6,7,8,9,10,11,12,15,16,17,])
    key = jax.random.PRNGKey(1)
    
    samples = []
    for i in range(5000):
        (m_convex, n_convex), (intersection_points, (y_prime1, n1) , (y_prime2, n2)),(area_lower_bound, area_upper_bound), sample, key =  approx_sampling_from_log_linear_interpolation_testing(x_values,key)   
        samples.append(sample)
    print(area_lower_bound.sum()  ,area_upper_bound.sum()) 
    import matplotlib.pyplot as plt
    f_,ax = plt.subplots(nrows = 1, ncols = 4, sharex = True, sharey = True)

    many_x = np.linspace(x_values[0],x_values[-1],1000)


    for i in range(len(x_values)):
        m,n = m_convex[i], n_convex[i]
        end_points = np.array([x_values[i],x_values[i+1]])
        x   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[0].plot(x,np.exp(m*x+n),c = 'black')
        ax[0].scatter(end_points,np.exp(m*end_points+n),marker ='*',c = 'black')
        
        #concave left
        m_left, n_left = y_prime1[i], n1[i]
        end_points         = np.array([x_values[i],intersection_points[i]]) 
        x_left   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[1].plot(x_left, np.exp(m_left * x_left + n_left), c = 'black')
        ax[1].scatter(end_points, np.exp(m_left * end_points + n_left), marker ='*',c = 'black')
        
        #mixed left
        ax[2].plot(x_left, np.exp((m_left + m)/2 * x_left + (n_left + n)/2), c = 'black')
        ax[2].scatter(end_points, np.exp((m_left + m)/2 * end_points + (n_left + n)/2), marker ='*',c = 'black')
            
        #concave right
        m_right, n_right = y_prime2[i], n2[i]
        end_points          = np.array([intersection_points[i], x_values[i+1]]) 
        x_right   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[1].plot(x_right, np.exp(m_right * x_right + n_right), c = 'black')
        ax[1].scatter(end_points, np.exp(m_right * end_points + n_right), marker ='*',c = 'black')
        
        #mixed right
        ax[2].plot(x_right, np.exp((m_right + m)/2 * x_right + (n_right + n)/2), c = 'black')
        ax[2].scatter(end_points, np.exp((m_right + m)/2 * end_points + (n_right + n)/2), marker ='*',c = 'black')
        
    #set titles
    ax[0].set_title('convex')    
    ax[1].set_title('concave')
    ax[2].set_title('mixed')
    #plot function
    ax[0].plot(many_x,g(many_x), label='function',c='r')
    ax[1].plot(many_x,g(many_x), label='function',c='r')
    ax[2].plot(many_x,g(many_x), label='function',c='r')
    f_.suptitle('function')


    f_,ax = plt.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True)

    many_x = np.linspace(x_values[0],x_values[-1],1000)


    for i in range(len(m_convex)):
        m,n = m_convex[i], n_convex[i]
        end_points = np.array([x_values[i],x_values[i+1]])
        x   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[0].plot(x, m*x+n,c = 'black')
        ax[0].scatter(end_points,m*end_points+n,marker ='*',c = 'black')
        
        #concave left
        m_left, n_left = y_prime1[i], n1[i]
        end_points         = np.array([x_values[i],intersection_points[i]]) 
        x_left   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[1].plot(x_left, m_left * x_left + n_left, c = 'black')
        ax[1].scatter(end_points, m_left * end_points + n_left, marker ='*',c = 'black')
        
        #mixed left
        ax[2].plot(x_left, (m_left + m)/2 * x_left + (n_left + n)/2, c = 'black')
        ax[2].scatter(end_points, (m_left + m)/2 * end_points + (n_left + n)/2, marker ='*',c = 'black')
            
        #concave right
        m_right, n_right = y_prime2[i], n2[i]
        end_points          = np.array([intersection_points[i], x_values[i+1]]) 
        x_right   = many_x[np.logical_and(many_x >= end_points[0], many_x <= end_points[1])]

        ax[1].plot(x_right, m_right * x_right + n_right, c = 'black')
        ax[1].scatter(end_points, m_right * end_points + n_right, marker ='*',c = 'black')
        
        #mixed left
        ax[2].plot(x_right, (m_right + m)/2 * x_right + (n_right + n)/2, c = 'black')
        ax[2].scatter(end_points, (m_right + m)/2 * end_points + (n_right + n)/2, marker ='*',c = 'black')
        
    #set titles
    ax[0].set_title('convex')    
    ax[1].set_title('concave')
    ax[2].set_title('mixed')
    #plot function
    ax[0].plot(many_x,f(many_x), label='function',c='r')
    ax[1].plot(many_x,f(many_x), label='function',c='r')
    ax[2].plot(many_x,f(many_x), label='function',c='r')
    f_.suptitle('log function')
    
    fig__, ax__ = plt.subplots()
    ax__.hist(samples, density = True, bins = 30)
    ax__.plot(jnp.sort(jnp.array(samples)), g(jnp.sort(jnp.array(samples))) / area_lower_bound.sum())


    
    #tangents_list = list(zip()) #[(intersection_point, (m1,n1),(m2,n2))]
    
    

# --------------------------- old ------------------------------------------------------------

# def construct_envelope(f, f_prime, x_values, left_tail_log_slope, righ_tail_log_slope):
#     """
#     f: log_pdf
#     f_prime: derivative of log_pdf"""
    
    
#     #x_values should be sorted already
#     assert isinstance(x_values, np.ndarray)
    
#     y_values      = f(x_values)
#     y_prime       = f_prime(x_values)
    
#     left_tail_log_slope = y_prime[0]
#     righ_tail_log_slope = y_prime[-1]
    
#     assert left_tail_log_slope > 0 and righ_tail_log_slope < 0

    
#     x1, x2              = x_values[:-1], x_values[1:]
#     y1, y2              = y_values[:-1], y_values[1:]
#     y_prime1, y_prime2  = y_prime[:-1],  y_prime[1:]
#     convexity_indicator = ((y_prime2 - y_prime1)>0) * 1

#     #construct convex envelopes
#     #y - y1 = (y2 - y1)/(x2 - x1) × (x - x1) <-> y = (y2 - y1)/(x2 - x1) * x + y1 - (y2 - y1)/(x2 - x1) * x1
#     m_convex = (y2 - y1) / (x2 - x1)
#     n_convex = y1 - m_convex * x1
#     area_below_secants = (np.exp(m_convex * x2 + n_convex) - np.exp(m_convex * x1 + n_convex))/m_convex
#     if np.any(m_convex == 0):
#         #raise ValueError('check before using')
#         area_below_secants[np.where(m_convex==0)[0]] = ((x2-x1) * np.exp(y1))[np.where(m_convex==0)[0]]

    
#     #construct concave envelopes
#     n1                  = y1 - y_prime1 * x1
#     n2                  = y2 - y_prime2 * x2
#     intersection_points = (n2 - n1) / (y_prime1 - y_prime2)
    
#     if np.any(y_prime1 == y_prime2 ):
#         intersection_points[np.where(y_prime1 == y_prime2)[0]] = (x1/2 + x2/2)[np.where(y_prime1 == y_prime2)[0]]
    
#     _area_1 = (np.exp(y_prime1 * intersection_points + n1) - np.exp(y_prime1 * x1 + n1))/y_prime1
#     _area_2 = (np.exp(y_prime2 * x2 + n2) - np.exp(y_prime2 * intersection_points + n2))/y_prime2
    
#     if np.any(y_prime1 == 0):
#         #raise ValueError('check before using')
#         _area_1[np.where(y_prime1==0)[0]] = ((intersection_points-x1) * np.exp(y1))[np.where(y_prime1==0)[0]]

#     if np.any(y_prime2 == 0):
#             #raise ValueError('check before using')
#         _area_2[np.where(y_prime2==0)[0]] = ((x2 - intersection_points) * np.exp(y2))[np.where(y_prime2==0)[0]]

    
#     area_below_tangents = _area_1 + _area_2
    
#     secants_list  = list(zip(m_convex, n_convex))
#     tangents_list = list(zip(intersection_points, zip(y_prime1, n1), zip(y_prime2, n2))) 
    
#     all_areas    = convexity_indicator * area_below_secants + (1 - convexity_indicator) * area_below_tangents
    
#     left_tail_area  =  np.exp(y_values[0])  /  left_tail_log_slope
#     right_tail_area = -np.exp(y_values[-1]) /  right_tail_log_slope
    
#     all_areas = np.insert(all_areas, 0, left_tail_area)
#     all_areas = np.append(arr = all_areas, values = right_tail_area)
#     total_area = np.sum(all_areas) # 1/ total_area is the accepantance rate
#     probabilities = all_areas / total_area
    
#     #only for testing purposes
#     #total_convex_area  = np.sum(area_below_secants)
#     #total_concave_area = np.sum(area_below_tangents)
#     return secants_list, tangents_list, convexity_indicator, total_area, probabilities
    