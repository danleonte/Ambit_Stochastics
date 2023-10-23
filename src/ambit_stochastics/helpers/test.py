# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:55:45 2023

@author: dleon
"""
# from adaptive_sampling import approx_sampling_from_log_linear_interpolation
from ambit_stochastics.helpers.mode_aux import find_mode_numerically_from_func
from ambit_stochastics.trawl import trawl
import matplotlib.pyplot as plt
# from ambit_stochastics.helpers.densities import pdf_or_log_pdf
import jax
import jax.numpy as jnp
import scipy
import numpy as np
from itertools import chain
from scipy.stats import norm
from scipy.stats import norminvgauss as scipy_norminvgauss
import tensorflow_probability.substrates.jax as tfp
norminvgauss = tfp.distributions.NormalInverseGaussian
nr_points = 25


def get_unnormalized_likelihood(distr_name, params, areas):
    """areas should be one of areas_1_i or areas_i"""
    # areas  = np.array(list(chain.from_iterable(areas)))
    a, b, loc, scale = params
    r = norminvgauss(a, b,  loc * jnp.array(areas),
                     scale * jnp.array(areas)).prob

    # the slices input x should be unraveld and with no nans

    return lambda x: jnp.prod(jnp.array(r(x)))


distr_name = 'norminvgauss'
params = (1, 0.5, 2., 0.75)
first_area, remaining_area = 0.35, 0.12
sum_ = 0.25

log_density_unnormalized = get_unnormalized_likelihood(distr_name=distr_name, params=params,
                                                       areas=jnp.array([first_area, remaining_area]))
# to double check
log_constant = get_unnormalized_likelihood(distr_name=distr_name, params=params,
                                           areas=first_area + remaining_area)(sum_)


def log_density(x): return log_density_unnormalized(jnp.array([x, sum_ - x])) / log_constant


a, b, loc, scale = params
b = 0

lower, upper = params[2] * first_area, sum_ - params[2] * remaining_area
v1 = scipy_norminvgauss(a=a * first_area, b=0, loc=loc *
                        first_area, scale=scale * first_area).stats(moments='v')
v2 = scipy_norminvgauss(a=a * remaining_area, b=0, loc=loc *
                        remaining_area, scale=scale * remaining_area).stats(moments='v')

if lower > upper:
    (lower, upper) = (upper, lower)
    (v1, v2) = (v2, v1)

mode = find_mode_numerically_from_func(
    func=log_density, lower=lower, upper=upper)
v1 = float(v1)
v2 = float(v2)
initial_interpolation_points = norm.isf(
    np.linspace(start=0.005, stop=0.995, num=nr_points))
initial_interpolation_points = jnp.array(initial_interpolation_points)
gaussian_variance = 1/(1/v1 + 1/v2)
interpolation_points1 = mode + initial_interpolation_points * gaussian_variance**0.5
interpolation_points2 = np.linspace(
    lower - 3 * v1**0.5, upper + 3 * v2**0.5, len(initial_interpolation_points))

interpolation_points = np.concatenate(
    [interpolation_points1, interpolation_points2])
interpolation_points.sort()
interpolation_points = jnp.array(interpolation_points)
key = jax.random.PRNGKey(4)


def g(x): return jnp.array(log_density(jnp.array(x)))


g_prime = jax.grad(g)

f = jax.jit(jax.vmap(lambda x: jnp.log(g(x))))
f_prime = jax.jit(jax.vmap(jax.grad(lambda x: jnp.log(g(x)))))


#@jax.jit
def approx_sampling_from_log_linear_interpolation(x_values, key):
    """f_prime is a componentwise differentiation, e.g. constructed by vmap"""

    #
    y_values = f(x_values)
    y_prime = f_prime(x_values)
    x1, x2 = x_values[:-1], x_values[1:]
    y1, y2 = y_values[:-1], y_values[1:]
    y_prime1, y_prime2 = y_prime[:-1],  y_prime[1:]
    convexity_indicator = jnp.where(y_prime2 - y_prime1 > 0, 1, 0)

    ########## construct convex envelopes ##########
    # y - y1 = (y2 - y1)/(x2 - x1) × (x - x1) <-> y = (y2 - y1)/(x2 - x1) * x + y1 - (y2 - y1)/(x2 - x1) * x1
    m_convex = (y2 - y1) / (x2 - x1)
    n_convex = y1 - m_convex * x1
    area_below_secants = (jnp.exp(m_convex * x2 + n_convex) -
                          jnp.exp(m_convex * x1 + n_convex))/m_convex

    # take care of 0 slopes, which we can't divide by
    zero_slope_convex_correction = jnp.where(
        m_convex == 0, (x2-x1) * jnp.exp(y1), 0)
    area_below_secants = jnp.nan_to_num(area_below_secants)
    # m_convex_equal_0 = jnp.where(m_convex == 0,1,0)    #area_below_secants = area_below_secants.at[m_convex_equal_0].set((x2-x1) * jnp.exp(y1)[m_convex_equal_0])
    area_below_secants += zero_slope_convex_correction

    ########## construct concave envelopes ##########
    n1 = y1 - y_prime1 * x1
    n2 = y2 - y_prime2 * x2
    intersection_points = (n2 - n1) / (y_prime1 - y_prime2)

    y_primes_equal_correction = jnp.where(y_prime1 == y_prime2, x1/2 + x2/2, 0)
    intersection_points = jnp.nan_to_num(intersection_points)
    intersection_points += y_primes_equal_correction

    _area_1 = (jnp.exp(y_prime1 * intersection_points + n1) -
               jnp.exp(y_prime1 * x1 + n1))/y_prime1
    _area_2 = (jnp.exp(y_prime2 * x2 + n2) -
               jnp.exp(y_prime2 * intersection_points + n2))/y_prime2

    # take care of 0 slopes, which we can't divide by
    y_prime1_correction = jnp.where(
        y_prime1 == 0, (intersection_points-x1) * jnp.exp(y1), 0)
    _area_1 = jnp.nan_to_num(_area_1)
    _area_1 += y_prime1_correction

    # take care of 0 slopes, which we can't divide by
    y_prime2_correction = jnp.where(
        y_prime2 == 0, (x2 - intersection_points) * jnp.exp(y2), 0)
    _area_2 = jnp.nan_to_num(_area_2)
    _area_2 += y_prime2_correction

    area_below_tangents = _area_1 + _area_2

    ###################################################
    # deal with tails

    left_tail_log_slope = y_prime[0]
    right_tail_log_slope = y_prime[-1]

    left_tail_area = jnp.exp(y_values[0]) / left_tail_log_slope
    right_tail_area = -jnp.exp(y_values[-1]) / right_tail_log_slope

    big_number = len(x_values) * 3 + 100
    x_values = jnp.insert(x_values, jnp.array(
        [0, big_number]), jnp.array([-jnp.inf, jnp.inf]))
    area_below_tangents = jnp.insert(area_below_tangents, jnp.array(
        [0, big_number]), jnp.array([left_tail_area, right_tail_area]))
    area_below_secants = jnp.insert(area_below_secants, jnp.array(
        [0, big_number]), jnp.array([left_tail_area, right_tail_area]))
    convexity_indicator = jnp.insert(
        convexity_indicator, jnp.array([0, big_number]), jnp.array([1, 1]))
    ####################################################

    area_upper_bound = convexity_indicator * area_below_secants + \
        (1 - convexity_indicator) * area_below_tangents
    area_lower_bound = convexity_indicator * area_below_tangents + \
        (1 - convexity_indicator) * area_below_secants

    key, subkey = jax.random.split(key)
    # jax.randon.choice(key = subkey, )
    u = jax.random.uniform(key=subkey)
    position = jnp.searchsorted(x_values, u)
    key, subkey = jax.random.split(key)

    # tail_sample =
    # body_sample =

    # jax.lax.cond(position > 0 and position < len(x_values)-1)

    return (m_convex, n_convex), (intersection_points, (y_prime1, n1), (y_prime2, n2)), (area_lower_bound, area_upper_bound)


x_values = interpolation_points
# approx_sampling_from_log_linear_interpolation(interpolation_points,key)
(m_convex, n_convex), (intersection_points, (y_prime1, n1), (y_prime2, n2)), (area_lower_bound,
                                                                              area_upper_bound) = approx_sampling_from_log_linear_interpolation(x_values, key)
print(area_lower_bound.sum(), area_upper_bound.sum())
f_, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

many_x = np.linspace(x_values[0], x_values[-1], 250)


for i in range(len(x_values)):
    m, n = m_convex[i], n_convex[i]
    end_points = np.array([x_values[i], x_values[i+1]])
    x = many_x[np.logical_and(many_x >= end_points[0],
                              many_x <= end_points[1])]

    ax[0].plot(x, np.exp(m*x+n), c='black')
    ax[0].scatter(end_points, np.exp(m*end_points+n), marker='*', c='black')

    # concave left
    m_left, n_left = y_prime1[i], n1[i]
    end_points = np.array([x_values[i], intersection_points[i]])
    x_left = many_x[np.logical_and(
        many_x >= end_points[0], many_x <= end_points[1])]

    ax[1].plot(x_left, np.exp(m_left * x_left + n_left), c='black')
    ax[1].scatter(end_points, np.exp(
        m_left * end_points + n_left), marker='*', c='black')

    # mixed left
    ax[2].plot(x_left, np.exp((m_left + m)/2 *
               x_left + (n_left + n)/2), c='black')
    ax[2].scatter(end_points, np.exp((m_left + m)/2 *
                  end_points + (n_left + n)/2), marker='*', c='black')

    # concave right
    m_right, n_right = y_prime2[i], n2[i]
    end_points = np.array([intersection_points[i], x_values[i+1]])
    x_right = many_x[np.logical_and(
        many_x >= end_points[0], many_x <= end_points[1])]

    ax[1].plot(x_right, np.exp(m_right * x_right + n_right), c='black')
    ax[1].scatter(end_points, np.exp(
        m_right * end_points + n_right), marker='*', c='black')

    # mixed right
    ax[2].plot(x_right, np.exp((m_right + m)/2 *
               x_right + (n_right + n)/2), c='black')
    ax[2].scatter(end_points, np.exp((m_right + m)/2 *
                  end_points + (n_right + n)/2), marker='*', c='black')


# set titles
ax[0].set_title('convex')
ax[1].set_title('concave')
ax[2].set_title('mixed')
# plot function

#many_x_2 = jnp.array(np.transpose([many_x, sum_ - many_x]))
# ax[0].plot(many_x,f(many_x_2), label='function',c='r')
# ax[1].plot(many_x,f(many_x_2), label='function',c='r')
# ax[2].plot(many_x,f(many_x_2), label='function',c='r')
# f_.suptitle('log function')

g = jax.vmap(g)

# plot function
ax[0].plot(many_x, g(many_x), label='function', c='r')
ax[1].plot(many_x, g(many_x), label='function', c='r')
ax[2].plot(many_x, g(many_x), label='function', c='r')
f_.suptitle('function')

# tangents_list = list(zip()) #[(intersec
# np.random.seed(243)

# nr_simulations   = 1
# tau              = 0.5
# nr_trawls        = 5
# jump_part_name   = 'norminvgauss'
# jump_part_params = (5,0,3,4.25)


# nig_trawl = trawl(nr_simulations = nr_simulations,tau = tau, nr_trawls = nr_trawls, \
#                   trawl_function = lambda x : np.exp(x) * (x<=0),
#                   jump_part_name = jump_part_name, jump_part_params = jump_part_params)

# nig_trawl.simulate('slice')
# values = nig_trawl.values

# x_1_3 = values[0,:3]
# x_1,x_2,x_3 = x_1_3
# areas_1         = nig_trawl.slice_areas_matrix[:,0]
# areas_2         = nig_trawl.slice_areas_matrix[:-1,1]
# areas_3         = nig_trawl.slice_areas_matrix[:-2,2]
