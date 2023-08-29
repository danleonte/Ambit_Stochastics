#jax imports
#from jax.config import config
#config.update("jax_enable_x64", True)
#from bfgs_for_cl_helper import do_modified_bfgs
#from   jax import random
#import jax.numpy as jnp
#import jax
import numpy as np

#default_precision = jnp.float64
#print('Default precision is: ',default_precision)


# trawl simulation parameters
tau = 0.1
nr_trawls = 2505
nr_simulations = 100
TRUE_GAMMA_PARAMS = (3.,
                     0.75)  # alpha, beta (rate) not alpha, theta (scale). trawl simulation uses alpha, scale, so need to pass 5, 1/2 there
envelope = 'gamma_H'
levy_seed = 'gamma'

def trawl_function_for_gmm_envelope_fit(params):
    return lambda x: params[0] / 1. * (1. - x / 1.) ** (-params[0] - 1.) * (x <= 0)


# trawl function parameters
TRUE_ENVELOPE_PARAMS = (2.5, 1.)

if envelope == 'gamma_H':
    aux_param = TRUE_ENVELOPE_PARAMS[1:]
    envelope2 = 'gamma'
elif envelope == 'exponential':
    aux_param = None
    envelope2 = 'exponential'

jax_seed = 23423545
#key = random.PRNGKey(jax_seed)
np.random.seed(seed=342151445)
np_random_seeds = np.random.randint(low=1, high=2 ** 31, size=1)

# inference params
nr_mc_samples_per_batch = 2 * 10 ** 3
nr_batches = 5  # giving nr_mc_samples_per_batch * nr_batches total samples
max_taylor_deg = 3  # degree of the taylor polynomial used as control variate

# bfgs params
max_iter_at_once_bfgs = 20
max_batches_bfgs = 1

lags_list = ((1, 3, 5, 7, 10), (1, 3, 5, 7), (1, 3, 5))
n_values = (2000, 1500, 1000, 750, 500)




    

