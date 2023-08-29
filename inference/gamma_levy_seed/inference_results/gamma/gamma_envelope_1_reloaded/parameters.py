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

levy_seed = 'gamma'
tau = 1
nr_trawls = 2500
nr_simulations = 100
TRUE_GAMMA_PARAMS = (
4, 3)  # alpha, beta (rate) not alpha, theta (scale). trawl simulation uses alpha, scale, so need to pass 5, 1/2 there
envelope = 'gamma'
jax_seed = 14243424
#key = random.PRNGKey(jax_seed)

# trawl function parameters
TRUE_ENVELOPE_PARAMS = (0.5,0.75)
np.random.seed(seed=11234429)

np_random_seeds = np.random.randint(low=1, high=2 ** 31, size=1)

# inference params
nr_mc_samples_per_batch = 1250
nr_batches = 15   # giving nr_mc_samples_per_batch * nr_batches total samples
max_taylor_deg = 3  # degree of the taylor polynomial used as control variate

# bfgs params
max_iter_at_once_bfgs = 20
max_batches_bfgs = 1

lags_list = ((1,3,5,10,15,20,30,40),)#,(1,5,10),(1,5,10,15),(1,5,10,15,20)),(1,3,5,10,15),(1,3,5,10,20))
n_values = (2000,1500,1000,750,500)#(1000,500,250,150)#,1000, 2500, 5000)#,750,1000,1500)
 



    

