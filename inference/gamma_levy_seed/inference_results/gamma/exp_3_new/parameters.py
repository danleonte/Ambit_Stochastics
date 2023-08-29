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




#trawl simulation parameters
tau = 1
nr_trawls = 2500
nr_simulations = 100
TRUE_GAMMA_PARAMS = (3,0.75) #alpha, beta (rate) not alpha, theta (scale). trawl simulation uses alpha, scale, so need to pass 5, 1/2 there
envelope = 'exponential'
jax_seed = 354925
levy_seed = 'gamma'
#key = random.PRNGKey(jax_seed)

#trawl function parameters
if envelope == 'exponential':
    TRUE_ENVELOPE_PARAMS = (0.4,)
    np.random.seed(seed = 324351189)

elif envelope == 'gamma':
    TRUE_ENVELOPE_PARAMS = (1.5,1.)
    np.random.seed(seed = 656881786)

#np_random_seeds = np.random.randint(low = 1, high = 2**31, size = 1)


#inference params
nr_mc_samples_per_batch =  1500
nr_batches =  7 #giving nr_mc_samples_per_batch * nr_batches total samples 
max_taylor_deg = 3  # degree of the taylor polynomial used as control variate

#bfgs params
max_iter_at_once_bfgs = 15
max_batches_bfgs      = 1


lags_list =  ((1,2,3,4,5),(1,2,3,4),(1,2,3))
n_values = (2000,1500,1000,750,500,250)







