import numpy as np
import matplotlib.pyplot as plt

                            
#from ambit_stochastics.trawl import trawl   

tau = 0.15
nr_trawls = 1000 
nr_simulations = 10
trawl_function = lambda x :   0.5 * np.exp(0.5 * x) * (x<=0)
#trawl_function= lambda x :  (x> -2) * (x<=0) * (2 - (-x) **2/2) 
#decorrelation_time =-2
gaussian_part_params = (0,0)
jump_part_params = (2,3)
jump_part_name   = 'gamma'
decorrelation_time = -np.inf
#mesh_size = 0.05
#truncation_grid = -2
#times_grid =  tau * np.arange(1,nr_trawls+1,1) #important to keep it this way

trawl_slice = trawl(nr_trawls = nr_trawls, nr_simulations = nr_simulations,
               trawl_function = trawl_function,tau =  tau,decorrelation_time =  decorrelation_time, 
               gaussian_part_params = gaussian_part_params,
               jump_part_name =  jump_part_name,jump_part_params = jump_part_params )   

trawl_slice.simulate('slice')
trawl_slice.fit_gmm(trawl_slice.values,'exponential','gamma',5)
