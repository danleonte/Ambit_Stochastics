"""Checks the inputs to the `trawl` and `simple_ambit_field classes`."""

import pandas as pd
import numpy as np

def check1(nr_trawls,nr_simulations):
    #assert (isinstance(tau,float) or isinstance(tau,int)) and tau > 0
    assert isinstance(nr_trawls,int) and nr_trawls >0
    assert isinstance(nr_simulations,int) and nr_simulations >0
    
def check_spatio_temporal_positions(x,tau,k_s,k_t,nr_simulations):
    assert (isinstance(x,float) or isinstance(x,int)) and x > 0
    assert (isinstance(tau,float) or isinstance(tau,int)) and tau > 0
    assert isinstance(k_s,int) and k_s >0
    assert isinstance(k_t,int) and k_t >0
    assert isinstance(nr_simulations,int) and nr_simulations >0


def check_trawl_function(phi):
    """Check if the function is increasing and 0 for strictly positive values"""
    assert callable(phi),'trawl_function is not a function'
    assert phi(0.000001) == 0,'trawl_function does not satisfy trawl_function(t)=0 for t >0'
        
    phi_values = phi(np.linspace(-100,0,10**5))
    assert pd.Series(phi_values).is_monotonic_increasing,'trawl_function is not increasing'   

#distributional checks
        
def check_gaussian_params(gaussian_part_params):
        """Check if the distribution of the jump part is supported and if the parameters 
        of the Gaussian part are numbers"""
        #gaussian part params
        assert isinstance(gaussian_part_params,tuple),'gaussian_part_params is not a tuple'
        assert all(isinstance(i,(int,float)) for i in gaussian_part_params),'parameters of the gaussian part are not numbers'
        
            
def check_jump_part_and_params(jump_part_name,jump_part_params):
        """Check if the distribution of the jump part is supported and if the parameters 
        of the Jump parts are numbers"""
        #jump part params
        assert isinstance(jump_part_params,tuple),'jump_part_params is not a tuple'
        assert all(isinstance(i,(int,float)) for i in jump_part_params),'parameters of the jump part are not numbers'
        
        #jump part_name
        if jump_part_name in ['invgauss','norminvgauss','geninvgauss','nbinom']: #to also add hyperbolic distributions 
            raise ValueError('distribution not yet supported')

        elif jump_part_name not in [None,'gamma','cauchy','poisson']:
            raise ValueError('unknown distribution')
            
def check_grid_params(mesh_size,truncation_grid,times_grid):
    assert  isinstance(mesh_size,(int,float)) and mesh_size >0,'please check mesh size'
    assert  isinstance(truncation_grid,(int,float)) and truncation_grid < 0
            
def check_cpp():
    pass
        


        