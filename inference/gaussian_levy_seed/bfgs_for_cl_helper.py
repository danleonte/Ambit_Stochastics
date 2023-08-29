from jax_aux_jitted_gaussian import minus_lambda_function_for_cl_and_grad_at_lags
from modified_minimize import modified_minimize
import jax


def do_modified_bfgs(trawl_path, envelope,  tau , nr_mc_samples_per_batch, nr_batches,
                     max_taylor_deg, key, lags_list, x0, max_iter_at_once_bfgs, max_batches_bfgs):

    bfgs_batch = 0
    nfev       = 0
    
    assert max_batches_bfgs > 0
    while bfgs_batch < max_batches_bfgs:
        
        
        minus_lambda_func = minus_lambda_function_for_cl_and_grad_at_lags(trawl_path, envelope, \
                                        tau , nr_mc_samples_per_batch, nr_batches, 
                                        max_taylor_deg, key, lags_list)
            
        resdd= modified_minimize(fun= minus_lambda_func, x0= x0, jac=True, method='BFGS',
                                     options = {'maxiter' : max_iter_at_once_bfgs})
        
        
        bfgs_batch += 1
        nfev += resdd.nfev
        x0 = resdd.x

        #update the randomness we're using
        for ___ in range(nfev):
            key, subkey = jax.random.split(key)  
            
        #if optimization is successful, return results 
        if resdd.success == True:
            return resdd, key
                

    
    return resdd, key