#generic imports
import numpy as np
import time
import statsmodels.api  as sm


#ambit stochastics imports
from generate_trawls_with_gamma_marginal import generate_gamma_seed_trawls
#gmm fitting of envelope params + gmm / mle for levy seed params
from ambit_stochastics.helpers.marginal_distribution_functions import fit_trawl_marginal
from ambit_stochastics.helpers.acf_functions import fit_trawl_envelope_gmm
import pickle

#jax imports
from jax.config import config
config.update("jax_enable_x64", True)
from bfgs_for_cl_helper import do_modified_bfgs
from   jax import random
import jax.numpy as jnp
import jax

default_precision = jnp.float64
print('Default precision is: ',default_precision)


#vanilla estimator using acf at lag 1
#this is np
def get_initial_value_for_1_param_trawl_function(tau_ ,values_,envelope,aux_param):
    assert len(values_.shape) < 2
    
    emp_corr_at_lag_1 =  np.array(sm.tsa.stattools.acf(values_,nlags =1))[[1]]
    
    #if envelope == 'exponential':
    #    assert aux_param == None
    #    return (-1/tau_) * np.log(emp_corr_at_lag_1)  

    assert envelope =='gamma_H'
    assert isinstance(aux_param,tuple) and len(aux_param) == 1
    delta = aux_param[0]
    return - np.log(emp_corr_at_lag_1) / np.log(1+ tau_/delta)

                         
if __name__ == "__main__":
    
    ##########################simulation study parameters#########################
    
    #trawl simulation parameters
    tau = 0.1
    nr_trawls = 2505
    nr_simulations = 100
    TRUE_GAMMA_PARAMS = (3., 0.75) #alpha, beta (rate) not alpha, theta (scale). trawl simulation uses alpha, scale, so need to pass 5, 1/2 there
    envelope = 'gamma_H'

    def trawl_function_for_gmm_envelope_fit(params):
        return        lambda x: params[0] / 1. * (1. - x / 1.) ** (-params[0] - 1.) * (x <= 0)

    #trawl function parameters
    TRUE_ENVELOPE_PARAMS = (0.5,1.)

    if envelope == 'gamma_H':
       aux_param = TRUE_ENVELOPE_PARAMS[1:]
       envelope2 = 'gamma'
    elif envelpe == 'exponential':
       aux_param = None
       envelope2 = exponential
    
    assert envelope == 'gamma_H'


    jax_seed = 14233515
    key = random.PRNGKey(jax_seed)
    np.random.seed(seed = 2324419459)
    np_random_seeds = np.random.randint(low = 1, high = 2**31, size = 1)

        
    #inference params
    nr_mc_samples_per_batch =   10**3
    nr_batches =  10 #giving nr_mc_samples_per_batch * nr_batches total samples 
    max_taylor_deg = 3  # degree of the taylor polynomial used as control variate
    
    #bfgs params
    max_iter_at_once_bfgs = 20
    max_batches_bfgs      = 1
    

    lags_list = ((1,3,5,10,20,30,40,50,75),(1,3,5,10,15,20,30,40,50),(1,3,5,10,15,20,30,40),(1,3,5,10,15,20,30))
    n_values = (2000,1500,1000,750,500)
    assert max(n_values) <= nr_trawls
    
    #results containers
    levy_seed_params_list = []
    d_gmm = dict()
    d_cl  = dict()
    d_env_initial_guess = dict()
    
    
    #simulate the trawl process
    #change the np_random_seed if doing a simulation study with more 
    trawl_instance =  generate_gamma_seed_trawls(tau = tau,nr_simulations = nr_simulations,
                      nr_trawls = nr_trawls, envelope = envelope2,envelope_params = TRUE_ENVELOPE_PARAMS,
                      jump_part_params = TRUE_GAMMA_PARAMS,np_seed = np_random_seeds[0])  
    #need to change np_random_seeds[-1] and the key in jax if doing a simulation study 
  
    all_values_not_to_use_in_general = trawl_instance.values

    with open('values_par2.npy', 'wb') as fff:
    	np.save(fff, all_values_not_to_use_in_general)
    

    #fit the gmm model and time it
    start_gmm = time.time()

    #marginal distribution gmm firstly
    for n_index in range(len(n_values)):
        n_to_use      = n_values[n_index]
        values_to_use = all_values_not_to_use_in_general[:,:n_to_use]
        levy_seed_params = fit_trawl_marginal(simulations = values_to_use, levy_seed = 'gamma', method='MM')
        levy_seed_params_list.append(levy_seed_params)
        
    #envelope gmm secondly
    for n_index in range(len(n_values)):

        with open("text.txt","a") as file:
           file.write('n is' +str(n_values[n_index]) + '\n')

        n_to_use      = n_values[n_index]
        values_to_use = all_values_not_to_use_in_general[:,:n_to_use]
        initial_guess = [get_initial_value_for_1_param_trawl_function(tau_ = tau, values_ = ppp, envelope = envelope,
                                                                         aux_param = aux_param)  for ppp in values_to_use]
        initial_guess = tuple([tuple(ppp) for ppp in initial_guess]) 
        d_env_initial_guess[n_to_use]    = initial_guess 

        for lags_index in range(len(lags_list)):
  
            lags_to_use     = lags_list[lags_index] 
            envelope_params = fit_trawl_envelope_gmm(s = tau,simulations = values_to_use, lags = lags_to_use, envelope = 'custom',
                                              envelope_function = trawl_function_for_gmm_envelope_fit, initial_guess = initial_guess)
                                               
            d_gmm[(lags_to_use,n_to_use)] = {'envelope_params':envelope_params,'levy_seed_params': levy_seed_params_list[n_index]}
            
    end_gmm = time.time()
    with open("text.txt","a") as file:
        file.write('gmm fitting finished, time taken: ' + str((end_gmm - start_gmm)//60) + ' minutes \n')      
    
    	        
    #fit the cl model
    for lags_index in range(len(lags_list)):
        start_current_lag = time.time()        
        
        for n_index in range(len(n_values)):
            with open("text.txt","a") as file:
                file.write('lags_index is: ' + str(lags_index) +'\n')
                file.write('n_index is: ' + str(n_index))


            #keep track of parameters and loss: not at the moment
            results_list     = []
            #loss_bfgs       = []
            #parameters_bfgs = []
            #hessian_list    = []
            
            for simulation_to_use in range(nr_simulations): 
                with open("text.txt","a") as file:

                    file.write('simulation ' + str(simulation_to_use) + ' / ' + str(nr_simulations) +'\n')
            
                lags_to_use   = lags_list[lags_index] 
                n_to_use      = n_values[n_index]
                values_to_use = all_values_not_to_use_in_general[simulation_to_use,:n_to_use]  
                
                #initialize model parameters with gmm result
                _ = d_gmm[(lags_to_use,n_to_use)] 
                #print(_['levy_seed_params'][simulation_to_use])
                
                initial_tensor = np.concatenate([[_['levy_seed_params'][simulation_to_use][0],
                                                 1/_['levy_seed_params'][simulation_to_use][1]],
                                                 d_env_initial_guess[n_to_use][simulation_to_use]])
                                                
                                                

                initial_log_tensor = jnp.log(initial_tensor.copy())
                

                try:
                    resdd, key = do_modified_bfgs(trawl_path = values_to_use, envelope = envelope,
                                    tau = tau, nr_mc_samples_per_batch = nr_mc_samples_per_batch, nr_batches = nr_batches,
                                    max_taylor_deg = max_taylor_deg, key = key, lags_list = lags_to_use,x0 = initial_log_tensor, 
                                    max_iter_at_once_bfgs = max_iter_at_once_bfgs, max_batches_bfgs = max_batches_bfgs)
                    
                    results_list.append(resdd)

                except ValueError:
                    for splitting_index in range(100):
                        key, subkey = jax.random.split(key)
                    try:
                        resdd, key = do_modified_bfgs(trawl_path = values_to_use, envelope = envelope,
                                    tau = tau, nr_mc_samples_per_batch = nr_mc_samples_per_batch, nr_batches = nr_batches,
                                    max_taylor_deg = max_taylor_deg, key = key, lags_list = lags_to_use,x0 = initial_log_tensor, 
                                    max_iter_at_once_bfgs = max_iter_at_once_bfgs, max_batches_bfgs = max_batches_bfgs)
                        
                        results_list.append(resdd)

                    except ValueError:
                        with open("text.txt","a") as file:
                            file.write('simulation ' + str(simulation_to_use) +  ' is very problematic')
                        results_list.append(np.nan)

                        #loss_bfgs_to_add,parameters_bfgs_to_add = np.nan,initial_tensor.copy()


                #loss_bfgs.append(loss_bfgs_to_add)
                #parameters_bfgs.append(parameters_bfgs_to_add)


            #d_cl[(lags_to_use,n_to_use)] = {'loss':loss_bfgs,'params': parameters_bfgs}
            d_cl[(lags_to_use,n_to_use)] = results_list

        
        end_current_lag = time.time()      
        
        with open("text.txt","a") as file:
            file.write('current lags time was: ' + str((end_current_lag - start_current_lag)//60) + '\n')
        
    end_cl = time.time()
    with open("text.txt","a") as file:
        file.write('cl fitting finished, time taken: ' +  str((end_cl - end_gmm)//60) + ' minutes \n')     
    with open("cl_dictionary.pickle", "wb") as output_file_cl:
        pickle.dump(d_cl, output_file_cl)
    with open("gmm_dictionary.pickle","wb") as output_file_gmm:
        pickle.dump(d_gmm, output_file_gmm)

    #write cl_time to disk
    cl_time = [end_current_lag - end_gmm]
    with open("cl_time.pickle", "wb") as output_cl_time:
        pickle.dump(cl_time, output_cl_time)



                
