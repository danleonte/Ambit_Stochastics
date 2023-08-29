import numpy as np
import scipy.stats
from dm_test import dm_test
from scipy.special import digamma
from scipy.special import loggamma
from import_file import import_file
import os
import ot
from pathlib import Path


generate_trawls_with_gamma_marginal = import_file(os.path.join(Path(os.getcwd()).parent,'generate_trawls_with_gamma_marginal.py')) #why need this?????
generate_gamma_seed_trawls          = generate_trawls_with_gamma_marginal.generate_gamma_seed_trawls


#check qlike definition
def qlike_func(forecast,observed):
    return observed/forecast - np.log(observed/forecast)  - 1

#
# `alpha1` and `alpha2` are Gamma shape parameters and
# `scale1` and `scale2` are scale parameters.
# (All, therefore, must be positive.)
#

def KL_a_b(a_p,b_p,a_q,b_q):
    return (a_p - a_q)*digamma(a_p) - loggamma(a_p) + loggamma(a_q) + a_q*(np.log(b_p) - np.log(b_q)) + a_p * (b_q - b_p) / b_p

def KL_k_theta(k_p,theta_p,k_q,theta_q):
    return KL_a_b(k_p, 1/theta_p, k_q, 1/theta_q)
    
#def approx_KL_divergence_a_b_from_samples(a1,b1,a2,b2,size):
#    rv1 = scipy.stats.gamma(a = a1, scale = 1/b1, loc = 0)
#    rv2 = scipy.stats.gamma(a = a2, scale = 1/b2, loc = 0)
#    
#    samples1 = rv1.rvs(size = size)
#    
#    return np.mean(rv1.logpdf(samples1) - rv2.logpdf(samples1))

#approx_KL_divergence_a_b_from_samples(1.65,0.5,1.25,2.25,10**7)
#KL_a_b(1.65,0.5,1.25,2.25)    

def compute_w_distance(list_params_1,list_params_2):
    
    tau1 , nr_simulations1, max_nr_trawls1, envelope1, envelope_params1, jump_part_params1, np_seed1 = list_params_1
    tau2 , nr_simulations2, max_nr_trawls2, envelope2, envelope_params2, jump_part_params2, np_seed2 = list_params_2
    
    assert tau1 == tau2 and max_nr_trawls1 == max_nr_trawls2 and envelope1 == envelope2


    values1 = generate_gamma_seed_trawls(tau1 , nr_simulations1, max_nr_trawls1, envelope1, envelope_params1,
                                        jump_part_params1, np_seed1).values
    
    values2 = generate_gamma_seed_trawls(tau2 , nr_simulations2, max_nr_trawls2, envelope2, envelope_params2,
                                        jump_part_params2, np_seed2).values
    
    M2 = ot.dist(values1,values2, metric = 'minkowski', p = 2)
    M1 = ot.dist(values1,values2, metric = 'minkowski', p = 1)
    
    a, b = np.ones((nr_simulations1,)) / nr_simulations1, np.ones((nr_simulations1,)) / nr_simulations1
    # uniform distribution on samples

    return ot.emd2(a, b, M1), ot.emd2(a, b, M2)

def compute_w_distance_from_dict_at_lag(tau,nr_simulations, max_nr_trawls, envelope, np_seed_list_1, np_seed_list_2, d,TRUE_GAMMA_PARAMS,
                                       TRUE_ENVELOPE_PARAMS):
    
    l = []
    
    for i in range(len(d['envelope_params'])):
    
        list_params_1 = (tau, nr_simulations, max_nr_trawls, envelope, tuple(d['envelope_params'][i]),
                                                            tuple(d['levy_seed_params'][i]),np_seed_list_1[i])
        list_params_2 = (tau, nr_simulations, max_nr_trawls, envelope, TRUE_ENVELOPE_PARAMS,
                                                                 TRUE_GAMMA_PARAMS,np_seed_list_2[i])
        
        l.append(compute_w_distance(list_params_1,list_params_2))
        
    return l
        

    

#check the ones below and the forecast as well from trawl ( where we actually do the forecast)
def compute_deterministic_losses(true_values,predicted_values_dict):
    #predicted_values_dict are the values based on which we make the prediction
    #mae - mean absolute error
    #medae - mean of the median absolute error
    #rmse - root mean squared error
    #qlike
    assert isinstance(predicted_values_dict,dict)
    assert isinstance(true_values,np.ndarray)
    
    assert len(true_values.shape) == 2
    mae,medae,rmse,mqlike = [],[],[],[]
    
    for h in predicted_values_dict.keys(): 
        #£print(predicted_values_dict.keys())
        #true_values = trawl_values[:,training_window + h-1:]
        #print(true_values.shape,predicted_values_dict[h].shape)
        true_values_to_use      = true_values[:,h:]
        predicted_values_to_use = predicted_values_dict[h][:,:-h]
        
        assert true_values_to_use.shape == predicted_values_to_use.shape
        
        #check what it does
        #mae
        mae.append( np.mean(np.abs(true_values_to_use - predicted_values_to_use)) )
        #mean median error
        medae.append( np.mean(np.median(np.abs(true_values_to_use - predicted_values_to_use),axis=1)) )
        #mse
        mse_vec = np.mean((true_values_to_use - predicted_values_to_use)**2,axis=1)
        #check
        assert len(mse_vec.shape) == 1 and  len(mse_vec) == predicted_values_to_use.shape[0]
        rmse.append(np.mean(mse_vec**0.5)) 
        #mean qlike loss
        mqlike.append(np.mean(qlike_func(predicted_values_to_use,true_values_to_use))) 
    #    nr_steps_ahead.append(h)
    #df = pd.DataFrame(np.array([mae,medae,rmse,mqlike,nr_steps_ahead]).T,\
    #                  columns = ['MAE', 'MedAE','rMSE','MQLIKE','nr_steps_ahead'])
    return np.array([mae,medae,rmse,mqlike]).T




 
def compute_dm_test_p_value(true_values,predicted_values_dict_1,predicted_values_dict_2):
    assert isinstance(predicted_values_dict_1,dict)
    assert isinstance(predicted_values_dict_2,dict)
    assert isinstance(true_values,np.ndarray)
    
    assert len(true_values.shape) == 2
    assert predicted_values_dict_1.keys() == predicted_values_dict_2.keys()
    
    d ={}
    
    for h in predicted_values_dict_1.keys(): 
        array = []
        #£print(predicted_values_dict.keys())
        #true_values = trawl_values[:,training_window + h-1:]
        #print(true_values.shape,predicted_values_dict[h].shape)
        true_values_to_use      = true_values[:,h:]
        list_1_to_use = predicted_values_dict_1[h][:,:-h]
        list_2_to_use = predicted_values_dict_2[h][:,:-h]

        assert list_1_to_use.shape == list_2_to_use.shape 
        print(list_1_to_use.shape,true_values_to_use.shape)
        assert list_1_to_use.shape == true_values_to_use.shape
        for i in range(list_1_to_use.shape[0]):
            print(h,i)
            assert not(np.isnan(true_values_to_use[i]).any())
            assert not(np.isnan(list_1_to_use[i]).any())
            assert not(np.isnan(list_2_to_use[i]).any())

            array.append((dm_test(actual_lst = list(true_values_to_use[i]),pred1_lst = list_1_to_use[i] ,
                             pred2_lst = list_2_to_use[i] ,h = h, crit="MSE"))[-1])
        d[h] = array
    return d
        
       
    
    
    