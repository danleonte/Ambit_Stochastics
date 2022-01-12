# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:22:18 2021

@author: dleon
"""
import numpy as np
import pandas as pd 

#check qlike definition
def qlike_func(forecast,observed):
    return observed/forecast - np.log(observed/forecast)  - 1

#check the ones below and the forecast as well from trawl ( where we actually do the forecast)
def loss_func(trawl_values,predicted_values_dict,training_window):
    #mae - mean absolute error
    #medae - mean of the median absolute error
    #rmse - root mean squared error
    #qlike
    assert isinstance(predicted_values_dict,dict)
    assert isinstance(trawl_values,np.ndarray)
    
    assert len(trawl_values.shape) == 2
    print('here')
    mae,medae,rmse,mqlike,nr_steps_ahead = [],[],[],[],[]
    
    for h in predicted_values_dict.keys(): 
        print(predicted_values_dict.keys())
        true_values = trawl_values[:,training_window + h-1:]
        print(true_values.shape,predicted_values_dict[h].shape)
        assert true_values.shape == predicted_values_dict[h].shape
        
        #check what it does
        #mae
        mae.append( np.mean(np.abs(true_values - predicted_values_dict[h])) )
        #mean median error
        medae.append( np.mean(np.median(np.abs(true_values - predicted_values_dict[h]),axis=1)) )
        #mse
        mse_vec = np.mean((true_values - predicted_values_dict[h])**2,axis=1)
        #check
        assert len(mse_vec.shape) == 1 and  len(mse_vec) == true_values.shape[0]
        rmse.append(np.mean(mse_vec**0.5)) 
        #mean qlike loss
        mqlike.append(np.mean(qlike_func(predicted_values_dict[h],true_values))) 
        nr_steps_ahead.append(h)
    df = pd.DataFrame(np.array([mae,medae,rmse,mqlike,nr_steps_ahead]).T,\
                      columns = ['mae', 'medae','rmse','mqlike','nr_steps_ahead'])
    return df

    
#trawl_values = np.array([[1,2,3,4,5],[5,6,3,1,2],[0.5,-2,3,-4,5]])    
#predicted_values_dict =     
    
    
    