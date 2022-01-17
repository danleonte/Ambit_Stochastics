# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:29:41 2022

@author: dleon
"""
import numpy  as np
from scipy.optimize  import fsolve
from scipy.integrate import quad

def empirical_covariance_matrix(values,max_lag_space,max_lag_time):
    """
    Args:
      values:
      max_lag_space:
      max_lag_time:
          
    Returns:
      result_cov:
      result_cor
    """
    nr_simulations, nr_rows, nr_columns = values.shape
    result_cov = np.zeros((nr_simulations,max_lag_space+1,max_lag_time+1))
    result_cor = np.zeros((nr_simulations,max_lag_space+1,max_lag_time+1))

    
    for row in range(max_lag_space+1):
        for column in range(max_lag_time+1):
            
            nr_elements = (nr_rows - row)*(nr_columns - column)
            
            sub_matrix_1 = values[:,:nr_rows - row, :nr_columns - column]
            sub_matrix_2 = values[:,row :, column :]
            #assert sub_matrix_1.shape == sub_matrix_2.shape
            
            mean_1  = np.einsum('ijk->i',sub_matrix_1) / nr_elements
            mean_2  = np.einsum('ijk->i',sub_matrix_2) / nr_elements
            
            variance_estimator_1 = np.array([np.var(sub_matrix_1[i,:,:]) for i in range(nr_simulations)])
            variance_estimator_2 = np.array([np.var(sub_matrix_2[i,:,:]) for i in range(nr_simulations)])

            
            sub_matrix_1 = sub_matrix_1 - mean_1[:,np.newaxis,np.newaxis]
            sub_matrix_2 = sub_matrix_2 - mean_2[:,np.newaxis,np.newaxis]
            
            covariances  = np.einsum('ijk,ijk->i',sub_matrix_1,sub_matrix_2) / nr_elements
            
            result_cov[:,row,column] = covariances
            result_cor[:,row,column] = covariances/(variance_estimator_1 * variance_estimator_2)**0.5
            
    return result_cov,result_cor

def covariance_matrix_theoretical(ambit_function,x,tau,max_lag_space,max_lag_time,total_nr_points,batch_size=10**5):
    
    #start at lag 0
    ambit_t_coords = tau * np.arange(1, max_lag_time+2,1)
    ambit_x_coords = x * np.arange(1, max_lag_space+2,1)
        
    low_x  = x
    high_x = x + ambit_function(0)
    
    low_t  = fsolve(lambda t: ambit_function(t)-x,x0=-1)[0] + tau
    high_t = tau
    
    areas_matrix = np.zeros((max_lag_space+1,max_lag_time+1))
    
    for batch_number in range(total_nr_points // batch_size):
        
        points_x = np.random.uniform(low=low_x, high=high_x, size=batch_size)
        points_t = np.random.uniform(low=low_t, high=high_t, size=batch_size)
        
        indicator_in_A_11 = points_x - x < ambit_function(points_t - tau)
        points_x = points_x[indicator_in_A_11]
        points_t = points_t[indicator_in_A_11]
        
        x_ik = np.subtract.outer(points_x, ambit_x_coords)
        x_ikl = np.repeat(x_ik[:, :, np.newaxis], repeats = max_lag_time + 1, axis=2)
        t_il = np.subtract.outer(points_t, ambit_t_coords)
        phi_t_ikl = np.repeat(ambit_function(t_il)[:, np.newaxis, :], repeats=max_lag_space+1, axis=1)
        range_indicator = x_ik > 0
        range_indicator = np.repeat(
            range_indicator[:, :, np.newaxis], repeats=max_lag_time+1, axis=2)

        indicator = (x_ikl < phi_t_ikl) * range_indicator
        
        areas_matrix += np.sum(indicator,axis=0)
    
    #correct for acf
    total_area          =  quad(ambit_function,a=-np.inf,b= 0)[0]
    correction_factor   =  (high_x-low_x)*(high_t-low_t) / (total_area * batch_size * (total_nr_points // batch_size))
    correlation_matrix  =  correction_factor * areas_matrix 


    correlation_matrix[0] =  np.array([quad(ambit_function,a=-np.inf,b= -i * tau)[0]
                                               for i in range(0,max_lag_time+1)])/total_area
    
    return correlation_matrix
    
    
    
            
            
            