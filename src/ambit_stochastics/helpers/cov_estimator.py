# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:11:06 2022

@author: dleon
"""

import numpy  as np

def compute_empirical_covariance_matrix(values,max_lag_space,max_lag_time):
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
