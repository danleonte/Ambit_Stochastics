# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 00:36:16 2022

@author: dleon
"""

x   = 0.15
tau = 0.2
max_lag_space = 7
max_lag_time = 6
total_nr_points =   10**7
ambit_function  = lambda x :np.exp(x) * (x<=0)
theoretical_corr_matrix = covariance_matrix_theoretical(ambit_function,x,tau,max_lag_space,max_lag_time,
                                            total_nr_points,batch_size=10**5)


sa = simple_ambit_field(x=x, tau=tau, k_s= 75, k_t=75, nr_simulations=2,
             ambit_function=ambit_function, decorrelation_time=-np.inf,
             gaussian_part_params=(1,1), jump_part_name='gamma', jump_part_params=(2,3),
             batch_size=10**5, total_nr_samples=10**7, values=None)

#sa.simulate()
emp_corr_matrix = empirical_covariance_matrix(sa.values,max_lag_space,max_lag_time)[1]
u = emp_corr_matrix[0]