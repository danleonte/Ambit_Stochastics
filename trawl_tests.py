# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:23:55 2022

@author: dleon
"""

### test compute_slice_areas_finite_decorrelation_time
tau = 0.25
nr_trawls = 10 
nr_simulations = 15
trawl_function= lambda x :  (x> -1) * (x<=0) * (1+x) 
decorrelation_time =-1 

trawl_example = trawl(tau, nr_trawls, nr_simulations, trawl_function, decorrelation_time)
self = trawl_example
self.I = math.ceil(-self.decorrelation_time/self.tau)
        
s_i1 = [quad(self.trawl_function,a=-i *self.tau, b = (-i+1) * self.tau)[0]
                for i in range(1,self.I+1)]
s_i2 = np.append(np.diff(s_i1[::-1]),s_i1[-1])
        
right_column = np.tile(s_i2[:,np.newaxis],(1,self.nr_trawls-1))
left_column  = (np.array(s_i1))[:,np.newaxis]
self.slice_areas_matrix  = np.concatenate([left_column,right_column],axis=1)

### test compute_slice_areas_finite_decorrelation_time

tau = 0.4
nr_trawls = 10 
nr_simulations = 15
trawl_function= lambda x :  (x> -1) * (x<=0) * (1+x) 
decorrelation_time =-1 
trawl_example = trawl(tau, nr_trawls, nr_simulations, trawl_function, decorrelation_time)
self = trawl_example
self.compute_slice_areas_finite_decorrelation_time()