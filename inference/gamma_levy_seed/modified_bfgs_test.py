# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 18:20:19 2023

@author: dleon
"""

from modified_minimize import modified_minimize
from scipy.optimize import rosen, rosen_der
import numpy as np

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
h =  np.abs(np.random.normal(5))

res = modified_minimize(rosen, x0, method='BFGS', jac=rosen_der, hess_inv_initial_estimate = h,
               options={'gtol': 1e-6, 'disp': True})
