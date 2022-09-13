# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 00:05:49 2021

@author: dleon
"""
import numpy as np

def tangents(x,a,b,nr_values,f,f_der):
    assert (a <= x and x <= b)
    if x == a:
        return f(a)
    elif x == b:
        return f(b)

    increment = (b-a)/ (nr_values-1)
    #a , a + i , a + 2*i ,..., a + nr_values * i
    interval_index = int(np.floor((nr_values-1) * (x-a)/(b-a)))


    meeting_point = (f(a + interval_index*increment) - f(a + (interval_index+1) * increment) +\
            (a + increment * (interval_index+1)) * f_der(a + (interval_index+1)*increment) -\
                (a + increment * interval_index) * f_der(a + interval_index * increment)) / \
            (f_der(a + increment*(interval_index+1)) - f_der(a + interval_index*increment))
            
    if x <= meeting_point:
            return f(a + interval_index*increment)  + f_der(a + (interval_index)*increment) * (x - (a + increment * interval_index)) 
    elif x > meeting_point:
            return f(a + increment * (interval_index+1))  + f_der(a + (interval_index+1)*increment) * (x - (a + increment * (interval_index+1)))  


def tangets_old(x,a,b,values,f_values,der_f_values):
    assert (a <= x and x <= b)
    if x == a:
        return f_values[0]
    elif x == b:
        return f_values[-1]
    else:
        interval_index = int(np.floor((len(values)-1) * (x-a)/(b-a)))   
        #if not (x - values[interval_index] >= 0) or not (values[interval_index+1] -x >= 0):
        #    print(x,values[interval_index],values[interval_index+1])
        
        
        #f(x0) + f'(x0) (x-x0) = f(x1) + f'(x1)(x-x1)  
        #f(x0) - f(x1) +x1 f'(x1) - x0 f'(x0) = x(f'(x1)-f'(x0))
        meeting_point = (f_values[interval_index] - f_values[interval_index+1] +\
            values[interval_index+1] * der_f_values[interval_index+1] -\
                values[interval_index] * der_f_values[interval_index]) / \
            (der_f_values[interval_index+1] - der_f_values[interval_index])
            
        if x <= meeting_point:
            return f_values[interval_index]  + der_f_values[interval_index] * (x - values[interval_index]) 
        elif x > meeting_point:
            return f_values[interval_index+1]  + der_f_values[interval_index+1] * (x - values[interval_index+1]) 

#### USAGE EXAMPLE####
a,b,nr_values = 1,20,5

f = lambda x : np.log(x+0.25)
f_der = lambda x : 1/ (x+0.25)
der_f = lambda x : 1/ (x+0.25)

values = np.linspace(a,b,nr_values)
f_values = f(values)
der_f_values = der_f(values)

import matplotlib.pyplot as plt
z = np.linspace(a,b,100)
zz = [tangets_old(i,a,b,values,f_values,der_f_values) for i in z]
#zz = [tangents(i,a,b,nr_values,f,f_der) for i in z]
plt.plot(z,zz,c='b')
plt.plot(z,f(z),c='r')
#print ((zz - f(z) >= 0).all() )
