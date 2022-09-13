# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 22:40:53 2021

@author: dleon
"""
import numpy as np

#values = f(y) where y= f(np.linspace(a,b,1000))

def secants(x,a,b,f,nr_values):
    assert (a <= x and x <= b)
    if x == a:
        return f(a)
    elif x == b:
        return f(b)
    
    increment = (b-a)/ (nr_values-1)
    #a , a + i , a + 2*i ,..., a + nr_values * i
    interval_index = int(np.floor((nr_values-1) * (x-a)/(b-a)))
    return f(a + interval_index * increment)  +\
        (f((interval_index+1) * increment) - f(interval_index * increment)) * (x - a - interval_index * increment) / increment

    
    
def secants_old(x,a,b,values,f_values):
    assert (a <= x and x <= b)
    if x == a:
        return f_values[0]
    elif x == b:
        return f_values[-1]
    else:
        increment = values[1]-values[0]
        interval_index = int(np.floor((len(values)-1) * (x-a)/(b-a)))
        #assert (x - values[interval_index] >= 0) and (values[interval_index+1] -x >= 0)
        return f_values[interval_index]  + (f_values[interval_index+1]-f_values[interval_index]) * (x - values[interval_index]) / increment

#### USAGE EXAMPLE####
f = lambda x : x**4 * (2-x)**1.25

a,b,nr_values = 0,1.25,3
values = np.linspace(a,b,nr_values)
f_values = f(values)

import matplotlib.pyplot as plt
z = np.linspace(a,b,100)
#zz = [secants_old(i,a,b,values,f_values) for i in z]
zz = [secants(i,a,b,f,nr_values) for i in z]
plt.plot(z,zz,c='b')
plt.plot(z,f(z),c='r')
#print ((zz - f(z) >= 0).all() )
