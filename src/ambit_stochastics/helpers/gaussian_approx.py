# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:19:42 2023

@author: dleon
"""

from scipy.stats import norm, norminvgauss
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def normal_approx(mu1, scale1, mu2, scale2):
    
    v1,v2 = scale1**2, scale2**2
    mu = (v1 * mu2 + v2 * mu1) / (v1+v2)
    v = 1 / (1/v1 + 1/v2)
    
    return mu, v**0.5


def test_normal_approx(mu1,scale1,mu2,scale2, x):
    
    
    mu, scale = normal_approx(mu1, scale1, mu2, scale2)
    norm_dist = norm(loc = mu, scale = scale)
    samples   = norm_dist.rvs(size = 10000)
    samples.sort()
    
    f = lambda samples: norm(loc = mu1 , scale = scale1).pdf(samples) * norm(loc = mu2 , scale = scale2).pdf(samples)
    
    y1 = norm_dist.pdf(samples)
    y2 = f(samples)
    plt.plot(samples, y1 / np.sum(y1), marker = 's')
    plt.plot(samples, y2 /np.sum(y2))
    
#test_normal_approx(0,2,5,0.1,3)

if __name__ == '__main__':
    
    params = np.array((1,0,6.15,0.15))
    area0, area1 = 0.5, 1.5
    area = area0 + area1
    
    a, b, loc, scale = params * area
    a0, b0, loc0, scale0 = params * area0
    a1, b1, loc1, scale1 = params * area1
    
    #nig_dist    = norminvgauss(a = a, b = b, loc = loc, scale = scale)
    #trawl_value = nig_dist.rvs()
    #k           = 1/nig_dist.pdf(trawl_value)
    
    nig_dist1 = norminvgauss(a = a1, b = b1, loc = loc1, scale = scale1)
    nig_dist0 = norminvgauss(a = a0, b = b0, loc = loc0, scale = scale0)
    
    m1,v1 = nig_dist1.stats(moments = 'mv')
    m1,v1 = float(m1), float(v1)
    
    m2,v2 = nig_dist1.stats(moments = 'mv')
    m2,v2 = float(m2), float(v2)
    
    normal_approx_params  = normal_approx(m1, v1**0.5, m2, v2**0.5) #scale, scale0, scale1 are used above
    
    samples  = norm(loc = normal_approx_params[0], scale = normal_approx_params[1]).rvs(size=10000)
    samples.sort()
    y_approx = norm(loc = normal_approx_params[0], scale = normal_approx_params[1]).pdf(samples)
    
    f = lambda x : nig_dist0.pdf(x) * nig_dist1.pdf(x)
    k = 1 / quad(f, -np.inf, np.inf)[0]
    
    y_true   = k * f(samples) 
    plt.plot(samples, y_approx,label='approx')
    plt.plot(samples, y_true, label = 'true')
    plt.legend()
    
    
        
    
    
#samples.sort()
#plt.plot(samples, nig_dist.pdf(samples),  label='nig')
#plt.plot(samples, norm_dist.pdf(samples), label='norm')
#plt.legend()
