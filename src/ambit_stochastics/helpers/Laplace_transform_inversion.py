"""Python implementation of R script from https://www.kent.ac.uk/smsas/personal/msr/webfiles/rlaptrans/rlaptranscount.r"""
###################################################################
#imports 

import numpy as np
import math
import warnings
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import gamma
###################################################################


def sample_from_laplace_transform(n, ltpdf, lt_params , tol=1e-4, x0=1, xinc=2, m=25, L=1, A=19, nburn=38):
    """Function for generating a random sample of size n from a distribution, given the Laplace transform of its p.d.f.
    
    Args:
        n:
    """
    
    
    maxiter = 100 #to increase maybe

    # -----------------------------------------------------
    # Derived quantities that need only be calculated once,
    # including the binomial coefficients
    # -----------------------------------------------------
    nterms = nburn + m*L
    seqbtL = np.arange(nburn-L,nterms,L)
    y = np.pi * (1j) * np.array(range(1,nterms+1)) / L
    #print('first y is',y)
    expy = np.exp(y)
    A2L = 0.5 * A / L
    expxt = np.exp(A2L) / L
    coef = np.array([math.comb(m,i) for i in range(m+1)]) / 2**m
    
    
    # --------------------------------------------------
    # Generate sorted uniform random numbers. xrand will
    # store the corresponding x values
    # --------------------------------------------------
    
    u = np.sort(np.random.uniform(low=0.0, high=1.0, size=n),kind = "quicksort")
    xrand = u.copy()
    #print('u is',u)
    

    #------------------------------------------------------------
    # Begin by finding an x-value that can act as an upper bound
    # throughout. This will be stored in upplim. Its value is
    # based on the maximum value in u. We also use the first
    # value calculated (along with its pdf and cdf) as a starting
    # value for finding the solution to F(x) = u_min. (This is
    # used only once, so doesn't need to be a good starting value
    #------------------------------------------------------------
    t = x0/xinc
    cdf = 0   
    kount0 = 0
    set1st = False
    while (kount0 < maxiter and cdf < u[n-1]): 
        t = xinc * t
        kount0 = kount0 + 1
        x = A2L / t
        z = x + y/t
        ltx = ltpdf(x, lt_params)
        #print('y is',y)
        if kount0 % 25 ==0 :
            print('kount0 is',kount0)

        #ltzexpy = ltpdf(z, lt_params) * expy #if ltpdf can be applied to a vector
        ltzexpy = np.array([ltpdf(i, lt_params) for i in z]) * expy

        par_sum = 0.5*np.real(ltx) + np.cumsum( np.real(ltzexpy) )
        par_sum2 = 0.5*np.real(ltx/x) + np.cumsum( np.real(ltzexpy/z) )
        
        #to check indeces
        pdf = expxt * np.sum(coef * par_sum[seqbtL]) / t
        cdf = expxt * np.sum(coef * par_sum2[seqbtL]) / t
        #print(cdf)
        
        if ((not set1st) and (cdf > u[0])): 
            print('aici')
            cdf1 = cdf
            pdf1 = pdf
            t1 = t
            set1st = True
    
        
    if kount0 >= maxiter:
        raise ValueError('Cannot locate upper quantile')

    upplim = t
    print('kount0 part 2 is',kount0)
    #--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------

    lower = 0
    t = t1
    cdf = cdf1
    pdf = pdf1
    kount = [0 for i in range(n)]
    maxiter = 1000
    
    for j in range(n) :

        #-------------------------------
        # Initial bracketing of solution
        #-------------------------------
        upper = upplim

        kount[j] = 0
        while (kount[j] < maxiter and abs(u[j]-cdf) > tol):
            kount[j] = kount[j] + 1

            #-----------------------------------------------
            # Update t. Try Newton-Raphson approach. If this 
            # goes outside the bounds, use midpoint instead
            #-----------------------------------------------
            t = t - (cdf-u[j])/pdf 
            if t < lower or t > upper:
               t = 0.5 * (lower + upper)
            
            #print(u[j]-cdf)
            #----------------------------------------------------
            # Calculate the cdf and pdf at the updated value of t
            #----------------------------------------------------
            x = A2L / t
            z = x + y/t
            ltx = ltpdf(x, lt_params)
            ltzexpy = np.array([ltpdf(i, lt_params) for i in z]) * expy
            par_sum  = 0.5 * np.real(ltx) + np.cumsum( np.real(ltzexpy) )
            par_sum2 = 0.5 * np.real(ltx/x) + np.cumsum( np.real(ltzexpy/z) )
            pdf = expxt * np.sum(coef * par_sum[seqbtL]) / t
            cdf = expxt * np.sum(coef * par_sum2[seqbtL]) / t

              #------------------
              # Update the bounds 
              #------------------
            if cdf <= u[j]:
                lower = t
            else: 
                upper = t
        
        if kount[j] >= maxiter:
           warnings.warn('Desired accuracy not achieved for F(x)=u.')
        
        xrand[j] = t
        lower = t
        
    meankount = (kount0 + np.sum(kount))/n
    if n > 1:
        rsample = np.random.permutation(xrand)
    else:
        rsample = xrand
    return rsample, meankount

    
#from scipy.stats import norm
#def gaussian_laplace_transform(t,aux_params):
#    mu,sigma = aux_params
#    return np.exp(0.5 * t**2 * sigma**2 - t * mu)  
#v = sample_from_laplace_transform(5, gaussian_laplace_transform, [0.,1.] )

#def gamma_laplace_transform(t,gamma_distr_params): !WRONG!
#    alpha,k = gamma_distr_params
#    return (1 + t/k) ** (-alpha)

#v2 = sample_from_laplace_transform(10000, gamma_laplace_transform, [2.5,1.] )
#qqplot(v2[0], dist= gamma, distargs=(2.5,), loc=0.,scale= 1.,line='45')