import numpy as np
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,geninvgauss,poisson 


def fit_trawl_marginal(simulations,levy_seed,method='MLE'):
    
    if levy_seed == 'gaussian':
        params = [norm.fit(data = simulation,method = method) for simulation in simulations]
    elif levy_seed == 'gamma':
        params = [gamma.fit(data = simulation,floc=0,method=method) for simulation in simulations]
        params = [[i[0],i[2]] for i in params]   #a, scale
        
    elif levy_seed == 'cauchy':
        params = [cauchy.fit(data = simulation,floc=0,method=method)[-1:] for simulation in simulations] #scale
        
    elif levy_seed == 'invgauss':
        raise ValueError('not yet implemented')
        
    elif levy_seed == 'norminvgauss':
        params = [norminvgauss.fit(data = simulation,method = method) for simulation in simulations] 
        
    elif levy_seed == 'geninvgauss':
        raise ValueError('not yet implemented')
        
    elif levy_seed == 'poisson':
        params =  [[np.mean(simulation)] for simulation in simulations]
    
    else:
        raise ValueError('not implemented yet')
    return np.array(params)