import numpy as np
from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,\
                            geninvgauss,poisson 

def fit_trawl_distribution_aux(values,jump_part_name,method='mle'):
    
    if jump_part_name == 'gaussian':
        params = [norm.fit(data = simulation,method = method) for simulation in values]
    elif jump_part_name == 'gamma':
        params = [gamma.fit(data = simulation,floc=0,method=method) for simulation in values]
        params = [[i[0],i[2]] for i in params]   #a, scale
        
    elif jump_part_name == 'cauchy':
        params = [cauchy.fit(data = simulation,floc=0,method=method)[-1:] for simulation in values] #scale
        
    elif jump_part_name == 'invgauss':
        raise ValueError('not yet implemented')
        
    elif jump_part_name == 'norminvgauss':
        raise ValueError('not yet implemented')
        
    elif jump_part_name == 'geninvgauss':
        raise ValueError('not yet implemented')
        
    elif jump_part_name == 'poisson':
        params =  [[np.mean(simulation)] for simulation in values]
    
    else:
        raise ValueError('not implemented yet')
    return np.array(params)
    