import numpy as np
def cumulative_and_diagonal_sums(a):
    k = a.shape[-1] #nr_trawls
    a = np.cumsum(a[::-1],axis=0)[::-1]
    return np.bincount(sum(np.indices(a.shape)).flat, a.flat)[:k]
