import numpy as np
def cumulative_and_diagonal_sums(a):
    k = a.shape[-1] #nr_trawls
    a = np.cumsum(a[::-1],axis=0)[::-1]
    return np.bincount(sum(np.indices(a.shape)).flat, a.flat)[:k]


#a = np.array([[0,1,0,1,1],[-1,0,1,2,1],[1,0,2,1,2],[1,0,2,1,2],[1,0,2,1,2]])
#cumulative_and_diagonal_sums(a)