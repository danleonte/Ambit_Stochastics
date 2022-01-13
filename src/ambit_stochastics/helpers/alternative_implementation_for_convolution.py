import numpy as np

def all_traces(x):
    """computes the sums on paralel diagonals, as described in [] of []"""
    assert x.shape[0] == x.shape[1],'the input matrix is not a square matrix'
    x = x[::-1,:]
    jj = np.tile(np.arange(x.shape[1]),x.shape[0])
    ii = (np.arange(x.shape[1])+np.arange(x.shape[0])[::-1,None]).ravel()
    z = np.zeros(((x.shape[0]+x.shape[1]-1),x.shape[1]),int)
    z[ii,jj] = x.ravel()
    return z.sum(axis=1)[:x.shape[0]]

