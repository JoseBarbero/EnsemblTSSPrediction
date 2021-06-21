import numpy as np
from numba import jit
import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes


def beta_k(d, k):
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    return 2*((d-k+1)/(d*(d+1)))

def fill_per_window(args):
        inirow, endrow = args
        tmp = np.ctypeslib.as_array(shared_array)

        for idx_x in range(inirow, endrow):
            for idx_y in range(n_cols):
                tmp[idx_x, idx_y] = get_K_value(X1_g[idx_x], X2_g[idx_y], L, d_g)


def parallel_wdkernel_gram_matrix(X1, X2):
    # https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html
    

    
    d = 10

    # Needed for multiprocessing
    global X1_g
    global X2_g
    global d_g
    global L
    global block_size
    global shared_array
    global n_rows
    global n_cols
    X1_g = X1
    X2_g = X2
    print('X1', X1.shape)
    print('X2', X2.shape)
    d_g = d
    L = len(X1[0])
    n_rows = X1.shape[0]
    n_cols = X2.shape[0]

    # Divide the matrix by rows
    cores = 20
    block_size = int(n_rows/cores)
    rows = [(startrow, startrow+block_size) if startrow+block_size <= n_rows  else (startrow, n_rows) for startrow in range(0, n_rows, block_size)]

    # Shared array
    K = np.ctypeslib.as_ctypes(np.zeros((n_rows, n_cols)))      
    shared_array = sharedctypes.RawArray(K._type_, K)

    
    print(rows)
    p = Pool()
    res = p.map(fill_per_window, rows)
    result = np.ctypeslib.as_array(shared_array)

    return result

def wdkernel_gram_matrix(X1, X2):
    '''
    Gets the gram matrix between X1 and X2.
    https://stackoverflow.com/questions/26962159/how-to-use-a-custom-svm-kernel

    x = sequences matrix (N, L)
    L = sequences length
    K = kernel matrix

    k = kmer length
    d = max kmer length
    beta_k = weighting coefficient
    I = indicator function (1 if true else 0)
    u = string of length k starting at position l of the sequence x

    E1 = First summatory in the formula
    E2 = Second summatory in the formula
    '''

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    L = len(X1[0])
    d = 3

    K = np.zeros((N1, N2))

    for i in range(N1):
        for j in range(N2):
            print(f'{N2*i+j:,}/{N1*N2:,}', end='\r', flush=True)
            K[i, j] = get_K_value(X1[i], X2[j], L, d)

    return K

@jit
def get_K_value(xi, xj, L, d):
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    # First SUM
    E1 = 0
    for k in range(1, d+1):
        beta = beta_k(d, k)
        # Second SUM
        E2 = 0
        for l in range(L-k+1):  #Aquí no sumo 1 a cada lado del range porque son posiciones de una lista
            E2 += int(xi[l:l+k] == xj[l:l+k])
        E1 += beta * E2
    return E1