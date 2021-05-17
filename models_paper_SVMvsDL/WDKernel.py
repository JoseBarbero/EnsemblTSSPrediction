import numpy as np
import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes


def beta_k(d, k):
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    return 2*((d-k+1)/(d*(d+1)))

def fill_per_window(args):
        print(args)
        (window_x, window_y), X1, X2, shared_array, L, d, block_size = args
        tmp = np.ctypeslib.as_array(shared_array)

        for idx_x in range(window_x, window_x + block_size):
            for idx_y in range(window_y, window_y + block_size):
                tmp[idx_x, idx_y] = X1[idx_x, idx_y]
                tmp[idx_x, idx_y] = get_K_value(X1[idx_x], X2[idx_y], L, d)


def parallel_wdkernel_gram_matrix(X1, X2):
    # https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html
    

    size = X1.shape[0]
    block_size = 100
    L = len(X1[0])
    d = 3

    K = np.ctypeslib.as_ctypes(np.zeros((size, size)))
    shared_array = sharedctypes.RawArray(K._type_, K)

    window_idxs = [(i, j) for i, j in itertools.product(range(0, size, block_size), 
                                                        range(0, size, block_size))]

    p = Pool()
    res = p.map(fill_per_window, [window_idxs, X1, X2, shared_array, L, d, block_size])
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
            get_K_value(X1[i], X2[j], L, d)

    return K


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