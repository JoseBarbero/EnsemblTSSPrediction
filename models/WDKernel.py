import numpy as np


def beta_k(d, k):
    return 2*((d-k+1)/d*(d+1))


def wdkernel(x, d):
    '''
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

    N = x.shape[0]
    L = len(x[0])

    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            print(f'{N*i+j}/{N*N}', end='\r', flush=True)
            get_K_value(x[i], x[j], L, d)

    return K

def get_K_value(xi, xj, L, d):
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    # First SUM
    E1 = 0
    for k in range(1, d):
        
        # Second SUM
        E2 = 0
        for l in range(L-k+1):
            ukl_xi = xi[l:l+k]
            ukl_xj = xj[l:l+k]
            I = int(ukl_xi == ukl_xj)
            E2 += I

        E1 += beta_k(d, k) * E2
    return E1