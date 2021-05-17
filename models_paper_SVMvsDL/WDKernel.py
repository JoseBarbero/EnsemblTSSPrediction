import numpy as np


def beta_k(d, k):
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    return 2*((d-k+1)/(d*(d+1)))


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
    print('X1', X1.shape)
    print('X2', X2.shape)
    L = len(X1[0])
    d = 3

    K = np.zeros((N1, N2))

    for i in range(N1):
        for j in range(N2):
            #print(f'{N2*i+j:,}/{N1*N2:,}', end='\r', flush=True)
            get_K_value(X1[i], X2[j], L, d)

    return K


def get_K_value(xi, xj, L, d):
    print('xi', len(xi), xi)
    print('xj', len(xj), xj)
    # Formula from https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    # First SUM
    E1 = 0
    for k in range(1, d+1):
        beta = beta_k(d, k)
        # Second SUM
        E2 = 0
        for l in range(L-k+1):  #Aquí no sumo 1 a cada lado del range porque son posiciones de una lista
            E2 += int(xi[l:l+k] == xj[l:l+k])
            print('xi[l:l+k]', len(xi[l:l+k]), xi[l:l+k])
            print('xj[l:l+k]', len(xj[l:l+k]), xj[l:l+k])
            input()
            #print(xi[l:l+k], xj[l:l+k], xi[l:l+k] == xj[l:l+k], int(xi[l:l+k] == xj[l:l+k]))
        E1 += beta * E2
    return E1