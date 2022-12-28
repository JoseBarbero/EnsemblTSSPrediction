import numpy as np
from multiprocessing import Pool  # Process pool
from multiprocessing import sharedctypes
from itertools import groupby

import sys
import ctypes as ct

RED = "\033[1m\033[91m"
GREEN = "\033[1m\033[92m"
END = "\033[0m"


#   #    #                                                       #
#   #   #       #    #    ##    #       #    #  ######   ####    #
#   #  #        #    #   #  #   #       #    #  #       #        #
#   ###         #    #  #    #  #       #    #  #####    ####    #
#   #  #        #    #  ######  #       #    #  #            #   #
#   #   #        #  #   #    #  #       #    #  #       #    #   #
#   #    #        ##    #    #  ######   ####   ######   ####    #


# ###########################--------------------------------------------------
#                           #
#  LOAD THE SHARED LIBRARY  #
#                           #
#############################

try:
    lib = ct.CDLL("./strkernel.so")
except Exception:
    print(f"{RED}Library strkernel.so not found.{END}")
    print(f"{RED}Compile it first with:{END}")
    print(f"{GREEN}   gcc -O2 -fPIC -shared strkernel.c -o strkernel.so{END}")
    print("or:")
    print(f"{GREEN}   make lib{END}")
    sys.exit(-1)

charptr = ct.POINTER(ct.c_char)

# long double cgo_get_K_value1(char *xi, char *xj, int L, int d)
lib.cgo_get_K_value1.argtypes = [charptr, charptr, ct.c_int, ct.c_int]
lib.cgo_get_K_value1.restype = ct.c_longdouble
cgo_get_K_value1 = lib.cgo_get_K_value1

# long double cgo_get_K_value2(char *xi, char *xj, int L, int d)
lib.cgo_get_K_value2.argtypes = [charptr, charptr, ct.c_int, ct.c_int]
lib.cgo_get_K_value2.restype = ct.c_longdouble
cgo_get_K_value2 = lib.cgo_get_K_value2

wcharptr = np.ctypeslib.ndpointer(dtype='<U1', ndim=1, flags="C")

# long double new_cgo_get_K_value(const wchar_t *xi, const wchar_t *xj, int L, int d)
lib.cgo_get_K_value3.argtypes = [wcharptr, wcharptr, ct.c_int, ct.c_int]
lib.cgo_get_K_value3.restype = ct.c_longdouble
cgo_get_K_value3 = lib.cgo_get_K_value3  # 👈 this is the preferred method

# -----------------------------------------------------------------------------


# CGO: número de valores parece finito y bajo, quizás mejor precalcularla.
# Alternativamente se podría decorar con functools.cache
def beta_k(d, k):
    # SEE: https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    return 2*((d-k+1)/(d*(d+1)))


def old_get_K_value(xi, xj, L, d, DEBUG=False):
    # 🔴  is expecting that xi and xj are strings
    # SEE: https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
    # First SUM
    E1 = 0
    for k in range(1, d+1):
        if DEBUG:
            print(f"***k: {k}")
        beta = beta_k(d, k)
        # Second SUM
        E2 = 0
        if DEBUG:
            print(f"range(L-k+1) => range({L}-{k}+1) ", end="")
            print(f"=> range({L-k+1}) => {list(range(L-k+1))}")
        for l in range(L-k+1):  # Aquí no sumo 1 a cada lado del range porque son posiciones de una lista
            if DEBUG:
                print(f"l: {l}")
            if DEBUG:
                print(f"xi: {xi}", end=", ")
                print(f"xi[{l}:{l+k}]: {xi[l:l+k]}", end=", ")
                print(f"xj[{l}:{l+k}]: {xj[l:l+k]}")
                if xi[l:l+k] == xj[l:l+k]:
                    print("+1")
            E2 += int(xi[l:l+k] == xj[l:l+k])
        E1 += beta * E2
        if DEBUG:
            print(f"E2: {E2}, E1: {E1}")
    return E1


def get_K_value(xi, xj, L, d):
    # Versión optimizada (mucho menos legible)
    # Más rápida en cadenas largas. Más lenta en cortas.
    E1 = 0
    groups = groupby(xi == xj)  # Para comparar carácter a carácter
    result = [(label, sum(1 for _ in group))
              for label, group in groups if label]
    E1 = np.zeros(d)
    for k in range(1, d+1):
        e1 = 0
        for _, n in result:
            if n >= k:
                e1 += n+1-k
        E1[k-1] = e1 * beta_k(d, k)
    return E1.sum()


Kvalue = cgo_get_K_value3  # 👈 👈  Default value for Kvalue


#   #    #                                            #
#   #   #   ######  #####   #    #  ######  #         #
#   #  #    #       #    #  ##   #  #       #         #
#   ###     #####   #    #  # #  #  #####   #         #
#   #  #    #       #####   #  # #  #       #         #
#   #   #   #       #   #   #   ##  #       #         #
#   #    #  ######  #    #  #    #  ######  ######    #
#                                                     #
#   #    #    ##    #####  #####   #  #    #          #
#   ##  ##   #  #     #    #    #  #   #  #           #
#   # ## #  #    #    #    #    #  #    ##            #
#   #    #  ######    #    #####   #    ##            #
#   #    #  #    #    #    #   #   #   #  #           #
#   #    #  #    #    #    #    #  #  #    #          #


def fill_per_window_same_matrix(args):
    # Si son iguales solo hace falta recorrer media matriz,
    # la otra mitad tiene los mismos resultados
    global Kvalue  # It is not modified, but this highlights it is a global
    inirow, endrow = args
    tmp = np.ctypeslib.as_array(shared_array)
    for idx_x in range(inirow, endrow):
        for idx_y in range(idx_x, n_cols):
            tmp[idx_x, idx_y] = Kvalue(X1_g[idx_x], X2_g[idx_y], L, d_g)
            tmp[idx_y, idx_x] = tmp[idx_x, idx_y]


def fill_per_window_same_matrix_with_cgo_get_K_value_version1(args):
    # This is just the previous function but calling
    # cgo_get_K_value1 which requires char strings as inputs
    inirow, endrow = args
    tmp = np.ctypeslib.as_array(shared_array)
    for idx_x in range(inirow, endrow):
        for idx_y in range(idx_x, n_cols):
            tmp[idx_x, idx_y] = cgo_get_K_value1(''.join(X1_g[idx_x]).encode(),
                                                 ''.join(X2_g[idx_y]).encode(),
                                                 L, d_g)
            tmp[idx_y, idx_x] = tmp[idx_x, idx_y]


def fill_per_window_different_matrix(args):
    inirow, endrow = args
    tmp = np.ctypeslib.as_array(shared_array)
    for idx_x in range(inirow, endrow):
        for idx_y in range(n_cols):
            tmp[idx_x, idx_y] = Kvalue(X1_g[idx_x], X2_g[idx_y], L, d_g)


def parallel_wdkernel_gram_matrix(X1, X2, ncores=32, d=10):
    # https://web.archive.org/web/20201211130940/https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html

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
    # X1_g = np.array(list(X1))
    # X2_g = np.array(list(X2))
    # print('X1', X1.shape)
    # print('X2', X2.shape)
    d_g = d
    L = len(X1[0])
    n_rows = X1.shape[0]
    n_cols = X2.shape[0]

    if X1_g is X2_g:
        fill_per_window = fill_per_window_same_matrix
    else:
        fill_per_window = fill_per_window_different_matrix

    # Divide the matrix by rows
    block_size = int(n_rows/ncores) + 1
    rows = [(startrow, startrow+block_size)
            if startrow+block_size <= n_rows else
            (startrow, n_rows)
            for startrow in range(0, n_rows, block_size)]

    # Shared array
    K = np.ctypeslib.as_ctypes(np.zeros((n_rows, n_cols)))
    shared_array = sharedctypes.RawArray(K._type_, K)

    # print(rows)
    p = Pool()
    _ = p.map(fill_per_window, rows)
    result = np.ctypeslib.as_array(shared_array)
    p.close()
    return result


def parallel_wdkernel_gram_matrix_with_cgo_get_K_value_version1(X1, X2, ncores=20, d=10):
    # This is just the previous function but calling
    # fill_per_window_same_matrix_with_cgo_get_K_value_version1
    # instead of fill_per_window_same_matrix
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
    d_g = d
    L = len(X1[0])
    n_rows = X1.shape[0]
    n_cols = X2.shape[0]
    if X1_g is X2_g:
        fill_per_window = fill_per_window_same_matrix_with_cgo_get_K_value_version1
    else:
        fill_per_window = fill_per_window_different_matrix
    block_size = int(n_rows/ncores) + 1
    rows = [(startrow, startrow+block_size)
            if startrow+block_size <= n_rows else
            (startrow, n_rows)
            for startrow in range(0, n_rows, block_size)]
    K = np.ctypeslib.as_ctypes(np.zeros((n_rows, n_cols)))
    shared_array = sharedctypes.RawArray(K._type_, K)
    p = Pool()
    _ = p.map(fill_per_window, rows)
    result = np.ctypeslib.as_array(shared_array)
    p.close()
    return result


def wdkernel_gram_matrix(X1, X2, d=10):
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
    global Kvalue  # It is not modified, but this highlights it is a global

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    L = len(X1[0])

    K = np.zeros((N1, N2))

    for i in range(N1):
        for j in range(N2):
            # print(f'{N2*i+j:,}/{N1*N2:,}', end='\r', flush=True)
            K[i, j] = Kvalue(X1[i], X2[j], L, d)

    return K


def wdkernel_gram_matrix_with_cgo_get_K_value_version1(X1, X2, d=10):
    # This is just the previous function but calling
    # cgo_get_K_value1 which requires char strings as inputs
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    L = len(X1[0])
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            K[i, j] = cgo_get_K_value1(''.join(X1[i]).encode(),
                                       ''.join(X2[j]).encode(), L, d)
    return K


#    #######                                    #
#       #     ######   ####   #####   ####      #
#       #     #       #         #    #          #
#       #     #####    ####     #     ####      #
#       #     #            #    #         #     #
#       #     #       #    #    #    #    #     #
#       #     ######   ####     #     ####      #
                                        

SEQ1 = "CGGTACACGACGGTGTGACCTGTGATGCGGCAGGAAGCCGCTCCCATGCCTTCCGCTAAATTATACGAGACGAGCGGTTAGGCACATAATTGAATCTGCTGCTGTCGATCGCTAAGCATCCGACTCGTGAATCATATAAACATGTCTACTTATGATCAATCAATCCCCCCTCACATTGAATCCGAGCTCCGTGACATCACATGGGATTTCGTAATTTGCATGTGACGACAGCAGTCCTACCCCATTTGCCTGAGCTATTTGTGGCACGGATAGCCCGCTCCTACGCCTGGTTTCTTACTACGCTGCGCGAAGGTCGTCTTTGGCTCTACAGACTGGTCTTGACCGGCCCTTCCAGATAGAGGCCGGACAGCGTGGCCTCTTCATGCGAAAATTCGGCAGGAGGGGTAGGGACGGGGACAATAGACAGCCATCTATCGTAGAAAACCCCACTGACTGGGATGGACCAGCAGCTGGGAGCTAGACCGAGGAAGCAGTCCGACCCGAGGTAGCCTCTGCTCGCCCGCGGCTGCACCGGAGGACTTTACAGTGGAATTCAAGTATCAGATAACTTGGTGTCGTCTACTCAGAGAACTTAATTACAATATCCTACCCCGCCCCGAGGCAATTGTTCTAGTTAGAGCCTAATCTACGTGTAGGGCGACGAGTACTTATTGGCCAGCATAGCTGGTAGCCTATCGGGGGTTCCATCGAACCACGGCTATCGCAGTACATGAACAAATGACCCGCCTGCATACTAACGGTCTCTATGAAACAAATTTATACTTAGTGATATCTACCATCTAAAGTTCGGCTCTAACTGTTCGCCGTCCGATAAGTGCTCGGAGGCGTGCAACAGGCCCAACCGTGAAGACCTTTAACCCATCCAAGACTATGTGCGGAATGGTTATGCACGCACGTATCGTCATGCCCTTTCGATTCCCTCGGCTCTCGCAAACGGAGTCCGTAGGCGAAGCGCGCGATGAAGGTAGGGGACGGAACCTGT"

SEQ2 = "ACGACCCGTGGGATTCTCTTTCTCCACAATTCACTGGGCCTTTAACCCGAATAACCTTCGATCGCTCCTACATCTTTTAGCTCCGCACTTGCGTTAGCGTAGAGGGGGCACCCGCTTTGCGGCGACCATCCAACTTCGTAGCTTGGTCGCGTGCGACAATGTCTCTACACTCACTGGAGATTGGCCTACAGCACTACCTATACACGGGGGAGCATCCCTAAAGCCTTTCCCTTGTCTGGGAGCCGCGGCGAATTCGAAGACTAGAGACTAGTAGCCGGATAGTCCGCAAGGAGCTCATTGCTGCACGAAACTAAACTCTCAAACCGCCCAAAAGTGATATCGAGATGCTGCACACTCAAGCTGACTACCCGGTTTCAGTATGTACACTGATAGCGATTACTAATCAACCCAGTACGTTGTTAGGATATCAATCGGTTTGCGTTGATGGACAGCGGCGGCAAATCCGGACATTATCAAACATAAGTCAGGTCTGTCCCGGCAGGGTGATCGGCCTCTGCCTAAGAATGGGGATCTGGATTGGCCACTGAAGATGAAGTTCTGTGTAAAAATGCTGTGTTCGCCACAATACTGCTGTGTCGTCGAGATGCGGCAGTTGGGATCTTACCCACACTCCGGCGACGTGGAGATCCTTTATTGGCGTACTCGCCGACTATTCGGTGAGGACGATAACCCTTGTGCTCAGCTCCGGATACGTAATCCCTAGGAGAGTTCTTCTCTCTTGAACTGTTATCGGGTACTGCCGTACTCGCATGGCCGGTGCGATTATCCCAGTCCCCTAAGAACCAGATGTTGTACGGCGACCTAGGGGCGAGCGTTTTTTGTGACAATATCCACTAGCCTGATCGCATGTTAGGAGTAGGACTACTATTACACCGGCGTTACTAGGTAAATTTGGATAGGGTTTGCGGTAGCACAGACATAAACAGGACACAAGATGGTCTACCCACTACTCGCATTGGACCTGATGGTCGCGTCCACTATC"


def genseq(length):
    import random
    return ''.join(random.choice('CGTA') for _ in range(length))


def genseqs(numseqs, length):
    return np.array([list(genseq(length)) for _ in range(numseqs)])

# SEQ1 = genseq(4000)
# SEQ2 = genseq(4000)


def test1():
    darray = (1, 2, 4)
    karray = (3, 4, 5)
    for d in darray:
        for k in karray:
            print(f"beta_k({d}, {k}): {beta_k(d, k)}")


def test2():
    import time
    print("Running")
    xi = SEQ1
    xj = SEQ2
    d = 800
    # xi, xj, d = "AAGGTT", "AGGTTT", 2
    # xi, xj, d = "AAGGTTAAGGTT", "AGGTTTAGGTTT", 10
    L = len(xi)
    print(f"L: {L}, d: {d}")
    start = time.time()
    res = old_get_K_value(xi, xj, L, d)
    end1 = time.time() - start
    print(f"Result is {res} in {end1} seconds.")

    xi = np.array(list(SEQ1))  # 👈 outside of the timing as it will be the native format
    xj = np.array(list(SEQ2))
    start = time.time()
    res = get_K_value(xi, xj, L, d)
    end2 = time.time() - start
    print(f"Result is {res} in {end2} seconds.")

    print(f"Version with NumPy is {end1/end2} faster than old version.")


def test3():
    import time

    # xi, xj, d = "AAGGTT", "AGGTTT", 2
    # xi, xj, d = "AAGGTTAAGGTT", "AGGTTTAGGTTT", 10

    seq1 = SEQ1*100
    seq2 = SEQ2*100

    d = 200
    L = len(seq1)
    print(f"L: {L}, d: {d}")

    xi = seq1
    xj = seq2
    start = time.time()
    res = old_get_K_value(xi, xj, L, d)
    endOld = time.time() - start
    print(f"Pyth result is {res} in {endOld:.15f} seconds.")

    xi = np.array(list(seq1))  # 👈 outside of the timing as it will be the native format
    xj = np.array(list(seq2))
    start = time.time()
    res = get_K_value(xi, xj, L, d)
    endNumpy = time.time() - start
    print(f"P.np result is {res} in {endNumpy:.15f} seconds.")

    start = time.time()
    xi = seq1.encode()  # 👈 inside the timing as v1 need char strings
    xj = seq2.encode()
    res = cgo_get_K_value1(xi, xj, L, d)
    endCv1 = time.time() - start
    print(f"C.v1 result is {res} in {endCv1:.15f} seconds.")

    start = time.time()
    xi = seq1.encode()  # 👈 inside the timing as v2 need char strings
    xj = seq2.encode()
    res = cgo_get_K_value2(xi, xj, L, d)
    endCv2 = time.time() - start
    print(f"C.v2 result is {res} in {endCv2:.15f} seconds.")

    xi = np.array(list(seq1))  # 👈 outside of the timing as it will be the native format
    xj = np.array(list(seq2))
    start = time.time()
    res = cgo_get_K_value3(xi, xj, L, d)
    endCv3 = time.time() - start
    print(f"C.v3 result is {res} in {endCv3:.15f} seconds.")

    print()
    print(f"{GREEN}Python.np is {endOld/endNumpy:.3f} times faster than Python!!!{END}")
    print(f"C.v1 is {endNumpy/endCv1:.3f} times faster than Python.np!!!")
    print(f"C.v2 is {endNumpy/endCv2:.3f} times faster than Python.np!!!")
    print(f"{GREEN}C.v3 is {endNumpy/endCv3:.3f} times faster than Python.np{END}!!!")
    print(f"With d={d}, C.v2 implementation was {endCv1/endCv2:.3f} faster than C.v1.")  
    print()
    print("It is not clear which C.v1 or C.v2 is faster (it depends on L and d).")
    print("C.v3 is marginally better than the others (but again, it depends on L and d).")


def test4():
    "Test times and that the result are the same for all implementations."
    import time
    global Kvalue
    oldKvalue = Kvalue

    CORES = 4
    d = 10
    L = 1003
    nseqs = 500
    X1 = genseqs(nseqs, L)
    print(f"nseqs: {nseqs}, L: {L}, d: {d}")

    start = time.time()
    Kvalue = get_K_value  # 👈 change global Kvalue
    res = wdkernel_gram_matrix(X1, X1, d=d)
    Kvalue = oldKvalue
    endPyNumpy = time.time() - start
    print(f"Python.np result is {res.sum()} in {endPyNumpy} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    start = time.time()
    res = wdkernel_gram_matrix_with_cgo_get_K_value_version1(X1, X1, d=d)
    endCv1 = time.time() - start
    print(f"C.v1 result is {res.sum()} in {endCv1} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    start = time.time()
    Kvalue = cgo_get_K_value3  # 👈 change global Kvalue
    res = wdkernel_gram_matrix(X1, X1, d=d)
    Kvalue = oldKvalue
    endCv3 = time.time() - start
    print(f"C.v3 result is {res.sum()} in {endCv3} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    start = time.time()
    Kvalue = get_K_value  # 👈 change global Kvalue
    res = parallel_wdkernel_gram_matrix(X1, X1, ncores=CORES, d=d)
    Kvalue = oldKvalue
    endPyNumpyPar = time.time() - start
    print(f"Python.par result is {res.sum()} in {endPyNumpyPar} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    start = time.time()
    res = parallel_wdkernel_gram_matrix_with_cgo_get_K_value_version1(X1, X1, ncores=CORES, d=d)
    endCParv1 = time.time() - start
    print(f"C.par.v1 result is {res.sum()} in {endCParv1} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    start = time.time()
    Kvalue = cgo_get_K_value3  # 👈 change global Kvalue
    res = parallel_wdkernel_gram_matrix(X1, X1, ncores=CORES, d=d)
    Kvalue = oldKvalue
    endCParv3 = time.time() - start
    print(f"C.par.v3 result is {res.sum()} in {endCParv3} seconds.")
    print(res.sum(axis=0)[:5])
    print()

    print(f"nseqs: {nseqs}, L: {L}, d: {d}")
    print()
    print(f"C.v1 is {endPyNumpy/endCv1:.3f} times faster than Python.np")
    print(f"C.v3 is {endPyNumpyPar/endCParv3:.3f} times faster than Python.np")
    print()
    print(f"C.par.v3 is {endPyNumpy/endCParv3:.3f} times faster than Python.np")
    print(f"{GREEN}C.par.v3 is {endPyNumpyPar/endCParv3:.3f} times faster than Python.par{END}")
    print()
    print(f"C.par.v3 is {endCv3/endCParv3:.3f} times faster than C.v3")
    print(f"{GREEN}Python.par is {endPyNumpy/endPyNumpyPar:.3f} times faster than Python.np{END}")


if __name__ == '__main__':
    test4()
    print()
