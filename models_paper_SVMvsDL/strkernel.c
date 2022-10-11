/**
 * Author:      César Ignacio García Osorio (cgosorio@ubu.es)
 * Created:     11/Oct/2022
 * Description: Implementation of the weighted degree string kernel as
 * described in https://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf
 *
 **/

#include <stdio.h>  // printf(), fpring(), stderr
#include <string.h> // strlen(), strcat()
#include <stddef.h> // size_t, wchar_t
#include <time.h>   // clock_t, clock(), CLOCKS_PER_SEC
#include <wchar.h>  // wchar_T, mbstowcs()
#include <stdlib.h>

#define BETA_K(d, k) 2*((1.0*d-k+1)/(d*(d+1)))

// This implementation is expecting strings of sequences, that means
// When called from Python, numpy array rows should be converted to strings
// first and later encoded, what adds an unnecessary overhead to the process.
long double cgo_get_K_value1(char *xi, char *xj, int L, int d)
{
    long double E1, beta;
    int E2;
    int k, l, j, eq;
    E1 = 0.0L;
    for (k=1; k<d+1; k++)
    {
        beta = BETA_K(d, k);
        E2 = 0;
        for (l=0; l<L-k+1; l++)
        {
            eq = 1;
            for (j=l; j<l+k; j++)
            {
                if (xi[j]!=xj[j]) 
                {
                    eq = 0;
                    break;
                }
            }
            E2 += eq;
        }
        E1 += beta*E2;
    }
    return E1;
}

// This implementation should be slightly faster than previous, but in
// practice it seems that it is only faster for d<=25. For bigger values
// this implementation is slower than previous.
// Still with the problem of numpy rows conversion.
long double cgo_get_K_value2(char *xi, char *xj, int L, int d)
{
    long double E1, beta;
    int E2;
    int k, l, j;
    char *xi_iniseq, *xj_iniseq;
    char *xi_currpos, *xj_currpos;
    E1 = 0.0L;
    for (k=1; k<d+1; k++)
    {
        beta = BETA_K(d, k);
        E2 = 0;
        for (l=0, xi_iniseq=xi, xj_iniseq=xj; l<L-k+1; l++, xi_iniseq++, xj_iniseq++)
        {
            for (j=l,   xi_currpos=xi_iniseq, xj_currpos=xj_iniseq;
                 j<l+k && *xi_currpos==*xj_currpos;
                 j++,   xi_currpos++, xj_currpos++) ;
            // Si he llegado hasta el final
            if (j==l+k) {
                E2++;
            }
        }
        E1 += beta*E2;
    }
    return E1;
}

// This is the fastest version when called from Python. Now the numpy array
// rows can be directly passed avoiding the overhead of transforming them to
// string and encoding them afterwards.
long double cgo_get_K_value3(const wchar_t *xi, const wchar_t *xj, int L, int
        d) { long double E1, beta; int E2; int k, l, j, eq;

    E1 = 0.0L;
    for (k=1; k<d+1; k++)
    {
        beta = BETA_K(d, k);
        E2 = 0;
        for (l=0; l<L-k+1; l++)
        {
            eq = 1;
            for (j=l; j<l+k; j++)
            {
                if (xi[j]!=xj[j]) 
                {
                    eq = 0;
                    break;
                }
            }
            E2 += eq;
        }
        E1 += beta*E2;
    }
    return E1;
}


int darray[] = {1, 2, 4};
int karray[] = {3, 4, 5};

// Testing that beta values are the same as in Python version
void test1()
{
    int d, k, dval, kval;
    for (d=0; d<3; d++)
    {
        dval = darray[d];
        for (k=0; k<3; k++)
        {
            kval = karray[k];
            printf("beta_k(%d, %d): %f\n", dval, kval, BETA_K(dval, kval));
        }
    }

}

char * SEQ1 = "CGGTACACGACGGTGTGACCTGTGATGCGGCAGGAAGCCGCTCCCATGCCTTCCGCTAAATTATACGAGACGAGCGGTTAGGCACATAATTGAATCTGCTGCTGTCGATCGCTAAGCATCCGACTCGTGAATCATATAAACATGTCTACTTATGATCAATCAATCCCCCCTCACATTGAATCCGAGCTCCGTGACATCACATGGGATTTCGTAATTTGCATGTGACGACAGCAGTCCTACCCCATTTGCCTGAGCTATTTGTGGCACGGATAGCCCGCTCCTACGCCTGGTTTCTTACTACGCTGCGCGAAGGTCGTCTTTGGCTCTACAGACTGGTCTTGACCGGCCCTTCCAGATAGAGGCCGGACAGCGTGGCCTCTTCATGCGAAAATTCGGCAGGAGGGGTAGGGACGGGGACAATAGACAGCCATCTATCGTAGAAAACCCCACTGACTGGGATGGACCAGCAGCTGGGAGCTAGACCGAGGAAGCAGTCCGACCCGAGGTAGCCTCTGCTCGCCCGCGGCTGCACCGGAGGACTTTACAGTGGAATTCAAGTATCAGATAACTTGGTGTCGTCTACTCAGAGAACTTAATTACAATATCCTACCCCGCCCCGAGGCAATTGTTCTAGTTAGAGCCTAATCTACGTGTAGGGCGACGAGTACTTATTGGCCAGCATAGCTGGTAGCCTATCGGGGGTTCCATCGAACCACGGCTATCGCAGTACATGAACAAATGACCCGCCTGCATACTAACGGTCTCTATGAAACAAATTTATACTTAGTGATATCTACCATCTAAAGTTCGGCTCTAACTGTTCGCCGTCCGATAAGTGCTCGGAGGCGTGCAACAGGCCCAACCGTGAAGACCTTTAACCCATCCAAGACTATGTGCGGAATGGTTATGCACGCACGTATCGTCATGCCCTTTCGATTCCCTCGGCTCTCGCAAACGGAGTCCGTAGGCGAAGCGCGCGATGAAGGTAGGGGACGGAACCTGT";

char * SEQ2 = "ACGACCCGTGGGATTCTCTTTCTCCACAATTCACTGGGCCTTTAACCCGAATAACCTTCGATCGCTCCTACATCTTTTAGCTCCGCACTTGCGTTAGCGTAGAGGGGGCACCCGCTTTGCGGCGACCATCCAACTTCGTAGCTTGGTCGCGTGCGACAATGTCTCTACACTCACTGGAGATTGGCCTACAGCACTACCTATACACGGGGGAGCATCCCTAAAGCCTTTCCCTTGTCTGGGAGCCGCGGCGAATTCGAAGACTAGAGACTAGTAGCCGGATAGTCCGCAAGGAGCTCATTGCTGCACGAAACTAAACTCTCAAACCGCCCAAAAGTGATATCGAGATGCTGCACACTCAAGCTGACTACCCGGTTTCAGTATGTACACTGATAGCGATTACTAATCAACCCAGTACGTTGTTAGGATATCAATCGGTTTGCGTTGATGGACAGCGGCGGCAAATCCGGACATTATCAAACATAAGTCAGGTCTGTCCCGGCAGGGTGATCGGCCTCTGCCTAAGAATGGGGATCTGGATTGGCCACTGAAGATGAAGTTCTGTGTAAAAATGCTGTGTTCGCCACAATACTGCTGTGTCGTCGAGATGCGGCAGTTGGGATCTTACCCACACTCCGGCGACGTGGAGATCCTTTATTGGCGTACTCGCCGACTATTCGGTGAGGACGATAACCCTTGTGCTCAGCTCCGGATACGTAATCCCTAGGAGAGTTCTTCTCTCTTGAACTGTTATCGGGTACTGCCGTACTCGCATGGCCGGTGCGATTATCCCAGTCCCCTAAGAACCAGATGTTGTACGGCGACCTAGGGGCGAGCGTTTTTTGTGACAATATCCACTAGCCTGATCGCATGTTAGGAGTAGGACTACTATTACACCGGCGTTACTAGGTAAATTTGGATAGGGTTTGCGGTAGCACAGACATAAACAGGACACAAGATGGTCTACCCACTACTCGCATTGGACCTGATGGTCGCGTCCACTATC";

/* char * SEQ1 = "AAGGTTAAGGTT"; */
/* char * SEQ2 = "AGGTTTAGGTTT"; */
/* char * SEQ1 = "AAGGTT"; */
/* char * SEQ2 = "AGGTTT"; */


// Utility function to get longer sequences by replicating one char string
char * repstr(char *input, int n)
{
    char * output = (char *)malloc(strlen(input)*n+1);
    if (n<1) {
        fprintf(stderr, "Number of replications should be larger than 0.");
        exit(-1);
    }
    strcpy(output, input);
    for(int i=1; i<n; i++) strcat(output, input);
    return output;
}

// Testing the times and that the results are always the same.
// The results are not too meaningful as here the methods are only
// compared in one sequence. The actual improvements will be seen when
// used within the calculation of the Kernel matrix in Python. There
// implementation 3 is clearly the winner as it avoids the overhead
// of transforming the numpy rows arrays into char strings.
void test2()
{
    int L, d;
    long double res;
    clock_t start, end;
    double time1, time2, time3;
    char * seq1, *seq2; 

    int n=5;

    seq1 = repstr(SEQ1, n);
    seq2 = repstr(SEQ2, n);

    L = strlen(seq1);
    d = (int)((float)L * 0.7);
    printf("L: %d, d: %d\n", L, d);

    start = clock();
    res = cgo_get_K_value1(seq1, seq2, L, d);
    end = clock();
    time1 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Res: %Lf in %f secs\n", res, time1);

    start = clock();
    res = cgo_get_K_value2(seq1, seq2, L, d);
    end = clock();
    time2 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Res: %Lf in %f secs\n", res, time2);

    // Transforming char string into wchar strings
    wchar_t seq1wchar[L+1];
    wchar_t seq2wchar[L+1];
    mbstowcs(seq1wchar, seq1, L);
    mbstowcs(seq2wchar, seq2, L);
    
    start = clock();
    res = cgo_get_K_value3(seq1wchar, seq2wchar, L, d);
    end = clock();
    time3 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Res: %Lf in %f secs\n", res, time3);

    printf("Version 3 is %f times faster than version 1\n", time1/time3);
    printf("Version 3 is %f times faster than version 2\n", time2/time3);

    free(seq1);
    free(seq2);
}

int main()
{
    test2();
    return 0;
}
