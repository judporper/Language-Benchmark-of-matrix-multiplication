#include <stdio.h>
#include <stdlib.h>
#include "matrix_multiplier.h"

double **generate_matrix(int n) {
    double **m = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        m[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            m[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return m;
}

double **multiply_matrices(double **a, double **b, int n) {
    double **c = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        c[i] = (double *)calloc(n, sizeof(double));
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

void free_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) {
        free(m[i]);
    }
    free(m);
}
