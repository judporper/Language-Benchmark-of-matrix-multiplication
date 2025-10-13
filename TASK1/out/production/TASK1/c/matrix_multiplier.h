#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

double **generate_matrix(int n);
double **multiply_matrices(double **a, double **b, int n);
void free_matrix(double **m, int n);

#endif
