#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

double **generate_matrix(int n);
double **multiply_basic(double **a, double **b, int n);
double **multiply_blocked(double **a, double **b, int n, int blockSize);
void free_matrix(double **m, int n);

typedef struct {
    int rows, cols, nnz;
    double *val;
    int *col_ind;
    int *row_ptr;
} CSRMatrix;

CSRMatrix *create_csr(int rows, int cols, int nnz);
void free_csr(CSRMatrix *A);

void spmv_csr(const CSRMatrix *A, const double *x, double *y);
CSRMatrix *csr_matmul(const CSRMatrix *A, const CSRMatrix *B);

CSRMatrix *load_csr_from_mtx(const char *path);

#endif
