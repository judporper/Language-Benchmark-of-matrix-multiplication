#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

double **multiply_basic(double **a, double **b, int n) {
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

double **multiply_blocked(double **a, double **b, int n, int blockSize) {
    double **c = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) c[i] = (double *)calloc(n, sizeof(double));

    for (int ii = 0; ii < n; ii += blockSize)
        for (int jj = 0; jj < n; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < ii+blockSize && i<n; i++)
                    for (int j = jj; j < jj+blockSize && j<n; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < kk+blockSize && k<n; k++)
                            sum += a[i][k] * b[k][j];
                        c[i][j] += sum;
                    }
    return c;
}

void free_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) free(m[i]);
    free(m);
}

CSRMatrix *create_csr(int rows, int cols, int nnz) {
    CSRMatrix *A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = rows; A->cols = cols; A->nnz = nnz;
    A->val = (double*)malloc(nnz * sizeof(double));
    A->col_ind = (int*)malloc(nnz * sizeof(int));
    A->row_ptr = (int*)malloc((rows+1) * sizeof(int));
    return A;
}

void free_csr(CSRMatrix *A) {
    if (!A) return;
    free(A->val); free(A->col_ind); free(A->row_ptr); free(A);
}

void spmv_csr(const CSRMatrix *A, const double *x, double *y) {
    for (int i=0; i<A->rows; i++) {
        double sum = 0.0;
        for (int k=A->row_ptr[i]; k<A->row_ptr[i+1]; k++) {
            sum += A->val[k] * x[A->col_ind[k]];
        }
        y[i] = sum;
    }
}

CSRMatrix *csr_matmul(const CSRMatrix *A, const CSRMatrix *B) {
    if (A->cols != B->rows) return NULL;
    int m = A->rows, n = B->cols;
    int max_nnz = A->nnz * B->nnz;
    CSRMatrix *C = create_csr(m, n, max_nnz);
    int nnz = 0;
    C->row_ptr[0] = 0;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            double sum = 0.0;
            for (int pa=A->row_ptr[i]; pa<A->row_ptr[i+1]; pa++) {
                int a_col = A->col_ind[pa];
                for (int pb=B->row_ptr[a_col]; pb<B->row_ptr[a_col+1]; pb++) {
                    if (B->col_ind[pb]==j) sum += A->val[pa]*B->val[pb];
                }
            }
            if (sum!=0.0) {
                C->col_ind[nnz] = j;
                C->val[nnz] = sum;
                nnz++;
            }
        }
        C->row_ptr[i+1] = nnz;
    }
    C->nnz = nnz;
    return C;
}

CSRMatrix *load_csr_from_mtx(const char *path) {
    FILE *f = fopen(path,"r");
    if (!f) {
        fprintf(stderr,"No se pudo abrir %s\n",path);
        return NULL;
    }
    char line[256];
    do {
        if (!fgets(line,256,f)) { fclose(f); return NULL; }
    } while(line[0]=='%');

    int rows, cols, nnz;
    sscanf(line,"%d %d %d",&rows,&cols,&nnz);

    int *row_counts = (int*)calloc(rows,sizeof(int));
    int *rowInd = (int*)malloc(nnz*sizeof(int));
    int *colInd = (int*)malloc(nnz*sizeof(int));
    double *vals = (double*)malloc(nnz*sizeof(double));

    for (int i=0;i<nnz;i++) {
        fgets(line,256,f);
        int r,c; double v;
        sscanf(line,"%d %d %lf",&r,&c,&v);
        r--; c--;
        rowInd[i]=r; colInd[i]=c; vals[i]=v;
        row_counts[r]++;
    }
    fclose(f);

    CSRMatrix *A = create_csr(rows,cols,nnz);
    A->row_ptr[0]=0;
    for (int i=0;i<rows;i++) A->row_ptr[i+1]=A->row_ptr[i]+row_counts[i];

    int *fill = (int*)calloc(rows,sizeof(int));
    for (int i=0;i<rows;i++) fill[i]=A->row_ptr[i];
    for (int i=0;i<nnz;i++) {
        int r=rowInd[i];
        int dst=fill[r]++;
        A->col_ind[dst]=colInd[i];
        A->val[dst]=vals[i];
    }

    free(row_counts); free(rowInd); free(colInd); free(vals); free(fill);
    return A;
}
