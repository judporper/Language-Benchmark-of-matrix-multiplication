import numpy as np
from scipy.sparse import csr_matrix

def multiply_basic(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def strassen(A, B):
    A = np.array(A)
    B = np.array(B)
    n = A.shape[0]

    if n == 1:
        return A * B

    mid = n // 2
    A11, A12, A21, A22 = A[:mid,:mid], A[:mid,mid:], A[mid:,:mid], A[mid:,mid:]
    B11, B12, B21, B22 = B[:mid,:mid], B[:mid,mid:], B[mid:,:mid], B[mid:,mid:]

    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

def multiply_blocked(A, B, block_size=64):
    A = np.array(A)
    B = np.array(B)
    n = A.shape[0]
    C = np.zeros((n, n))
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii+block_size, n)):
                    for j in range(jj, min(jj+block_size, n)):
                        sum_val = 0
                        for k in range(kk, min(kk+block_size, n)):
                            sum_val += A[i, k] * B[k, j]
                        C[i, j] += sum_val
    return C

def multiply_sparse(A_sparse, B_sparse):
    if not isinstance(A_sparse, csr_matrix):
        A_sparse = csr_matrix(A_sparse)
    if not isinstance(B_sparse, csr_matrix):
        B_sparse = csr_matrix(B_sparse)
    return A_sparse.dot(B_sparse)

def generate_sparse_matrix(n, sparsity=0.9):
    density = 1.0 - sparsity
    nnz = int(density * n * n)
    rows, cols, vals = [], [], []
    for _ in range(nnz):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        rows.append(i)
        cols.append(j)
        vals.append(np.random.random())
    return csr_matrix((vals, (rows, cols)), shape=(n, n))
