import os
import numpy as np
from scipy.sparse import csr_matrix

def get_blas_threads():
    num = None
    try:
        from threadpoolctl import threadpool_info
        infos = threadpool_info()
        for info in infos:
            if info.get("internal_api") in ("mkl", "openblas", "blis"):
                num = info.get("num_threads", num)
    except Exception:
        pass
    if num is None:
        for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if key in os.environ:
                try:
                    num = int(os.environ[key])
                    break
                except Exception:
                    pass
    return num

def multiply_basic(A, B):
    n = len(A)
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def strassen(A, B):
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
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
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for ii in range(0, n, block_size):
        for kk in range(0, n, block_size):
            Ablk = A[ii:ii+block_size, kk:kk+block_size]
            for jj in range(0, n, block_size):
                Bblk = B[kk:kk+block_size, jj:jj+block_size]
                C[ii:ii+block_size, jj:jj+block_size] += Ablk @ Bblk
    return C

def multiply_numpy(A, B):
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    return A @ B  # BLAS

def multiply_sparse(A_sparse, B_sparse):
    if not isinstance(A_sparse, csr_matrix):
        A_sparse = csr_matrix(A_sparse)
    if not isinstance(B_sparse, csr_matrix):
        B_sparse = csr_matrix(B_sparse)
    return A_sparse.dot(B_sparse)

def generate_sparse_matrix(n, sparsity=0.9, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, n, size=int((1.0 - sparsity) * n * n))
        cols = rng.integers(0, n, size=rows.size)
        vals = rng.random(size=rows.size)
    else:
        nnz = int((1.0 - sparsity) * n * n)
        rows = np.random.randint(0, n, size=nnz)
        cols = np.random.randint(0, n, size=nnz)
        vals = np.random.random(size=nnz)
    return csr_matrix((vals, (rows, cols)), shape=(n, n))

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _basic_numba(A, B):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C

    @njit(parallel=True, fastmath=True)
    def _parallel_numba(A, B):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for i in prange(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C

    @njit(parallel=True, fastmath=True)
    def _blocked_numba(A, B, block_size):
        n = A.shape[0]
        C = np.zeros((n, n), dtype=np.float64)
        for ii in range(0, n, block_size):
            iimax = min(ii + block_size, n)
            for kk in range(0, n, block_size):
                kkmax = min(kk + block_size, n)
                for jj in range(0, n, block_size):
                    jjmax = min(jj + block_size, n)
                    for i in prange(ii, iimax):
                        for k in range(kk, kkmax):
                            aik = A[i, k]
                            for j in range(jj, jjmax):
                                C[i, j] += aik * B[k, j]
        return C

def multiply_numba_basic(A, B, threads=None):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba no disponible")
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    if threads:
        set_num_threads(threads)
    return _basic_numba(A, B)

def multiply_numba_parallel(A, B, threads=None):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba no disponible")
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    if threads:
        set_num_threads(threads)
    return _parallel_numba(A, B)

def multiply_numba_blocked(A, B, block_size=64, threads=None):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba no disponible")
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    if threads:
        set_num_threads(threads)
    return _blocked_numba(A, B, block_size)

def get_numba_threads():
    if NUMBA_AVAILABLE:
        try:
            return get_num_threads()
        except Exception:
            return None
    return None
