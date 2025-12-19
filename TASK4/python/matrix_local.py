import numpy as np
from multiprocessing import Pool, cpu_count


def baseline_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def _row_task(args):
    row, A, B = args
    n = B.shape[1]
    row_res = np.zeros(n)
    for j in range(n):
        s = 0.0
        for k in range(A.shape[1]):
            s += A[row, k] * B[k, j]
        row_res[j] = s
    return row, row_res


def parallel_multiply(A: np.ndarray, B: np.ndarray, workers: int | None = None) -> np.ndarray:
    if workers is None:
        workers = cpu_count()

    n = A.shape[0]
    C = np.zeros((n, n))
    tasks = [(i, A, B) for i in range(n)]

    with Pool(workers) as p:
        for row, row_res in p.imap_unordered(_row_task, tasks):
            C[row, :] = row_res

    return C
