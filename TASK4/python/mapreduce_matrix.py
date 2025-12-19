import numpy as np
import time
from multiprocessing import Pool, cpu_count

BLOCK_SIZE = 256


def map_task(args):
    bi, bj, A_block, B_block = args
    return bi, bj, np.dot(A_block, B_block)


def distributed_multiply(A: np.ndarray, B: np.ndarray,
                         workers: int | None = None,
                         block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, dict]:
    if workers is None:
        workers = cpu_count()

    n = A.shape[0]
    assert n % block_size == 0, "n debe ser múltiplo del tamaño de bloque"

    tasks = []
    num_blocks = n // block_size

    prep_start = time.time()
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                A_block = A[bi*block_size:(bi+1)*block_size,
                          bk*block_size:(bk+1)*block_size]
                B_block = B[bk*block_size:(bk+1)*block_size,
                          bj*block_size:(bj+1)*block_size]
                tasks.append((bi, bj, A_block, B_block))
    prep_end = time.time()

    map_start = time.time()
    with Pool(workers) as p:
        results = list(p.imap_unordered(map_task, tasks))
    map_end = time.time()

    reduce_start = time.time()
    C = np.zeros((n, n))
    for bi, bj, block_res in results:
        i0 = bi * block_size
        j0 = bj * block_size
        C[i0:i0+block_size, j0:j0+block_size] += block_res
    reduce_end = time.time()

    stats = {
        "prep_s": prep_end - prep_start,
        "map_s": map_end - map_start,
        "reduce_s": reduce_end - reduce_start,
        "total_s": reduce_end - prep_start,
        "workers": workers,
        "block_size": block_size,
        "size": n,
    }
    return C, stats


if __name__ == "__main__":
    A = np.random.rand(512, 512)
    B = np.random.rand(512, 512)
    C, stats = distributed_multiply(A, B)
    print(C.shape, stats)
