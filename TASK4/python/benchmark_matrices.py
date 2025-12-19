import time
import numpy as np

from matrix_local import baseline_multiply, parallel_multiply
from mapreduce_matrix import distributed_multiply

SIZES = [256, 512, 1024]
REPEATS = 1


def checksum(M: np.ndarray) -> float:
    return float(M.sum())


def run():
    print(f"NumPy version: {np.__version__}")
    for n in SIZES:
        print(f"\n===== MATRIX SIZE: {n} =====")
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        for r in range(REPEATS):
            print(f"--- Run {r+1} ---")

            t1 = time.time()
            C1 = baseline_multiply(A, B)
            t2 = time.time()
            baseline_t = (t2 - t1) * 1000
            print(f"Baseline local: {baseline_t:.1f} ms")

            t1 = time.time()
            C2 = parallel_multiply(A, B)
            t2 = time.time()
            parallel_t = (t2 - t1) * 1000
            print(f"Parallel local: {parallel_t:.1f} ms")

            t1 = time.time()
            C3, stats = distributed_multiply(A, B)
            t2 = time.time()
            dist_total_ms = (t2 - t1) * 1000
            print(f"Distributed MapReduce total: {dist_total_ms:.1f} ms")
            print(f"  prep:   {stats['prep_s']*1000:.1f} ms")
            print(f"  map:    {stats['map_s']*1000:.1f} ms")
            print(f"  reduce: {stats['reduce_s']*1000:.1f} ms")

            print("Checksums:",
                  checksum(C1), checksum(C2), checksum(C3))


if __name__ == "__main__":
    run()
