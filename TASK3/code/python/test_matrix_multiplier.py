import random
import time
import os
import psutil
import sys
import csv
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from matrix_multiplier import (
    multiply_basic, strassen, multiply_blocked, multiply_sparse, generate_sparse_matrix,
    multiply_numpy, multiply_numba_basic, multiply_numba_parallel, multiply_numba_blocked,
    get_numba_threads, get_blas_threads,
)

matrix_sizes = [128, 256, 512, 1024]
runs = 5
warmup_runs = 1
thread_sweep = [1, 2, 4, 8]
block_size = 64

if len(sys.argv) >= 2:
    matrix_sizes = [int(sys.argv[1])]
if len(sys.argv) >= 3:
    runs = int(sys.argv[2])
if len(sys.argv) >= 4:
    warmup_runs = int(sys.argv[3])

def generate_matrix(n, seed=None):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(n)] for _ in range(n)]

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "benchmark_python_results.csv")

def benchmark(name, func, A, B, runs=3, warmup=1, metadata=None):
    wall_times, cpu_times = [], []
    peak_mem = 0
    process = psutil.Process(os.getpid())

    for _ in range(warmup):
        _ = func(A, B)

    for r in range(runs):
        mem_before = process.memory_info().rss / 1024
        cpu_before = process.cpu_times().user + process.cpu_times().system
        start = time.time()
        C = func(A, B)
        end = time.time()
        cpu_after = process.cpu_times().user + process.cpu_times().system
        mem_after = process.memory_info().rss / 1024

        wall_time = end - start
        cpu_time = cpu_after - cpu_before
        wall_times.append(wall_time)
        cpu_times.append(cpu_time)
        peak_mem = max(peak_mem, mem_after)

        print(f" [{name}] Run {r+1}: {wall_time:.6f}s | CPU {cpu_time:.6f}s | Peak {peak_mem:.2f} KB")

    avg_wall = sum(wall_times)/runs
    avg_cpu = sum(cpu_times)/runs
    return avg_wall, avg_cpu, peak_mem, metadata or {}

def write_row(writer, approach, n, avg_wall, avg_cpu, peak_mem,
              threads=None, speedup=None, efficiency=None, extra=None):
    writer.writerow({
        "Approach": approach,
        "MatrixSize": n,
        "AverageWall": f"{avg_wall:.6f}",
        "AverageCPU": f"{avg_cpu:.6f}",
        "PeakMemoryKB": f"{peak_mem:.2f}",
        "Threads": threads if threads is not None else "",
        "Speedup_vs_Basic": f"{speedup:.3f}" if isinstance(speedup,(int,float)) else "",
        "Efficiency_per_thread": f"{efficiency:.3f}" if isinstance(efficiency,(int,float)) else "",
        "Extra": extra or ""
    })

def compute_speedup(baseline_time, method_time):
    return baseline_time / method_time if method_time > 0 else float("inf")

print("Detectando hilos BLAS (NumPy):", get_blas_threads())
print("Detectando hilos Numba:", get_numba_threads())

with open(output_file, "w", newline="") as f:
    fieldnames = ["Approach","MatrixSize","AverageWall","AverageCPU","PeakMemoryKB","Threads","Speedup_vs_Basic","Efficiency_per_thread","Extra"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for n in matrix_sizes:
        print(f"\nMatrix {n}x{n}")
        A = generate_matrix(n, seed=42)
        B = generate_matrix(n, seed=1337)

        base_wall, base_cpu, base_mem, _ = benchmark("Basic", multiply_basic, A, B, runs=runs, warmup=warmup_runs)
        write_row(writer, "Basic", n, base_wall, base_cpu, base_mem, threads=1, speedup=1.0, efficiency=1.0)

        if n & (n-1) == 0:
            s_wall, s_cpu, s_mem, _ = benchmark("Strassen", strassen, A, B, runs=runs, warmup=warmup_runs)
            s_sp = compute_speedup(base_wall, s_wall)
            write_row(writer, "Strassen", n, s_wall, s_cpu, s_mem, threads=None, speedup=s_sp, efficiency=None)

        blk_wall, blk_cpu, blk_mem, _ = benchmark("Blocked", lambda X, Y: multiply_blocked(X, Y, block_size), A, B, runs=runs, warmup=warmup_runs)
        blk_sp = compute_speedup(base_wall, blk_wall)
        write_row(writer, "Blocked", n, blk_wall, blk_cpu, blk_mem, threads=None, speedup=blk_sp, efficiency=None)

        np_wall, np_cpu, np_mem, meta = benchmark("NumPy_BLAS", multiply_numpy, A, B, runs=runs, warmup=warmup_runs, metadata={"blas_threads": get_blas_threads()})
        np_sp = compute_speedup(base_wall, np_wall)
        eff_np = (np_sp/meta["blas_threads"]) if meta.get("blas_threads") else None
        write_row(writer, "NumPy_BLAS", n, np_wall, np_cpu, np_mem, threads=meta.get("blas_threads"), speedup=np_sp, efficiency=eff_np)

        try:
            nb1_wall, nb1_cpu, nb1_mem, _ = benchmark("Numba_Basic_1t", lambda X, Y: multiply_numba_basic(X, Y, threads=1), A, B, runs=runs, warmup=warmup_runs)
            nb1_sp = compute_speedup(base_wall, nb1_wall)
            write_row(writer, "Numba_Basic_1t", n, nb1_wall, nb1_cpu, nb1_mem, threads=1, speedup=nb1_sp, efficiency=nb1_sp)
        except Exception as e:
            print("Numba_Basic_1t error:", e)

        for t in thread_sweep:
            try:
                nbp_wall, nbp_cpu, nbp_mem, _ = benchmark(f"Numba_Parallel_{t}t", lambda X, Y: multiply_numba_parallel(X, Y, threads=t), A, B, runs=runs, warmup=warmup_runs)
                sp = compute_speedup(base_wall, nbp_wall)
                eff = sp / t
                write_row(writer, "Numba_Parallel", n, nbp_wall, nbp_cpu, nbp_mem, threads=t, speedup=sp, efficiency=eff)
            except Exception as e:
                print(f"Numba_Parallel_{t}t error:", e)

        for t in thread_sweep:
            try:
                nbb_wall, nbb_cpu, nbb_mem, _ = benchmark(f"Numba_Blocked_{t}t", lambda X, Y: multiply_numba_blocked(X, Y, block_size=block_size, threads=t), A, B, runs=runs, warmup=warmup_runs)
                sp = compute_speedup(base_wall, nbb_wall)
                eff = sp / t
                write_row(writer, "Numba_Blocked", n, nbb_wall, nbb_cpu, nbb_mem, threads=t, speedup=sp, efficiency=eff, extra=f"block_size={block_size}")
            except Exception as e:
                print(f"Numba_Blocked_{t}t error:", e)

    try:
        print("\nSparse Matrix mc2depi")
        sparse_path_mat = "../../mc2depi.mat"
        if os.path.exists(sparse_path_mat):
            mat_data = loadmat(sparse_path_mat, struct_as_record=False, squeeze_me=True)
            if "Problem" in mat_data:
                problem_struct = mat_data["Problem"]
                if hasattr(problem_struct, "A"):
                    A_sparse = csr_matrix(problem_struct.A)
                else:
                    raise ValueError("El struct 'Problem' no contiene campo 'A'")
            else:
                raise ValueError("No se encontró 'Problem' en mc2depi.mat")
        else:
            raise FileNotFoundError("No se encontró mc2depi.mat")

        s_wall, s_cpu, s_mem, _ = benchmark("Sparse_mc2depi", multiply_sparse, A_sparse, A_sparse, runs=runs, warmup=warmup_runs)
        write_row(writer, "Sparse_mc2depi", A_sparse.shape[0], s_wall, s_cpu, s_mem, threads=None, speedup=None, efficiency=None)

    except Exception as e:
        print("No se pudo cargar mc2depi:", e)

    print("\nSynthetic Sparse Matrices (varying sparsity)")
    sparsity_levels = [0.1, 0.5, 0.9]
    for sparsity in sparsity_levels:
        A_sparse = generate_sparse_matrix(500, sparsity=sparsity, seed=123)
        s_wall, s_cpu, s_mem, _ = benchmark(f"SparseSynthetic_{int(sparsity*100)}pctZeros", multiply_sparse, A_sparse, A_sparse, runs=runs, warmup=warmup_runs)
        write_row(writer, f"SparseSynthetic_{int(sparsity*100)}pctZeros", A_sparse.shape[0], s_wall, s_cpu, s_mem, threads=None, speedup=None, efficiency=None)

print(f"\nResultados guardados en: {output_file}")
