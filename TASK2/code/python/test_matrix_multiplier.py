import random
import time
import os
import psutil
import sys
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from matrix_multiplier import multiply_basic, strassen, multiply_blocked, multiply_sparse, generate_sparse_matrix

matrix_sizes = [50, 100, 500, 1024]
runs = 3

if len(sys.argv) >= 2:
    matrix_sizes = [int(sys.argv[1])]
if len(sys.argv) >= 3:
    runs = int(sys.argv[2])

def generate_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "benchmark_python_results.csv")

def benchmark(name, func, A, B, runs=3):
    wall_times, cpu_times = [], []
    peak_mem = 0
    process = psutil.Process(os.getpid())

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

        delta_mem = max(0, mem_after - mem_before)
        if mem_after > peak_mem:
            peak_mem = mem_after

        print(f" [{name}] Run {r+1}: {wall_time:.6f}s | CPU {cpu_time:.6f}s | Mem {delta_mem:.2f} KB")

    return sum(wall_times)/runs, sum(cpu_times)/runs, peak_mem

with open(output_file, "w") as f:
    f.write("Approach,MatrixSize,AverageWall,AverageCPU,PeakMemoryKB\n")

    for n in matrix_sizes:
        print(f"\nMatrix {n}x{n}")
        A = generate_matrix(n)
        B = generate_matrix(n)

        avg_wall, avg_cpu, peak_mem = benchmark("Basic", multiply_basic, A, B, runs)
        f.write(f"Basic,{n},{avg_wall:.6f},{avg_cpu:.6f},{peak_mem:.2f}\n")

        if n & (n-1) == 0:
            avg_wall, avg_cpu, peak_mem = benchmark("Strassen", strassen, A, B, runs)
            f.write(f"Strassen,{n},{avg_wall:.6f},{avg_cpu:.6f},{peak_mem:.2f}\n")

        avg_wall, avg_cpu, peak_mem = benchmark("Blocked", multiply_blocked, A, B, runs)
        f.write(f"Blocked,{n},{avg_wall:.6f},{avg_cpu:.6f},{peak_mem:.2f}\n")

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

        avg_wall, avg_cpu, peak_mem = benchmark("Sparse_mc2depi", multiply_sparse, A_sparse, A_sparse, runs)
        f.write(f"Sparse_mc2depi,{A_sparse.shape[0]},{avg_wall:.6f},{avg_cpu:.6f},{peak_mem:.2f}\n")

    except Exception as e:
        print("No se pudo cargar mc2depi:", e)

    print("\nSynthetic Sparse Matrices (varying sparsity)")
    sparsity_levels = [0.1, 0.5, 0.9]
    for sparsity in sparsity_levels:
        A_sparse = generate_sparse_matrix(500, sparsity=sparsity)
        avg_wall, avg_cpu, peak_mem = benchmark(f"SparseSynthetic_{int(sparsity*100)}pctZeros", multiply_sparse, A_sparse, A_sparse, runs)
        f.write(f"SparseSynthetic_{int(sparsity*100)}pctZeros,{A_sparse.shape[0]},{avg_wall:.6f},{avg_cpu:.6f},{peak_mem:.2f}\n")
