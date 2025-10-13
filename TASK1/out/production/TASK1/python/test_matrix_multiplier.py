import random
import time
import os
import psutil
import sys
from matrix_multiplier import multiply_matrices

matrix_sizes = [50, 100, 500, 1024, 2000]
runs = 5

if len(sys.argv) >= 2:
    matrix_sizes = [int(sys.argv[1])]
if len(sys.argv) >= 3:
    runs = int(sys.argv[2])

def generate_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "benchmark_python_results.csv")

with open(output_file, "w") as f:
    f.write("MatrixSize")
    for r in range(1, runs+1): f.write(f",Run{r}_Time")
    f.write(",AverageTime")
    for r in range(1, runs+1): f.write(f",Run{r}_CPU")
    f.write(",AverageCPU,PeakMemoryKB\n")

    for n in matrix_sizes:
        print(f"\nMatrix {n}x{n}")
        A = generate_matrix(n)
        B = generate_matrix(n)
        wall_times = []
        cpu_times = []
        peak_mem = 0

        for r in range(runs):
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024
            cpu_before = process.cpu_times().user + process.cpu_times().system
            start = time.time()

            C = multiply_matrices(A, B)

            end = time.time()
            cpu_after = process.cpu_times().user + process.cpu_times().system
            mem_after = process.memory_info().rss / 1024

            wall_time = end - start
            cpu_time = cpu_after - cpu_before
            wall_times.append(wall_time)
            cpu_times.append(cpu_time)
            if mem_after > peak_mem: peak_mem = mem_after

            print(f" Run {r+1}: {wall_time:.6f}s (wall) | {cpu_time:.6f}s (CPU) | Peak memory: {mem_after-mem_before:.2f} KB")

        avg_wall = sum(wall_times)/runs
        avg_cpu = sum(cpu_times)/runs
        row = ",".join([f"{t:.6f}" for t in wall_times])
        row_cpu = ",".join([f"{t:.6f}" for t in cpu_times])
        f.write(f"{n},{row},{avg_wall:.6f},{row_cpu},{avg_cpu:.6f},{peak_mem:.2f}\n")
        print(f" Average: {avg_wall:.6f}s (wall), {avg_cpu:.6f}s (CPU), Peak memory: {peak_mem:.2f} KB")
