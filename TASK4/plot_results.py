import matplotlib.pyplot as plt
import numpy as np

sizes = np.array([256, 512, 1024])

java_naive = np.array([19, 172, 2665])
java_parallel = np.array([35, 39, 795])
java_distributed = np.array([1600, 280, 1402])

java_dist_distribution = np.array([121, 61, 216])
java_dist_compute = np.array([46, 178, 1150])
java_dist_gather = np.array([12, 16, 21])

# ============================
# DATOS DE PYTHON (MapReduce)
# ============================

py_naive = np.array([4120.5, 35334.2, 275173.7])
py_parallel = np.array([1661.7, 12132.2, 74859.8])
py_distributed = np.array([567.3, 817.2, 944.6])

py_dist_prep = np.array([0.0, 0.0, 1.0])
py_dist_map = np.array([566.3, 812.9, 924.4])
py_dist_reduce = np.array([0.0, 2.0, 13.1])

# ============================
# 1. TIEMPOS JAVA
# ============================

plt.figure(figsize=(8,5))
plt.plot(sizes, java_naive, marker='o', label="Java Naive")
plt.plot(sizes, java_parallel, marker='o', label="Java Parallel")
plt.plot(sizes, java_distributed, marker='o', label="Java Distributed (Hazelcast)")
plt.xlabel("Matrix size (N)")
plt.ylabel("Time (ms)")
plt.title("Java: Execution Time vs Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 2. TIEMPOS PYTHON
# ============================

plt.figure(figsize=(8,5))
plt.plot(sizes, py_naive, marker='o', label="Python Naive")
plt.plot(sizes, py_parallel, marker='o', label="Python Parallel")
plt.plot(sizes, py_distributed, marker='o', label="Python Distributed (MapReduce)")
plt.xlabel("Matrix size (N)")
plt.ylabel("Time (ms)")
plt.title("Python: Execution Time vs Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 3. OVERHEAD HAZELCAST (Java)
# ============================

plt.figure(figsize=(8,5))
plt.plot(sizes, java_dist_distribution, marker='o', label="Distribution")
plt.plot(sizes, java_dist_compute, marker='o', label="Compute")
plt.plot(sizes, java_dist_gather, marker='o', label="Gather")
plt.xlabel("Matrix size (N)")
plt.ylabel("Time (ms)")
plt.title("Java Hazelcast Overhead Breakdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 4. OVERHEAD PYTHON MAPREDUCE
# ============================

plt.figure(figsize=(8,5))
plt.plot(sizes, py_dist_prep, marker='o', label="Prep")
plt.plot(sizes, py_dist_map, marker='o', label="Map")
plt.plot(sizes, py_dist_reduce, marker='o', label="Reduce")
plt.xlabel("Matrix size (N)")
plt.ylabel("Time (ms)")
plt.title("Python MapReduce Overhead Breakdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 5. COMPARACIÓN JAVA vs PYTHON (Distribuido)
# ============================

plt.figure(figsize=(8,5))
plt.plot(sizes, java_distributed, marker='o', label="Java Distributed")
plt.plot(sizes, py_distributed, marker='o', label="Python Distributed")
plt.xlabel("Matrix size (N)")
plt.ylabel("Time (ms)")
plt.title("Distributed Performance: Java vs Python")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
