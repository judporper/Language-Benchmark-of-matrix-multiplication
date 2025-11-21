import pandas as pd
import matplotlib.pyplot as plt
import os

files = {
    "C": "data/benchmark_c_results.csv",
    "Python": "data/benchmark_python_results.csv",
    "Java": "data/java_results.csv"
}

plot_dir = os.path.join("data", "plots")
os.makedirs(plot_dir, exist_ok=True)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    if "AverageWall" in df.columns:
        df["AverageTime"] = df["AverageWall"]
    return df

data = {}
for lang, file in files.items():
    if os.path.exists(file):
        data[lang] = load_data(file)
    else:
        print(f"Not found: {file}")

dense_methods = ["Basic", "Blocked", "Strassen"]

for lang, df in data.items():
    df_dense = df[df["Approach"].isin(dense_methods)]
    plt.figure(figsize=(10,6))
    for method in dense_methods:
        df_m = df_dense[df_dense["Approach"] == method]
        if not df_m.empty:
            plt.plot(df_m["MatrixSize"], df_m["AverageTime"], marker='o', label=method)
    plt.title(f"{lang} - Wall Time (Métodos Dense)")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Wall Time (s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_dense_wall_time.png"), dpi=300)
    plt.close()

for lang, df in data.items():
    df_sparse = df[~df["Approach"].isin(dense_methods)]
    if not df_sparse.empty:
        plt.figure(figsize=(10,6))
        plt.bar(df_sparse["Approach"], df_sparse["AverageTime"], color="skyblue")
        plt.title(f"{lang} - Wall Time (Métodos Sparse)")
        plt.xlabel("Método Sparse")
        plt.ylabel("Wall Time (s)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_sparse_wall_time.png"), dpi=300)
        plt.close()

plt.figure(figsize=(10,6))
for lang, df in data.items():
    df_dense = df[df["Approach"] == "Basic"]
    if not df_dense.empty:
        plt.plot(df_dense["MatrixSize"], df_dense["AverageTime"], marker='o', label=f"{lang} Basic")
plt.title("Comparativa Wall Time - Método Basic")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Wall Time (s)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "comparativa_basic_wall_time.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,6))
for lang, df in data.items():
    df_dense = df[df["Approach"] == "Basic"]
    if not df_dense.empty:
        plt.plot(df_dense["MatrixSize"], df_dense["AverageCPU"], marker='x', linestyle='--', label=f"{lang} CPU")
plt.title("Comparativa CPU Time - Método Basic")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("CPU Time (s)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "comparativa_basic_cpu_time.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,6))
for lang, df in data.items():
    df_dense = df[df["Approach"] == "Basic"]
    if not df_dense.empty:
        plt.plot(df_dense["MatrixSize"], df_dense["PeakMemoryKB"], marker='s', label=f"{lang} Memoria")
plt.title("Comparativa Peak Memory - Método Basic")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Memoria Pico (KB)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "comparativa_basic_memory.png"), dpi=300)
plt.close()
