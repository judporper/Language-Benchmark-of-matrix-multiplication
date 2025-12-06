import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-colorblind")

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

    for col in ["MatrixSize", "AverageWall", "AverageCPU", "PeakMemoryKB",
                "Threads", "Speedup_vs_Basic", "Efficiency_per_thread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AverageWall" in df.columns:
        df["AverageTime"] = df["AverageWall"]

    return df

data = {}
for lang, file in files.items():
    if os.path.exists(file):
        data[lang] = load_data(file)

dense_methods = ["Basic", "Blocked", "Strassen", "NumPy_BLAS",
                 "Numba_Parallel", "Numba_Blocked"]

for lang, df in data.items():
    df_dense = df[df["Approach"].isin(dense_methods)]
    if not df_dense.empty:
        plt.figure(figsize=(10, 6))
        for method in dense_methods:
            df_m = df_dense[df_dense["Approach"] == method]
            if not df_m.empty:
                plt.plot(df_m["MatrixSize"], df_m["AverageTime"], marker='o', label=method)
        plt.title(f"{lang} – Dense Methods: Wall Time")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Wall Time (s)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_dense_wall_time.png"), dpi=300)
        plt.close()

for lang, df in data.items():
    if "Approach" in df.columns and "AverageTime" in df.columns:
        df_sparse = df[df["Approach"].str.contains("Sparse", na=False)]
        if not df_sparse.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(df_sparse["Approach"], df_sparse["AverageTime"])
            plt.title(f"{lang} – Sparse Methods")
            plt.xlabel("Método Sparse")
            plt.ylabel("Wall Time (s)")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle="--")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_sparse_wall_time.png"), dpi=300)
            plt.close()

plt.figure(figsize=(10, 6))
for lang, df in data.items():
    if "Approach" in df.columns and "AverageTime" in df.columns:
        df_basic = df[df["Approach"] == "Basic"]
        if not df_basic.empty:
            plt.plot(df_basic["MatrixSize"], df_basic["AverageTime"], marker='o', label=f"{lang}")
plt.title("Comparativa Lenguajes – Método Basic (Wall Time)")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Wall Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "comparativa_basic_wall_time.png"), dpi=300)
plt.close()

for lang, df in data.items():
    if "Speedup_vs_Basic" in df.columns and df["Speedup_vs_Basic"].notna().any():
        df_sp = df[df["Speedup_vs_Basic"].notna()]
        if not df_sp.empty:
            plt.figure(figsize=(10, 6))
            for approach in df_sp["Approach"].unique():
                dfa = df_sp[df_sp["Approach"] == approach]
                if len(dfa) > 0 and any(x in approach for x in ["Numba", "BLAS", "Parallel"]):
                    plt.plot(dfa["MatrixSize"], dfa["Speedup_vs_Basic"], marker='o', label=approach)
            plt.title(f"{lang} – Speedup frente a Basic")
            plt.xlabel("Matrix Size (N)")
            plt.ylabel("Speedup")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_speedup.png"), dpi=300)
            plt.close()

for lang, df in data.items():
    if "Efficiency_per_thread" in df.columns and df["Efficiency_per_thread"].notna().any():
        df_eff = df[df["Efficiency_per_thread"].notna()]
        if not df_eff.empty:
            plt.figure(figsize=(10, 6))
            for n in df_eff["MatrixSize"].unique():
                df_n = df_eff[df_eff["MatrixSize"] == n]
                plt.plot(df_n["Threads"], df_n["Efficiency_per_thread"], marker='o', label=f"N={n}")
            plt.title(f"{lang} – Eficiencia por Hilo")
            plt.xlabel("Threads")
            plt.ylabel("Efficiency (Speedup / Threads)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_efficiency_per_thread.png"), dpi=300)
            plt.close()

for lang, df in data.items():
    if "Threads" in df.columns and df["Threads"].notna().any() and "AverageTime" in df.columns:
        df_thr = df[df["Threads"].notna()]
        if not df_thr.empty:
            plt.figure(figsize=(10, 6))
            for n in df_thr["MatrixSize"].unique():
                df_n = df_thr[df_thr["MatrixSize"] == n]
                plt.plot(df_n["Threads"], df_n["AverageTime"], marker='o', label=f"N={n}")
            plt.title(f"{lang} – Wall Time vs Threads")
            plt.xlabel("Número de hilos")
            plt.ylabel("Wall Time (s)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{lang.lower()}_threads_vs_time.png"), dpi=300)
            plt.close()

print("\nTodas las gráficas generadas en:", plot_dir)
