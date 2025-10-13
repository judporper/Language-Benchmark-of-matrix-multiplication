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
    if "AverageTime" not in df.columns:
        run_cols = [c for c in df.columns if "Run" in c and "CPU" not in c]
        df["AverageTime"] = df[run_cols].mean(axis=1)
    if "AverageCPU" not in df.columns:
        run_cpu_cols = [c for c in df.columns if "CPU" in c]
        if run_cpu_cols:
            df["AverageCPU"] = df[run_cpu_cols].mean(axis=1)
    return df

data = {}
for lang, file in files.items():
    if os.path.exists(file):
        data[lang] = load_data(file)
    else:
        print(f"⚠ Not found: {file}")

# --- Función para plotear ---
def plot_metric(metric, ylabel, filename, languages=None, markers=None, linestyles=None):
    plt.figure(figsize=(10,6))
    langs_to_plot = languages if languages else data.keys()
    for i, lang in enumerate(langs_to_plot):
        df = data[lang]
        m = markers[i] if markers else 'o'
        ls = linestyles[i] if linestyles else '-'
        if metric in df.columns:
            plt.plot(df["MatrixSize"], df[metric], marker=m, linestyle=ls, label=lang)
    plt.title(f"{ylabel}")
    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()
    print(f"{filename} saved")

plot_metric("AverageTime", "Average Wall Time (s)", "benchmark_wall_time.png")

plot_metric("AverageCPU", "Average CPU Time (s)", "benchmark_cpu_time.png")

plot_metric("PeakMemoryKB", "Peak Memory (KB)", "benchmark_memory.png")

plt.figure(figsize=(10,6))
for i, lang in enumerate(data.keys()):
    df = data[lang]
    if "AverageCPU" in df.columns:
        plt.plot(df["MatrixSize"], df["AverageCPU"], marker='x', linestyle='--', label=f"{lang} CPU")
    plt.plot(df["MatrixSize"], df["AverageTime"], marker='o', linestyle='-', label=f"{lang} Wall")
plt.title("Wall Time vs CPU Time")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (s)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "benchmark_wall_vs_cpu.png"), dpi=300)
plt.close()
print("benchmark_wall_vs_cpu.png saved")

langs_CJava = ["C", "Java"]
plot_metric("AverageTime", "Average Wall Time (s) (C vs Java)", "benchmark_wall_time_c_java.png", languages=langs_CJava, markers=['o','s'])
plot_metric("AverageCPU", "Average CPU Time (s) (C vs Java)", "benchmark_cpu_time_c_java.png", languages=langs_CJava, markers=['o','s'])
plot_metric("PeakMemoryKB", "Peak Memory (KB) (C vs Java)", "benchmark_memory_c_java.png", languages=langs_CJava, markers=['o','s'])
