#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <string.h>
#endif

#include "matrix_multiplier.h"

double get_time_in_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

long get_memory_usage_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (long)(pmc.PeakWorkingSetSize / 1024);
    }
    return -1;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (long)usage.ru_maxrss;
#endif
}

double get_cpu_time_seconds() {
#ifdef _WIN32
    FILETIME createTime, exitTime, kernelTime, userTime;
    if (GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime)) {
        ULARGE_INTEGER kt, ut;
        kt.LowPart = kernelTime.dwLowDateTime; kt.HighPart = kernelTime.dwHighDateTime;
        ut.LowPart = userTime.dwLowDateTime;  ut.HighPart = userTime.dwHighDateTime;
        return (kt.QuadPart + ut.QuadPart) / 10000000.0;
    }
    return -1.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (double)usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6
         + (double)usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6;
#endif
}

typedef struct {
    double avg_wall;
    double avg_cpu;
    long peak_mem;
} Result;

Result run_benchmark(const char *name,
                     double **(*func)(double **, double **, int),
                     double **A, double **B, int n, int runs) {
    double wall_times[runs], cpu_times[runs];
    long peakMem = 0;

    for (int r=0; r<runs; r++) {
        double cpu_before = get_cpu_time_seconds();
        double start = get_time_in_seconds();

        double **C = func(A,B,n);

        double end = get_time_in_seconds();
        double cpu_after = get_cpu_time_seconds();
        wall_times[r] = end - start;
        cpu_times[r] = cpu_after - cpu_before;

        long mem = get_memory_usage_kb();
        if(mem > peakMem) peakMem = mem;

        free_matrix(C, n);
        printf(" [%s] Run %d: %.6f s (wall) | %.6f s (CPU) | PeakMem %ld KB\n",
               name, r+1, wall_times[r], cpu_times[r], peakMem);
    }
    Result res = {0};
    for (int r=0; r<runs; r++) { res.avg_wall += wall_times[r]; res.avg_cpu += cpu_times[r]; }
    res.avg_wall /= runs; res.avg_cpu /= runs; res.peak_mem = peakMem;
    return res;
}

double **wrapper_basic(double **A, double **B, int n) { return multiply_basic(A,B,n); }
double **wrapper_blocked(double **A, double **B, int n) { return multiply_blocked(A,B,n,64); }
double **wrapper_simd(double **A, double **B, int n) {
    double **BT = transpose_matrix(B,n);
    double **C = multiply_simd_transposed(A,BT,n);
    free_matrix(BT,n);
    return C;
}
double **wrapper_openmp_rows(double **A, double **B, int n) { return multiply_openmp_rows(A,B,n,4); } // ejemplo con 4 hilos
double **wrapper_openmp_blocked(double **A, double **B, int n) { return multiply_openmp_blocked(A,B,n,64,4); }

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    int sizes[] = {128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int runs = 5;
    int warmup = 1;

    if (argc >= 2) runs = atoi(argv[1]);
#ifdef _WIN32
    system("mkdir ..\\..\\data >nul 2>&1");
#else
    system("mkdir -p ../../data");
#endif

    FILE *output = fopen("../../data/benchmark_c_results.csv", "w");
    if (!output) { perror("Couldn't open file "); return 1; }
    fprintf(output, "Approach,MatrixSize,AverageWall,AverageCPU,PeakMemoryKB\n");

    for (int s=0; s<num_sizes; s++) {
        int n = sizes[s];
        printf("\nDense Matrix %dx%d\n", n, n);

        double **A = generate_matrix(n);
        double **B = generate_matrix(n);

        for (int w=0; w<warmup; w++) {
            double **Cw = multiply_basic(A,B,n);
            free_matrix(Cw,n);
        }

        Result base = run_benchmark("Basic", wrapper_basic, A, B, n, runs);
        fprintf(output, "Basic,%d,%.6f,%.6f,%ld\n", n, base.avg_wall, base.avg_cpu, base.peak_mem);

        Result blk = run_benchmark("Blocked", wrapper_blocked, A, B, n, runs);
        fprintf(output, "Blocked,%d,%.6f,%.6f,%ld\n", n, blk.avg_wall, blk.avg_cpu, blk.peak_mem);

        Result simd = run_benchmark("SIMD_AVX2", wrapper_simd, A, B, n, runs);
        fprintf(output, "SIMD_AVX2,%d,%.6f,%.6f,%ld\n", n, simd.avg_wall, simd.avg_cpu, simd.peak_mem);

#ifdef _OPENMP
        Result pr = run_benchmark("OpenMP_Rows", wrapper_openmp_rows, A, B, n, runs);
        fprintf(output, "OpenMP_Rows,%d,%.6f,%.6f,%ld\n", n, pr.avg_wall, pr.avg_cpu, pr.peak_mem);

        Result pblk = run_benchmark("OpenMP_Blocked", wrapper_openmp_blocked, A, B, n, runs);
        fprintf(output, "OpenMP_Blocked,%d,%.6f,%.6f,%ld\n", n, pblk.avg_wall, pblk.avg_cpu, pblk.peak_mem);
#else
        printf("OpenMP no habilitado en compilación; saltando variantes paralelas.\n");
#endif

        free_matrix(A, n);
        free_matrix(B, n);
    }

    fclose(output);
    printf("\nResults written to ../../data/benchmark_c_results.csv\n");
    return 0;
}
