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
        kt.LowPart = kernelTime.dwLowDateTime;
        kt.HighPart = kernelTime.dwHighDateTime;
        ut.LowPart = userTime.dwLowDateTime;
        ut.HighPart = userTime.dwHighDateTime;
        return (kt.QuadPart + ut.QuadPart) / 10000000.0; // convertir a segundos
    }
    return -1.0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (double)usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6
         + (double)usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6;
#endif
}

void benchmark_dense(FILE *output, int n, int runs) {
    double wall_times[runs], cpu_times[runs];
    long peakMem = 0;

    double **A = generate_matrix(n);
    double **B = generate_matrix(n);

    // Basic
    for(int r=0;r<runs;r++){
        double cpu_before = get_cpu_time_seconds();
        double start = get_time_in_seconds();

        double **C = multiply_basic(A,B,n);

        double end = get_time_in_seconds();
        double cpu_after = get_cpu_time_seconds();
        wall_times[r] = end - start;
        cpu_times[r] = cpu_after - cpu_before;

        long mem = get_memory_usage_kb();
        if(mem > peakMem) peakMem = mem;

        free_matrix(C,n);
        printf(" [Basic] Run %d: %.6f s (wall) | %.6f s (CPU) | PeakMem %ld KB\n",
               r+1, wall_times[r], cpu_times[r], peakMem);
    }
    double avg_wall=0, avg_cpu=0;
    for(int r=0;r<runs;r++){ avg_wall+=wall_times[r]; avg_cpu+=cpu_times[r]; }
    avg_wall/=runs; avg_cpu/=runs;
    fprintf(output,"Basic,%d,%.6f,%.6f,%ld\n",n,avg_wall,avg_cpu,peakMem);

    // Blocked
    peakMem=0; avg_wall=0; avg_cpu=0;
    for(int r=0;r<runs;r++){
        double cpu_before = get_cpu_time_seconds();
        double start = get_time_in_seconds();

        double **C = multiply_blocked(A,B,n,64);

        double end = get_time_in_seconds();
        double cpu_after = get_cpu_time_seconds();
        double wt = end - start;
        double ct = cpu_after - cpu_before;

        long mem = get_memory_usage_kb();
        if(mem > peakMem) peakMem = mem;

        free_matrix(C,n);
        avg_wall += wt; avg_cpu += ct;
        printf(" [Blocked] Run %d: %.6f s (wall) | %.6f s (CPU) | PeakMem %ld KB\n",
               r+1, wt, ct, peakMem);
    }
    avg_wall/=runs; avg_cpu/=runs;
    fprintf(output,"Blocked,%d,%.6f,%.6f,%ld\n",n,avg_wall,avg_cpu,peakMem);

    free_matrix(A,n);
    free_matrix(B,n);
}

// 🔹 Sparse sintéticas
CSRMatrix *generate_sparse_matrix(int n, double sparsity) {
    double density = 1.0 - sparsity;
    int nnz_est = (int)(density * n * n);
    CSRMatrix *A = create_csr(n, n, nnz_est);

    int nnz = 0;
    A->row_ptr[0] = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (((double)rand()/RAND_MAX) > sparsity) {
                A->col_ind[nnz] = j;
                A->val[nnz] = (double)rand()/RAND_MAX;
                nnz++;
            }
        }
        A->row_ptr[i+1] = nnz;
    }
    A->nnz = nnz;
    return A;
}

void benchmark_sparse_synthetic(FILE *output, int runs) {
    double sparsity_levels[] = {0.1, 0.5, 0.9};
    int n = 500;
    for (int s=0; s<3; s++) {
        double sparsity = sparsity_levels[s];
        CSRMatrix *A = generate_sparse_matrix(n, sparsity);
        printf("\nSynthetic Sparse Matrix %dx%d (%.0f%% zeros)\n", n, n, sparsity*100);

        double *x = (double*)malloc(A->cols*sizeof(double));
        double *y = (double*)malloc(A->rows*sizeof(double));
        for (int j=0;j<A->cols;j++) x[j] = (double)rand()/RAND_MAX;

        double wall_times[runs], cpu_times[runs];
        long peakMem = 0;

        for(int r=0;r<runs;r++){
            double cpu_before = get_cpu_time_seconds();
            double start = get_time_in_seconds();

            spmv_csr(A, x, y);

            double end = get_time_in_seconds();
            double cpu_after = get_cpu_time_seconds();
            wall_times[r] = end - start;
            cpu_times[r] = cpu_after - cpu_before;

            long mem = get_memory_usage_kb();
            if(mem > peakMem) peakMem = mem;

            printf(" [SparseSynthetic %.0f%%] Run %d: %.6f s (wall) | %.6f s (CPU) | PeakMem %ld KB\n",
                   sparsity*100, r+1, wall_times[r], cpu_times[r], peakMem);
        }

        double avg_wall=0, avg_cpu=0;
        for(int r=0;r<runs;r++){ avg_wall+=wall_times[r]; avg_cpu+=cpu_times[r]; }
        avg_wall/=runs; avg_cpu/=runs;
        fprintf(output,"SparseSynthetic_%.0f%%,%d,%.6f,%.6f,%ld\n",
                sparsity*100, A->rows, avg_wall, avg_cpu, peakMem);

        free(x); free(y);
        free_csr(A);
    }
}

void benchmark_sparse_mtx(FILE *output, const char *path, int runs) {
    CSRMatrix *A = load_csr_from_mtx(path);
    if (!A) return;
    printf("\nSparse Matrix %s (%dx%d, nnz=%d)\n", path, A->rows, A->cols, A->nnz);

    double *x = (double*)malloc(A->cols*sizeof(double));
    double *y = (double*)malloc(A->rows*sizeof(double));
    for (int j=0;j<A->cols;j++) x[j] = (double)rand()/RAND_MAX;

    double wall_times[runs], cpu_times[runs];
    long peakMem = 0;

    for(int r=0;r<runs;r++){
        double cpu_before = get_cpu_time_seconds();
        double start = get_time_in_seconds();

        spmv_csr(A, x, y);

        double end = get_time_in_seconds();
        double cpu_after = get_cpu_time_seconds();
        wall_times[r] = end - start;
        cpu_times[r] = cpu_after - cpu_before;

        long mem = get_memory_usage_kb();
        if(mem > peakMem) peakMem = mem;

        printf(" [Sparse MTX SpMV] Run %d: %.6f s (wall) | %.6f s (CPU) | PeakMem %ld KB\n",
               r+1, wall_times[r], cpu_times[r], peakMem);
    }

    double avg_wall=0, avg_cpu=0;
    for(int r=0;r<runs;r++){ avg_wall+=wall_times[r]; avg_cpu+=cpu_times[r]; }
        avg_wall/=runs;
        avg_cpu/=runs;
        fprintf(output,"Sparse_MTX,%d,%.6f,%.6f,%ld\n",
                A->rows, avg_wall, avg_cpu, peakMem);

        free(x);
        free(y);
        free_csr(A);
    }

    int main(int argc, char **argv) {
        srand((unsigned int)time(NULL));
        int sizes[] = {50, 100, 500, 1024};
        int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        int runs = 3;

        if (argc >= 2) runs = atoi(argv[1]);

    #ifdef _WIN32
        system("mkdir ..\\..\\data >nul 2>&1");
    #else
        system("mkdir -p ../../data");
    #endif

        FILE *output = fopen("../../data/benchmark_c_results.csv", "w");
        if (!output) { perror("Couldn't open file "); return 1; }
        fprintf(output, "Approach,MatrixSize,AverageWall,AverageCPU,PeakMemoryKB\n");

        for(int s=0;s<num_sizes;s++){
            int n = sizes[s];
            printf("\nDense Matrix %dx%d\n", n, n);
            benchmark_dense(output, n, runs);
        }

        benchmark_sparse_synthetic(output, runs);

        benchmark_sparse_mtx(output, "../../mc2depi.mtx", runs);

        fclose(output);
        printf("\nResults written to ../../data/benchmark_c_results.csv\n");
        return 0;
    }
