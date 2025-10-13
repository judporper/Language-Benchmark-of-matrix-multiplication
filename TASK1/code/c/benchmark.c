#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <windows.h>
#include <psapi.h>
#include "matrix_multiplier.h"

double get_time_in_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

long get_memory_usage_kb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (long)(pmc.PeakWorkingSetSize / 1024);
    }
    return -1;
}

double get_cpu_time_seconds() {
    FILETIME createTime, exitTime, kernelTime, userTime;
    if (GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime)) {
        ULARGE_INTEGER kt, ut;
        kt.LowPart = kernelTime.dwLowDateTime; kt.HighPart = kernelTime.dwHighDateTime;
        ut.LowPart = userTime.dwLowDateTime;   ut.HighPart = userTime.dwHighDateTime;
        return (kt.QuadPart + ut.QuadPart) / 10000000.0;
    }
    return -1.0;
}

int main() {
    srand(time(NULL));
    int sizes[] = {50, 100, 500, 1024, 2000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int runs = 5;

    system("mkdir -p ../../data");
    FILE *output = fopen("../../data/benchmark_c_results.csv", "w");
    if (!output) { perror("Couldn't open file "); return 1; }

    fprintf(output, "MatrixSize");
    for(int r=1;r<=runs;r++) fprintf(output,",Run%d_Time",r);
    fprintf(output,",AverageTime");
    for(int r=1;r<=runs;r++) fprintf(output,",Run%d_CPU",r);
    fprintf(output,",AverageCPU,PeakMemoryKB\n");

    for(int s=0;s<num_sizes;s++){
        int n = sizes[s];
        double wall_times[runs], cpu_times[runs];
        long peakMem = 0;

        double **A = generate_matrix(n);
        double **B = generate_matrix(n);

        for(int r=0;r<runs;r++){
            double cpu_before = get_cpu_time_seconds();
            double start = get_time_in_seconds();

            double **C = multiply_matrices(A,B,n);

            double end = get_time_in_seconds();
            double cpu_after = get_cpu_time_seconds();
            wall_times[r] = end - start;
            cpu_times[r] = cpu_after - cpu_before;

            long mem = get_memory_usage_kb();
            if(mem > peakMem) peakMem = mem;

            free_matrix(C,n);
        }

        double avg_wall=0, avg_cpu=0;
        for(int r=0;r<runs;r++){ avg_wall+=wall_times[r]; avg_cpu+=cpu_times[r]; }
        avg_wall/=runs; avg_cpu/=runs;

        fprintf(output,"%d",n);
        for(int r=0;r<runs;r++) fprintf(output,",%.6f",wall_times[r]);
        fprintf(output,",%.6f",avg_wall);
        for(int r=0;r<runs;r++) fprintf(output,",%.6f",cpu_times[r]);
        fprintf(output,",%.6f,%ld\n",avg_cpu,peakMem);

        printf("%dx%d Average: %.6f s (wall), %.6f s (CPU), Peak memory: %ld KB\n",
               n,n,avg_wall,avg_cpu,peakMem);
        free_matrix(A,n);
        free_matrix(B,n);
    }

    fclose(output);
    printf("\nResults in../../data/benchmark_c_results.csv\n");
    return 0;
}
