package Java;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class MatrixBenchmark {

    static class CSRMatrix {
        int rows, cols;
        int[] rowPtr;
        int[] colInd;
        double[] values;
        CSRMatrix(int rows, int cols, int[] rowPtr, int[] colInd, double[] values) {
            this.rows = rows; this.cols = cols;
            this.rowPtr = rowPtr; this.colInd = colInd; this.values = values;
        }
        void multiply(double[] x, double[] y) {
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) sum += values[k] * x[colInd[k]];
                y[i] = sum;
            }
        }
    }

    static CSRMatrix loadSparseFromMtx(String path) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        do { line = br.readLine(); } while (line != null && line.startsWith("%"));
        if (line == null) throw new IOException("Archivo vacío o inválido");
        String[] parts = line.trim().split("\\s+");
        int rows = Integer.parseInt(parts[0]);
        int cols = Integer.parseInt(parts[1]);
        int nnz = Integer.parseInt(parts[2]);

        int[] rowCounts = new int[rows];
        int[] rowInd = new int[nnz];
        int[] colInd = new int[nnz];
        double[] values = new double[nnz];

        for (int i=0;i<nnz;i++) {
            line = br.readLine();
            parts = line.trim().split("\\s+");
            int r = Integer.parseInt(parts[0])-1;
            int c = Integer.parseInt(parts[1])-1;
            double v = Double.parseDouble(parts[2]);
            rowInd[i]=r; colInd[i]=c; values[i]=v; rowCounts[r]++;
        }
        br.close();

        int[] rowPtr = new int[rows+1]; rowPtr[0]=0;
        for (int i=0;i<rows;i++) rowPtr[i+1]=rowPtr[i]+rowCounts[i];

        int[] csrColInd = new int[nnz]; double[] csrVal = new double[nnz];
        int[] fill = Arrays.copyOf(rowPtr, rows);
        for (int i=0;i<nnz;i++) {
            int r=rowInd[i]; int dst=fill[r]++;
            csrColInd[dst]=colInd[i]; csrVal[dst]=values[i];
        }
        return new CSRMatrix(rows, cols, rowPtr, csrColInd, csrVal);
    }

    public static void main(String[] args) throws Exception {
        int[] defaultSizes = {128, 256, 512, 1024};
        int runs = 5;
        int warmup = 1;
        int[] threadSweep = {1, 2, 4, 8}; // ajusta según tus cores
        int blockSize = 64;

        if (args.length >= 1) defaultSizes = new int[]{Integer.parseInt(args[0])};
        if (args.length >= 2) runs = Integer.parseInt(args[1]);
        if (args.length >= 3) warmup = Integer.parseInt(args[2]);

        File dataDir = new File("data"); if (!dataDir.exists()) dataDir.mkdirs();
        File outputFile = new File(dataDir, "java_results.csv");

        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeSupported = bean.isCurrentThreadCpuTimeSupported();
        if(cpuTimeSupported) bean.setThreadCpuTimeEnabled(true);

        try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {
            pw.print("Approach,MatrixSize,AverageWall,AverageCPU,PeakMemoryKB,Threads,Speedup_vs_Basic,Efficiency_per_thread,Extra\n");

            for (int n : defaultSizes) {
                System.out.println("\nDense Matrix " + n + "x" + n);
                double[][] A = MatrixMultiplier.generateMatrix(n);
                double[][] B = MatrixMultiplier.generateMatrix(n);

                Result base = benchmark("Basic", runs, warmup, () -> MatrixMultiplier.multiplyBasic(A,B), bean, cpuTimeSupported);
                writeRow(pw, "Basic", n, base, 1, 1.0, 1.0, "");

                if ((n & (n-1)) == 0) {
                    Result s = benchmark("Strassen", runs, warmup, () -> MatrixMultiplier.strassen(A,B), bean, cpuTimeSupported);
                    double sp = base.avgWall / s.avgWall;
                    writeRow(pw, "Strassen", n, s, null, sp, null, "");
                }

                Result blk = benchmark("Blocked", runs, warmup, () -> MatrixMultiplier.multiplyBlocked(A,B,blockSize), bean, cpuTimeSupported);
                writeRow(pw, "Blocked", n, blk, null, base.avgWall/blk.avgWall, null, "block_size="+blockSize);

                for (int t : threadSweep) {
                    int threads = t;
                    Result pr = benchmark("ParallelRows_"+threads+"t", runs, warmup,
                            () -> {
                                try { MatrixMultiplier.multiplyParallelRows(A,B,threads); }
                                catch (InterruptedException e) { throw new RuntimeException(e); }
                            }, bean, cpuTimeSupported);
                    double sp = base.avgWall / pr.avgWall;
                    double eff = sp / threads;
                    writeRow(pw, "ParallelRows", n, pr, threads, sp, eff, "");
                }

                for (int t : threadSweep) {
                    int threads = t;
                    Result pblk = benchmark("BlockedParallel_"+threads+"t", runs, warmup,
                            () -> {
                                try { MatrixMultiplier.multiplyBlockedParallel(A,B,blockSize,threads); }
                                catch (InterruptedException e) { throw new RuntimeException(e); }
                            }, bean, cpuTimeSupported);
                    double sp = base.avgWall / pblk.avgWall;
                    double eff = sp / threads;
                    writeRow(pw, "BlockedParallel", n, pblk, threads, sp, eff, "block_size="+blockSize);
                }

                Result vec = benchmark("Vectorized", runs, warmup, () -> MatrixMultiplier.multiplyVectorized(A,B), bean, cpuTimeSupported);
                writeRow(pw, "Vectorized", n, vec, null, base.avgWall/vec.avgWall, null, "transpose+tight-loops");
            }

            System.out.println("\nSparse Matrix mc2depi.mtx");
            try {
                CSRMatrix A_sparse = loadSparseFromMtx("mc2depi.mtx");
                Result sm = benchmark("Sparse_mc2depi", runs, warmup, () -> {
                    double[] x = new double[A_sparse.cols];
                    double[] y = new double[A_sparse.rows];
                    for (int i=0;i<x.length;i++) x[i]=Math.random();
                    A_sparse.multiply(x,y);
                }, bean, cpuTimeSupported);
                writeRow(pw, "Sparse_mc2depi", A_sparse.rows, sm, null, null, null, "");
            } catch (IOException e) {
                System.out.println("Working dir: " + System.getProperty("user.dir"));
                System.err.println("No se pudo cargar mc2depi.mtx: " + e.getMessage());
            }

            System.out.println("\nSynthetic Sparse Matrices (varying sparsity)");
            double[] sparsityLevels = {0.1, 0.5, 0.9};
            for (double sparsity : sparsityLevels) {
                MatrixMultiplier.CSRMatrix A_sparse = MatrixMultiplier.generateSparseMatrix(500, 500, sparsity);
                Result sm = benchmark("SparseSynthetic_"+(int)(sparsity*100)+"pctZeros", runs, warmup, () -> {
                    double[] x = new double[A_sparse.cols];
                    double[] y = new double[A_sparse.rows];
                    for (int i=0;i<x.length;i++) x[i]=Math.random();
                    A_sparse.multiplyParallel(x, y, Runtime.getRuntime().availableProcessors());
                }, bean, cpuTimeSupported);
                writeRow(pw, "SparseSynthetic_"+(int)(sparsity*100)+"pctZeros", A_sparse.rows, sm, null, null, null, "");
            }
        }

        System.out.println("\nResultados guardados en: " + outputFile.getAbsolutePath());
    }

    static class Result {
        double avgWall, avgCPU;
        long peakMemKB;
        Result(double avgWall, double avgCPU, long peakMemKB) {
            this.avgWall = avgWall; this.avgCPU = avgCPU; this.peakMemKB = peakMemKB;
        }
    }

    private static Result benchmark(String name, int runs, int warmup, Runnable func,
                                    ThreadMXBean bean, boolean cpuTimeSupported) {
        double[] times = new double[runs];
        double[] cpuTimes = new double[runs];
        long peakMem = 0;

        for (int w=0; w<warmup; w++) func.run();

        for (int r=0; r<runs; r++) {
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long memBefore = runtime.totalMemory() - runtime.freeMemory();
            long cpuBefore = cpuTimeSupported ? bean.getCurrentThreadCpuTime() : 0;

            long start = System.nanoTime();
            func.run();
            long end = System.nanoTime();

            long cpuAfter = cpuTimeSupported ? bean.getCurrentThreadCpuTime() : 0;
            long memAfter = runtime.totalMemory() - runtime.freeMemory();
            peakMem = Math.max(peakMem, memAfter);

            times[r] = (end - start)/1e9;
            cpuTimes[r] = cpuTimeSupported ? (cpuAfter - cpuBefore)/1e9 : 0.0;

            long deltaMem = Math.max(0, memAfter - memBefore);
            System.out.printf(Locale.US,
                    " [%s] Run %d: %.6f s (wall) | %.6f s (CPU) | Mem %d KB%n",
                    name, r+1, times[r], cpuTimes[r], deltaMem/1024);
        }

        double avgTime = Arrays.stream(times).average().orElse(0.0);
        double avgCPU = Arrays.stream(cpuTimes).average().orElse(0.0);
        long peakMemKB = peakMem/1024;

        System.out.printf(Locale.US,
                " Average [%s]: %.6f s (wall), %.6f s (CPU), Peak memory: %d KB%n",
                name, avgTime, avgCPU, peakMemKB);

        return new Result(avgTime, avgCPU, peakMemKB);
    }

    private static void writeRow(PrintWriter pw, String approach, int n, Result r,
                                 Integer threads, Double speedup, Double efficiency, String extra) {
        pw.printf(Locale.US, "%s,%d,%.6f,%.6f,%d,%s,%s,%s,%s%n",
                approach,
                n,
                r.avgWall,
                r.avgCPU,
                r.peakMemKB,
                threads == null ? "" : threads.toString(),
                speedup == null ? "" : String.format(Locale.US, "%.3f", speedup),
                efficiency == null ? "" : String.format(Locale.US, "%.3f", efficiency),
                (extra == null ? "" : extra));
    }
}
