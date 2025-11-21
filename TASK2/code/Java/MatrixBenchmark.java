package Java;

import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.*;

public class MatrixBenchmark {
    static class CSRMatrix {
        int rows, cols;
        int[] rowPtr;
        int[] colInd;
        double[] values;

        CSRMatrix(int rows, int cols, int[] rowPtr, int[] colInd, double[] values) {
            this.rows = rows;
            this.cols = cols;
            this.rowPtr = rowPtr;
            this.colInd = colInd;
            this.values = values;
        }

        void multiply(double[] x, double[] y) {
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
                    sum += values[k] * x[colInd[k]];
                }
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
            rowInd[i]=r; colInd[i]=c; values[i]=v;
            rowCounts[r]++;
        }
        br.close();

        int[] rowPtr = new int[rows+1];
        rowPtr[0]=0;
        for (int i=0;i<rows;i++) rowPtr[i+1]=rowPtr[i]+rowCounts[i];

        int[] csrColInd = new int[nnz];
        double[] csrVal = new double[nnz];
        int[] fill = Arrays.copyOf(rowPtr, rows);
        for (int i=0;i<nnz;i++) {
            int r=rowInd[i];
            int dst=fill[r]++;
            csrColInd[dst]=colInd[i];
            csrVal[dst]=values[i];
        }

        return new CSRMatrix(rows, cols, rowPtr, csrColInd, csrVal);
    }

    public static void main(String[] args) throws Exception {
        int[] defaultSizes = {50, 100, 500, 1024};
        int runs = 3;
        int[] sizes = defaultSizes;

        File dataDir = new File("data");
        if (!dataDir.exists()) dataDir.mkdirs();
        File outputFile = new File(dataDir, "java_results.csv");

        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeSupported = bean.isCurrentThreadCpuTimeSupported();
        if(cpuTimeSupported) bean.setThreadCpuTimeEnabled(true);

        try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {
            pw.print("Approach,MatrixSize,AverageWall,AverageCPU,PeakMemoryKB\n");
            for (int n : sizes) {
                System.out.println("\nDense Matrix " + n + "x" + n + " ");
                double[][] A = MatrixMultiplier.generateMatrix(n);
                double[][] B = MatrixMultiplier.generateMatrix(n);

                benchmarkAndWrite(pw, "Basic", n, runs, () -> MatrixMultiplier.multiplyBasic(A,B), bean, cpuTimeSupported);
                if ((n & (n-1)) == 0)
                    benchmarkAndWrite(pw, "Strassen", n, runs, () -> MatrixMultiplier.strassen(A,B), bean, cpuTimeSupported);
                benchmarkAndWrite(pw, "Blocked", n, runs, () -> MatrixMultiplier.multiplyBlocked(A,B,64), bean, cpuTimeSupported);
            }

            System.out.println("\nSparse Matrix mc2depi.mtx");
            try {
                CSRMatrix A_sparse = loadSparseFromMtx("mc2depi.mtx");
                benchmarkAndWrite(pw, "Sparse_mc2depi", A_sparse.rows, runs, () -> {
                    double[] x = new double[A_sparse.cols];
                    double[] y = new double[A_sparse.rows];
                    for (int i=0;i<x.length;i++) x[i]=Math.random();
                    A_sparse.multiply(x,y);
                }, bean, cpuTimeSupported);
            } catch (IOException e) {
                System.out.println("Working dir: " + System.getProperty("user.dir"));
                System.err.println("No se pudo cargar mc2depi.mtx: " + e.getMessage());
            }

            System.out.println("\nSynthetic Sparse Matrices (varying sparsity)");
            double[] sparsityLevels = {0.1, 0.5, 0.9};
            for (double sparsity : sparsityLevels) {
                MatrixMultiplier.CSRMatrix A_sparse = MatrixMultiplier.generateSparseMatrix(500, 500, sparsity);
                benchmarkAndWrite(pw, "SparseSynthetic_"+(int)(sparsity*100)+"pctZeros", A_sparse.rows, runs, () -> {
                    double[] x = new double[A_sparse.cols];
                    double[] y = new double[A_sparse.rows];
                    for (int i=0;i<x.length;i++) x[i]=Math.random();
                    A_sparse.multiply(x,y);
                }, bean, cpuTimeSupported);
            }
        }
    }

    private static void benchmarkAndWrite(PrintWriter pw, String name, int n, int runs, Runnable func,
                                          ThreadMXBean bean, boolean cpuTimeSupported) {
        double[] times = new double[runs];
        double[] cpuTimes = new double[runs];
        long peakMem = 0;

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
            cpuTimes[r] = (cpuAfter - cpuBefore)/1e9;

            long deltaMem = Math.max(0, memAfter - memBefore);

            System.out.printf(Locale.US,
                    " [%s] Run %d: %.6f s (wall) | %.6f s (CPU) | Mem %d KB%n",
                    name, r+1, times[r], cpuTimes[r], deltaMem/1024);
        }

        double avgTime = 0, avgCPU = 0;
        for (int r=0; r<runs; r++) {
            avgTime += times[r];
            avgCPU += cpuTimes[r];
        }
        avgTime /= runs;
        avgCPU /= runs;

        pw.printf(Locale.US, "%s,%d,%.6f,%.6f,%d%n",
                name, n, avgTime, avgCPU, peakMem/1024);

        System.out.printf(Locale.US,
                " Average: %.6f s (wall), %.6f s (CPU), Peak memory: %d KB%n",
                avgTime, avgCPU, peakMem/1024);
    }
}
