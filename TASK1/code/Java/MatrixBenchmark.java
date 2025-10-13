package Java;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.Locale;

public class MatrixBenchmark {
    public static void main(String[] args) throws Exception {
        int[] defaultSizes = {50, 100, 500, 1024, 2000};
        int runs = 5;
        int[] sizes = defaultSizes;

        if (args.length >= 1) sizes = new int[]{Integer.parseInt(args[0])};
        if (args.length >= 2) runs = Integer.parseInt(args[1]);

        File dataDir = new File("data");
        if (!dataDir.exists()) dataDir.mkdirs();
        File outputFile = new File(dataDir, "java_results.csv");

        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        boolean cpuTimeSupported = bean.isCurrentThreadCpuTimeSupported();
        if(cpuTimeSupported) bean.setThreadCpuTimeEnabled(true);

        try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {
            pw.print("MatrixSize");
            for (int r=1; r<=runs; r++) pw.print(",Run"+r+"_Time");
            pw.print(",AverageTime");
            for (int r=1; r<=runs; r++) pw.print(",Run"+r+"_CPU");
            pw.print(",AverageCPU,PeakMemoryKB\n");

            for (int n : sizes) {
                System.out.println("\nMatrix " + n + "x" + n);
                double[][] A = MatrixMultiplier.generateMatrix(n);
                double[][] B = MatrixMultiplier.generateMatrix(n);

                double[] times = new double[runs];
                double[] cpuTimes = new double[runs];
                long peakMem = 0;

                for (int r=0; r<runs; r++) {
                    Runtime runtime = Runtime.getRuntime();
                    runtime.gc();
                    long memBefore = runtime.totalMemory() - runtime.freeMemory();
                    long cpuBefore = cpuTimeSupported ? bean.getCurrentThreadCpuTime() : 0;

                    long start = System.nanoTime();
                    double[][] C = MatrixMultiplier.multiply(A, B);
                    long end = System.nanoTime();

                    long cpuAfter = cpuTimeSupported ? bean.getCurrentThreadCpuTime() : 0;
                    long memAfter = runtime.totalMemory() - runtime.freeMemory();
                    if(memAfter > peakMem) peakMem = memAfter;

                    times[r] = (end - start)/1e9;
                    cpuTimes[r] = (cpuAfter - cpuBefore)/1e9;
                }

                double avgTime = 0; double avgCPU = 0;
                for (int r=0; r<runs; r++) { avgTime += times[r]; avgCPU += cpuTimes[r]; }
                avgTime /= runs; avgCPU /= runs;

                pw.printf(Locale.US, "%d", n);
                for(double t: times) pw.printf(Locale.US, ",%.6f", t);
                pw.printf(Locale.US, ",%.6f", avgTime);
                for(double t: cpuTimes) pw.printf(Locale.US, ",%.6f", t);
                pw.printf(Locale.US, ",%.6f,%d\n", avgCPU, peakMem/1024);

                System.out.printf(Locale.US, "Average: %.6f s (wall), %.6f s (CPU), Peak memory: %d KB%n",
                        avgTime, avgCPU, peakMem/1024);
            }
        }
    }
}
