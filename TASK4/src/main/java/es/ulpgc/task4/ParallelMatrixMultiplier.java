package es.ulpgc.task4;

import java.util.concurrent.*;

public class ParallelMatrixMultiplier {

    public static double[][] multiply(double[][] A, double[][] B, int N) throws Exception {
        double[][] C = new double[N][N];
        int threads = Runtime.getRuntime().availableProcessors();

        ExecutorService exec = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < N; i++) {
            final int row = i;
            exec.submit(() -> {
                for (int j = 0; j < N; j++) {
                    double sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += A[row][k] * B[k][j];
                    }
                    C[row][j] = sum;
                }
            });
        }

        exec.shutdown();
        exec.awaitTermination(1, TimeUnit.HOURS);
        return C;
    }
}
