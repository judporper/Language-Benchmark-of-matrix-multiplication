package es.ulpgc.task4;

public class BenchmarkRunner {

    public static void main(String[] args) throws Exception {

        int[] sizes = {256, 512, 1024};
        System.out.println("Available processors: " +
                Runtime.getRuntime().availableProcessors());

        for (int N : sizes) {

            System.out.println("\n===== MATRIX SIZE: " + N + " =====");

            double[][] A = MatrixGenerator.randomMatrix(N);
            double[][] B = MatrixGenerator.randomMatrix(N);

            long t1 = System.currentTimeMillis();
            LocalBaseline.multiply(A, B, N);
            long t2 = System.currentTimeMillis();
            System.out.println("Local naive: " + (t2 - t1) + " ms");

            t1 = System.currentTimeMillis();
            ParallelMatrixMultiplier.multiply(A, B, N);
            t2 = System.currentTimeMillis();
            System.out.println("Parallel local: " + (t2 - t1) + " ms");

            t1 = System.currentTimeMillis();
            double[][] C = DistributedMatrixMultiplier.multiply(A, B, N);
            t2 = System.currentTimeMillis();
            System.out.println("Distributed Hazelcast (end-to-end): " + (t2 - t1) + " ms");

            System.out.println("Result checksum: " + checksum(C));
        }
    }

    private static double checksum(double[][] M) {
        double s = 0.0;
        for (double[] row : M)
            for (double v : row)
                s += v;
        return s;
    }
}
