package es.ulpgc.task4;

public class LocalBaseline {

    public static double[][] multiply(double[][] A, double[][] B, int N) {
        double[][] C = new double[N][N];

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }
}
