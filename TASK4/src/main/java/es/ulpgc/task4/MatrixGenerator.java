package es.ulpgc.task4;

import java.util.Random;

public class MatrixGenerator {

    public static double[][] randomMatrix(int N) {
        double[][] M = new double[N][N];
        Random r = new Random();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                M[i][j] = r.nextDouble();
        return M;
    }
}
