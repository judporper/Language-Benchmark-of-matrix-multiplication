package es.ulpgc.task4;

import com.hazelcast.map.IMap;

public class MatrixUtils {

    public static final int BLOCK_SIZE = 256;

    public static MatrixBlock extractBlock(double[][] M, int bi, int bj) {
        double[][] block = new double[BLOCK_SIZE][BLOCK_SIZE];

        int rowStart = bi * BLOCK_SIZE;
        int colStart = bj * BLOCK_SIZE;

        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                block[i][j] = M[rowStart + i][colStart + j];
            }
        }
        return new MatrixBlock(bi, bj, block);
    }

    public static void multiplyBlocks(double[][] A, double[][] B, double[][] C) {
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                double sum = 0;
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] += sum;
            }
        }
    }

    public static double[][] combineBlocks(IMap<String, MatrixBlock> blocks, int N) {
        double[][] C = new double[N][N];
        int numBlocks = N / BLOCK_SIZE;

        for (int bi = 0; bi < numBlocks; bi++) {
            for (int bj = 0; bj < numBlocks; bj++) {
                MatrixBlock block = blocks.get(bi + "," + bj);
                if (block == null) continue;

                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[bi * BLOCK_SIZE + i][bj * BLOCK_SIZE + j] = block.data[i][j];
                    }
                }
            }
        }
        return C;
    }
}
