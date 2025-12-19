package es.ulpgc.task4;

import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.map.IMap;

public class DistributedMatrixMultiplier {

    public static double[][] multiply(double[][] A, double[][] B, int N) {

        HazelcastInstance hz = HazelcastNode.startNode();

        IMap<String, MatrixBlock> blocksA = hz.getMap("A");
        IMap<String, MatrixBlock> blocksB = hz.getMap("B");
        IMap<String, MatrixBlock> blocksC = hz.getMap("C");

        // Limpieza entre tamaños
        blocksA.clear();
        blocksB.clear();
        blocksC.clear();

        int blockSize = MatrixUtils.BLOCK_SIZE;
        int numBlocks = N / blockSize;

        long transferStart = System.currentTimeMillis();
        long bytesTransferred = 0L;

        // Distribuir bloques A y B en el cluster
        for (int bi = 0; bi < numBlocks; bi++) {
            for (int bj = 0; bj < numBlocks; bj++) {
                blocksA.put(bi + "," + bj, MatrixUtils.extractBlock(A, bi, bj));
                blocksB.put(bi + "," + bj, MatrixUtils.extractBlock(B, bi, bj));
                bytesTransferred += 2L * blockSize * blockSize * 8;
            }
        }

        long transferEnd = System.currentTimeMillis();
        System.out.println("Data distribution time: " + (transferEnd - transferStart) + " ms");
        System.out.println("Approx data transferred: " + (bytesTransferred / 1_000_000.0) + " MB");

        // Cálculo "distribuido" pero ejecutado localmente leyendo de Hazelcast
        long computeStart = System.currentTimeMillis();

        for (int bi = 0; bi < numBlocks; bi++) {
            for (int bj = 0; bj < numBlocks; bj++) {

                double[][] result = new double[blockSize][blockSize];

                for (int bk = 0; bk < numBlocks; bk++) {
                    MatrixBlock aBlock = blocksA.get(bi + "," + bk);
                    MatrixBlock bBlock = blocksB.get(bk + "," + bj);
                    MatrixUtils.multiplyBlocks(aBlock.data, bBlock.data, result);
                }

                blocksC.put(bi + "," + bj, new MatrixBlock(bi, bj, result));
            }
        }

        long computeEnd = System.currentTimeMillis();
        System.out.println("Distributed compute time (local over Hazelcast maps): " + (computeEnd - computeStart) + " ms");

        // Reconstruir C desde blocksC
        long gatherStart = System.currentTimeMillis();
        System.out.println("blocksC size: " + blocksC.size());
        double[][] C = MatrixUtils.combineBlocks(blocksC, N);
        long gatherEnd = System.currentTimeMillis();
        System.out.println("Result gather time: " + (gatherEnd - gatherStart) + " ms");

        System.out.println("Cluster nodes: " + hz.getCluster().getMembers().size());
        System.out.println("Used memory (MB): " +
                (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1_000_000);

        return C;
    }
}
