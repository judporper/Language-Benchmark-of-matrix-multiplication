package es.ulpgc.task4;

import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.map.IMap;

import java.io.Serializable;
import java.util.concurrent.Callable;

public class BlockMultiplyTask implements Callable<Void>, Serializable, HazelcastInstanceAware {

    private transient HazelcastInstance hz;

    private final int bi, bj, numBlocks, blockSize;

    public BlockMultiplyTask(int bi, int bj, int numBlocks, int blockSize) {
        this.bi = bi;
        this.bj = bj;
        this.numBlocks = numBlocks;
        this.blockSize = blockSize;
    }

    @Override
    public void setHazelcastInstance(HazelcastInstance hazelcastInstance) {
        this.hz = hazelcastInstance;
    }

    @Override
    public Void call() {
        IMap<String, MatrixBlock> A = hz.getMap("A");
        IMap<String, MatrixBlock> B = hz.getMap("B");
        IMap<String, MatrixBlock> C = hz.getMap("C");

        double[][] result = new double[blockSize][blockSize];

        for (int k = 0; k < numBlocks; k++) {
            MatrixBlock a = A.get(bi + "," + k);
            MatrixBlock b = B.get(k + "," + bj);
            MatrixUtils.multiplyBlocks(a.data, b.data, result);
        }

        C.put(bi + "," + bj, new MatrixBlock(bi, bj, result));
        return null;
    }
}
