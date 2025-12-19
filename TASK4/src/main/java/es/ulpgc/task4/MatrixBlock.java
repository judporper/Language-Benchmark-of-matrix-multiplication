package es.ulpgc.task4;

import java.io.Serializable;

public class MatrixBlock implements Serializable {

    public final int rowBlock;
    public final int colBlock;
    public final double[][] data;

    public MatrixBlock(int rowBlock, int colBlock, double[][] data) {
        this.rowBlock = rowBlock;
        this.colBlock = colBlock;
        this.data = data;
    }
}
