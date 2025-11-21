package Java;
import java.util.Random;

public class MatrixMultiplier {
	private static final Random random = new Random();

	public static double[][] generateMatrix(int n) {
		double[][] m = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<n; j++)
				m[i][j] = random.nextDouble();
		return m;
	}

	public static CSRMatrix generateSparseMatrix(int rows, int cols, double sparsity) {
		int nnzEstimate = (int)((1.0 - sparsity) * rows * cols);
		int[] rowPtr = new int[rows+1];
		int[] colInd = new int[nnzEstimate];
		double[] values = new double[nnzEstimate];

		int nnz = 0;
		for (int i=0; i<rows; i++) {
			rowPtr[i] = nnz;
			for (int j=0; j<cols; j++) {
				if (random.nextDouble() > sparsity) {
					if (nnz >= values.length) {
						int newSize = values.length * 2;
						colInd = java.util.Arrays.copyOf(colInd, newSize);
						values = java.util.Arrays.copyOf(values, newSize);
					}
					colInd[nnz] = j;
					values[nnz] = random.nextDouble();
					nnz++;
				}
			}
		}
		rowPtr[rows] = nnz;
		colInd = java.util.Arrays.copyOf(colInd, nnz);
		values = java.util.Arrays.copyOf(values, nnz);

		return new CSRMatrix(rows, cols, rowPtr, colInd, values);
	}

	public static double[][] multiplyBasic(double[][] a, double[][] b) {
		int n = a.length;
		double[][] c = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<n; j++)
				for (int k=0; k<n; k++)
					c[i][j] += a[i][k] * b[k][j];
		return c;
	}

	public static double[][] strassen(double[][] A, double[][] B) {
		int n = A.length;
		if (n == 1) return new double[][]{{A[0][0] * B[0][0]}};
		int mid = n/2;
		double[][] A11 = subMatrix(A,0,mid,0,mid);
		double[][] A12 = subMatrix(A,0,mid,mid,n);
		double[][] A21 = subMatrix(A,mid,n,0,mid);
		double[][] A22 = subMatrix(A,mid,n,mid,n);

		double[][] B11 = subMatrix(B,0,mid,0,mid);
		double[][] B12 = subMatrix(B,0,mid,mid,n);
		double[][] B21 = subMatrix(B,mid,n,0,mid);
		double[][] B22 = subMatrix(B,mid,n,mid,n);

		double[][] M1 = strassen(add(A11,A22), add(B11,B22));
		double[][] M2 = strassen(add(A21,A22), B11);
		double[][] M3 = strassen(A11, subtract(B12,B22));
		double[][] M4 = strassen(A22, subtract(B21,B11));
		double[][] M5 = strassen(add(A11,A12), B22);
		double[][] M6 = strassen(subtract(A21,A11), add(B11,B12));
		double[][] M7 = strassen(subtract(A12,A22), add(B21,B22));

		double[][] C11 = add(subtract(add(M1,M4),M5),M7);
		double[][] C12 = add(M3,M5);
		double[][] C21 = add(M2,M4);
		double[][] C22 = add(subtract(add(M1,M3),M2),M6);

		return join(C11,C12,C21,C22);
	}

	public static double[][] multiplyBlocked(double[][] A, double[][] B, int blockSize) {
		int n = A.length;
		double[][] C = new double[n][n];
		for (int ii=0; ii<n; ii+=blockSize)
			for (int jj=0; jj<n; jj+=blockSize)
				for (int kk=0; kk<n; kk+=blockSize)
					for (int i=ii; i<Math.min(ii+blockSize,n); i++)
						for (int j=jj; j<Math.min(jj+blockSize,n); j++) {
							double sum=0;
							for (int k=kk; k<Math.min(kk+blockSize,n); k++)
								sum += A[i][k]*B[k][j];
							C[i][j]+=sum;
						}
		return C;
	}

	private static double[][] add(double[][] A, double[][] B) {
		int n=A.length; double[][] C=new double[n][n];
		for(int i=0;i<n;i++) for(int j=0;j<n;j++) C[i][j]=A[i][j]+B[i][j];
		return C;
	}
	private static double[][] subtract(double[][] A, double[][] B) {
		int n=A.length; double[][] C=new double[n][n];
		for(int i=0;i<n;i++) for(int j=0;j<n;j++) C[i][j]=A[i][j]-B[i][j];
		return C;
	}
	private static double[][] subMatrix(double[][] A,int r1,int r2,int c1,int c2){
		double[][] M=new double[r2-r1][c2-c1];
		for(int i=r1;i<r2;i++) for(int j=c1;j<c2;j++) M[i-r1][j-c1]=A[i][j];
		return M;
	}
	private static double[][] join(double[][] C11,double[][] C12,double[][] C21,double[][] C22){
		int mid=C11.length; int n=mid*2; double[][] C=new double[n][n];
		for(int i=0;i<mid;i++) for(int j=0;j<mid;j++) {
			C[i][j]=C11[i][j]; C[i][j+mid]=C12[i][j];
			C[i+mid][j]=C21[i][j]; C[i+mid][j+mid]=C22[i][j];
		}
		return C;
	}

	public static class CSRMatrix {
		public int rows, cols;
		public int[] rowPtr;
		public int[] colInd;
		public double[] values;
		public CSRMatrix(int rows, int cols, int[] rowPtr, int[] colInd, double[] values) {
			this.rows = rows; this.cols = cols;
			this.rowPtr = rowPtr; this.colInd = colInd; this.values = values;
		}
		public void multiply(double[] x, double[] y) {
			for (int i=0;i<rows;i++) {
				double sum=0;
				for (int k=rowPtr[i]; k<rowPtr[i+1]; k++) {
					sum += values[k]*x[colInd[k]];
				}
				y[i]=sum;
			}
		}
	}
}
