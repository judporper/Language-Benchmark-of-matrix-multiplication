package Java;
import java.util.Random;

public class MatrixMultiplier {
	public static double[][] generateMatrix(int n) {
		double[][] m = new double[n][n];
		Random random = new Random();
		for (int i=0; i<n; i++)
			for (int j=0; j<n; j++)
				m[i][j] = random.nextDouble();
		return m;
	}

	public static double[][] multiply(double[][] a, double[][] b) {
		int n = a.length;
		double[][] c = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<n; j++)
				for (int k=0; k<n; k++)
					c[i][j] += a[i][k] * b[k][j];
		return c;
	}
}
