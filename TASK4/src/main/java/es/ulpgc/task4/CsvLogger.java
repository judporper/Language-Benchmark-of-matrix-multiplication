package es.ulpgc.task4;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class CsvLogger {

    public static synchronized void log(
            String filename,
            int size,
            long naiveMs,
            long parallelMs,
            long distributedTotalMs) {

        boolean appendHeader = new java.io.File(filename).length() == 0;

        try (PrintWriter pw = new PrintWriter(new FileWriter(filename, true))) {
            if (appendHeader) {
                pw.println("size,naive_ms,parallel_ms,distributed_ms");
            }
            pw.printf("%d,%d,%d,%d%n",
                    size, naiveMs, parallelMs, distributedTotalMs);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
