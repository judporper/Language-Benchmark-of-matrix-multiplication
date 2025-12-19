package es.ulpgc.task4;

import java.util.*;
import java.util.stream.Collectors;

public class FrequentItemsetMapReduce {

    public static Map<String, Integer> count(List<List<String>> transactions) {

        List<Map<String, Integer>> mapped = transactions.parallelStream()
                .map(t -> {
                    Map<String, Integer> m = new HashMap<>();
                    for (String item : t)
                        m.put(item, m.getOrDefault(item, 0) + 1);
                    return m;
                }).toList();

        Map<String, Integer> result = new HashMap<>();
        for (Map<String, Integer> m : mapped) {
            for (var e : m.entrySet()) {
                result.put(e.getKey(),
                        result.getOrDefault(e.getKey(), 0) + e.getValue());
            }
        }
        return result;
    }

    public static Map<String, Integer> frequentItems(
            List<List<String>> transactions,
            int minSupport) {

        Map<String, Integer> counts = count(transactions);
        return counts.entrySet().stream()
                .filter(e -> e.getValue() >= minSupport)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    public static void main(String[] args) {
        List<List<String>> transactions = List.of(
                List.of("milk", "bread"),
                List.of("milk", "butter"),
                List.of("bread", "butter"),
                List.of("milk", "bread", "butter")
        );

        Map<String, Integer> freq = frequentItems(transactions, 2);
        System.out.println(freq);
    }
}
