import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count


def mapper(transactions):
    counts = Counter()
    for t in transactions:
        for item in t:
            counts[item] += 1
    return counts


def reducer(counters):
    total = Counter()
    for c in counters:
        total.update(c)
    return total


def frequent_items(transactions, workers=None, min_support=1):
    if workers is None:
        workers = cpu_count()

    chunks = np.array_split(transactions, workers)
    with Pool(workers) as p:
        mapped = p.map(mapper, chunks)
    counts = reducer(mapped)
    return {item: cnt for item, cnt in counts.items() if cnt >= min_support}


if __name__ == "__main__":
    transactions = [
        ["milk", "bread"],
        ["milk", "butter"],
        ["bread", "butter"],
        ["milk", "bread", "butter"]
    ]
    print(frequent_items(transactions, min_support=2))
