"""
Take dataset as parameter, display info for all numerical features.
Info to display: count, mean, std, min, 25%, 50%, 75%, Max.
Each feature gets a column, each info gets a row.
Cannot use function that calculates info by itself.
"""
import sys
import pandas as pd
import numpy as np
import math

def calculate_max(series):
    pass

def calculate_75(series):
    pass

def calculate_50(series):
    pass

def calculate_25(series):
    pass

def calculate_min(series):
    pass

def calculate_std(series):
    clean = series.dropna()
    count = len(clean)
    mean = sum(clean) / count
    total_squared = 0
    for value in clean:
        diff_squared = (value - mean) ** 2
        total_squared += diff_squared
    div = total_squared / (count - 1)
    return math.sqrt(div)


def calculate_means(series):
    clean = series.dropna()
    return sum(clean) / len(clean)

def calculate_count(series):
    count = 0
    for value in series:
        if not pd.isna(value):
            count +=1
    return count

def describe(dataset: pd.DataFrame):
    num = dataset.select_dtypes(include= np.number)
    counts = num.apply(calculate_count)
    print('COUNTS')
    print(counts)
    means = num.apply(calculate_means)
    print('MEANS')
    print(means)
    std = num.apply(calculate_std)
    print('STD')
    print(std)

def main():
    try:
        assert len(sys.argv) == 2, "wrong argument count"
        assert sys.argv[1].endswith('.csv'), "wrong file type"
        dataset = pd.read_csv(sys.argv[1], index_col="Index")
        describe(dataset)
    except AssertionError as error:
        print("AssertionError:", error)

if __name__ == "__main__":
    main()