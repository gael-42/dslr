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
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    max = 0
    for value in series:
        if value > max:
            max = value
    return max


def calculate_75(series):
    clean = series.dropna().sort_values().to_numpy()
    if len(clean) == 0:
        return np.nan
    clean.sort()
    n = len(clean)
    q = 0.75
    idx = q * (n - 1)
    k = int(np.floor(idx))
    f = idx - k
    if k >= n - 1:
        return float(clean[-1])
    percentile = clean[k] + f * (clean[k + 1] - clean[k])
    return percentile

def calculate_50(series):
    clean = series.dropna().sort_values().to_numpy()
    if len(clean) == 0:
        return np.nan
    clean.sort()
    n = len(clean)
    q = 0.50
    idx = q * (n - 1)
    k = int(np.floor(idx))
    f = idx - k
    if k >= n - 1:
        return float(clean[-1])
    percentile = clean[k] + f * (clean[k + 1] - clean[k])
    return percentile

def calculate_25(series):
    clean = series.dropna().sort_values().to_numpy()
    if len(clean) == 0:
        return np.nan
    clean.sort()
    n = len(clean)
    q = 0.25 # = 25 percentile
    idx = q * (n - 1)
    k = int(np.floor(idx)) #round down to nearest integer
    f = idx - k
    if k >= n - 1:
        return float(clean[-1])
    percentile = clean[k] + f * (clean[k + 1] - clean[k])
    return percentile

def calculate_min(series):
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    min = 0
    for value in clean:
        if value < min:
            min = value
    return min

def calculate_std(series):
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
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
    if len(clean) == 0:
        return np.nan
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
    print('25')
    percent_25 = num.apply(calculate_25)
    print(percent_25)
    print('50')
    percent_50 = num.apply(calculate_50)
    print(percent_50)
    print('75')
    percent_75 = num.apply(calculate_75)
    print(percent_75)

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