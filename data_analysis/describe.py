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
    q = 0.25  # = 25 percentile
    idx = q * (n - 1)
    k = int(np.floor(idx))  # round down to nearest integer
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
            count += 1
    return count


def describe(dataset: pd.DataFrame):
    num = dataset.select_dtypes(include=np.number)
    counts = num.apply(calculate_count)
    means = num.apply(calculate_means)
    stds = num.apply(calculate_std)
    mins = num.apply(calculate_min)
    percent_25 = num.apply(calculate_25)
    percent_50 = num.apply(calculate_50)
    percent_75 = num.apply(calculate_75)
    maxs = num.apply(calculate_max)
    description = pd.DataFrame({
        'Count': counts,
        'Mean': means,
        'Std': stds,
        'Min': mins,
        '25%': percent_25,
        '50%': percent_50,
        '75%': percent_75,
        'Max': maxs
    }).T
    print(description)
    # print(num.describe())


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
