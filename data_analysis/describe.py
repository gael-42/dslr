"""
Take dataset as parameter, display info for all numerical features.
Info to display: count, mean, std, min, 25%, 50%, 75%, Max.
Each feature gets a column, each info gets a row.
Cannot use function that calculates info by itself.
"""
import sys
import pandas as pd
import numpy as np

def calculate_count(series):
    count = 0
    for value in series:
        if not pd.isna(value):
            count +=1
    return count

def calculate_means(series):
    pass

def describe(dataset: pd.DataFrame):
    num = dataset.select_dtypes(include= np.number)
    counts = num.apply(calculate_count)
    print('COUNTS')
    print(counts)
    # means = num.apply(calculate_means)
    # print(means)

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