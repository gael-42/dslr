"""
Take dataset as parameter, display info for all numerical features.
Info to display: count, mean, std, min, 25%, 50%, 75%, Max.
Each feature gets a column, each info gets a row.
Cannot use function that calculates info by itself.
"""
import sys
import pandas as pd
import numpy as np

def describe(dataset: pd.DataFrame):
    num = dataset.select_dtypes(include= np.number)
    #print(num.head())
    for column in num:
        columnSeries = num[column]
        print('name = ', column)
        print('content = ', columnSeries.values)

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