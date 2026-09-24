import sys
import pandas as pd
import numpy as np
import math

"""
- apply function over each column
- separate data by house
- calculate median score per house
- calculate standard deviation between house median scores
- return
- compare standard deviation between each course
- select course with least deviation
- create histogram
"""

def getStd():
    pass


def getMedian(scores):
    clean = scores.dropna().sort_values()
    size = clean.count()
    if size % 2 != 0:
        return clean[size // 2]  # divide and discard remainder
    return (clean[(size - 1) // 2] + clean[size // 2] / 2.0)


def findEvenDistrib(dataset):
    """
    - select rows for each house
    - of those rows take only the scores
    - get median for scores
    """
    hufflepuff = dataset[dataset['Hogwarts House'] == 'Hufflepuff']
    ravenclaw = dataset[dataset['Hogwarts House'] == 'Ravenclaw']
    slytherin = dataset[dataset['Hogwarts House'] == 'Slytherin']
    gryffindor = dataset[dataset['Hogwarts House'] == 'Gryffindor']
    huffmedian = getMedian(hufflepuff.select_dtypes(include=np.number))
    print(huffmedian, hufflepuff.select_dtypes(include=np.number).median())



def main():
    dataset = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    findEvenDistrib(dataset)



if __name__ == "__main__":
    main()