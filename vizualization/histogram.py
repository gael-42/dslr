import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    hufflepuff = dataset[dataset['Hogwarts House'] == 'Hufflepuff']
"""

def createHistogram(dataset):
    # grades = dataset.select_dtypes(include=np.number)
    # huff = dataset.loc[dataset['Hogwarts House'] == 'Hufflepuff', 'Arithmancy']
    # gryff = dataset.loc[dataset['Hogwarts House'] == 'Gryffindor', 'Arithmancy']
    # colors = ['yellow', 'red']
    # plt.hist(huff, alpha=0.8, label='Hufflepuff', color='yellow')
    # plt.hist(gryff, alpha=0.8, label='Gryffindor', color='red')
    # plt.legend(loc='upper right')
    # plt.title('Arithmancy')
    grades = dataset.select_dtypes(include=np.number)
    ncols = 5 #  have 5 histograms per line
    nrows = 3 #  have 3 rows of histograms
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, tight_layout=True, figsize=(15, 10))
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            if counter < len(grades.columns):
                ax.hist(grades[grades.columns[counter]], label='{}'.format(grades.columns[counter]))
            else:
                ax.set_axis_off()
            counter+=1
    plt.show()

    plt.show()

def main():
    dataset = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    createHistogram(dataset)



if __name__ == "__main__":
    main()