import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
What are the two features that are similar?
"""

def createScatterPlot(dataset):
    grades = dataset.select_dtypes(include=np.number)
    house_colors = {
    'Hufflepuff': 'yellow',
    'Gryffindor': 'red',
    'Ravenclaw': 'blue',
    'Slytherin': 'green'
    }
    for house, color in house_colors.items():
        data = dataset.loc[dataset['Hogwarts House'] == house]
        plt.scatter(data['Defense Against the Dark Arts'], data['Astronomy'], color=color, label=house, alpha=0.5)
    plt.show()

def main():
    dataset = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    createScatterPlot(dataset)


if __name__ == "__main__":
    main()