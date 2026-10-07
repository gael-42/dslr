import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def createScatterPlot(dataset):
    grades = dataset.select_dtypes(include=np.number)
    house_colors = {
    'Hufflepuff': 'yellow',
    'Gryffindor': 'red',
    'Ravenclaw': 'blue',
    'Slytherin': 'green'
    }

    corr_matrix = grades.corr(method='pearson')  #Dataframe of correlation between every course
    pairs = corr_matrix.abs().unstack()  # get absolute value and collapse grid into a MultiIndex ex Astronomy Divination 0.9 Herbology 0.7
    pairs = pairs[pairs < 1.0].sort_values(ascending=False) # filter out self-correlation and sort values
    top = pairs.index[0]  # get highest pair at the top of the series
    highest = corr_matrix.loc[top] # get value of the correlation on highest pair

    fig, ax = plt.subplots()
    for house, color in house_colors.items():
        data = dataset.loc[dataset['Hogwarts House'] == house]
        plt.scatter(data[top[0]], data[top[1]], color=color, alpha=0.7, label=house)
    ax.legend()
    ax.set(xlabel=top[0], ylabel=top[1])
    ax.set_title(f"Pearson correlation coefficient: {highest}")
    plt.show()

def main():
    dataset = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    createScatterPlot(dataset)


if __name__ == "__main__":
    main()