import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def createPairPlot(dataset):
    grades = dataset.select_dtypes(include=np.number)
    house_colors = {
        'Hufflepuff': 'yellow',
        'Gryffindor': 'red',
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
    }
    plot = sns.pairplot(dataset, vars=grades.columns, hue='Hogwarts House', palette=house_colors, plot_kws={'alpha': 0.6, 's': 20})
    if plot._legend:
        plot._legend.remove()
    for ax in plot.axes.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        ax.label_outer()
    for ax in plot.axes[:, 0]:
        if ax is not None:
            ax.set_ylabel(
                ax.get_ylabel(),
                rotation=0,
                labelpad=5,
                ha='right',
                va='center',
                fontsize=8,
            )
    for ax in plot.axes[-1, :]:
        if ax is not None:
            ax.set_xlabel(
                ax.get_xlabel(),
                fontsize=8,
            )
    plt.subplots_adjust(left=0.10, bottom=0.05)
    plt.show()


def main():
    dataset = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    createPairPlot(dataset)

if __name__ == "__main__":
    main()