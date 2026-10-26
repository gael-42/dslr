import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def createHistogram(dataset):
    grades = dataset.select_dtypes(include=np.number)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25, 10))
    axes = axes.flatten()

    house_colors = {
    'Hufflepuff': 'yellow',
    'Gryffindor': 'red',
    'Ravenclaw': 'blue',
    'Slytherin': 'green'
    }

    for i, column in enumerate(grades.columns):
        for house, color in house_colors.items():
            data = dataset.loc[dataset['Hogwarts House'] == house, column]
            data.hist(ax=axes[i], color=color, alpha=0.7, grid=False, label=house)
        axes[i].set_title(column)
    
    for ax in axes[len(grades.columns):]:
        ax.set_visible(False)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir.parent / 'dataset_train.csv'
    dataset = pd.read_csv(dataset_path, index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    createHistogram(dataset)


if __name__ == "__main__":
    main()