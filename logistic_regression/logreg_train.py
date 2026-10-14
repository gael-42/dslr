import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

def logisticRegression(data):
    # separate classes (houses) from rest of data
    y = data['Hogwarts House']
    x = data.drop(["Hogwarts House"], axis=1)
    # split data into train and test, keep 25% of data to test model after training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

def main():
    data = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    logisticRegression(data)

if __name__ == "__main__":
    main()