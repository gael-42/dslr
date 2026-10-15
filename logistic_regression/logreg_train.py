import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
"""
Courses to exclude:
- hist:
Care of Magical Creatures
Arithmancy
- Scatter:
Defense against dark arts x astronomy
- pair:
care magical creatures x arithmancy
potions x arithmancy
potions x care magical creatures
Final list:
- Care of Magical Creatures
- Defense Against the Dark Arts
- Potions
- Arithmancy
"""

class LogisticRegression:
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def sigmoid(z):
        # sigmoid function used to get probability between 0 and 1
        y_head = 1 / (1+np.exp(-z))
        return y_head

    def cost(self, h, y):
        # calculate how well the model is performing
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def train(self, x, y):
        row, col = x.shape
        self.weights = np.zeros(col)

        for _ in range(self.iterations):
            # multiply data points by weights and add bias
            z = np.dot(x, self.weights) + self.bias
            # turn into 0 to 1 odd probability using sigmoid function
            h = self.sigmoid(z)


def logisticRegression(data):
    # separate classes (houses) from rest of data
    y = data['Hogwarts House']
    x = data.select_dtypes(include=np.number)
    # remove features that muddy the data, based on pair plot
    x = x.drop(['Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    # split data into train and test, keep 25% of data to test model after training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)



def main():
    data = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    logisticRegression(data)


if __name__ == "__main__":
    main()