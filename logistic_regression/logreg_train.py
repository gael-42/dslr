import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
"""
take dataset_train.csv as parameter
Generate file containing weights and bias
"""

class LogisticRegression:
    def __init__(self, weights, bias, learning_rate=0.01, iterations=1000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = weights
        self.bias = bias
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def train(self, x, y):
        m, n = x.shape
        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            z = np.dot(x, self.weights) + self.bias
            h = self.sigmoid(z)
            d_weights = (1 / m) * np.dot(x.T, (h - y))
            d_bias = (1 / m) * np.sum(h - y)
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            self.cost_history.append(self.cost(h, y))


def trainModel(data):
    data = data.dropna()
    y = (data['Hogwarts House'] == 'Slytherin').astype(int)
    x = data.select_dtypes(include=np.number)
    x = x.drop(['Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(None, 0, 0.1, 1000)
    model.train(x_train, y_train)
    with open("weights.txt", "a") as f:
        np.savetxt("weights.txt", model.weights)
        f.write("\n")
        f.write(str(model.bias))

def main():
    try:
        assert len(sys.argv) == 2, "wrong argument count"
        assert sys.argv[1].find("dataset_train.csv") != -1, "wrong file, use dataset_train.csv"
        dataset = pd.read_csv(sys.argv[1], index_col="Index")
        trainModel(dataset)
    except AssertionError as error:
        print("AssertionError:", error)


if __name__ == "__main__":
    main()