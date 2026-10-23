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
        # sigmoid function used to get probability between 0 and 1
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        # calculate how well the model is performing
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def train(self, x, y):
        m, n = x.shape
        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            # multiply data points by weights and add bias
            z = np.dot(x, self.weights) + self.bias
            # turn into 0 to 1 odd probability using sigmoid function
            h = self.sigmoid(z)
            #use gradient descent to adjust weights and bias
            d_weights = (1 / m) * np.dot(x.T, (h - y))
            d_bias = (1 / m) * np.sum(h - y)
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            self.cost_history.append(self.cost(h, y))

    def predict(self, x):
        return (self.sigmoid(np.dot(x, self.weights) + self.bias) >= 0.5).astype(int)

def trainModel(data):
    houses = {
        'Hufflepuff',
        'Gryffindor',
        'Ravenclaw',
        'Slytherin'
    }
    # remove rows with missing data
    data = data.dropna()
    x = data.select_dtypes(include=np.number)
    x = x.drop(['Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    for house in houses:
        print(house)
        y = (data['Hogwarts House'] == house).astype(int)
        model = LogisticRegression(None, 0, 0.1, 1000)
        model.train(x, y)
        with open("weights.txt", "a") as f:
            f.write(house)
            f.write("\nWeights:\n")
            np.savetxt(f, model.weights)
            f.write("\nBias:\n")
            f.write(str(model.bias) + "\n\n")

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