import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0
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


def logisticRegression(data):
    # separate classes (houses) from rest of data
    data = data.dropna()
    y = (data['Hogwarts House'] == 'Slytherin').astype(int)
    x = data.select_dtypes(include=np.number)
    # remove features that muddy the data, based on pair plot
    x = x.drop(['Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    # split data into train and test, keep 25% of data to test model after training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.train(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    plt.plot(model.cost_history)
    plt.title("Cost Function Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()
    print(model.weights)
    print(model.bias)


def main():
    data = pd.read_csv("../dataset_train.csv", index_col="Index")  # not ideal doesn't work if execute script from outside vizualize directory
    logisticRegression(data)


if __name__ == "__main__":
    main()