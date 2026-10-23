import sys
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logreg_train as lgt
"""
take dataset_test.csv as parameter and file containing weights
generate file houses.csv with dataframe containing index and predicted house
"""

def predict(model, x):
    return (model.sigmoid(np.dot(x, model.weights) + model.bias) >= 0.5).astype(int)

def makePredictions(data, weights, bias):
    num = data.select_dtypes(include=np.number)
    x = num.drop(['Hogwarts House', 'Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    x = x.fillna(x.mean()) # replace missing values with mean

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    model = lgt.LogisticRegression(weights, bias, 0.01, 1000)
    predictions =predict(model, x)
    print('Predictions:')
    print(predictions) 



def main():
    try:
        assert len(sys.argv) == 3, "wrong argument count"
        assert sys.argv[1].find("dataset_test.csv") != -1 and  sys.argv[2].find("weights.txt") != -1 , "wrong file, use dataset_test.csv and weights.txt"
        dataset = pd.read_csv(sys.argv[1], index_col="Index")
        with open(sys.argv[2], 'r') as f:
            r_weights, r_bias = f.read().strip().split('\n\n', 1)
            weights = np.genfromtxt(io.StringIO(r_weights))
            bias = float(r_bias)
        makePredictions(dataset, weights, bias)
    except AssertionError as error:
        print("AssertionError:", error)


if __name__ == "__main__":
    main()