import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logreg_train as lgt
import json
"""
take dataset_test.csv as parameter and file containing weights
generate file houses.csv with dataframe containing index and predicted house
"""

def predict(model, x):
    return model.sigmoid(np.dot(x, model.weights) + model.bias)

def startModels(file, dataset):
    num = dataset.select_dtypes(include=np.number)
    x = num.drop(['Hogwarts House', 'Care of Magical Creatures', 'Defense Against the Dark Arts', 'Potions', 'Arithmancy'], axis=1)
    x = x.fillna(x.mean())

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    models = {
        d["house"]: {"weights": np.array(d["weights"]), "bias": d["bias"]}
        for d in map(json.loads, open(file))
    }
    loaded_models = {
        house: lgt.LogisticRegression(weights=params['weights'], bias=params["bias"])
        for house, params in models.items()
    }
    predictions = {
        house: predict(loaded_models[house], x)
        for house in loaded_models.keys()
    }
    choixpeaux = pd.DataFrame(predictions).idxmax(axis=1)
    choixpeaux.index.name = 'Index'
    choixpeaux.name = 'Hogwarts House'
    choixpeaux.to_csv("houses.csv")

def main():
    try:
        assert len(sys.argv) == 3, "wrong argument count"
        assert sys.argv[1].find("dataset_test.csv") != -1 and  sys.argv[2].find("weights.jsonl") != -1 , "wrong file, use dataset_test.csv and weights.jsonl"
        dataset = pd.read_csv(sys.argv[1], index_col="Index")
        startModels(sys.argv[2], dataset)
    except AssertionError as error:
        print("AssertionError:", error)


if __name__ == "__main__":
    main()