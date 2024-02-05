import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

class LogisticRegression:
    
    weights = None
    bias = None
    allPredictions = []
    lastAccuracy = 0

    def __init__(self, lr = 0.01, nIter = 1000) -> None:
        self.lr = lr
        self.nIter = nIter

        plt.ion()

        fig, ax = plt.subplots(1,sharex=True)
        self.ax = ax

    def fit(self, X, y):

        nSamples, nFeatures = X.shape
        self.weights = np.zeros(nFeatures)
        self.bias = 0

        for _ in range(self.nIter):
            linearPred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linearPred)

            dw = (1/nSamples) * np.dot(X.T, (predictions - y))
            db = (1/nSamples) * np.sum((predictions - y))

            self.weights = self.weights - (dw * self.lr)
            self.bias = self.bias - (db * self.lr)

            self.showPlots( X, y, _)

    def predict(self, X):
        liPred = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(liPred)

        return [0 if y <= 0.5 else 1 for y in predictions ]
    
    def _sigmoid(self, X):
        return 1/ (1 + np.exp(-X))
    
    def showPlots(self, X, Y, currentIter):

        if currentIter % 20 == 0 :
        # self.ax.clear()
            y_pred = self.predict(X)
            thisLastAccuracy = accuracy(y_pred, Y)
            self.allPredictions.append(thisLastAccuracy)
            mean = np.mean(self.allPredictions) * 100
            if mean > self.lastAccuracy:
                self.lastAccuracy = mean
            # self.ax.plot(X, (np.dot(X, self.weights) + self.bias))
            self.ax.scatter(currentIter , self.lastAccuracy)
            plt.pause(0.002)
    



wine_data = datasets.load_breast_cancer()

X_train, X_test, Y_train, Y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.2, random_state=764)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)



# print(y_pred)
print(accuracy(y_pred, Y_test))

# examples based on https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/