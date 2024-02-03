import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    
    weights = None
    bias = None

    def __init__(self, lr = 0.001, nIter = 100) -> None:
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
            yPred= np.dot(X, self.weights) + self.bias

            dw = (1/nSamples) * np.dot(X.T, (yPred - y))
            db = (1/nSamples) * np.sum((yPred - y))

            self.weights = self.weights - (dw * self.lr)
            self.bias = self.bias - (db * self.lr)

            self.showPlots(X, y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def showPlots(self, X, y):

        self.ax.clear()
        self.ax.plot(X, (np.dot(X, self.weights) + self.bias))
        self.ax.scatter(X , y)

        plt.pause(0.02)
    


X = np.array([np.array(np.linspace(1, 10, 100)) for _ in range(1)])

Y = np.array([np.random.uniform(i + 10, i + 30) for i in range(100) for _ in range(1)])

lr = LinearRegression()
lr.fit(X.T, Y)