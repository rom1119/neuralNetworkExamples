import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

class Node:

    def __init__(self, feature = None, threshold = None, left = None, right= None, *, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
      

class DecisionTree:
    
    allPredictions = []
    lastAccuracy = 0

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

        plt.ion()

        fig, ax = plt.subplots(1,sharex=True)
        self.ax = ax

    def _split(self, X_column, split_threshold):
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()

        return left_idx, right_idx

    def _entropy(self,  y):
        hist = np.bincount(y)
        ps = hist / len(y)

        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _information_gain(self,  y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)
        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        #calculate the weight avg. entropy of chldren

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy

        return information_gain


    def _best_split(self, X,  y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find the best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)
        # create child nodes

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)  

        # self.showPlots( X, y, _)

    def _treverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._treverse_tree(x, node.left)
        
        return self._treverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._treverse_tree(x, self.root) for x in X])
    
    
    # def showPlots(self, X, Y, currentIter):

    #     if currentIter % 20 == 0 :
    #     # self.ax.clear()
    #         y_pred = self.predict(X)
    #         thisLastAccuracy = accuracy(y_pred, Y)
    #         self.allPredictions.append(thisLastAccuracy)
    #         mean = np.mean(self.allPredictions) * 100
    #         if mean > self.lastAccuracy:
    #             self.lastAccuracy = mean
    #         # self.ax.plot(X, (np.dot(X, self.weights) + self.bias))
    #         self.ax.scatter(currentIter , self.lastAccuracy)
    #         plt.pause(0.002)
    



wine_data = datasets.load_breast_cancer()

X_train, X_test, Y_train, Y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.2, random_state=764)

lr = DecisionTree()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)



# print(Y_train)
print(accuracy(y_pred, Y_test))