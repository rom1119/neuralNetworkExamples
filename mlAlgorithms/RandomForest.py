from  DecisionTree import DecisionTree
from  DecisionTree import accuracy
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    
    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         n_features=self.n_features
                         )
            
            X_sample, Y_sample = self._boot_samples(X, y)

            tree.fit(X_sample, Y_sample)

            self.trees.append(tree)

    def _most_common_label(self, y):
        c = Counter(y)
        return c.most_common(1)[0][0]

    def _boot_samples(seslf, X, y):
        n_samples =X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)

        return X[idxs], y[idxs]
    
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(preds, 0, 1)
        preds = np.array([self._most_common_label(p) for p in tree_preds])

        return preds

cancer_data = datasets.load_breast_cancer()

X_train, X_test, Y_train, Y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=764)

lr = RandomForest()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)


# print(X_train[0])
# print(X_test)
# X_test = X_test + 3
# print(X_test)
# print(Y_train)
print(accuracy(y_pred, Y_test))

# examples based on https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/