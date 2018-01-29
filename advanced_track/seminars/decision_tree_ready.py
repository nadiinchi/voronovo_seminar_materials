from sklearn.base import BaseEstimator
import numpy as np

class MyDecisionTree(BaseEstimator):
    def __init__(self, min_leaf_size=1):
        self.min_leaf_size_ = min_leaf_size
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = Tree(X, y, self.min_leaf_size_)

    def predict(self, X):
        return self.tree_.predict(X)

class Tree:
    def __init__(self, X, y, min_leaf_size):
        self.y_ = y
        self.left_child = None
        self.right_child = None

        self.feature, self.border = find_best_split(X, y)
        if self.feature == None:
            return

        split_mask = X[:, self.feature] <= self.border

        # TO DO:
        # Implement stopping criteria based on min_leaf_size
        if np.sum(split_mask) < min_leaf_size or \
            (len(split_mask) - np.sum(split_mask)) < min_leaf_size:
            return

        self.left_child = Tree(X[split_mask], y[split_mask], min_leaf_size)
        self.right_child = Tree(X[np.logical_not(split_mask)],
                                y[np.logical_not(split_mask)], min_leaf_size)

    def predict(self, X):
        if self.left_child == None:
            return np.zeros(len(X)) + np.mean(self.y_)

        # TO DO: implement predciction function
        split_mask = X[:, self.feature] <= self.border
        predictions = np.zeros(len(X))
        predictions[split_mask] = self.left_child.predict(X[split_mask])
        predictions[np.logical_not(split_mask)] = \
                self.right_child.predict(X[np.logical_not(split_mask)])

        return predictions

def find_best_split(X, y):
    best_error = np.sum(np.power(y - np.mean(y), 2))
    best_border = None
    best_feature = None

    for feature in range(X.shape[1]):
        for i in range(X.shape[0]):
            split_mask = X[:, feature] <= X[i, feature]
            if np.sum(split_mask) == 0 or np.sum(split_mask) == len(split_mask):
                continue

            error = y[split_mask] - np.mean(y[split_mask])
            left_error = np.sum(np.power(error, 2))

            error = y[np.logical_not(split_mask)] - \
                    np.mean(y[np.logical_not(split_mask)])
            right_error = np.sum(np.power(error, 2))
            if left_error + right_error < best_error:
                best_feature = feature
                best_border = X[i, feature]
                best_error = left_error + right_error

    return best_feature, best_border
