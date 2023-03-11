import numpy as np
from numpy import ndarray


class LinearModel:
    def __init__(self):
        self._beta = None

    def train(self, X: ndarray, y: ndarray):
        """ beta = (X^T X)^-1 * X^T *y
        """
        assert isinstance(X, ndarray), "X should be a ndarray"
        assert isinstance(y, ndarray), "y should be a ndarray"
        self._beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X: ndarray):
        assert isinstance(X, ndarray), "X should be a ndarray"
        return X.dot(self.beta)

    def score(self, X: ndarray, y: ndarray):
        pass

    @property
    def beta(self):
        return self._beta
