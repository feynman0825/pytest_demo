import os

import numpy as np
from numpy import ndarray


def get_data(path: str) -> ndarray:
    assert os.path.exists(path), "Should be a valid path."
    return np.load(path)

def etl(X: ndarray) -> ndarray:
    assert isinstance(X, ndarray), "X should be a ndarray."
    X = X.copy()
    X = np.concatenate([X, np.zeros((X.shape[0], 2))], axis=1)
    X[:, 10] = X[:, 2] + X[:, 4]
    X[:, 11] = X[:, 0] ** 2
    X = np.delete(X, 4, 1)
    X = np.delete(X, 2, 1)
    X = np.delete(X, 0, 1)
    return X
