import numpy as np
import pytest
from numpy import ndarray

from pytest_demo.model import LinearModel

TEST_Xs = [np.array([[0, 1],
                     [0, 0]]),
            np.array([[0, 1],
                     [1, 0]]),
            np.array([[1, 1],
                     [1, 0]]),
            ]

TEST_ys = [np.array([1, 0]),
           np.array([1, 1]),
           np.array([2, 1])]

class TestLinearModel:

    @pytest.mark.model
    def test_model_train(self):
        # function - scenario
        # Given
        X = np.array([[1, 0],
                      [0, 1]])
        y = np.array([1, 1])
        ans = np.array([1, 1])

        # When
        model = LinearModel()
        model.train(X, y)
        beta = model.beta

        # Then
        assert isinstance(beta, ndarray), "Beta should be a numpy array."
        assert np.allclose(beta, ans)

    @pytest.mark.model
    @pytest.mark.parametrize("X, y", list(zip(TEST_Xs, TEST_ys)))
    def test_model_predict(self, X: ndarray, y: ndarray):
        # Given
        model = LinearModel()
        model._beta = np.array([1, 1])

        # When
        pred = model.predict(X)

        # Then
        assert isinstance(pred, ndarray), "Predict should be a numpy array."
        assert np.allclose(pred, y), "Model predictions are wrong."


    @pytest.mark.model
    @pytest.mark.skip
    def test_model_score(self):
        assert True == False