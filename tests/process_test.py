import numpy as np
import pytest
from numpy import ndarray

from pytest_demo import process


@pytest.mark.ingestion
def test_get_data():
    # Given
    path = 'pytest_demo/data/data.npy'
    shape= (100, 11)

    # When
    data = process.get_data(path)
    
    # Then
    assert isinstance(data, ndarray), "Data should be a numpy array."
    assert data.shape == shape, f"Data should have shape {shape}"
    assert np.isclose(np.min(data[:, -1]), 0) and \
        np.isclose(np.max(data[:, -1]), 1), "Data should be between [0, 1]."

@pytest.mark.ingestion
def test_get_data_with_wrong_input():
    # Given
    path = "pytest_demo/data/fake.npy"
    
    # When & Then
    with pytest.raises(AssertionError) as e_info:
        data = process.get_data(path)

@pytest.mark.etl
def test_etl(etl_test_data):
    # Given
    X = etl_test_data[:, :10]
    expected_shape = (100, 9)
    
    # When
    processed_X = process.etl(X)

    # Then
    assert isinstance(processed_X, ndarray), "Data should be an numpy array."
    assert processed_X.shape == expected_shape, f"Data should have shape {expected_shape}."
    assert np.alltrue(processed_X[:, 8] >= 0), "The feature should be non-negative."

@pytest.mark.scope
def test_fixture_inc(test_object):
    # Given
    expected_ans = 2

    # When
    test_object.inc()

    # Then
    assert test_object.a == expected_ans

@pytest.mark.scope
def test_fixture_dec(test_object):
    # Given
    expected_ans = 0

    # When
    test_object.dec()

    # THen
    assert test_object.a == expected_ans

@pytest.mark.etl
def test_etl_with_wrong_shape():
    # Given
    X = np.zeros((100, 5))

    # When/THen
    with pytest.raises(IndexError) as e_info:
        processed_X = process.etl(X)