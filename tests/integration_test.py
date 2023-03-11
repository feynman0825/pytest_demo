import numpy as np
import pytest
from unittest.mock import patch

from pytest_demo.pipeline import go

@pytest.mark.pipeline
@patch('pytest_demo.pipeline.get_data')
def test_pipeline(get_data):
    get_data.return_value = np.load('pytest_demo/data/small_data.npy')
    go()