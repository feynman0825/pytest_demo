import random

import numpy as np
import pytest


## 5 Scopes of pytest fixtures
## function, class, module, package, session
@pytest.fixture(scope='function')
def etl_test_data():
    random.seed(42)
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(100, 10))
    y = np.random.binomial(1, 0.5, size=(100, 1))
    data = np.concatenate([X, y], axis=1)
    return data

@pytest.fixture(scope='module')
def test_object():
    # closure
    class Test:
        a = 1
        def inc(self):
            self.a += 1
        def dec(self):
            self.a -= 1
    return Test()
