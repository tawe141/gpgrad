import pytest
from gpgrad.kernels import *
import jax.numpy as np


def test_rbf():
    a = np.linspace(0, 10)
    k = RBF(1.0)
    result = k(a, a)
    assert result.shape == (len(a), len(a))
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(k(a, a), k(a.reshape(-1, 1), a.reshape(-1, 1)))
