import pytest
from gpgrad.kernels import *
import jax.numpy as np
from jax import jacfwd, jacrev
import numpy as onp
import jax


def test_rbf():
    a = np.linspace(0, 10).reshape(-1, 1)
    k = RBF(1.0)
    result = k(a, a)
    assert result.shape == (len(a), len(a))
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(k(a, a), k(a.reshape(-1, 1), a.reshape(-1, 1)))


def test_rbfgrad():
    a = np.linspace(0, 10).reshape(-1, 1)
    b = np.linspace(0, 10).reshape(-1, 1)

    k = RBF(1.0)
    gk = GradKernel(k)
    result = gk(a, b)
    assert result.shape == (2*len(a), 2*len(a))
    # diagonals of kernel matrix should be 1.0
    assert np.allclose(np.diag(result[:len(a), :len(a)]), 1.0)
    # diagonals of derivative matrices wrt. x1 or x2 should be 0.0
    assert np.allclose(np.diag(result[:len(a), len(a):]), 0.0)
    assert np.allclose(np.diag(result[len(a):, :len(a)]), 0.0)
    # diagonals of hessian wrt x1 and x2 should be 1.0
    assert np.allclose(np.diag(result[len(a):, len(a):]), 1.0)

    # upper right block should be equal to the transpose of the lower left block
    assert np.allclose(
        result[:len(a), len(a):],
        result[len(a):, :len(a)].T
    )

