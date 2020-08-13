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


def test_pairwise_gradients():
    f = lambda x,y: np.linalg.norm(np.expand_dims(x, 1) - np.expand_dims(y, 0), axis=-1)
    a = np.array(onp.random.rand(10)).reshape(-1, 1)
    b = np.array(onp.random.rand(10)).reshape(-1, 1)

    assert f(a, b).shape == (10, 10)
    dfdx = jacfwd(f, argnums=0)
    assert dfdx(a, b).shape == (10, 10, 10, 1)

    def slow_pairwise_dist(x, y):
        d = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                d = jax.ops.index_update(d, (i, j), np.linalg.norm(x[i] - y[j]))
        return d

    assert np.allclose(f(a, a), slow_pairwise_dist(a, a))

    dslowdx = jacfwd(slow_pairwise_dist, argnums=0)
    assert dslowdx(a, a).shape == (10, 10, 10, 1)

    df2 = jacfwd(jacrev(f, argnums=0), argnums=1)
    print(df2(a, a).shape)
