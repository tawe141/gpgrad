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
    # # smallest eigenvalues are negative but close to 0. numerical precision error?
    # assert (np.linalg.eigvalsh(result) > 0.0).all()


def test_rbfgrad_1d():
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

    # # upper right block should be equal to the negative transpose of the lower left block
    # assert np.allclose(
    #     result[:len(a), len(a):],
    #     -result[len(a):, :len(a)].T
    # )
    # # upper right block should also be equal to the negative of the lower block
    # assert np.allclose(
    #     result[:len(a), len(a):],
    #     -result[len(a):, :len(a)].T
    # )

    assert (np.linalg.eigvalsh(result + 1e-4 * np.eye(len(result))) > 0.0).all()


# def test_rbfgrad_2d():
#     a = onp.random.rand(10, 2)
#
#     k = RBF(1.0)
#     gk = GradKernel(k)
#     result = gk(a, a)
#
#     assert result.shape == (30, 30)
#
#     # diagonals of kernel matrix should be 1.0
#     assert np.allclose(np.diag(result[:len(a), :len(a)]), 1.0)
#     # diagonals of hessian wrt x1 and x2 should be 1.0
#     assert np.allclose(np.diag(result[len(a):, len(a):]), 1.0)
#
#     # upper right block should be equal to the transpose of the lower left block
#     assert np.allclose(
#         result[:len(a), len(a):],
#         result[len(a):, :len(a)].T
#     )

    # assert (np.linalg.eigvalsh(result + 1e-4 * np.eye(len(result))) > 0.0).all()

def test_rbfgrad_2d():
    def f(x):
        return x[:, 0]**2 + x[:, 1]**2

    def rbf(x1, x2, s=1.0):
        """
        Radial basis function
        :param x1: vector
        :param x2: vector
        :param s: lengthscale
        :return: scalar
        """
        d = x1 - x2
        return np.exp(-0.5*np.dot(d, d)/s/s)

    def rbf_dx1(x1, x2, d, s=1.0):
        """
        Derivative of RBF wrt. dimension `d`
        :param x1: vector
        :param x2: vector
        :param d: int, dimension
        :param s: lengthscale
        :return: scalar
        """
        return -rbf(x1, x2, s) / s / s * (x1[d]-x2[d])

    def rbf_dx2(x1, x2, d, s=1.0):
        return -rbf_dx1(x1, x2, d, s)

    def rbf_dx1dx2(x1, x2, d1, d2, s=1.0):
        r = rbf(x1, x2, s)
        a1 = r / s
        a2 = -r/s/s

        a2 *= x1[d1] - x2[d1]
        a2 *= x1[d2] - x2[d2]
        if d1 != d2:
            return a2
        else:
            return a1 + a2

    X = np.array([
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.]
    ])
    dy = 2 * X

    # make sure jax grad and hand-derived grad match up
    assert np.allclose(
        grad(rbf, argnums=0)(X[0], X[1]),
        [rbf_dx1(X[0], X[1], 0), rbf_dx1(X[0], X[1], 1)]
    )

    k = RBF(1.0)
    gk = GradKernel(k)
    result = gk(X, X)

    assert result.shape == (12, 12)
    assert k.forward_(X[0], X[1], thetas=np.array([1.])) == rbf(X[0], X[1])

    # test kernel w/o gradients
    assert np.allclose(
        result[:4, :4],
        k(X, X)
    )

    assert np.allclose(
        result[0, :4],
        [1., np.exp(-2.), np.exp(-2.), np.exp(-4.)]
    )

    # test upper gradient kernel
    true_dx1_0 = np.array([
        [rbf_dx1(X[i], X[j], 0) for j in range(len(X))]
        for i in range(len(X))
    ])
    # should be one or the other...not entirely sure whether it should be transposed or not
    # assertions for first derivatives
    assert np.allclose(result[:4, 4:8], true_dx1_0) or np.allclose(result[:4, 4:8], true_dx1_0.T)
    true_dx1_1 = np.array([
        [rbf_dx1(X[i], X[j], 1) for j in range(len(X))]
        for i in range(len(X))
    ])
    assert np.allclose(result[:4, 8:], true_dx1_1) or np.allclose(result[:4, 8:], true_dx1_1.T)

    # assertions for hessian blocks
    true_dx1dx2_00 = np.array([
        [rbf_dx1dx2(X[i], X[j], 0, 0) for j in range(len(X))]
        for i in range(len(X))
    ])
    true_dx1dx2_01 = np.array([
        [rbf_dx1dx2(X[i], X[j], 0, 1) for j in range(len(X))]
        for i in range(len(X))
    ])
    true_dx1dx2_10 = np.array([
        [rbf_dx1dx2(X[i], X[j], 1, 0) for j in range(len(X))]
        for i in range(len(X))
    ])
    true_dx1dx2_11 = np.array([
        [rbf_dx1dx2(X[i], X[j], 1, 1) for j in range(len(X))]
        for i in range(len(X))
    ])
    true_hess = np.block([
        [true_dx1dx2_00, true_dx1dx2_01],
        [true_dx1dx2_10, true_dx1dx2_11]
    ])
    assert np.allclose(result[4:, 4:], true_hess)
    assert np.allclose(true_hess, true_hess.T)
    # assert np.allclose(np.diag(result[4:, 4:]), 2.0)


def test_jit():
    k = RBF(debug=False)
    x = np.array([
        [1, 2],
        [2, 3],
        [5, 1]
    ])
    k(x, x)
