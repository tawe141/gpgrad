import pytest
from gpgrad.gp import GP, GPGrad
from gpgrad.kernels import RBF
import jax.numpy as np
from jax import jit, vmap, grad, jacfwd
from functools import reduce
import numpy as onp
# use sklearn's GPs as ground truth
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SklRBF


def mean_squared_error(x, y):
    return np.mean((x - y) ** 2)


def smoothed_herbie_(x):
    def f(a):
        return np.exp(-(a - 1) ** 2) + np.exp(-0.8 * (a + 1) ** 2)
    return reduce(np.multiply, f(x).T)


smoothed_herbie = jit(vmap(smoothed_herbie_))
smoothed_herbie_dy = jit(vmap(grad(smoothed_herbie_)))


def test_gp():
    model = GP()
    x = np.linspace(0.0, np.pi, 6).reshape(-1, 1)
    y = np.sin(x).squeeze()
    model.fit(x, y)
    mu, var = model.predict(x)
    assert np.allclose(mu, y, rtol=1e-3, atol=1e-5)
    assert np.allclose(var, 0.0, atol=1e-5)

    more_x = np.linspace(0.0, np.pi, 50).reshape(-1, 1)
    mu, var = model.predict(more_x)
    true_y = np.sin(more_x)
    mse = mean_squared_error(mu, true_y)
    print('Sine wave MSE: %f' % mse)
    assert mse > 1e-4


def test_gp_2d():
    model = GP(kernel=RBF(1.0, debug=False), debug=False)

    x, y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    X = np.stack((x.flatten(), y.flatten()), axis=1)
    z = smoothed_herbie_(X)

    model.fit(X, z)
    mu, var = model.predict(X)
    assert np.allclose(mu, z)
    assert np.allclose(var, np.zeros_like(var), atol=1e-6)


def test_gp_optimize_thetas():
    model = GP(kernel=RBF(1e-1), alpha=1e-5)
    x = np.linspace(0.0, np.pi, 10).reshape(-1, 1)
    y = np.sin(x).squeeze()
    model.fit(x, y)

    test_x = np.linspace(0.0, np.pi, 50).reshape(-1, 1)
    test_y = np.sin(test_x).squeeze()
    predict_before_opt, _ = model.predict(test_x)

    mse1 = mean_squared_error(predict_before_opt, test_y)

    model._optimize_theta()
    predict_after_opt, _ = model.predict(test_x)
    mse2 = mean_squared_error(predict_after_opt, test_y)
    assert mse2 < mse1

    skl = GaussianProcessRegressor(kernel=SklRBF(length_scale_bounds=(0.001, 10000.)), alpha=1e-5)
    skl.fit(x, y)
    assert np.allclose(model.kernel.thetas, np.exp(skl.kernel_.theta), atol=0.01)


def test_gpgrad_1d():
    model = GPGrad(kernel=RBF(1))

    # true y(x) is a sine curve, but training points are only where y=0
    x = np.linspace(0.0, 3 * np.pi, 4).reshape(-1, 1)
    y = np.sin(x).squeeze()
    dy = np.cos(x)
    model.fit(x, y, dy)
    mu, mu_dy = model.predict(x)
    assert np.allclose(mu.squeeze(), y)
    assert np.allclose(mu_dy, dy, atol=1e-3)

    normal_model = GP(kernel=RBF(1))
    normal_model.fit(x, y)

    # gradient-enhanced GP should have a lower MSE than normal GP
    # interpolation
    more_x = np.linspace(0.0, np.pi).reshape(-1, 1)

    gpgrad_y = model.predict_mu(more_x)
    normal_y, _ = normal_model.predict(more_x)
    true_y = np.sin(more_x)

    gpgrad_mse = mean_squared_error(gpgrad_y, true_y)
    normal_mse = mean_squared_error(normal_y, true_y)
    print("Gradient-enhanced GP MSE: %f" % gpgrad_mse)
    print("Normal GP MSE: %f" % normal_mse)

    assert gpgrad_mse < normal_mse


def test_gpgrad_2d():
    model = GPGrad(kernel=RBF(1))

    x, y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    X = np.stack((x.flatten(), y.flatten()), axis=1)
    z = smoothed_herbie(X)
    dz = smoothed_herbie_dy(X)

    model.fit(X, z, dz)
    # prediction = model.predict_(X[0])
    # assert np.isclose(prediction, z[0])

    mu, mu_d = model.predict(X)
    assert np.allclose(mu, z, atol=1e-4)
    assert np.allclose(mu_d, dz, atol=1e-4)


def test_gpgrad_predict_dy():
    model = GPGrad(kernel=RBF(1e-1), alpha=1e-4)
    x = np.linspace(0.0, np.pi, 10).reshape(-1, 1)
    y = np.sin(x).squeeze()
    dy = np.cos(x)
    model.fit(x, y, dy)

    assert np.allclose(model.predict_(x[0]), 0.0)
    assert np.allclose(model.predict_dy_(x[0]), 1.0)


def test_gpgrad_optimize_thetas():
    model = GPGrad(kernel=RBF(1e-1), alpha=1e-4)
    x = np.linspace(0.0, np.pi, 4).reshape(-1, 1)
    y = np.sin(x).squeeze()
    dy = np.cos(x)
    model.fit(x, y, dy)

    test_x = np.linspace(0.0, np.pi, 50).reshape(-1, 1)
    test_y = np.sin(test_x)
    mse1 = mean_squared_error(model.predict_mu(test_x), test_y)

    model._optimize_theta()
    mse2 = mean_squared_error(model.predict_mu(test_x), test_y)

    assert mse2 < mse1
