import pytest
from gpgrad.gp import GP, GPGrad
from gpgrad.kernels import RBF
import jax.numpy as np


def mean_squared_error(x, y):
    return np.mean((x - y)**2)


def test_gp():
    model = GP()
    x = np.linspace(0.0, np.pi, 6).reshape(-1, 1)
    y = np.sin(x).squeeze()
    model.fit(x, y)
    mu, var = model.predict(x)
    assert np.allclose(mu, y, rtol=1e-3, atol=1e-5)
    assert np.allclose(var, 0.0, atol=1e-5)

    more_x = np.linspace(0.0, np.pi, 50)
    mu, var = model.predict(more_x)
    true_y = np.sin(more_x)
    mse = mean_squared_error(mu, true_y)
    print('Sine wave MSE: %f' % mse)
    assert mse < 1e-4


def test_gpgrad():
    model = GPGrad(kernel=RBF(1e-1))
    x = np.linspace(0.0, np.pi, 10).reshape(-1, 1)
    y = np.sin(x).squeeze()
    dy = np.cos(x)
    model.fit(x, y, dy)
    mu = model.predict(x)
    assert np.allclose(mu.squeeze(), y)
    mu_dy = model.predict_dy(x)
    assert np.allclose(mu_dy, dy)

    normal_model = GP(kernel=RBF(1e-1))
    normal_model.fit(x, y)

    # gradient-enhanced GP should have a lower MSE than normal GP
    # interpolation
    more_x = np.linspace(0.0, np.pi).reshape(-1, 1)

    gpgrad_y = model.predict(more_x)
    normal_y, _ = normal_model.predict(more_x)
    true_y = np.sin(more_x)

    gpgrad_mse = mean_squared_error(gpgrad_y, true_y)
    normal_mse = mean_squared_error(normal_y, true_y)
    print("Gradient-enhanced GP MSE: %f" % gpgrad_mse)
    print("Normal GP MSE: %f" % normal_mse)

    assert gpgrad_mse < normal_mse

    # # extrapolation
    # # NOTE: the MSE of both methods are very similar here
    # more_x = np.linspace(np.pi, 1.5 * np.pi).reshape(-1, 1)
    #
    # gpgrad_y = model.predict(more_x)
    # normal_y, _ = normal_model.predict(more_x)
    # true_y = np.sin(more_x)
    #
    # gpgrad_mse = mean_squared_error(gpgrad_y, true_y)
    # normal_mse = mean_squared_error(normal_y, true_y)
    # print("Gradient-enhanced GP MSE: %f" % gpgrad_mse)
    # print("Normal GP MSE: %f" % normal_mse)
    #
    # assert gpgrad_mse < normal_mse


def test_gpgrad_predict_dy():
    model = GPGrad(kernel=RBF(1e-1), alpha=0.0)
    x = np.linspace(0.0, np.pi, 10).reshape(-1, 1)
    y = np.sin(x).squeeze()
    dy = np.cos(x)
    model.fit(x, y, dy)

    assert np.allclose(model.predict_(x[0]), 0.0)
    assert np.allclose(model.predict_dy_(x[0]), 1.0)  # fails here; opposite sign. took derivative wrt wrong x?
