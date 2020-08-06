import pytest
from gpgrad.gp import GP
import jax.numpy as np


def test_gp():
    model = GP()
    x = np.linspace(0.0, np.pi, 6)
    y = np.sin(x)
    model.fit(x, y)
    mu, var = model.predict(x)
    assert np.allclose(mu, y, rtol=1e-3, atol=1e-5)
    assert np.allclose(var, 0.0, atol=1e-5)

    more_x = np.linspace(0.0, np.pi, 50)
    mu, var = model.predict(more_x)
    true_y = np.sin(more_x)
    mse = np.mean((mu - true_y)**2)
    print('Sine wave MSE: %f' % mse)
    assert mse < 1e-4
