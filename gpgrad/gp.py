import jax.numpy as np
from .kernels import RBF
from jax.scipy.linalg import solve


class GP:
    def __init__(self, alpha=1e-8, kernel=RBF):
        self.alpha = alpha
        self.kernel = kernel()
        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Obtains the hyperparameters that maximizes the log-likelihood of the observations
        TODO: implement this; current implementation only saves x and y without optimizing hyperparams

        :param x: array shape (Nxd), independent variables
        :param y: array shape (Nx1), dependent variable
        :return:
        """
        self.x = x
        self.y = y
        self.K = self.kernel(x, x) + self.alpha * np.eye(len(x))
        self.K_inv = np.linalg.inv(self.K)

    def predict(self, x: np.ndarray):
        K_s = self.kernel(x, self.x)
        K_ss = self.kernel(x, x)

        solved = solve(self.K, K_s, sym_pos=True).T

        mu = solved @ self.y
        cov = K_ss - (solved @ K_s)

        return mu, cov
