import jax.numpy as np
from .kernels import RBF, GradKernel, Kernel
from jax.scipy.linalg import solve


class GP:
    def __init__(self, alpha=1e-8, kernel: Kernel = RBF()):
        self.alpha = alpha
        self.kernel = kernel
        self.x = None
        self.y = None
        self.K = None

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

    def predict(self, x: np.ndarray):
        K_s = self.kernel(x, self.x)
        K_ss = self.kernel(x, x)

        solved = solve(self.K, K_s, sym_pos=True).T

        mu = solved @ self.y
        cov = K_ss - (solved @ K_s)

        return mu, cov


class GPGrad:
    # TODO: this should probably inherit from GP
    def __init__(self, alpha=1e-8, kernel: Kernel = RBF()):
        self.alpha = alpha
        self.kernel = GradKernel(kernel)
        self.x = None
        self.x_ = None
        self.y = None
        self.K = None

    def fit(self, x: np.ndarray, y: np.ndarray, dydx: np.ndarray):
        """
        Obtains the hyperparameters that maximizes the log-likelihood of the observations
        TODO: implement this; current implementation only saves x, y, and dydx without optimizing hyperparams

        :param x:
        :param y:
        :param dydx:
        :return:
        """
        self.x = x
        self.y = np.concatenate((y.reshape(-1, 1), dydx))
        self.K = self.kernel(x, x) + self.alpha * np.eye(len(x) + len(x)*x.shape[1])

    def predict(self, x: np.ndarray):
        """
        Returns mean and covariance from posterior distribution
        # TODO: implement covariance

        :param x:
        :return:
        """
        K_s = self.kernel.k(x, self.x)
        dK_s = self.kernel.dkdx2(x, self.x, self.kernel.k.thetas)
        dK_s = np.concatenate([dK_s[:, :, i] for i in range(dK_s.shape[2])])
        r = np.concatenate((K_s, dK_s))
        return solve(self.K, r, sym_pos=True).T @ self.y
