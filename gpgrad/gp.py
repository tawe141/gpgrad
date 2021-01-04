import jax.numpy as np
from .kernels import RBF, GradKernel, Kernel
from jax.scipy.linalg import solve, cho_solve, cho_factor, cholesky, solve_triangular
from jax import vmap, grad, jit, partial
from typing import Tuple


@jit
def cholesky_solve(A, b):
    L = cholesky(A, lower=True)
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y)


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
        self.U = solve(self.K, self.y)

    # @partial(jit, static_argnums=(0,))
    def predict(self, x: np.ndarray):
        K_s = self.kernel(x, self.x).T
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
        self.predict_mu = vmap(self.predict_)
        self.predict_dy_ = grad(self.predict_, argnums=0)
        self.predict_dy = vmap(self.predict_dy_)
        # self.predict_dy_ = grad(self.predict_, argnums=0)
        # self.predict_dy = vmap(self.predict_dy_)

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
        # dy = np.concatenate([dydx[:, i] for i in range(dydx.shape[1])])
        self.y = np.concatenate((y, dydx.T.flatten()))
        self.K = self.kernel(x, x) + self.alpha * np.eye(len(x) + len(x)*x.shape[1])
        self.U = solve(self.K, self.y)

    # def predict(self, x: np.ndarray):
    #     """
    #     Returns mean and covariance from posterior distribution
    #     # TODO: implement covariance
    #
    #     :param x:
    #     :return:
    #     """
    #     K_s = self.kernel.k(x, self.x)
    #     dK_s = self.kernel.dkdx1(x, self.x, self.kernel.k.thetas)
    #     dK_s = np.concatenate([dK_s[:, :, i] for i in range(dK_s.shape[2])])
    #     r = np.concatenate((K_s, dK_s))
    #     return r.T @ self.U

    def predict_(self, x: np.ndarray) -> float:
        """
        Returns mean and covariance for a single example point `x`

        :param x: array of shape `(d, )`, where `d` is the dimensionality of the problem
        :return: posterior estimation at `x`
        """
        r = self.kernel.k(self.x, x)
        dr = self.kernel.dkdx1_a(self.x, x, self.kernel.k.thetas)
        # dr = np.concatenate([dr[:, i] for i in range(dr.shape[1])])
        return np.dot(np.concatenate((r, dr.T.flatten())), self.U)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_mu(x), self.predict_dy(x)


    # def predict_dy_(self, x: np.ndarray):

