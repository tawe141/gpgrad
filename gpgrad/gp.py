import jax.numpy as np
from .kernels import RBF, GradKernel, Kernel
from jax.scipy.linalg import solve
from jax import vmap, grad, jit, partial
from typing import Tuple
from .utils import method_jit


class GP:
    def __init__(self, alpha=1e-8, kernel: Kernel = RBF(), debug=True):
        self.alpha = alpha
        self.kernel = kernel
        self.x = None
        self.y = None
        self.K = None
        self.predict_mu = vmap(self.predict_)
        self.predict_var = vmap(self.predict_var_)

        if debug is False:
            self.predict_mu = method_jit(self.predict_mu)
            self.predict_var = method_jit(self.predict_var)

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

    # # @partial(jit, static_argnums=(0,))
    # def predict(self, x: np.ndarray):
    #     K_s = self.kernel(x, self.x).T
    #     K_ss = self.kernel(x, x)
    #
    #     solved = solve(self.K, K_s, sym_pos=True).T
    #
    #     mu = solved @ self.y
    #     cov = K_ss - (solved @ K_s)
    #
    #     return mu, cov
    def predict_(self, x: np.ndarray) -> float:
        return self.kernel(x, self.x) @ self.U

    def predict_var_(self, x: np.ndarray) -> float:
        K_train = self.kernel(x, self.x)
        return self.kernel(x, x) - K_train.T @ solve(self.K, K_train)

    def predict(self, x: np.ndarray):
        return self.predict_mu(x), self.predict_var(x)


class GPGrad:
    # TODO: this should probably inherit from GP
    def __init__(self, alpha=1e-8, kernel: Kernel = RBF(), debug=True):
        self.alpha = alpha
        self.kernel = GradKernel(kernel)
        self.x = None
        self.y = None
        self.K = None
        self.U = None
        self.predict_mu = vmap(self.predict_)
        self.predict_dy_ = grad(self.predict_, argnums=0)
        self.predict_dy = vmap(self.predict_dy_)

        if debug is False:
            # self.fit = method_jit(self.fit)
            self.predict_mu = method_jit(self.predict_mu)
            self.predict_dy = method_jit(self.predict_dy)

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

