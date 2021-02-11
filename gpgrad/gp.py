import jax.numpy as np
from .kernels import RBF, GradKernel, Kernel
from jax.scipy.linalg import solve
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize
from jax import vmap, grad, jit, partial, hessian
from typing import Tuple
from .utils import method_jit


@jit
def add_jitter(mat: np.ndarray, alpha: float) -> np.ndarray:
    """
    Adds "jitter", a small number along the diagonal of `mat` to increase numerical stability upon inversion of `mat`.
    Increase `alpha` if `mat` is ill-conditioned.
    :param mat: np.ndarray
    :param alpha: float
    :return:
    """
    return mat + alpha * np.eye(len(mat))


@jit
def jit_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    JIT compiled version of `jax.scipy.linalg.solve`. Used in the `fit` function because `GP.fit` cannot be JITed
    :param A: rank 2 matrix
    :param b: vector
    :return: np.ndarray, A^-1 @ b
    """
    return solve(A, b, sym_pos=True)


def posdef_logdet(m: np.ndarray) -> float:
    """
    Returns the log-determinant of a positive definite matrix
    :param m:
    :return:
    """
    L = np.linalg.cholesky(m)
    return 2 * np.sum(np.log(np.diag(L)))


@jit
def _nll(y: np.ndarray, K: np.ndarray, Kinv_y: np.ndarray) -> float:
    ld = posdef_logdet(K)
    return 0.5 * (np.vdot(y, Kinv_y) + ld)


class GP:
    def __init__(self, alpha: float = 1e-8, kernel: Kernel = RBF(), debug: bool = True):
        """
        Constructor for base Gaussian process class.

        :param alpha: (float) nugget parameter, increase if matrix inverses fail due to ill-conditioning. Default: 1e-8
        :param kernel: (Kernel)
        :param debug: (bool) if False, JIT compiles `self.predict_mu` and `self.predict_var`
        TODO: figure out why you can't JIT `self.fit`
        """
        self.alpha = alpha
        self.kernel = kernel
        self.x = None
        self.y = None
        self.K = None
        self.U = None
        self.predict_mu = vmap(self.predict_)
        self.predict_var = vmap(self.predict_var_)

        if debug is False:
            self.predict_mu = method_jit(self.predict_mu)
            self.predict_var = method_jit(self.predict_var)

    def fit(self, x: np.ndarray, y: np.ndarray, *args):
        """
        Obtains the hyperparameters that maximizes the log-likelihood of the observations
        TODO: implement this; current implementation only saves x and y without optimizing hyperparams

        :param x: array shape (Nxd), independent variables
        :param y: array shape (Nx1), dependent variable
        :return:
        """
        self.x = x
        self.y = y
        self.K = add_jitter(self.kernel(x, x), self.alpha)
        self.U = jit_solve(self.K, self.y)

    def predict_(self, x: np.ndarray) -> float:
        """
        Returns prediction of a *single sample* `x`. See `self.predict_mu` for the corresponding vectorized function

        :param x: vector of shape (d,), where d is the dimensionality of the input
        :return: float
        """
        return self.kernel(x, self.x) @ self.U

    def predict_var_(self, x: np.ndarray) -> float:
        """
        Returns covariance of a *single sample* `x`. See `self.predict_var` for the corresponding vectorized function

        :param x: vector of shape (d,), where d is the dimensionality of the input
        :return: float
        """
        K_train = self.kernel(x, self.x)
        return self.kernel(x, x) - K_train.T @ solve(self.K, K_train)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns prediction and variance of the GP given inputs `x`. Simple wrapper calling `self.predict_mu` and `self.predict_var`

        :param x: vector of shape(N, d) inputs
        :return:
        """
        return self.predict_mu(x), self.predict_var(x)

    def nll(self, thetas):
        self.kernel.thetas = thetas
        self.fit(self.x, self.y)
        # logdet = posdef_logdet(self.K)
        # return 0.5 * (np.vdot(self.y, self.U) + logdet)
        return _nll(self.y, self.K, self.U)

    def _optimize_theta(self, init: np.ndarray = None):
        jac_fn = jit(grad(self.nll))
        hess_fn = jit(hessian(self.nll))
        if init is None:
            init = self.kernel.thetas
        res = minimize(
            self.nll,
            jac=jac_fn,
            hess=hess_fn,
            x0=init,
            method='Newton-CG',
            options={
                'disp': 1
            }
        )
        print(res)
        self.kernel.thetas = np.array(res.x)
#
#
# class NewGPGrad(GP):
#     def __init__(self, alpha=1e-8, kernel: Kernel = RBF(), debug=True):
#         gkernel = GradKernel(kernel)
#         super(NewGPGrad, self).__init__(alpha, gkernel, debug)
#
#     def fit(self, x: np.ndarray, y: np.ndarray, dydx: np.ndarray, *args):
#         y = np.concatenate((y, dydx.T.flatten()))
#         super().fit(x, y)
#
#     def predict_(self, x: np.ndarray) -> float:
#         raise NotImplementedError()
#
#     def predict_var_(self, x: np.ndarray) -> float:
#         raise NotImplementedError()
#
#     def predict_mu(self, x: np.ndarray) -> np.ndarray:
#         r = self.kernel.k(self.x, x)
#         dr = self.kernel.dkdx1(self.x, x)
#         return np.concatenate((r, dr.T.flatten())) @ self.U
#
#     def predict_var(self, x: np.ndarray) -> np.ndarray:
#         k = self.kernel.k(self.x, x)
#         k_grad = self.kernel(self.x, x)
#         return k - k_grad.T @ solve(self.K, k_grad)


class GPGrad:
    alpha: float
    kernel: GradKernel
    x: np.ndarray
    dydx: np.ndarray
    y: np.ndarray
    K: np.ndarray
    U: np.ndarray

    def __init__(self, alpha=1e-8, kernel: Kernel = RBF(), debug=True):
        self.alpha = alpha
        self.kernel = GradKernel(kernel)
        self.x = None
        self.dydx = None
        self.y = None
        self.K = None
        self.U = None
        self.predict_mu = vmap(self.predict_)
        self.predict_dy_ = grad(self.predict_, argnums=0)
        self.predict_dy = vmap(self.predict_dy_)
        self.predict_var = vmap(self.predict_var_)

        if debug is False:
            # self.fit = method_jit(self.fit)
            self.predict_mu = method_jit(self.predict_mu)
            self.predict_dy = method_jit(self.predict_dy)
            self.predict_var = method_jit(self.predict_var)

    def fit(self, x: np.ndarray, y: np.ndarray, dydx: np.ndarray, optimize_thetas: bool = False):
        """
        Obtains the hyperparameters that maximizes the log-likelihood of the observations
        TODO: implement this; current implementation only saves x, y, and dydx without optimizing hyperparams

        :param x:
        :param y:
        :param dydx:
        :return:
        """
        self.x = x
        self.dydx = dydx
        # dy = np.concatenate([dydx[:, i] for i in range(dydx.shape[1])])
        self._fit(x, np.concatenate((y, dydx.T.flatten())))
        if optimize_thetas:
            self._optimize_theta()

    def _fit(self, x: np.ndarray, y: np.ndarray):
        self.y = y
        self.K = add_jitter(self.kernel(x, x), self.alpha)
        self.U = jit_solve(self.K, self.y)

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

    def predict_var_(self, x: np.ndarray) -> float:
        """
        Returns the variance of the prediction at x
        :param x:
        :return:
        """
        K_train = self.kernel.k(x, self.x)
        K = self.K[:len(self.x), :len(self.x)]
        return self.kernel.k(x, x) - K_train.T @ solve(K, K_train)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_mu(x), self.predict_dy(x)

    def nll(self, thetas):
        self.kernel.k.thetas = thetas
        self._fit(self.x, self.y)
        return _nll(self.y, self.K, self.U)

    def _optimize_theta(self, init: np.ndarray = None):
        jac_fn = jit(grad(self.nll))
        # hess_fn = jit(hessian(self.nll))
        if init is None:
            init = self.kernel.k.thetas
        res = minimize(
            self.nll,
            jac=jac_fn,
            # hess=hess_fn,
            x0=init,
            method='BFGS',
            options={
                'disp': 1
            }
        )
        print(res)
        self.kernel.thetas = np.array(res.x)

