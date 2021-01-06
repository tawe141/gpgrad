import jax.numpy as np
from jax import jit, jacrev, jacfwd, vmap, grad, partial
from abc import ABC, abstractmethod
from .utils import method_jit


class Kernel(ABC):
    def __init__(self, thetas: np.ndarray, name: str, debug=True):
        self.thetas = thetas
        self.name = name
        self.debug = debug
        # two different forward functions: forward_b is vectorized for `x2` while forward is vectorized for both `x`'s.
        # forward_b is useful for the GPGrad implementation
        self.forward_b = vmap(self.forward_, in_axes=(None, 0, None))
        self.forward_a = vmap(self.forward_, in_axes=(0, None, None))

        self.forward = vmap(
            self.forward_b,
            in_axes=(0, None, None)
        )
        if debug is False:
            self.forward_b = method_jit(self.forward_b)
            self.forward_a = method_jit(self.forward_a)
            self.forward = method_jit(self.forward)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        if len(x1.shape) == 1 and len(x2.shape) == 2:
            return self.forward_b(x1, x2, self.thetas)
        elif len(x2.shape) == 1 and len(x1.shape) == 2:
            return self.forward_a(x1, x2, self.thetas)
        else:
            return self.forward(x1, x2, self.thetas)

    @abstractmethod
    def forward_(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        """
        Forward pass of kernel function for two individual samples `x1` and `x2`. Becomes vectorized via JAX.
        Output must be scalar.

        :param x1:
        :param x2:
        :param thetas:
        :return:
        """
        raise NotImplementedError()


class GradKernel:
    """
    Based largely on the report by Sandia found here:
    https://core.ac.uk/download/pdf/192883491.pdf
    """
    def __init__(self, kernel: Kernel):
        self.k = kernel

        self.dkdx1_ = grad(self.k.forward_, argnums=0)
        self.dkdx1_b = vmap(self.dkdx1_, in_axes=(None, 0, None))
        self.dkdx1_a = vmap(self.dkdx1_, in_axes=(0, None, None))
        self.dkdx1 = vmap(
            self.dkdx1_b,
            in_axes=(0, None, None)
        )

        self.dkdx2_ = grad(self.k.forward_, argnums=1)
        self.dkdx2_b = vmap(self.dkdx2_, in_axes=(None, 0, None))
        self.dkdx2 = vmap(
            self.dkdx2_b,
            in_axes=(0, None, None)
        )
        self.dk2dx1dx2_ = jacrev(
            grad(self.k.forward_, argnums=0),
            argnums=1
        )
        self.dk2dx1dx2 = vmap(
            vmap(
                self.dk2dx1dx2_,
                in_axes=(0, None, None)
            ),
            in_axes=(None, 0, None)
        )
        if self.k.debug is False:
            self.forward = method_jit(self.forward)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.forward(x1, x2, self.k.thetas)

    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        dim = x1.shape[1]
        assert dim == x2.shape[1]

        K = self.k.forward(x1, x2, thetas)
        dx2 = self.dkdx2(x1, x2, thetas)
        # upper = np.concatenate([dx2[:, i, :] for i in range(dx2.shape[1])], axis=1)
        upper = np.concatenate([dx2[:, :, i] for i in range(dx2.shape[2])], axis=1)
        dx1 = self.dkdx1(x1, x2, thetas)
        left = np.concatenate([dx1[:, :, i] for i in range(dx1.shape[2])], axis=0)
        # left = -np.transpose(upper)
        dx1dx2 = self.dk2dx1dx2(x1, x2, thetas)

        hess = np.block([
            [dx1dx2[:, :, i, j] for j in range(dim)]
            for i in range(dim)
        ])

        # form the overall covariance matrix
        # [
        #     [K,       dK/dx2      ],
        #     [dK/dx1,  dK^2/dx1dx2 ]
        # ]
        return np.block([
            [K, upper],
            [left, hess]
        ])


class RBF(Kernel):
    def __init__(self, length_scale=1.0, debug=True):
        self.length_scale = length_scale
        super(RBF, self).__init__(np.array([length_scale]), 'RBF', debug=debug)

    # @partial(jit, static_argnums=(0,)
    def forward_(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        assert thetas.shape == (1,)
        length_scale = thetas[0]

        dist_sq = np.vdot(x1, x1) + np.vdot(x2, x2) - 2.0 * np.vdot(x1, x2)
        # dist_sq = np.dot(x1-x2, x1-x2)
        return np.exp(-0.5 * dist_sq / length_scale / length_scale)

