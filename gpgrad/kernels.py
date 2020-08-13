import jax.numpy as np
from jax import jit, jacrev, jacfwd, vmap, grad
from abc import ABC, abstractmethod


class Kernel(ABC):
    def __init__(self, thetas: np.ndarray, name: str, debug=True):
        self.thetas = thetas
        self.name = name
        self.debug = debug
        self.forward = vmap(
            vmap(
                self.forward_,
                in_axes=(0, None, None)
            ),
            in_axes=(None, 0, None)
        )
        if debug is False:
            self.forward = jit(self.forward)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.forward(x1, x2, self.theta)

    @abstractmethod
    def forward_(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        raise NotImplementedError()


class GradKernel:
    def __init__(self, kernel: Kernel):
        self.k = kernel
        self.dkdx1 = vmap(
            vmap(
                grad(self.k.forward_, argnums=0),
                in_axes=(0, None, None)
            ),
            in_axes=(None, 0, None)
        )
        self.dkdx2 = vmap(
            vmap(
                grad(self.k.forward_, argnums=1),
                in_axes=(0, None, None)
            ),
            in_axes=(None, 0, None)
        )
        self.dk2dx1dx2 = vmap(
            vmap(
                jacrev(
                    grad(self.k.forward_, argnums=0),
                    argnums=1
                ),
                in_axes=(0, None, None)
            ),
            in_axes=(None, 0, None)
        )
        if self.k.debug is False:
            self.dkdx1 = jit(self.dkdx1)
            self.dkdx2 = jit(self.dkdx2)
            self.dk2dx1dx2 = jit(self.dk2dx1dx2)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.forward(x1, x2, self.k.thetas)

    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        K = self.k.forward(x1, x2, thetas)
        dx2 = self.dkdx2(x1, x2, thetas)
        upper = np.concatenate([dx2[:, :, i] for i in range(dx2.shape[-1])], axis=1)
        dx1 = self.dkdx1(x1, x2, thetas)
        left = np.concatenate([dx1[:, :, i] for i in range(dx1.shape[-1])], axis=0)
        dx1dx2 = self.dk2dx1dx2(x1, x2, thetas)
        dx2_concatenated = np.concatenate([
            dx1dx2[:, :, :, i]
            for i in range(dx1dx2.shape[-1])
        ], axis=1)
        hess = np.concatenate([
            dx2_concatenated[:, :, i]
            for i in range(dx2_concatenated.shape[-1])
        ], axis=0)

        # form the overall covariance matrix
        # [
        #     [K,       dK/dx2      ],
        #     [dK/dx1,  dK^2/dx1dx2 ]
        # ]
        return np.concatenate([
            np.concatenate([K, upper], axis=1),
            np.concatenate([left, hess], axis=1)],
            axis=0
        )


class RBF(Kernel):
    def __init__(self, length_scale=1.0, debug=True):
        self.length_scale = length_scale
        super(RBF, self).__init__(np.array([length_scale]), 'RBF', debug=debug)

    def forward_(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        assert thetas.shape == (1,)
        length_scale = thetas[0]

        dist_sq = np.vdot(x1, x1) + np.vdot(x2, x2) - 2.0 * np.vdot(x1, x2)
        return np.exp(-0.5 * dist_sq / length_scale / length_scale)

