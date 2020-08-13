import jax.numpy as np
from jax import jit, jacrev, jacfwd, vmap
from abc import ABC, abstractmethod


class Kernel(ABC):
    def __init__(self, name: str, debug=True):
        self.name = name

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        raise NotImplementedError()


# def pairwise_dist(x1: np.ndarray, x2: np.ndarray):



class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
        super(RBF, self).__init__('RBF')

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.forward(x1, x2, np.array([self.length_scale]))

    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        assert thetas.shape == (1,)
        length_scale = thetas[0]

        # d = np.linalg.norm(
        #     np.expand_dims(x1, 0) - np.expand_dims(x2, 1),
        #     axis=-1
        # ) / length_scale
        # return np.exp(-0.5 * (d * d))
        d = (np.expand_dims(x1, 0) - np.expand_dims(x2, 1)) / length_scale
        d = d * d
        d = np.sum(d, axis=-1)
        return np.exp(-0.5 * d)


class RBFGrad(RBF):
    def __init__(self, length_scale=1.0):
        super(RBFGrad, self).__init__(length_scale)
        self.dkdx1 = jacfwd(super(RBFGrad, self).forward, argnums=0)
        self.dkdx2 = jacfwd(super(RBFGrad, self).forward, argnums=1)
        self.dk2dx1dx2 = vmap(
            jacfwd(jacrev(super(RBFGrad, self).forward, argnums=0), argnums=1),
            in_axes=(0, None, None),
            out_axes=0
        )

    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        K = super().forward(x1, x2, thetas)
        dx2 = self.dkdx2(x1, x2, thetas).sum(-2)
        upper = np.concatenate([dx2[:, :, i] for i in range(dx2.shape[-1])], axis=1)
        dx1 = self.dkdx1(x1, x2, thetas).sum(-2)
        left = np.concatenate([dx1[:, :, i] for i in range(dx1.shape[-1])], axis=0)
        dx1dx2 = self.dk2dx1dx2(x1, x2, thetas).sum(2).sum(-2)
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
