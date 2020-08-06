import jax.numpy as np
from jax import jit, jacrev, jacfwd
from abc import ABC, abstractmethod


class Kernel(ABC):
    def __init__(self, name: str, debug=True):
        self.name = name

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        raise NotImplementedError()


class RBF(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
        super(RBF, self).__init__('RBF')
        self.dkdx1 = jacfwd(self.forward, argnums=0)
        self.dkdx2 = jacfwd(self.forward, argnums=1)
        self.dk2dx1dx2 = jacrev(self.dkdx1, argnums=1)

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        return self.forward(x1, x2, np.array([self.length_scale]))

    def forward(self, x1: np.ndarray, x2: np.ndarray, thetas: np.ndarray):
        assert thetas.shape == (1,)
        length_scale = thetas[0]

        if len(x1.shape) == 1:
            x1 = x1.reshape(-1, 1)
        if len(x2.shape) == 1:
            x2 = x2.reshape(-1, 1)

        d = np.linalg.norm(
            np.expand_dims(x1, 0) - np.expand_dims(x2, 1),
            axis=-1
        ) / length_scale
        return np.exp(-0.5 * (d * d))

