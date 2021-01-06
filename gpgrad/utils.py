from jax import jit, partial


def method_jit(fun):
    """
    JAX JIT a method. Assumes first argument, `self`, is not JITable
    :param fun: method
    :return:
    """
    return jit(fun, static_argnums=(0,))
