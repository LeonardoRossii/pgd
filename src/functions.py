import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class TestFunction:
    """
    Representation of a benchmark function
    used to test optimization algorithms.

    This class stores a callable objective
    function together with metadata describing
    its search domain and the coordinates of 
    its known global minima.

    :param fn: Objective function to minimize
    :type fn: Callable
    :param search_domain: Bounds for each dimension of the search space
    :type search_domain: tuple[tuple[float, float], ...]
    :param global_minima: Global minima coordinates
    :type global_minima: tuple[tuple[float, ...], ...]
    """

    fn: Callable
    search_domain: tuple[tuple[float, float], ...]
    global_minima: tuple[tuple[float, ...], ...]

    def value(self, z: jax.Array) -> jax.Array:
        """
        Evaluate the value of the
        test function at point z.

        :param z: Input vector
        :type z: jax.Array
        :return: Function value at z
        :rtype: jax.Array
        """
        return self.fn(z)


    def gradient(self, z: jax.Array) -> jax.Array:
        """
        Evaluate the gradient of the
        test function at point z.

        :param z: Input vector
        :type z: jax.Array
        :return: Gradient at z
        :rtype: jax.Array
        """
        return jax.grad(self.fn)(z)
    

def himmelblau(z):
    """
    Evaluate Himmelblau's function at point z.

    :param z: Input vector.
    :type z: jax.Array
    :return: Function value at z.
    :rtype: jax.Array
    """
    x, y = z
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def three_hump_camel(z):
    """
    Evaluate Three-Hump Camel function at point z.

    :param z: Input vector.
    :type z: jax.Array
    :return: Function value at z.
    :rtype: jax.Array
    """
    x, y = z
    return 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2


Himmelblau = TestFunction(
    fn = himmelblau,
    search_domain = ((-5.0, 5.0), (-5.0, 5.0)),
    global_minima = (
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    )
)

ThreeHumpCamel = TestFunction(
    fn = three_hump_camel,
    search_domain = ((-5.0, 5.0), (-5.0, 5.0)),
    global_minima = ((0.0, 0.0),)
)