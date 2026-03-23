import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


class ConstraintSet(ABC):
    """
    Abstract base class for constraint sets used in optimization.

    A constraint defines a feasible set together with a projection
    operator that maps any point onto that set. Subclasses must
    implement the :meth:`project` method.
    """

    @abstractmethod
    def project(self, z: jax.Array) -> jax.Array:
        """
        Project a point onto the constraint set.

        :param z: Point to be projected.
        :type z: jax.Array
        :return: Projected point lying in the constraint set.
        :rtype: jax.Array
        :raises NotImplementedError: If the method is not implemented
            in a subclass.
        """
        raise NotImplementedError
    

from dataclasses import dataclass
from dataclasses import field
    
@dataclass(frozen=True)
class DiskConstraintSet2D(ConstraintSet):
    """
    Constraint set representing a Euclidean disk in 2D.
    

    :param radius: Radius of the disk.
    :type radius: float
    :param center: Center of the disk.
    :type center: jax.Array
    """

    disk_radius: float = 1.0
    disk_center: jax.Array = field(default_factory=lambda: jnp.zeros(2))

    def project(self, z: jax.Array) -> jax.Array:
        """
        Project a vector onto the disk constraint set.

        If the vector lies outside the disk, it is scaled back
        to the boundary along the radial direction from the center.
        Otherwise, it is returned unchanged.

        :param z: Input vector.
        :type z: jax.Array
        :return: Projected vector whose norm is at most ``disk_radius``.
        :rtype: jax.Array
        """

        dist = jnp.linalg.norm(z - self.disk_center)
        scale = jnp.minimum(1.0, self.disk_radius / (dist + 1e-12))
        z_proj = self.disk_center + scale * (z - self.disk_center)

        return z_proj
    

@dataclass(frozen=True)
class BoxConstraintSet2D(ConstraintSet):
    """
    Constraint set representing a box in 2D defined by lower and upper bounds.

    :param lo: Lower bound for each dimension.
    :type lo: jax.Array
    :param hi: Upper bound for each dimension.
    :type hi: jax.Array
    """

    lo: jax.Array = field(default_factory=lambda: jnp.array([-1.0, -1.0]))
    hi: jax.Array = field(default_factory=lambda: jnp.array([1.0, 1.0]))

    def project(self, z: jax.Array) -> jax.Array:
        """
        Project a vector onto the box constraint set.

        Each component of the input vector is clipped to lie within
        the corresponding lower and upper bounds.

        :param z: Input vector.
        :type z: jax.Array
        :return: Projected vector lying within the box defined by ``lo`` and ``hi``.
        :rtype: jax.Array
        """
        return jnp.clip(z, self.lo, self.hi)
