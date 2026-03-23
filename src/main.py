import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functions import Himmelblau, ThreeHumpCamel
from constraints import DiskConstraintSet2D
from solver import ProjectedGradientDescent
from plot import plot_optimization


def run_experiment(func, solver, constraint, key, label):
    """
    Run an optimization experiment with the given function, solver, and constraint.

    :param func: The objective function to optimize.
    :param solver: The optimization solver to use.
    :param constraint: The constraint set for the optimization.
    :param key: JAX random key for initialization.
    :param label: Label for the experiment.
    """

    bounds = jnp.array(func.search_domain)
    z_init = jax.random.uniform(key, shape=(2,), minval=bounds[:, 0], maxval=bounds[:, 1])

    solution, history = solver.solve(func, constraint, z_init)

    print(f"=== {label} ===")
    print(f"Initial point: {z_init}")
    print(f"Solution: {solution}")
    print(f"Function value at solution: {func.value(solution):.6f}")
    print(f"Number of iterations: {len(history) - 1}")
    print()

    plot_optimization(func, constraint, history)
    plt.show()


if __name__ == "__main__":

    # Himmelblau with unit disk constraint
    disk_constraint = DiskConstraintSet2D()
    solver = ProjectedGradientDescent(alpha=0.1, max_iter=1000, tol=1e-8)
    key = jax.random.PRNGKey(42)
    run_experiment(Himmelblau, solver, disk_constraint, key, "Himmelblau Function with Disk Constraint")

    # Three Hump Camel with unit disk constraint
    disk_constraint = DiskConstraintSet2D()
    solver = ProjectedGradientDescent(alpha=0.1, max_iter=1000, tol=1e-8)
    key = jax.random.PRNGKey(123)
    run_experiment(ThreeHumpCamel, solver, disk_constraint, key, "Three Hump Camel Function with Disk Constraint")
