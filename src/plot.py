import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import jax
import jax.numpy as jnp
import numpy as np

from functions import TestFunction
from constraints import ConstraintSet

def plot_optimization(func: TestFunction, constraint: ConstraintSet, history: jax.Array):
    """
    Visualize the optimization trajectory over the objective function contour plot.

    :param func: Test function to visualize.
    :param constraint: Constraint set.
    :param history: Array of iterates from the optimization.
    :return: Figure and axes objects.
    """

    xlim = func.search_domain[0]
    ylim = func.search_domain[1]

    xs = jnp.linspace(*xlim, 100)
    ys = jnp.linspace(*ylim, 100)

    # Create a grid of points over the search domain
    x_jnp, y_jnp = jnp.meshgrid(xs, ys)
    grid_jnp = jnp.stack([x_jnp, y_jnp], axis=-1)

    # Evaluate the function on the grid using JAX's vectorization
    z_jnp = jax.vmap(jax.vmap(func.value))(grid_jnp)

    # Convert JAX arrays to NumPy for plotting
    x_np = np.array(x_jnp) 
    y_np = np.array(y_jnp)
    z_np = np.array(z_jnp)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the objective function level curves
    contour = ax.contourf(x_np, y_np, z_np, levels=50, alpha=0.8)
    ax.contour(x_np, y_np, z_np, levels=50, colors=[(1,1,1)], linewidths=0.3, alpha=0.5)
    plt.colorbar(contour, ax=ax, label="f(x, y)")

    # Plot the constraint set boundary
    patch = Circle(constraint.disk_center, constraint.disk_radius, fill=False, linewidth=2.5, 
                   edgecolor=(0.99, 0.99, 0.99), linestyle=(0, (5, 3)), label="Constraint Set Boundary")
    ax.add_patch(patch)

    # Plot the optimization trajectory and key points
    history_np = np.array(history)
    ax.plot(history_np[:, 0], history_np[:, 1], color=(1,0,0), linestyle="-", 
            marker=".", linewidth=1.5, markersize=4, zorder=5, label="Optimization Trajectory")

    ax.scatter(*history_np[0], c=[(1,0,0)], s=100, zorder=6, marker="s",
               edgecolors=[(1,0,0)], linewidths=1.5, label="Initial Point")
    
    ax.scatter(*history_np[1], c=[(1,0,0)], s=100, zorder=6, marker="o",
               edgecolors=[(1,0,0)], linewidths=1.5, label="Initial Projection")

    ax.scatter(*history_np[-1], c=[(1,0,0)], s=150, marker="*", zorder=8, 
               edgecolors=[(1,0,0)], linewidths=1.5, label="PGD Solution")
    
    # Plot the known global minima
    minima = np.array(func.global_minima)
    ax.scatter(minima[:, 0], minima[:, 1], c=[(0.99, 0.99, 0.99)], s=200, marker="X", 
               zorder=7, edgecolors=[(0.99, 0.99, 0.99)], linewidths=0.5, label="Global Minimum")
     
    ax.set(xlim=xlim, ylim=ylim, xlabel="x", ylabel="y", aspect="equal")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax