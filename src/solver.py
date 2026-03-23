import jax
import jax.numpy as jnp
from functions import TestFunction
from constraints import ConstraintSet

class ProjectedGradientDescent:
    """
    Projected gradient descent solver for constrained optimization problems.
    This optimizer minimizes an objective function over a feasible set by
    combining gradient steps with projection onto the constraint set.
    A backtracking line search is used to adapt the step size.
    """

    def __init__(self, alpha: float =0.01, max_iter: int =10000, tol: float =1e-6, max_backtracking_steps: int =50):
        """
        Initialize the projected gradient descent solver.

        :param alpha: Initial step size used in the line search.
        :type alpha: float
        :param max_iter: Maximum number of optimization iterations.
        :type max_iter: int
        :param tol: Stopping tolerance
        :type tol: float
        :param max_backtracking_steps: Maximum number of backtracking reductions.
        :type max_backtracking_steps: int
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.max_backtracking_steps = max_backtracking_steps


    def back_tracking_line_search(self, func: TestFunction, constraint: ConstraintSet, z: jax.Array, grad: jax.Array, beta: float =0.5):
        """
        Perform backtracking line search for a projected gradient step.

        Starting from the current step size, this method
        repeatedly reduces the step size by a factor until the
        sufficient decrease condition is satisfied or the maximum 
        number of backtracking steps is reached.

        :param func: Objective function to minimize.
        :type func: TestFunction
        :param constraint: Constraint set with projection operator.
        :type constraint: ConstraintSet
        :param z: Current iterate.
        :type z: jax.Array
        :param grad: Gradient of the objective.
        :type grad: jax.Array
        :param beta: Multiplicative reduction factor for the step size.
        :type beta: float
        :return: Accepted step size and projected next iterate.
        :rtype: tuple[float, jax.Array]
        """

        # Get current step size
        alpha = self.alpha
        
        fz = func.value(z)

        for _ in range(self.max_backtracking_steps):

            # Compute the projected gradient step
            z_new = constraint.project(z - alpha * grad)
            diff = z_new - z

            # Compute the quadratic approximation of the objective
            q = fz + jnp.dot(grad, diff) + (jnp.dot(diff, diff) / (2.0 * alpha))
            
            # If the sufficient decrease condition is satisfied:
            if func.value(z_new) <= q:

                # Update current step size and new point
                self.alpha = alpha
                return alpha, z_new
            
            # Otherwise reduce the step size
            alpha *= beta
        
        z_new = constraint.project(z - alpha * grad)
        self.alpha = alpha
        
        return alpha, z_new


    def solve(self, func: TestFunction, constraint: ConstraintSet, z_init: jax.Array, use_back_tracking_line_search: bool =True):
        """
        Solve the constrained optimization problem.

        The initial point is first projected onto the feasible set. At each
        iteration, a projected gradient step is computed using backtracking
        line search. The method stops when the distance between consecutive
        iterates falls below the specified tolerance or when the maximum
        number of iterations is reached.
    
        :param func: Objective function to minimize.
        :type func: TestFunction
        :param constraint: Constraint set.
        :type constraint: ConstraintSet
        :param z_init: Initial point.
        :type z_init: jax.Array
        :param use_back_tracking_line_search: Whether to use backtracking line search (default: True).
        :type use_back_tracking_line_search: bool
        :return: Final iterate and optimization history.
        :rtype: tuple[jax.Array, jax.Array]
        """

        # To store the optimization history (for visualization)
        history = [z_init]

        # Project the initial point onto the constraint set
        z = constraint.project(z_init)

        history.append(z)

        for _ in range(self.max_iter):

            if use_back_tracking_line_search:
                _, z_new = self.back_tracking_line_search(func, constraint, z, func.gradient(z))
            else:
                z_new = constraint.project(z - self.alpha * func.gradient(z))

            # Check for convergence
            if jnp.linalg.norm(z_new - z) < self.tol:
                z = z_new
                history.append(z)
                break

            # Update
            z = z_new
            history.append(z)
        
        return z, jnp.stack(history)