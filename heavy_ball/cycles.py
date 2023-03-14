from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def cycle_heavy_ball_momentum(mu, L, gamma, beta, n):
    """
    Verify existence or not of cycle on heavy ball method.

    Args:
        mu (float): strong convexity parameter
        L (float): smoothness parameter
        gamma (float): step-size
        beta (float): momentum parameter
        n (float): number of steps / length of searched cycle

    Returns (cvxpy.Problem): the cvxpy problem to solve

    """

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()

    # Then define the starting point x1 of the algorithm as well as corresponding function value f1
    x1 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((x1 - x0) ** 2 >= 1)

    # Run one step of the heavy ball method
    x_new = x1
    x_old = x0

    for _ in range(n):
        x_next = x_new - gamma * func.gradient(x_new) + beta * (x_new - x_old)
        x_old = x_new
        x_new = x_next

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(- ((x_new - x1) ** 2 + (x_old - x0) ** 2))

    return problem
