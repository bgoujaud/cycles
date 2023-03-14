from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def cycle_accelerated_gradient_strongly_convex(mu, L, gamma, beta, n):
    """
    Verify existence or not of cycle on fast gradient method.

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

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    x1 = problem.set_initial_point()

    # Set the initial constraint that is a well-chosen distance between x0 and x^*
    problem.set_initial_condition((x1 - x0)**2 >= 1)

    # Run n steps of the fast gradient method
    x_old = x0
    x_new = x1
    y = x_new + beta * (x_new - x_old)
    for i in range(n):
        x_old = x_new
        x_new = y - gamma * func.gradient(y)
        y = x_new + beta * (x_new - x_old)

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(- ((x_new - x1) ** 2 + (x_old - x0) ** 2))

    return problem
