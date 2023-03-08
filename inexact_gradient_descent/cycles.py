from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.primitive_steps import inexact_gradient_step


def cycle_inexact_gradient_descent(L, mu, alpha, beta, n):
    """
    Verify existence or not of cycle on inexact gradient descent.

    Args:
        mu (float): strong convexity parameter
        L (float): smoothness parameter
        alpha (float): step-size
        beta (float): oracle relative error
        n (float): number of steps / length of searched cycle

    Returns (cvxpy.Problem): the cvxpy problem to solve

    """

    # gamma is alpha
    # epsilon is beta
    gamma = alpha
    epsilon = beta

    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Then define the starting point x0 of the algorithm
    # as well as corresponding inexact gradient and function value g0 and f0
    x0 = problem.set_initial_point()
    x1, dx, fx = inexact_gradient_step(x0, func, gamma=gamma, epsilon=epsilon, notion='relative')

    # Set the initial constraint that is the distance between f0 and f_*
    problem.set_initial_condition((x1 - x0)**2 >= 1)

    # Run n steps of the inexact gradient method
    x = x1
    for i in range(n-1):
        x, dx, fx = inexact_gradient_step(x, func, gamma=gamma, epsilon=epsilon, notion='relative')

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(-(x - x0)**2)

    return problem
