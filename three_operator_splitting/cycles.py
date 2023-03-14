from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from PEPit.operators import CocoerciveOperator
from PEPit.operators import MonotoneOperator
from PEPit.primitive_steps import proximal_step


def cycle_three_operator_splitting(L, mu, gamma, beta, n):

    # Instantiate PEP
    problem = PEP()

    # Declare a monotone operator
    A = problem.declare_function(MonotoneOperator)
    B = problem.declare_function(CocoerciveOperator, beta=L)
    C = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)

    # Then define the starting point w0
    w0 = problem.set_initial_point()

    # Compute one step of the Three Operator Splitting starting from w0
    x1, _, _ = proximal_step(w0, B, gamma)
    y1, _, _ = proximal_step(2 * x1 - w0 - C.gradient(x1), A, gamma)
    w1 = w0 - beta * (x1 - y1)

    # Set the initial constraint that is the distance between w0 and w1
    problem.set_initial_condition((w1 - w0) ** 2 >= 1)

    w = w1
    for i in range(n-1):
        x, _, _ = proximal_step(w, B, gamma)
        y, _, _ = proximal_step(2 * x - w - C.gradient(x), A, gamma)
        w = w - beta * (x - y)

    # Set the performance metric to the distance between z0 and z1
    problem.set_performance_metric(-(w - w0) ** 2)

    return problem
