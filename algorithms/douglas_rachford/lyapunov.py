from math import inf
import numpy as np
import cvxpy as cp

from tools.interpolation_conditions import square, interpolation_combination


def lyapunov_douglas_rachford(beta, gamma, mu, L, rho):

    # Initialize
    x0, y0, y1, Ay1, Bx1, As, Ay0, Bx0 = list(np.eye(8))
    ws = np.zeros(8)
    Bs = - As

    # Run algorithm
    # y1 = J_{gamma A} (2 x1 - w0)
    x1 = y1 + gamma * Ay1 + gamma * Bx1

    # x1 = J_{gamma B} (w0)
    w0 = x1 + gamma * Bx1

    # w1 = w0 - beta (x1 - y1)
    w1 = w0 - beta * (x1 - y1)

    # Lyapunov
    G = cp.Variable((7, 7), symmetric=True)
    list_of_cvxpy_constraints = []

    VG = np.array([x0 - ws, y0 - ws, w0 - ws, Ay0, Bx0, As, Bs]).T @ G @ np.array([x0 - ws, y0 - ws, w0 - ws, Ay0, Bx0, As, Bs])
    VG_plus = np.array([x1 - ws, y1 - ws, w1 - ws, Ay1, Bx1, As, Bs]).T @ G @ np.array([x1 - ws, y1 - ws, w1 - ws, Ay1, Bx1, As, Bs])

    # Write problem
    list_of_points_A = [(ws, As, 0), (y0, Ay0, 0), (y1, Ay1, 0)]
    list_of_points_B = [(ws, Bs, 0), (x0, Bx0, 0), (x1, Bx1, 0)]

    matrix_combination1, _, dual1 = interpolation_combination(list_of_points_A, 0, L, function_class="lipschitz strongly monotone operator")
    matrix_combination2, _, dual2 = interpolation_combination(list_of_points_B, mu, inf, function_class="strongly monotone operator")
    list_of_cvxpy_constraints.append(VG_plus - rho * VG << matrix_combination1 + matrix_combination2)
    list_of_cvxpy_constraints.append(dual1 >= 0)
    list_of_cvxpy_constraints.append(dual2 >= 0)

    matrix_combination1, _, dual1 = interpolation_combination(list_of_points_A, 0, L, function_class="lipschitz strongly monotone operator")
    matrix_combination2, _, dual2 = interpolation_combination(list_of_points_B, mu, inf, function_class="strongly monotone operator")
    list_of_cvxpy_constraints.append(square(w1 - ws) - VG_plus << matrix_combination1 + matrix_combination2)
    list_of_cvxpy_constraints.append(dual1 >= 0)
    list_of_cvxpy_constraints.append(dual2 >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=list_of_cvxpy_constraints)
    try:
        value = prob.solve(solver="MOSEK")
    except cp.error.SolverError:
        value = prob.solve(solver="SCS")
    return value
