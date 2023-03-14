from math import inf
import numpy as np
import cvxpy as cp

from tools.interpolation_conditions import square, interpolation_combination


def lyapunov_three_operator_splitting(beta, gamma, mu, L, rho):
    # Initialize
    x0, y0, y1, Ay1, Bx1, As, Bs, Ay0, Bx0, gx0, gx1 = list(np.eye(11))
    ws = np.zeros(11)
    gs = - As - Bs
    fx0, fx1 = list(np.eye(2))
    fs = np.zeros(2)

    # Run algorithm
    # y1 = J_{gamma A} (2 x1 - w0 - g1)
    x1 = y1 + gamma * Ay1 + gamma * Bx1 + gx1

    # x1 = J_{gamma B} (w0)
    w0 = x1 + gamma * Bx1

    # w1 = w0 - beta (x1 - y1)
    w1 = w0 - beta * (x1 - y1)

    # Lyapunov
    G = cp.Variable((7, 7), symmetric=True)
    F = cp.Variable((1,))
    list_of_cvxpy_constraints = []

    VG = np.array([x0 - ws, y0 - ws, w0 - ws, Ay0, Bx0, As, Bs]).T @ G @ np.array([x0 - ws, y0 - ws, w0 - ws, Ay0, Bx0, As, Bs])
    VG_plus = np.array([x1 - ws, y1 - ws, w1 - ws, Ay1, Bx1, As, Bs]).T @ G @ np.array([x1 - ws, y1 - ws, w1 - ws, Ay1, Bx1, As, Bs])
    VF = np.array([fx0 - fs]).T @ F
    VF_plus = np.array([fx1 - fs]).T @ F

    # Write problem
    list_of_points_A = [(ws, As, 0), (y0, Ay0, 0), (y1, Ay1, 0)]
    list_of_points_B = [(ws, Bs, 0), (x0, Bx0, 0), (x1, Bx1, 0)]
    list_of_points_f = [(ws, gs, fs), (x0, gx0, fx0), (x1, gx1, fx1)]

    matrix_combination1, _, dual1 = interpolation_combination(list_of_points_A, 0, inf, function_class="strongly monotone operator")
    matrix_combination2, _, dual2 = interpolation_combination(list_of_points_B, 0, L, function_class="cocoercive operator")
    matrix_combination3, vector_combination3, dual3 = interpolation_combination(list_of_points_f, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(VG_plus - rho * VG << matrix_combination1 + matrix_combination2 + matrix_combination3)
    list_of_cvxpy_constraints.append(VF_plus - rho * VF <= vector_combination3)
    list_of_cvxpy_constraints.append(dual1 >= 0)
    list_of_cvxpy_constraints.append(dual2 >= 0)
    list_of_cvxpy_constraints.append(dual3 >= 0)

    matrix_combination1, _, dual1 = interpolation_combination(list_of_points_A, 0, inf, function_class="strongly monotone operator")
    matrix_combination2, _, dual2 = interpolation_combination(list_of_points_B, 0, L, function_class="cocoercive operator")
    matrix_combination3, vector_combination3, dual3 = interpolation_combination(list_of_points_f, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(square(w1 - ws) - VG_plus << matrix_combination1 + matrix_combination2 + matrix_combination3)
    list_of_cvxpy_constraints.append(- VF_plus <= vector_combination3)
    list_of_cvxpy_constraints.append(dual1 >= 0)
    list_of_cvxpy_constraints.append(dual2 >= 0)
    list_of_cvxpy_constraints.append(dual3 >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=list_of_cvxpy_constraints)
    try:
        value = prob.solve(solver="MOSEK")
    except cp.error.SolverError:
        value = prob.solve(solver="SCS")
    return value
