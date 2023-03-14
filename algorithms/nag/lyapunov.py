import numpy as np
import cvxpy as cp

from tools.interpolation_conditions import interpolation_combination


def lyapunov_accelerated_gradient_strongly_convex(beta, gamma, mu, L, rho):

    # Initialize
    x0, g0, x1, g1, g2, gy = list(np.eye(6))
    xs = np.zeros(6)
    gs = np.zeros(6)
    f0, f1, f2, fy = list(np.eye(4))
    fs = np.zeros(4)

    # Run algorithm
    y = x1 + beta * (x1 - x0)
    x2 = y - gamma * gy

    # Lyapunov
    G = cp.Variable((4, 4), symmetric=True)
    F = cp.Variable((2,))
    list_of_cvxpy_constraints = []

    VG = np.array([x0 - xs, g0, x1 - xs, g1]).T @ G @ np.array([x0 - xs, g0, x1 - xs, g1])
    VG_plus = np.array([x1 - xs, g1, x2 - xs, g2]).T @ G @ np.array([x1 - xs, g1, x2 - xs, g2])
    VF = np.array([f0 - fs, f1 - fs]).T @ F
    VF_plus = np.array([f1 - fs, f2 - fs]).T @ F

    # Write problem
    list_of_points = [(xs, gs, fs), (x0, g0, f0), (x1, g1, f1), (x2, g2, f2), (y, gy, fy)]

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(VG_plus - rho * VG << matrix_combination)
    list_of_cvxpy_constraints.append(VF_plus - rho * VF <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points, mu, L, function_class="smooth strongly convex")
    list_of_cvxpy_constraints.append(- VG_plus << matrix_combination)
    list_of_cvxpy_constraints.append(f2 - fs - VF_plus <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    prob = cp.Problem(objective=cp.Minimize(0), constraints=list_of_cvxpy_constraints)
    try:
        value = prob.solve(solver="MOSEK")
    except cp.error.SolverError:
        value = prob.solve(solver="SCS")
    return value
