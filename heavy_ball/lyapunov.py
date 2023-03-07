from tqdm import tqdm

from math import sqrt, inf
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def inner_product(u, v):
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    return inner_product(u, u)


def interpolation_i_j(pointi, pointj, mu=.1, L=1):
    xi, gi, fi = pointi
    xj, gj, fj = pointj

    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
    F = fj - fi

    return G, F


def interpolation(list_of_points):
    list_of_matrices = []
    list_of_vectors = []

    for i, pointi in enumerate(list_of_points):
        for j, pointj in enumerate(list_of_points):
            if i != j:
                G, F = interpolation_i_j(pointi, pointj)
                list_of_matrices.append(G)
                list_of_vectors.append(F)

    return list_of_matrices, list_of_vectors


def interpolation_combination(list_of_points):
    list_of_matrices, list_of_vectors = interpolation(list_of_points)
    nb_constraints = len(list_of_matrices)
    dual = cp.Variable((nb_constraints,))
    matrix_combination = cp.sum([dual[i] * list_of_matrices[i] for i in range(nb_constraints)])
    vector_combination = cp.sum([dual[i] * list_of_vectors[i] for i in range(nb_constraints)])

    return matrix_combination, vector_combination, dual


def lyap_hb(beta = .9, gamma = 1, mu=.1, L=1, rho = .999):

    # Initialize
    x0, g0, x1, g1, g2 = list(np.eye(5))
    xs = np.zeros(5)
    gs = np.zeros(5)
    f0, f1, f2 = list(np.eye(3))
    fs = np.zeros(3)

    # Run algorithm
    x2 = x1 + beta * (x1 - x0) - gamma * g1

    # Lyapunov
    G = cp.Variable((4, 4), symmetric=True)
    F = cp.Variable((2,))
    list_of_cvxpy_constraints = [cp.trace(G) == 1]

    VG = np.array([x0 - xs, g0, x1 - xs, g1]).T @ G @ np.array([x0 - xs, g0, x1 - xs, g1])
    VG_plus = np.array([x1 - xs, g1, x2 - xs, g2]).T @ G @ np.array([x1 - xs, g1, x2 - xs, g2])
    VF = np.array([f0 - fs, f1 - fs]).T @ F
    VF_plus = np.array([f1 - fs, f2 - fs]).T @ F

    # Write problem
    list_of_points = [(xs, gs, fs), (x0, g0, f0), (x1, g1, f1), (x2, g2, f2)]

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points=list_of_points)
    list_of_cvxpy_constraints.append(VG_plus - rho * VG << matrix_combination)
    list_of_cvxpy_constraints.append(VF_plus - rho * VF <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    matrix_combination, vector_combination, dual = interpolation_combination(list_of_points=list_of_points)
    list_of_cvxpy_constraints.append(- VG_plus << matrix_combination)
    list_of_cvxpy_constraints.append(f2 - fs - VF_plus <= vector_combination)
    list_of_cvxpy_constraints.append(dual >= 0)

    # 0 if there exists a Lyapunov
    # inf otherwise
    value = cp.Problem(objective=cp.Minimize(0), constraints=list_of_cvxpy_constraints).solve()
    return value


def conditional_bisection_search(mu, L, nb_points, precision):
    if mu > 0:
        rho = .999
    else:
        rho = 1
    betas = np.linspace(0, 1, nb_points, endpoint=False)
    gammas_min_lyap = np.zeros_like(betas) 
    gammas_max_lyap = 2 * (1 + betas) / L
    
    gammas_lyap = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        gamma_min_lyap = gammas_min_lyap[it]
        gamma_max_lyap = gammas_max_lyap[it]

        while gamma_max_lyap - gamma_min_lyap > precision:
            gamma = (gamma_min_lyap + gamma_max_lyap) / 2
            lyap = lyap_hb(beta=beta, gamma=gamma, mu=mu, L=L, rho=rho)
            if lyap != inf:
                gamma_min_lyap = gamma
            else:
                gamma_max_lyap = gamma

        gammas_lyap.append(gamma_max_lyap)

    with open("hb_lyap_kappa={}.txt".format(mu / L), "w") as f:
        f.write("alpha\tbeta\n")
        for gamma, beta in zip(gammas_lyap, betas):
            f.write("{}\t{}\n".format(gamma, beta))

    plt.figure(figsize=(15, 9))
    plt.plot(gammas_lyap, betas)
    plt.savefig("hb_lyap_kappa={}.png".format(mu / L))


if __name__ == "__main__":
    conditional_bisection_search(mu=.1, L=1, nb_points=10, precision=.01)
