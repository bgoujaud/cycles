from math import inf
import cvxpy as cp


def inner_product(u, v):
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    return inner_product(u, u)


def smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L):
    xi, gi, fi = pointi
    xj, gj, fj = pointj

    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
    F = fj - fi

    return G, F


def lipschitz_operator_interpolation_i_j(pointi, pointj, L):
    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L ** 2 * square(xi - xj)
    F = 0

    return G, F


def strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu):
    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = mu * square(xi - xj) - inner_product(gi - gj, xi - xj)
    F = 0

    return G, F


def cocoercive_operator_interpolation_i_j(pointi, pointj, L):
    xi, gi, _ = pointi
    xj, gj, _ = pointj

    G = square(gi - gj) - L * inner_product(xi - xj, gi - gj)
    F = 0

    return G, F


def interpolation(list_of_points, mu, L, function_class):
    list_of_matrices = []
    list_of_vectors = []

    for i, pointi in enumerate(list_of_points):
        for j, pointj in enumerate(list_of_points):
            if i != j:
                if function_class == "smooth strongly convex":
                    G, F = smooth_strongly_convex_interpolation_i_j(pointi, pointj, mu, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)
                elif function_class == "lipschitz strongly monotone operator":
                    G, F = lipschitz_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "strongly monotone operator":
                    assert L == inf
                    G, F = strongly_monotone_operator_interpolation_i_j(pointi, pointj, mu)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

                elif function_class == "cocoercive operator":
                    assert mu == 0
                    G, F = cocoercive_operator_interpolation_i_j(pointi, pointj, L)
                    list_of_matrices.append(G)
                    list_of_vectors.append(F)

    return list_of_matrices, list_of_vectors


def interpolation_combination(list_of_points, mu, L, function_class):
    list_of_matrices, list_of_vectors = interpolation(list_of_points, mu, L, function_class)
    nb_constraints = len(list_of_matrices)
    dual = cp.Variable((nb_constraints,))
    matrix_combination = cp.sum([dual[i] * list_of_matrices[i] for i in range(nb_constraints)])
    vector_combination = cp.sum([dual[i] * list_of_vectors[i] for i in range(nb_constraints)])

    return matrix_combination, vector_combination, dual
