import cvxpy as cp


def inner_product(u, v):
    matrix = u.reshape(-1, 1) * v.reshape(1, -1)
    return (matrix + matrix.T) / 2


def square(u):
    return inner_product(u, u)


def interpolation_i_j(pointi, pointj, mu, L):
    xi, gi, fi = pointi
    xj, gj, fj = pointj

    G = inner_product(gj, xi - xj) + 1 / (2 * L) * square(gi - gj) + mu / (2 * (1 - mu / L)) * square(
        xi - xj - 1 / L * gi + 1 / L * gj)
    F = fj - fi

    return G, F


def interpolation(list_of_points, mu, L):
    list_of_matrices = []
    list_of_vectors = []

    for i, pointi in enumerate(list_of_points):
        for j, pointj in enumerate(list_of_points):
            if i != j:
                G, F = interpolation_i_j(pointi, pointj, mu, L)
                list_of_matrices.append(G)
                list_of_vectors.append(F)

    return list_of_matrices, list_of_vectors


def interpolation_combination(list_of_points, mu, L):
    list_of_matrices, list_of_vectors = interpolation(list_of_points, mu, L)
    nb_constraints = len(list_of_matrices)
    dual = cp.Variable((nb_constraints,))
    matrix_combination = cp.sum([dual[i] * list_of_matrices[i] for i in range(nb_constraints)])
    vector_combination = cp.sum([dual[i] * list_of_vectors[i] for i in range(nb_constraints)])

    return matrix_combination, vector_combination, dual
