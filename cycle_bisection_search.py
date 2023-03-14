from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import cvxpy as cp

from tools.file_management import write_result_file
from algorithms.heavy_ball.cycles import cycle_heavy_ball_momentum
from algorithms.nag.cycles import cycle_accelerated_gradient_strongly_convex
from algorithms.inexact_gradient_descent.cycles import cycle_inexact_gradient_descent
from algorithms.douglas_rachford.cycles import cycle_douglas_rachford_splitting
from algorithms.three_operator_splitting.cycles import cycle_three_operator_splitting


def cycle_bisection_search(method, mu, L, nb_points, precision, cycle_length):
    """
    Search for cycles of a given length for all parametrization of a given method applied on a given class.
    Produce a txt file in folder "./results".

    Args:
        method (str): name of the method
        mu (float): strong convexity parameter
        L (float): smoothness parameter
        nb_points (int): number of bisection search performed (1 per beta value)
        precision (float): maximal absolute error accepted on gamma
        cycle_length (int): the length of the searched cycle

    """
    betas = np.linspace(0, 1, nb_points, endpoint=False)[1:]
    gammas_min_cycle = np.zeros_like(betas)
    if method == "HB":
        gammas_max_cycle = 2 * (1 + betas) / L
        cycle_search = cycle_heavy_ball_momentum
    elif method == "NAG":
        gammas_max_cycle = (1 + 1 / (1 + betas)) / L
        cycle_search = cycle_accelerated_gradient_strongly_convex
    elif method == "GD":
        gammas_max_cycle = 2 * np.ones_like(betas) / L
        cycle_search = cycle_inexact_gradient_descent
    elif method == "DR":
        # Here we set a gamma by default, but it is not even clear that there is cycles in 2/L
        gammas_max_cycle = 2 * np.ones_like(betas) / L
        cycle_search = cycle_douglas_rachford_splitting
    elif method == "TOS":
        # Here we set a gamma by default, but it is not even clear that there is cycles in 2/L
        gammas_max_cycle = 2 / L / betas
        cycle_search = cycle_three_operator_splitting
    else:
        raise ValueError
    gammas_cycle = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        gamma_min_cycle = gammas_min_cycle[it]
        gamma_max_cycle = gammas_max_cycle[it]

        while gamma_max_cycle - gamma_min_cycle > precision:
            gamma = (gamma_min_cycle + gamma_max_cycle) / 2

            problem = cycle_search(mu=mu, L=L, gamma=gamma, beta=beta, n=cycle_length)

            # Solve the PEP
            # A small cycle metric means there is a cycle
            try:
                pepit_cycle_metric = -problem.solve(verbose=0, solver="MOSEK")
            except cp.error.SolverError:
                pepit_cycle_metric = -problem.solve(verbose=0, solver="SCS")

            # Update search interval
            if pepit_cycle_metric < 10 ** -3:
                gamma_max_cycle = gamma
            else:
                gamma_min_cycle = gamma

        gammas_cycle.append(gamma_max_cycle)

    write_result_file(file_path="results/cycles/{}_mu{:.2f}_L{:.0f}_K{:.0f}.txt".format(method, mu, L, cycle_length),
                      gammas=gammas_cycle, betas=betas)


if __name__ == "__main__":
    methods = list()
    mus = list()
    cycle_lengths = list()
    for method in ["HB", "NAG", "GD", "DR", "TOS"]:
        for mu in [0, .01, .1, .2]:
            for cycle_length in range(3, 20 + 1):
                methods.append(method)
                mus.append(mu)
                cycle_lengths.append(cycle_length)

    Parallel(n_jobs=-1)(delayed(cycle_bisection_search)(method=methods[i],
                                                        mu=mus[i],
                                                        L=1,
                                                        nb_points=500,
                                                        precision=10 ** -4,
                                                        cycle_length=cycle_lengths[i],
                                                        ) for i in range(len(methods)))
