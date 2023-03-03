from tqdm import tqdm
from joblib import Parallel, delayed

from math import log2, ceil
import numpy as np
import matplotlib.pyplot as plt

from results.tools import write_result_file
from heavy_ball.cycles import cycle_heavy_ball_momentum
from nag.cycles import cycle_accelerated_gradient_strongly_convex
from inexact_gradient_descent.cycles import cycle_inexact_gradient_descent


def conditional_bisection_search(method, mu, L, nb_points, precision, cycle_length):
    betas = np.linspace(0, 1, nb_points, endpoint=False)
    alphas_min_cycle = np.zeros_like(betas)
    if method == "HB":
        alphas_max_cycle = 2 * (1 + betas) / L
        cycle_search = cycle_heavy_ball_momentum
    elif method == "NAG":
        alphas_max_cycle = (1 + 1 / (1 + betas)) / L
        cycle_search = cycle_accelerated_gradient_strongly_convex
    elif method == "GD":
        alphas_max_cycle = 2 * np.ones_like(betas) / L
        cycle_search = cycle_inexact_gradient_descent
    else:
        raise ValueError
    alphas_cycle = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        alpha_min_cycle = alphas_min_cycle[it]
        alpha_max_cycle = alphas_max_cycle[it]

        while alpha_max_cycle - alpha_min_cycle > precision:
            alpha = (alpha_min_cycle + alpha_max_cycle) / 2
            cycle = cycle_search(mu=mu, L=L, alpha=alpha, beta=beta,
                                 n=cycle_length, threshold=10 ** -3, verbose=-1)
            if cycle:
                alpha_max_cycle = alpha
            else:
                alpha_min_cycle = alpha

        alphas_cycle.append(alpha_max_cycle)

    write_result_file(file_path="results/cycles/{}_mu{:.2f}_L{:.0f}_K{:.0f}.txt".format(method, mu, L, cycle_length),
                      alphas=alphas_cycle, betas=betas)


if __name__ == "__main__":

    methods = list()
    mus = list()
    cycle_lengths = list()
    for method in ["HB", "NAG", "GD"]:
        for mu in [0, .01, .1, .2]:
            for cycle_length in range(3, 13):
                methods.append(method)
                mus.append(mu)
                cycle_lengths.append(cycle_length)

    def run(method, mu, cycle_length):
        conditional_bisection_search(method=method, mu=mu, L=1, nb_points=500, precision=10**-3, cycle_length=cycle_length)

    Parallel(n_jobs=36)(delayed(run)(methods[i], mus[i], cycle_lengths[i]) for i in range(120))
