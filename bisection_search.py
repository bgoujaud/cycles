from tqdm import tqdm
from joblib import Parallel, delayed

from math import log2, ceil
import numpy as np
import matplotlib.pyplot as plt

from results.tools import write_result_file

from heavy_ball.contraction import wc_heavy_ball_momentum
from heavy_ball.cycles import cycle_heavy_ball_momentum

from nag.contraction import wc_accelerated_gradient_strongly_convex
from nag.cycles import cycle_accelerated_gradient_strongly_convex

from inexact_gradient_descent.contraction import wc_inexact_gradient_descent
from inexact_gradient_descent.cycles import cycle_inexact_gradient_descent


def bisection_search_hb(mu, L, nb_points, precision, max_cycle_length):
    betas = np.linspace(0, 1, nb_points, endpoint=False)
    alphas_min_cycle = np.zeros_like(betas)
    alphas_max_cycle = 2 * (1 + betas) / L
    alphas_cycle = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        alpha_min_cycle = alphas_min_cycle[it]
        alpha_max_cycle = alphas_max_cycle[it]

        depth = ceil(log2((alpha_max_cycle - alpha_min_cycle) / precision))
        for k in range(depth):
            alpha = (alpha_min_cycle + alpha_max_cycle) / 2
            cycle = False
            for n in range(3, max_cycle_length + 1):
                if cycle is False:
                    cycle = cycle_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta,
                                                      n=n, threshold=10 ** -3, verbose=-1)
            if cycle:
                alpha_max_cycle = alpha
            else:
                alpha_min_cycle = alpha

        alphas_cycle.append(alpha_max_cycle)

    write_result_file(file_path="results/cycles/HB_mu{:.2f}_L{:.0f}.txt".format(mu, L), alphas=alphas_cycle, betas=betas)


def bisection_search_nag(mu, L, nb_points, precision, max_cycle_length):
    betas = np.linspace(0, 1, nb_points, endpoint=False)
    alphas_min_cycle = np.zeros_like(betas)
    alphas_max_cycle = (1 + 1 / (1 + betas)) / L
    alphas_cycle = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        alpha_min_cycle = alphas_min_cycle[it]
        alpha_max_cycle = alphas_max_cycle[it]

        depth = ceil(log2((alpha_max_cycle - alpha_min_cycle) / precision))
        for k in range(depth):
            alpha = (alpha_min_cycle + alpha_max_cycle) / 2
            cycle = False
            for n in range(3, max_cycle_length + 1):
                if cycle is False:
                    cycle = cycle_accelerated_gradient_strongly_convex(mu=mu, L=L, alpha=alpha, beta=beta,
                                                                       n=n, threshold=10 ** -3, verbose=-1)
            if cycle:
                alpha_max_cycle = alpha
            else:
                alpha_min_cycle = alpha

        alphas_cycle.append(alpha_max_cycle)

    write_result_file(file_path="results/cycles/NAG_mu{:.2f}_L{:.0f}.txt".format(mu, L), alphas=alphas_cycle, betas=betas)


def bisection_search_gd_inexact(mu, L, nb_points, precision, max_cycle_length):
    # beta is epsilon
    betas = np.linspace(0, 1, nb_points, endpoint=False)

    # alpha is gamma
    alphas_min_cycle = 1.9 / (L*(1+betas))
    alphas_max_cycle = 2 * np.ones_like(betas) / L
    alphas_cycle = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        alpha_min_cycle = alphas_min_cycle[it]
        alpha_max_cycle = alphas_max_cycle[it]

        depth = ceil(log2((alpha_max_cycle - alpha_min_cycle) / precision))
        for k in range(depth):
            alpha = (alpha_min_cycle + alpha_max_cycle) / 2
            cycle = False
            for n in range(3, max_cycle_length + 1):
                if cycle is False:
                    cycle = cycle_inexact_gradient_descent(mu=mu, L=L, gamma=alpha, epsilon=beta,
                                                           n=n, threshold=10 ** -3, verbose=-1)
            if cycle:
                alpha_max_cycle = alpha
            else:
                alpha_min_cycle = alpha

        alphas_cycle.append(alpha_max_cycle)

    write_result_file(file_path="results/cycles/GD_mu{:.2f}_L{:.0f}.txt".format(mu, L), alphas=alphas_cycle, betas=betas)


if __name__ == "__main__":

    methods = list()
    mus = list()
    for method in ["HB", "NAG", "GD"]:
        for mu in [0, .01, .1, .2]:
            methods.append(method)
            mus.append(mu)

    def run(method, mu):
        if method == "HB":
            bisection_search_hb(mu=mu, L=1, nb_points=100, precision=10**-2, max_cycle_length=8)
        elif method == "NAG":
            bisection_search_nag(mu=mu, L=1, nb_points=100, precision=10**-2, max_cycle_length=8)
        elif method == "GD":
            bisection_search_gd_inexact(mu=mu, L=1, nb_points=100, precision=10**-2, max_cycle_length=8)
        else:
            raise Exception

    Parallel(n_jobs=12)(delayed(run)(methods[i], mus[i]) for i in range(12))
