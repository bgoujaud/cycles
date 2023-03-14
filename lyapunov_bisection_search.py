from tqdm import tqdm
from joblib import Parallel, delayed

from math import inf
import numpy as np

from tools.file_management import write_result_file
from heavy_ball.lyapunov import lyapunov_heavy_ball_momentum
from nag.lyapunov import lyapunov_accelerated_gradient_strongly_convex
from inexact_gradient_descent.lyapunov import lyapunov_inexact_gradient_descent
from douglas_rachford.lyapunov import lyapunov_douglas_rachford
from three_operator_splitting.lyapunov import lyapunov_three_operator_splitting


def lyapunov_bisection_search(method, mu, L, nb_points, precision, rho=1):
    betas = np.linspace(0, 1, nb_points, endpoint=False)
    gammas_min_lyap = np.zeros_like(betas)
    if method == "HB":
        gammas_max_lyap = 2 * (1 + betas) / L
        lyapunov_search = lyapunov_heavy_ball_momentum
    elif method == "NAG":
        gammas_max_lyap = (1 + 1 / (1 + betas)) / L
        lyapunov_search = lyapunov_accelerated_gradient_strongly_convex
    elif method == "GD":
        gammas_max_lyap = 2 * np.ones_like(betas) / L
        lyapunov_search = lyapunov_inexact_gradient_descent
    elif method == "DR":
        gammas_max_lyap = 2 * np.ones_like(betas) / L
        lyapunov_search = lyapunov_douglas_rachford
    elif method == "TOS":
        gammas_max_lyap = 2 * np.ones_like(betas) / L
        lyapunov_search = lyapunov_three_operator_splitting
    else:
        raise ValueError
    gammas_lyap = list()

    for it in tqdm(range(len(betas))):

        beta = betas[it]
        gamma_min_lyap = gammas_min_lyap[it]
        gamma_max_lyap = gammas_max_lyap[it]

        while gamma_max_lyap - gamma_min_lyap > precision:
            gamma = (gamma_min_lyap + gamma_max_lyap) / 2
            lyap = lyapunov_search(beta=beta, gamma=gamma, mu=mu, L=L, rho=rho)
            if lyap != inf:
                gamma_min_lyap = gamma
            else:
                gamma_max_lyap = gamma

        gammas_lyap.append(gamma_min_lyap)

    write_result_file(file_path="results/lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L),
                      alphas=gammas_lyap, betas=betas)


if __name__ == "__main__":
    methods = list()
    mus = list()
    for method in ["HB", "NAG", "GD", "DR", "TOS"]:
        for mu in [0, .01, .1, .2]:
            methods.append(method)
            mus.append(mu)

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search)(method=methods[i],
                                                           mu=mus[i],
                                                           L=1,
                                                           nb_points=500,
                                                           precision=10 ** -3,
                                                           rho=1,
                                                           ) for i in range(len(methods)))
