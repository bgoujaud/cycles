from tqdm import tqdm
from joblib import Parallel, delayed

from math import inf
import numpy as np

from tools.file_management import bound, write_result_file
from algorithms.heavy_ball.lyapunov import lyapunov_heavy_ball_momentum
from algorithms.nag.lyapunov import lyapunov_accelerated_gradient_strongly_convex
from algorithms.inexact_gradient_descent.lyapunov import lyapunov_inexact_gradient_descent
from algorithms.three_operator_splitting.lyapunov import lyapunov_three_operator_splitting


def lyapunov_bisection_search(method, mu, L, nb_points, precision, rho=1):
    if method == "TOS":
        betas = np.linspace(0, 2, nb_points + 1, endpoint=False)[1:]
    else:
        betas = np.linspace(0, 1, nb_points + 1, endpoint=False)[1:]
    gammas_min_lyap = np.zeros_like(betas)
    gammas_max_lyap = [bound(method=method, L=L, beta=beta) for beta in betas]
    if method == "HB":
        lyapunov_search = lyapunov_heavy_ball_momentum
    elif method == "NAG":
        lyapunov_search = lyapunov_accelerated_gradient_strongly_convex
    elif method == "GD":
        lyapunov_search = lyapunov_inexact_gradient_descent
    elif method == "TOS":
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
                      gammas=gammas_lyap, betas=betas)


if __name__ == "__main__":

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search)(method=method,
                                                           mu=0,
                                                           L=1,
                                                           nb_points=300,
                                                           precision=10 ** -4,
                                                           rho=1,
                                                           ) for method in ["HB", "NAG", "GD", "TOS"])
