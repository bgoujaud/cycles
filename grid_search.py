from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from heavy_ball.cycles import cycle_heavy_ball_momentum
from nag.cycles import cycle_accelerated_gradient_strongly_convex
from inexact_gradient_descent.cycles import cycle_inexact_gradient_descent


def make_grid_hb(L, nb_points):

    beta_list = []
    alpha_list = []

    for beta in np.linspace(0, .99, nb_points):
        for alpha in np.arange(.05 / L, 1.95 * (1 + beta) / L, 2 / L / nb_points):
            beta_list.append(beta)
            alpha_list.append(alpha)

    beta_list = np.array(beta_list)
    alpha_list = np.array(alpha_list)

    return beta_list, alpha_list


def make_grid_nag(L, nb_points):

    beta_list = []
    alpha_list = []

    for beta in np.linspace(0, .99, nb_points):
        for alpha in np.arange(.05 / L, (0.95 + 1/(1+beta)) / L, 1.9 / L / nb_points):
            beta_list.append(beta)
            alpha_list.append(alpha)

    beta_list = np.array(beta_list)
    alpha_list = np.array(alpha_list)

    return beta_list, alpha_list


def make_grid_gd_inexact(L, nb_points):

    epsilon_list = []
    gamma_list = []

    for epsilon in np.linspace(0.01, .99, nb_points):
        for gamma in np.linspace(.05 / L, 1.95 / L, nb_points):
            epsilon_list.append(epsilon)
            gamma_list.append(gamma)

    epsilon_list = np.array(epsilon_list)
    gamma_list = np.array(gamma_list)

    return epsilon_list, gamma_list


def grid_search_hb(mu, L, nb_points, max_length):
    beta_list, alpha_list = make_grid_hb(L=L, nb_points=nb_points)
    cv_list = []
    cycle_list = []

    for i in tqdm(range(len(beta_list))):
        beta = beta_list[i]
        alpha = alpha_list[i]
        cv = False
        cycle = False
        for n in range(3, max_length + 1):
            if cv is False and cycle is False:
                cv = wc_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=n, verbose=-1)
            if cv is False and cycle is False:
                cycle = cycle_heavy_ball_momentum(mu=mu, L=L, alpha=alpha, beta=beta, n=n, threshold=10 ** -3, verbose=-1)
        cv_list.append(cv)
        cycle_list.append(cycle)

    cv_list = np.array(cv_list)
    cycle_list = np.array(cycle_list)

    return beta_list, alpha_list, cv_list, cycle_list


def grid_search_nag(mu, L, nb_points, max_length):
    beta_list, alpha_list = make_grid_nag(L=L, nb_points=nb_points)
    cv_list = []
    cycle_list = []

    for i in tqdm(range(len(beta_list))):
        beta = beta_list[i]
        alpha = alpha_list[i]
        cv = False
        cycle = False
        for n in range(3, max_length + 1):
            if cv is False and cycle is False:
                cv = wc_accelerated_gradient_strongly_convex(mu=mu, L=L, alpha=alpha, beta=beta, n=n, verbose=-1)
            if cv is False and cycle is False:
                cycle = cycle_accelerated_gradient_strongly_convex(mu=mu, L=L, alpha=alpha, beta=beta, n=n, threshold=10 ** -3, verbose=-1)
        cv_list.append(cv)
        cycle_list.append(cycle)

    cv_list = np.array(cv_list)
    cycle_list = np.array(cycle_list)

    return beta_list, alpha_list, cv_list, cycle_list


def grid_search_inexact_grad(mu, L, nb_points, max_length):
    epsilon_list, gamma_list = make_grid_gd_inexact(L=L, nb_points=nb_points)
    cv_list = []
    cycle_list = []

    for i in tqdm(range(len(epsilon_list))):
        epsilon = epsilon_list[i]
        gamma = gamma_list[i]
        cv = False
        cycle = False
        for n in range(3, max_length + 1):
            if cv is False and cycle is False:
                cv = wc_inexact_gradient_descent(mu=mu, L=L, epsilon=epsilon, gamma=gamma, n=n, verbose=-1)
            if cv is False and cycle is False:
                cycle = cycle_inexact_gradient_descent(mu=mu, L=L, epsilon=epsilon, gamma=gamma, n=n, threshold=10 ** -3, verbose=-1)
        cv_list.append(cv)
        cycle_list.append(cycle)

    cv_list = np.array(cv_list)
    cycle_list = np.array(cycle_list)

    return epsilon_list, gamma_list, cv_list, cycle_list


def get_graphic(mu, L, nb_points, max_length, alg):

    if alg == "hb":
        beta_list, alpha_list, cv_list, cycle_list = grid_search_hb(mu, L, nb_points, max_length)
        plt.figure(figsize=(15, 9))
        plt.plot(alpha_list[cv_list], beta_list[cv_list], '.g')
        plt.plot(alpha_list[cycle_list], beta_list[cycle_list], '.r')
        plt.plot(alpha_list[~cv_list & ~cycle_list], beta_list[~cv_list & ~cycle_list], '.k')
        plt.savefig("hb.png")

    elif alg == "nag":
        beta_list, alpha_list, cv_list, cycle_list = grid_search_nag(mu, L, nb_points, max_length)
        plt.figure(figsize=(15, 9))
        plt.plot(alpha_list[cv_list], beta_list[cv_list], '.g')
        plt.plot(alpha_list[cycle_list], beta_list[cycle_list], '.r')
        plt.plot(alpha_list[~cv_list & ~cycle_list], beta_list[~cv_list & ~cycle_list], '.k')
        plt.savefig("nag.png")

    elif alg == "gd":
        epsilon_list, gamma_list, cv_list, cycle_list = grid_search_inexact_grad(mu, L, nb_points, max_length)
        plt.figure(figsize=(15, 9))
        plt.plot(gamma_list[cv_list], epsilon_list[cv_list], '.g')
        plt.plot(gamma_list[cycle_list], epsilon_list[cycle_list], '.r')
        plt.plot(gamma_list[~cv_list & ~cycle_list], epsilon_list[~cv_list & ~cycle_list], '.k')
        plt.savefig("gd_inexact.png")

    else:
        raise Exception


if __name__ == "__main__":

    def get_the_right_graphic(alg):
        get_graphic(mu=.1, L=1., nb_points=10, max_length=10, alg=alg)

    Parallel(n_jobs=3)(delayed(get_the_right_graphic)(alg) for alg in ["hb", "nag", "gd"])
