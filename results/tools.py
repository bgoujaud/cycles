import numpy as np
import matplotlib.pyplot as plt


def read_result_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()[1:]

    alphas = list()
    betas = list()

    for line in lines:
        alpha, beta = line.split("\t")[:2]
        alphas.append(float(alpha))
        betas.append(float(beta))

    return alphas, betas


def write_result_file(file_path, alphas, betas):
    with open(file_path, "w") as f:
        f.write("alpha\tbeta\n")
        for alpha, beta in zip(alphas, betas):
            f.write("{}\t{}\n".format(alpha, beta))


def bound(method, L, beta):
    if method == "HB":
        return 2 * (1 + beta) / L
    elif method == "NAG":
        return (1 + 1 / (1 + 2 * beta)) / L
    elif method == "GD":
        return 2 / L
    else:
        raise Exception


def get_graphics(method, mu, L):
    alphas_lyap, betas_lyap = read_result_file(file_path="lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
    alphas_cycle, betas_cycle = read_result_file(file_path="cycles/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))

    x_green = list()
    y_green = list()
    for alpha_max, beta in zip(alphas_lyap, betas_lyap):
        x_green += list(np.linspace(0, alpha_max, 500))
        y_green += [beta] * 500

    x_red = list()
    y_red = list()
    for alpha_min, beta in zip(alphas_cycle, betas_cycle):
        x_red += list(np.linspace(alpha_min, bound(method, L, beta), 500))
        y_red += [beta] * 500

    plt.figure(figsize=(15, 9))
    plt.plot(x_green, y_green, '.g')
    plt.plot(x_red, y_red, '.r')
    plt.savefig("figures/{}_mu{:.2f}_L{:.0f}.png".format(method, mu, L))


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD"]:
        for mu in [0., .01, .1, .2]:
            try:
                get_graphics(method=method, mu=mu, L=1)
            except FileNotFoundError:
                pass
