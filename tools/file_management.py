import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
#
# mpl.backend_bases.register_backend("pdf", FigureCanvasPgf)
# size = 19
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "font.family": "serif",
#     "font.serif": "Times",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "font.size": size,
#     "axes.labelsize": size,
#     "axes.titlesize": size,
#     "figure.titlesize": size,
#     "xtick.labelsize": size,
#     "ytick.labelsize": size,
#     "legend.fontsize": size,
# })
import scienceplots
plt.style.use('science')



def read_result_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()[1:]

    gammas = list()
    betas = list()

    for line in lines:
        gamma, beta = line.split("\t")[:2]
        gammas.append(float(gamma))
        betas.append(float(beta))

    return gammas, betas


def write_result_file(file_path, gammas, betas):
    with open(file_path, "w") as f:
        f.write("gamma\tbeta\n")
        for gamma, beta in zip(gammas, betas):
            f.write("{}\t{}\n".format(gamma, beta))


def bound(method, L, beta):
    if method == "HB":
        return 2 * (1 + beta) / L
    elif method == "NAG":
        return (1 + 1 / (1 + 2 * beta)) / L
    elif method == "GD":
        return 2 / L
    elif method == "DR":
        # Set by default, not even clear that there is cycle in 2 / L
        return 2 / L
    elif method == "TOS":
        # Set by default, not even clear that there is cycle in 2 / L
        return 2 / L / beta
    else:
        raise Exception


def get_colored_graphics(method, mu, L, max_cycle_length, folder="results/"):
    plt.figure(figsize=(15, 9))

    gammas_lyap, betas_lyap = read_result_file(file_path=folder + "lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
    x_green = list()
    y_green = list()
    for gamma_max, beta in zip(gammas_lyap, betas_lyap):
        x_green += list(np.linspace(0, gamma_max, 500))
        y_green += [beta] * 500
    plt.plot(x_green, y_green, '.', color="yellowgreen")
    legends = ["convergence"]

    color_map = plt.get_cmap('OrRd')
    for K in range(max_cycle_length, 2, -1):
        try:
            gammas_cycle, betas_cycle = read_result_file(
                file_path=folder + "cycles/{}_mu{:.2f}_L{:.0f}_K{:.0f}.txt".format(method, mu, L, K))
            x_red = list()
            y_red = list()
            for gamma_min, beta in zip(gammas_cycle, betas_cycle):
                if gamma_min <= bound(method, L, beta):
                    x_red += list(np.linspace(gamma_min, bound(method, L, beta), 500))
                    y_red += [beta] * 500
            color_scale = (max_cycle_length + 1 - K) / (max_cycle_length - 1)
            color = color_map((2 + 3 * color_scale) / 5)
            plt.plot(x_red, y_red, '.', color=color)
            legends.append("cycle of length {}".format(K))
        except FileNotFoundError:
            pass

    if method == "GD":
        best_loc = "lower left"
    elif method == "HB":
        best_loc = "lower right"
    elif method == "NAG":
        best_loc = "upper right"
    elif method == "TOS":
        best_loc = "upper right"
    else:
        raise ValueError

    plt.legend(legends, loc=best_loc)
    plt.savefig(folder + "figures/{}_mu{:.2f}_L{:.0f}_colored.png".format(method, mu, L))


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD", "TOS"]:
        for mu in [0., .01, .1]:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=25, folder="../results/")
            except FileNotFoundError:
                pass
