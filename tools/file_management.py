import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

size = 19
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.serif": "Times",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": size,
    "axes.labelsize": size,
    "axes.titlesize": size,
    "figure.titlesize": size,
    "xtick.labelsize": size,
    "ytick.labelsize": size,
    "legend.fontsize": size,
})


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
    elif method == "TOS":
        return 2 / L
    else:
        raise Exception


def get_colored_graphics(method, mu, L, max_cycle_length, folder="results/"):
    fig = plt.figure(figsize=(15, 9))
    plt.xlabel(r"$\gamma$")
    if method == "GD":
        plt.ylabel(r"$\varepsilon$", rotation=0, labelpad=10)
    else:
        plt.ylabel(r"$\beta$", rotation=0, labelpad=10)

    ax = plt.subplot(111)

    if method == "HB":
        axins = zoomed_inset_axes(ax, zoom=3.5, loc="lower right")

    gammas_lyap, betas_lyap = read_result_file(
        file_path=folder + "lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
    x_green = list()
    y_green = list()
    for gamma_max, beta in zip(gammas_lyap, betas_lyap):
        x_green += list(np.linspace(0, gamma_max, 500))
        y_green += [beta] * 500
    ax.plot(x_green, y_green, '.', color="yellowgreen", label="convergence")

    if method == "HB":
        axins.plot(x_green, y_green, ".", color="yellowgreen")

    legends = ["convergence"]
    colors = ["yellowgreen"]

    color_map = plt.get_cmap('YlOrRd')
    for K in range(max_cycle_length, 1, -1):
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
            color = color_map(color_scale)

            ax.plot(x_red, y_red, '.', color=color, label="cycle of length {}".format(K))
            if method == "HB":
                axins.plot(x_red, y_red, ".", color=color)
            legends.append("cycle of length {}".format(K))
            colors.append(color)
        except FileNotFoundError:
            pass
    if method == "HB":
        x1 = -0.01
        x2 = 0.25
        y1 = 0.95
        y2 = 1.002
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.7")
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=0.6, color="grey")

    bounds = list(range(len(colors[1:])))
    norm = mpl.colors.BoundaryNorm(np.array(bounds) + 2, len(colors[1:]))
    position = fig.add_axes([0.28, 0.92, 0.6, 0.02])  # [x_init, y_init, width, height]
    clbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap(colors[1:][::-1])),
                         cax=position, orientation="horizontal", shrink=0.8, fraction=0.1, aspect=50)
    clbar.ax.set_title("Length of the shortest cycle")
    handles = [mlines.Line2D([], [], color=colors[0], marker="s",
                             linestyle="None", markersize=12, markeredgecolor="black")]
    labels = [""]
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 1), title_fontsize=20, title="Convergence", frameon=False)
    plt.savefig(folder + "figures/{}_mu{:.2f}_L{:.0f}_colored.png".format(method, mu, L), bbox_inches="tight")


if __name__ == "__main__":
    for method in ["HB", "NAG", "GD", "TOS"]:
        for mu in [0]:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=25, folder="../results/")
            except FileNotFoundError:
                pass
