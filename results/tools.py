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
        f.write("alpha\tbeta")
        for alpha, beta in zip(alphas, betas):
            f.write("{}\t{}".format(alpha, beta))


def get_graphics(method, mu, L):
    alphas_lyap, betas_lyap = read_result_file(file_path="lyapunov/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
    alphas_cycle, betas_cycle = read_result_file(file_path="cycles/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))

    plt.figure(figsize=(15, 9))
    plt.plot(alphas_lyap, betas_lyap, 'g')
    plt.plot(alphas_cycle, betas_cycle, 'r')
    plt.savefig("figures/{}_mu{:.2f}_L{:.0f}.txt".format(method, mu, L))
