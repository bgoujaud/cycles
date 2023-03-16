from joblib import Parallel, delayed

from lyapunov_bisection_search import lyapunov_bisection_search
from cycle_bisection_search import cycle_bisection_search
from tools.file_management import get_colored_graphics


def run_all(list_algos, list_mus, nb_points, precision, max_cycle_length):
    methods = list()
    mus = list()
    for method in list_algos:
        for mu in list_mus:
            methods.append(method)
            mus.append(mu)

    Parallel(n_jobs=-1)(delayed(lyapunov_bisection_search)(method=methods[i],
                                                           mu=mus[i],
                                                           L=1,
                                                           nb_points=nb_points,
                                                           precision=precision,
                                                           rho=1,
                                                           ) for i in range(len(methods)))

    methods = list()
    mus = list()
    cycle_lengths = list()
    for method in list_algos:
        for mu in list_mus:
            for cycle_length in range(2, max_cycle_length + 1):
                methods.append(method)
                mus.append(mu)
                cycle_lengths.append(cycle_length)

    Parallel(n_jobs=-1)(delayed(cycle_bisection_search)(method=methods[i],
                                                        mu=mus[i],
                                                        L=1,
                                                        nb_points=nb_points,
                                                        precision=precision,
                                                        cycle_length=cycle_lengths[i],
                                                        ) for i in range(len(methods)))

    for method in list_algos:
        for mu in list_mus:
            try:
                get_colored_graphics(method=method, mu=mu, L=1, max_cycle_length=max_cycle_length)
            except FileNotFoundError:
                pass


if __name__ == "__main__":

    run_all(list_algos=["HB", "NAG", "GD", "TOS"], list_mus=[0], nb_points=300, precision=10**-4, max_cycle_length=15)
