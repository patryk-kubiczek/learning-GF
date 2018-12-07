import numpy as np
from time import time
#from mpi_tools import *
from strong_coupling import strong_coupling_gf
from exact_diagonalization import exact_diagonalization_gf
from generate_params import read_params, name, samples_per_file, files_per_core, beta

for n in range(files_per_core):

    Delta_filename = name("Delta", beta, n)
    Deltas = np.loadtxt(Delta_filename, delimiter=",")
    tau = np.linspace(0., beta, num=Deltas.shape[1], endpoint=True)
    G_strong_filenames = [name(prefix, beta, n)
                   for prefix in ["G_0_strong", "G_strong"]]
    G_ED_filename = name("G_ED", beta, n)
    params = read_params(name("params", beta, n))

    T_start = time()
    for p, Delta in zip(params, Deltas):
        strong_coupling_gf(beta, p["U"], p["eps"], tau, Delta,
                           filenames=G_strong_filenames, only_gf=True)

    print("Generated {} strong coupling GFs in {:.2f} s".format(
        len(params), time() - T_start))

    T_start = time()
    for p in params:
        exact_diagonalization_gf(beta, p["U"], p["eps"], p["e_list"], p["V_list"],
                                 filename=G_ED_filename, only_gf=True)

    print("Generated {} exact GFs in {:.2f} s".format(
        len(params), time() - T_start))
