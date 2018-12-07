import numpy as np
from time import time
#from mpi_tools import *
from strong_coupling import strong_coupling_gf
from exact_diagonalization import exact_diagonalization_gf
from generate_params import read_params_cont_bath, name, beta


Delta_filename = name("Delta_cont_bath", beta, 0, parent="data_cont_bath/")
Deltas = np.loadtxt(Delta_filename, delimiter=",")
tau = np.linspace(0., beta, num=Deltas.shape[1], endpoint=True)
G_strong_filenames = [name(prefix, beta, 0, parent="data_cont_bath/")
               for prefix in ["G_0_strong_cont_bath", "G_strong_bath"]]
params = read_params_cont_bath(name("params_cont_bath", beta, 0, parent="data_cont_bath/"))

T_start = time()
for p, Delta in zip(params, Deltas):
    strong_coupling_gf(beta, p["U"], p["eps"], tau, Delta,
                       filenames=G_strong_filenames, only_gf=True)

print("Generated {} strong coupling GFs in {:.2f} s".format(
    len(params), time() - T_start))
