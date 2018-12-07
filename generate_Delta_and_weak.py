from time import time
#from mpi_tools import *
from weak_coupling import export_Delta_tau, weak_coupling_gf
from generate_params import read_params, name, samples_per_file, files_per_core, beta

for n in range(files_per_core):

    Delta_filename = name("Delta", beta, n)
    G_filenames = [name(prefix, beta, n)
                   for prefix in ["G_0_weak", "G_weak"]]
    params = read_params(name("params", beta, n))

    T_start = time()

    for p in params:
        export_Delta_tau(beta, discrete_bath=[p["e_list"], p["V_list"]],
                         filename=Delta_filename, only_gf=True)
        weak_coupling_gf(beta, p["U"], p["eps"],
                         discrete_bath=[p["e_list"], p["V_list"]],
                         filenames=G_filenames, only_gf=True)

    print("Generated {} weak coupling GFs in {:.2f} s".format(
        len(params), time() - T_start))


