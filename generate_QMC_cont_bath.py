from time import time
from qmc import qmc_gf
from generate_params import read_params_cont_bath, name, beta
from params import Gamma_func, integrated_Gamma, E_max
from mpi_tools import mpi_broadcast

G_filename_QMC = name("G_QMC", beta, 0, parent="data_cont_bath/")
G_filename_QMC = mpi_broadcast(G_filename_QMC)

params_filename = name("params_cont_bath", beta, 0, parent="data_cont_bath/")
params_filename = mpi_broadcast(params_filename)

params = read_params_cont_bath(params_filename)

T_start = time()

for i, p in enumerate(params):
    print("Param set", i)
    qmc_gf(beta, p["U"], p["eps"], continuous_bath=[
        Gamma_func(p["D"]), integrated_Gamma(p["D"]), E_max(p["D"])],
           filename=G_filename_QMC, only_gf=True)

print("Generated {} QMC GFs in {:.2f} s".format(
    len(params), time() - T_start))
