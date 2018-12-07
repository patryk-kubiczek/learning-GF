from time import time
from weak_coupling import export_Delta_tau, weak_coupling_gf
from generate_params import read_params_cont_bath, name, beta
from params import Gamma_func, integrated_Gamma, E_max

Delta_filename = name("Delta_cont_bath", beta, 0, parent="data_cont_bath/")
G_filenames = [name(prefix, beta, 0, parent="data_cont_bath/")
               for prefix in ["G_0_weak_cont_bath", "G_weak_cont_bath"]]
params = read_params_cont_bath(name("params_cont_bath", beta, 0, parent="data_cont_bath/"))

T_start = time()

for p in params:
    export_Delta_tau(beta,
                     continuous_bath=[
                         Gamma_func(p["D"]), integrated_Gamma(p["D"]), E_max(p["D"])],
                     filename=Delta_filename, only_gf=True)
    weak_coupling_gf(beta, p["U"], p["eps"],
                     continuous_bath=[
                         Gamma_func(p["D"]), integrated_Gamma(p["D"]), E_max(p["D"])],
                     filenames=G_filenames, only_gf=True)

print("Generated {} weak coupling GFs in {:.2f} s".format(
    len(params), time() - T_start))


