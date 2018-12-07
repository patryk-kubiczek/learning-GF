from __future__ import print_function, division
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
from pytriqs.applications.impurity_solvers.cthyb import Solver
from qmc import save_gf

indices = [0]
n_l = 11
n_iter = 7
n_cycles, cycle_length = 2 * 5000000, 25
alpha = 0.80
n_tau = 6001
n_iw = 1001
skip_factor = 3

# DMFT parameters

def DMFT(U, eps, t, beta, filename="DMFT_QMC.h5"):
    # Construct the CT-HYB-QMC solver
    S = Solver(beta = beta, gf_struct = {'up':[0], 'down':[0]}, n_tau=n_tau, n_iw=n_iw, n_l=n_l)

    # Initialize Delta
    Delta = GfImFreq(beta = beta, indices = [0], n_points=n_iw)
    Delta_prev = GfImFreq(beta = beta, indices = [0], n_points=n_iw)
    Delta_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau)

    Delta << t ** 2 * SemiCircular(half_bandwidth = 2 * t)
    Delta_tau << InverseFourier(Delta)
    if mpi.is_master_node():
        with HDFArchive(filename) as Results:
            Results["Delta_iter0"] = Delta
            save_gf(DMFT_Delta_filename("QMC_0"), Delta_tau, skip_factor=skip_factor)

    # Now do the DMFT loop
    for iter in range(1, n_iter):

        # Compute new S.G0_iw
        for name, g0 in S.G0_iw:
            g0 << inverse(iOmega_n - eps + U / 2. - Delta)
        # Run the solver
        S.solve(h_int = U * n('up',0) * n('down',0),    # Local Hamiltonian
                n_cycles = n_cycles,                      # Number of QMC cycles
                length_cycle = cycle_length,                     # Length of a cycle
                n_warmup_cycles = n_cycles // 20,                 # How many warmup cycles
                measure_g_l = True)
        # Compute new Delta with the self-consistency condition while imposing paramagnetism
        g_l = 0.5 * (S.G_l['up'] + S.G_l['down'])
        # g_iw = 0.5 * (S.G_iw['up'] + S.G_iw['down'])
        Delta_prev << Delta
        Delta.set_from_legendre(t**2 * g_l)
        # Delta << t**2 * g_iw
        Delta << (1 - alpha) * Delta_prev + alpha * Delta
        Delta_tau << InverseFourier(Delta)

        # Intermediate saves
        if mpi.is_master_node():
            with HDFArchive(filename) as Results:
                Results["G_tau_iter{}".format(iter)] = S.G_tau
                Results["G_iw_iter{}".format(iter)] = S.G_iw
                Results["G_l_iter{}".format(iter)] = S.G_l
                Results["Sigma_iter{}".format(iter)] = S.Sigma_iw
                Results["Delta_iter{}".format(iter)] = Delta
                save_gf(DMFT_Delta_filename("QMC_" + str(iter)), Delta_tau,
                        skip_factor=skip_factor)

    if mpi.is_master_node():
        with HDFArchive(filename) as Results:
            Results["G_tau"] = S.G_tau
            Results["G_iw"] = S.G_iw
            Results["G_l"] = S.G_l
            Results["Sigma"] = S.Sigma_iw
            Results["Delta"] = Delta

if __name__ == "__main__":
    from params import *
    DMFT(U, eps, t, beta)
