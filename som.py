from pytriqs.gf.local import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
import numpy as np
from pytriqs.applications.analytical_continuation.som import Som
from params import *

indices = [0]
beta = 1

n_tau = 501                 # Number of tau-slices for the input GF
n_w = 250
energy_window = (-8.0, 8.0)  # Energy window to search the solution in

# Parameters for Som.run()
run_params = {'energy_window' : energy_window}
# Verbosity level
run_params['verbosity'] = 2
# Number of particular solutions to accumulate
run_params['l'] = 5000
# Number of global updates
run_params['f'] = 100
# Number of local updates per global update
run_params['t'] = 50
# Accumulate histogram of the objective function values
run_params['make_histograms'] = False


def SOM(G_filename, A_filename="spectral_function.txt"):

    # Read G(\tau) from file
    tau, G = np.loadtxt(G_filename, unpack=True)

    G = G / t**2
    g_tau = GfImTime(indices=indices, beta=beta, n_points=len(G))
    g_tau.data[:,0,0] = G

    # Prepare input data: reduce the number of points to n_tau
    g_tau_rebinned = rebinning_tau(g_tau, n_tau)

    # Set the weight function S to a constant (all points of G_tau are equally important)
    S_tau = g_tau_rebinned.copy()
    S_tau.data[:] = 1.0

    # Construct a SOM object
    cont = Som(g_tau_rebinned, S_tau, kind = "FermionGf")

    # Run!
    cont.run(**run_params)

    # Create a real frequency GF obtained with SOM
    g_w = GfReFreq(window=energy_window, n_points=n_w, indices=indices)
    g_w << cont

    A_w = -1. / np.pi * np.imag(g_w.data[:,0,0])
    w = np.linspace(energy_window[1], energy_window[0], n_w)

    # G(\tau) reconstructed from the SOM solution
    #g_rec_tau = g_tau_rebinned.copy()
    #g_rec_tau << cont

    # On master node, save results to an archive
    if mpi.is_master_node():
        np.savetxt(A_filename, np.array([w, A_w]).T)

    return w, A_w

if __name__ == "__main__":
    w, A_QMC_w = SOM("Delta_DMFT_QMC_6.txt", "spectral_function_QMC.txt")
    w, A_w = SOM("Delta_DMFT_6.txt", "spectral_function.txt")

    if mpi.is_master_node():
        from matplotlib import pyplot as plt
        plt.plot(w, A_w, label="NN")
        plt.plot(w, A_QMC_w, label="QMC")
        plt.legend(loc="best")
        plt.show()
