from __future__ import print_function, division
import numpy as np
from pytriqs.gf.local import *
from pytriqs.gf.local.tools import *
from pytriqs.dos import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.operators import n
from pytriqs.utility import mpi
from math import sqrt, pi

# Hamiltonian
# H = sum_sigma eps * n_sigma + U * (n_up - 0.5) * (n_down - 0.5) + H_bath + H_hyb

# Hard-coded parameters
indices = [0]  # only PM case considered
n_l = 21
n_tau = 6001
n_iw = 1000
rebinning_factor = 10 # rebin gf from 6001 to 601 bins
skip_factor = 3 # save only every 3rd rebinned gf point, i.e. 201 points

# Tools
def get_data(G):
    return [np.asarray(list(G.mesh)).real, G.data[:,0,0].real]

def save_gf(filename, G, only_gf=False, skip_factor=1):
    data = np.asarray(get_data(G))
    if not only_gf:
        np.savetxt(filename, data.T)
    else:
        with open(filename, 'ab') as f:
            np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

def plot_gf(G):
    data = get_data(G)
    plt.plot(data[0], data[1], label=G.name)


# Generating Delta
def Delta_from_func(Gamma_func, integrated_Gamma, E_max, beta):
    d = DOSFromFunction(lambda x: Gamma_func(x) / integrated_Gamma,
                        -E_max, E_max, n_pts=1000)
    HT = HilbertTransform(d)
    Sigma0 = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    Sigma0.zero()
    return integrated_Gamma / pi * HT(Sigma=Sigma0)

def Delta_from_lists(e_list, V_list):
    return sum(v**2 * inverse(iOmega_n - e) for e, v in zip(e_list, V_list))

# Generating G_0
def generate_G_0(eps, Delta_iw):
    return inverse(iOmega_n - eps - Delta_iw)

# QMC solution for G
def get_G_tau(eps, U, Delta_iw, measure_gl=False, rebinning_factor=1):
    n_cycles, cycle_length = 5000000, 25

    beta = Delta_iw.beta

    S = Solver(beta=beta, gf_struct={'up': [0], 'dn': [0]}, n_tau=n_tau, n_iw=n_iw, n_l=n_l)
    for name, g0 in S.G0_iw:
        g0 << generate_G_0(eps - U / 2., Delta_iw)

    S.solve(h_int=U*n('up', 0)*n('dn', 0),
            n_cycles=n_cycles,
            length_cycle=cycle_length,
            n_warmup_cycles=n_cycles//20,
            measure_g_l=measure_gl)

    G_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau, name=r"$G(\tau)$")
    if measure_gl:
        G_l = 0.5 * (S.G_l['up'] + S.G_l['dn'])
        # if mpi.is_master_node():
        #     plot_gf(G_l)
        #     plot_gf(-1j * G_l)
        #     plt.show()
        #     plt.close()
        G_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
        G_iw << LegendreToMatsubara(G_l)
        G_tau << InverseFourier(G_iw)
    else:
        G_tau << 0.5 * (S.G_tau['up'] + S.G_tau['dn'])
        G_tau = rebinning_tau(G_tau, (n_tau - 1) // rebinning_factor + 1)
    return G_tau

# Function to export

def qmc_gf(beta, U, eps, discrete_bath=None, continuous_bath=None,
           Delta=None, Delta_tau=None,
           filename="G_QMC.txt", plot=False, measure_gl=True, only_gf=True):

    Delta_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)

    if(discrete_bath != None):
        e_list, V_list = discrete_bath
        Delta_iw << Delta_from_lists(e_list, V_list)

    if(continuous_bath != None):
        Gamma_func, integrated_Gamma, E_max = continuous_bath
        Delta_iw << Delta_from_func(Gamma_func, integrated_Gamma, E_max, beta)

    if(Delta != None):
        Delta_iw << Delta

    if(Delta_tau != None):
        Delta_iw << Fourier(Delta_tau)

    G_tau = get_G_tau(eps, U, Delta_iw, measure_gl=measure_gl, rebinning_factor=rebinning_factor)

    if mpi.is_master_node():
        save_gf(filename, G_tau, only_gf=only_gf,
                skip_factor=skip_factor*(rebinning_factor if measure_gl else 1))

    if plot and mpi.is_master_node():
        plot_gf(G_tau)
        plt.legend()
        plt.title("QMC solution")
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Input - example
    # beta = 1.0
    # # Local
    # U = 4
    # eps = 0
    # # Bath
    # D = 5.0
    # Gamma = 1.0
    # # Discrete bath
    # N = 3
    # e_list = [-D + (2 * D) / (N - 1) * i for i in range(N)] if N != 1 else [0]
    # V_list = [sqrt(2 * D * Gamma / pi / N)] * N if N > 0 else []
    # # Continuous bath
    # E_max = D
    # def Gamma_func(x):
    #     return Gamma if abs(x) <= D else 0
    # integrated_Gamma = 2 * D * Gamma

    from params import *

    #qmc_gf(beta, U, eps, discrete_bath=[e_list, V_list], plot=False)
    qmc_gf(beta, U, eps, continuous_bath=[Gamma_func(D), integrated_Gamma(D), E_max(D)], plot=True, only_gf=True)
