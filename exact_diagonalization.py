from quspin.operators import hamiltonian, exp_op
from quspin.basis import spinful_fermion_basis_general
import numpy as np
from time import time
from itertools import product
from math import exp, sqrt, tanh, pi

verbose = False

# Hamiltonian
# H = sum_sigma eps * n_sigma + U * (n_up - 0.5) * (n_down - 0.5) + H_bath + H_hyb

# Hard-coded parameters
n_tau = 201

def calculate_G(beta, U, eps, e_list, V_list, only_up_gf=False):
    spins = ["up", "down"]
    L = len(e_list) + 1
    bath_sites = range(1, L)
    N_list = list(range(L + 1))

    def index(site, spin):
        return site + (0 if spin == "up" else L)

    def create_basis_and_hamiltonian(Nf, E0=0):
        basis = spinful_fermion_basis_general(L, simple_symm=False, Nf=Nf)
        # define site-coupling lists
        hyb_in_list = [[v, index(0, spin), index(i, spin)]
                       for v, i in zip(V_list, bath_sites) for spin in spins]
        hyb_out_list = [[-v, index(0, spin), index(i, spin)]
                        for v, i in zip(V_list, bath_sites) for spin in spins]
        pot_list = [[eps, index(0, spin)] for spin in spins]
        pot_list += [[e, index(i, spin)]
                     for e, i in zip(e_list, bath_sites) for spin in spins]
        int_list = [[U, index(0, "up"), index(0, "down")]]
        E0_list = [[-E0, 0]]
        # create static and dynamic lists for hamiltonian
        h_static = [
            ["+-", hyb_in_list],
            ["-+", hyb_out_list],
            ["n", pot_list],
            ["zz", int_list],
            ["I", E0_list],
        ]
        # create hamiltonian
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        H = hamiltonian(h_static, [], basis=basis, dtype=np.float64,
                        **no_checks)
        return basis, H

    def contribution_to_gf(N_up, N_down, gf_spin, E0=0):
        # Create basis
        Nf = [(N_up, N_down)] + ([(N_up + 1, N_down)] if gf_spin == "up"
                                 else [(N_up, N_down + 1)])
        basis, H = create_basis_and_hamiltonian(Nf, E0)
        if verbose:
            print("Sector (N_up={}, N_down={}). Extended basis size = {}".format(
            N_up, N_down, basis.Ns))
        # create operators
        cr = [["+", [[1, index(0, gf_spin)]]]]
        # an = [["-", [[1, index(0, gf_spin)]]]]
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        CR = hamiltonian(cr, [], basis=basis, dtype=np.float64, **no_checks)
        # AN = hamiltonian(an, [], basis=basis, dtype=np.float64, **no_checks)
        # Transform all operators to the eigenbasis
        E, W = H.eigh()
        CR = W.T @ CR.toarray() @ W
        AN = CR.T
        # Determine evolution operators in the eigenbasis
        tau = np.linspace(0, beta, n_tau, endpoint=True)
        U_tau = np.exp(np.outer(E, -tau))
        U_beta_minus_tau = np.fliplr(U_tau)
        # Calculate contributions to gf
        g = np.zeros(n_tau)
        if verbose: T_start = time()
        # g(tau) = -Tr[U(beta-tau) * AN * U(tau) * CR]
        g[:] = -np.einsum("ni,nm,mi,mn->i", U_beta_minus_tau, AN, U_tau, CR)
        if verbose:
            print("Calculation of {} GF in sector ({}, {}) completed in {:.2f} s".format(
            gf_spin, N_up, N_down, time() - T_start))
        return g

        # # Create imaginary time evolution operators
        # U_tau = exp_op(H, a=-1., start=0., stop=beta, num=n_tau,
        #                endpoint=True, iterate=True)
        # U_beta_minus_tau = exp_op(H, a=-1., start=beta, stop=0., num=n_tau,
        #                           endpoint=True, iterate=True)
        # # Fix the bug
        # for E in [U_tau, U_beta_minus_tau]:
        #     E._grid, E._step = E._step, E._grid
        # g = np.zeros(n_tau)
        # for i, (A, B) in enumerate(zip(U_beta_minus_tau.dot(AN, shift=0 * beta * E0),
        #                                U_tau.dot(CR, shift=0 * beta * E0))):
        #     g[i] = -A.dot(B).trace()
        # return g

    # Find ground state energy
    E0_list = []
    for N_up, N_down in product(N_list, N_list):
        basis, H = create_basis_and_hamiltonian((N_up, N_down))
        E = H.eigvalsh()
        E0_list.append(min(E))
    E0 = min(E0_list)

    # Sum contributions to GF from all sectors
    G = np.zeros((2, n_tau))
    for g, gf_spin in zip(G, spins):
        if only_up_gf and gf_spin == "down": continue
        for N_up, N_down in product(N_list, N_list):
            if (N_up if gf_spin == "up" else N_down) == L: continue
            g[:] += contribution_to_gf(N_up, N_down, gf_spin, E0)
    # Normalize GF by demanding g(0) + g(beta) = -1
    Z = -(G[0, 0] + G[0, -1])
    G /= Z
    if only_up_gf:
        G[1] = G[0]

    return G

# Tools

from matplotlib import pyplot as plt

def plot_gf(tau, G, label=""):
    for spin, g in zip([r"\uparrow", r"\downarrow"], G):
        plt.plot(tau, g, label="$G_{}$".format(spin) + label)

def save_gf(filename, tau, G, single_spin=True, only_gf=False, skip_factor=1):
    if single_spin == True:
        data = np.asarray([tau, G[0]])
    else:
        data = np.asarray([tau, G[0], G[1]])
    if not only_gf:
        np.savetxt(filename, data.T)
    else:
        with open(filename, 'ab') as f:
            np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")

# Function to export

def exact_diagonalization_gf(beta, U, eps, e_list, V_list,
                             filename="G_ED.txt", plot=False, only_gf=False):
    tau = np.linspace(0., beta, n_tau, endpoint=True)
    G = calculate_G(beta, U, eps, e_list, V_list, only_up_gf=True)
    save_gf(filename, tau, G, only_gf=only_gf)

    if plot:
        plot_gf(tau, G)
        plt.legend()
        plt.title("ED")
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Input - example
    # beta = 1.0
    # # Local
    # U = 4.
    # eps = 0.
    # # Bath
    # D = 5.0
    # Gamma = 1.0
    # # Discrete bath
    # N = 3
    # e_list = [-D + (2 * D) / (N - 1) * i for i in range(N)] if N != 1 else [0]
    # V_list = [sqrt(2 * D * Gamma / pi / N)] * N if N > 0 else []

    from params import *

    exact_diagonalization_gf(beta, U, eps, e_list, V_list, plot=False)

