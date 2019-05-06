from __future__ import print_function, division
import numpy as np
from pytriqs.gf.local import *
from pytriqs.gf.local.tools import *
from pytriqs.dos import *
from math import sqrt, pi

# Hamiltonian
# H = sum_sigma eps * n_sigma + U * (n_up - 0.5) * (n_down - 0.5) + H_bath + H_hyb

# Hard-coded parameters
indices = [0]  # only PM case considered
n_tau = 2001
n_iw = 1000
skip_factor = 10 # save only every tenth gf point, i.e. 201 points
skip_factor_Delta = 5 # skip factor for Delta exported to NCA code (401 points)

# Operations on GF
def reverse_tau(G_tau, statistic="Fermion"):
    sign = -1 if statistic == "Fermion" else 1
    G_minus_tau = GfImTime(indices=indices, beta=G_tau.beta, n_points=n_tau)
    G_minus_tau.data[:,0,0] = sign * np.flipud(G_tau.data[:,0,0])
    for m in range(G_tau.tail.order_min, G_tau.tail.order_max + 1):
        G_minus_tau.tail[m] = (-1)**m * G_tau.tail[m]
    return G_minus_tau

def trapez(X, dtau):
    if len(X) < 2: return 0
    I = dtau * np.sum(X[1:-1])
    I += 0.5 * dtau * (X[0] + X[-1])
    return I

def integration(X_tau):
    dtau = X_tau.beta / (n_tau - 1)
    return trapez(X_tau.data[:,0,0], dtau)

def change_statistic(X_tau):
    statistic = X_tau.statistic
    if statistic == "Fermion":
        new_stat = "Boson"
    else:
        new_stat = "Fermion"
    Y_tau = GfImTime(indices=indices, beta=X_tau.beta, n_points=n_tau, statistic=new_stat)
    Y_tau.data[:] = X_tau.data[:]
    Y_tau.tail = X_tau.tail
    return Y_tau

def convolution(X_tau, Y_tau, statistic="Fermion"):
    X_iw = GfImFreq(indices=indices, beta=X_tau.beta, n_points=n_iw, statistic=statistic)
    Y_iw = GfImFreq(indices=indices, beta=X_tau.beta, n_points=n_iw, statistic=statistic)
    X_iw << Fourier(X_tau if X_tau.statistic == statistic else change_statistic(X_tau))
    Y_iw << Fourier(Y_tau if Y_tau.statistic == statistic else change_statistic(Y_tau))
    Z_tau = GfImTime(indices=indices, beta=X_tau.beta, n_points=n_tau, statistic=statistic)
    Z_tau << InverseFourier(X_iw * Y_iw)
    return Z_tau if X_tau.statistic == statistic else change_statistic(Z_tau)

def convolution_direct(X_tau, Y_tau, statistic="Fermion"):
    X, Y = X_tau.data[:,0,0], Y_tau.data[:,0,0]
    dtau = X_tau.beta / (n_tau - 1)
    sign = -1 if statistic == "Fermion" else 1
    Z = np.zeros(n_tau)
    for i in range(n_tau):
        Z[i] = (trapez(np.flipud(X[:i+1]) * Y[:i+1], dtau)
                + sign * trapez(np.flipud(X[i:]) * Y[i:], dtau)).real
    Z_tau = GfImTime(indices=indices, beta=X_tau.beta, n_points=n_tau)
    Z_tau.data[:,0,0] = Z
    # Warning: the tail is not set properly!
    return Z_tau

def test_convolution(G_tau):
    G_minus_tau = reverse_tau(G_tau)
    def print_results(C_tau):
        print(C_tau.data[:3,0,0].real, C_tau.data[20:23,0,0].real, C_tau.data[-3:,0,0].real)
    C_tau = convolution_direct(G_tau * G_tau, G_tau * G_tau, "Boson")
    print_results(C_tau)
    C_tau << convolution(G_tau * G_tau, G_tau * G_tau, "Boson")
    print_results(C_tau)
    C_tau = convolution_direct(G_minus_tau * G_tau, G_minus_tau * G_tau, "Boson")
    print_results(C_tau)
    C_tau << convolution(G_minus_tau * G_tau, G_minus_tau * G_tau, "Boson")
    print_results(C_tau)
    C_tau = convolution_direct(G_tau, G_tau)
    print_results(C_tau)
    C_tau << convolution(G_tau, G_tau)
    print_results(C_tau)
    C_tau = convolution_direct(G_minus_tau, G_minus_tau)
    print_results(C_tau)
    C_tau << convolution(G_minus_tau, G_minus_tau)
    print_results(C_tau)
    C_tau = convolution_direct(G_tau * G_tau * G_minus_tau, G_tau)
    print_results(C_tau)
    C_tau << convolution(G_tau * G_tau * G_minus_tau, G_tau)
    print_results(C_tau)

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

# Self energy (iw dependent)
def first_order_Sigma(G_tau, U):
    n = G_tau.data[0,0,0].real + 1
    return U * (n - 0.5)

def second_order_Sigma(G_tau, U, only_skeleton=False):
    Sigma_tau = GfImTime(indices=indices, beta=G_tau.beta, n_points=n_tau)
    G_minus_tau = reverse_tau(G_tau)
    Sigma_tau << -U**2 * G_tau * G_tau * G_minus_tau
    # non-skeleton contributions
    Hartree = 0
    if not only_skeleton:
        Hartree = U * first_order_Sigma(G_tau, U) * integration(G_minus_tau * G_tau)
    return Fourier(Sigma_tau) + Hartree

def third_order_Sigma(G_tau, U, only_skeleton=False):
    Sigma_tau = GfImTime(indices=indices, beta=G_tau.beta, n_points=n_tau)
    G_minus_tau = reverse_tau(G_tau)
    # skeleton contributions 3a and 3b
    Sigma = U**3 * G_tau * convolution(G_tau * G_minus_tau, G_tau * G_minus_tau, "Boson")
    Sigma +=  U**3 * G_minus_tau * convolution(G_tau * G_tau, G_tau * G_tau, "Boson")
    # non-skeleton contributions
    Hartree = 0
    if not only_skeleton:
        tadpole = first_order_Sigma(G_tau, U)
        # Diagrams 3c and 3e
        X_tau = convolution(G_tau, G_tau)
        Sigma += -tadpole * U**2 * G_tau * G_minus_tau * X_tau * 2
        # Diagram 3d
        Sigma += -tadpole * U**2 * G_tau * G_tau * reverse_tau(X_tau)
        # Hartree diagrams 3a, 3b, 3c
        Hartree += tadpole * U**2 * integration(G_minus_tau * G_tau)**2
        Hartree += tadpole**2 * U * integration(G_minus_tau * X_tau)
        X_tau = convolution(G_tau * G_tau * G_minus_tau, G_tau)
        Hartree += -U**3 * integration(G_minus_tau * X_tau)
    Sigma_tau << Sigma
    return Fourier(Sigma_tau) + Hartree

# Solve Dyson equation for G
def solve_Dyson_for_G(eps, Delta_iw, Sigma_iw):
    return inverse(iOmega_n - eps - Delta_iw - Sigma_iw)

def get_G_tau(eps, U, Delta_iw, self_consistent=False):
    tol, n_iter_max = 1e-5, 40

    beta = Delta_iw.beta
    G_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau, name=r"$G(\tau)$")
    G_tau_prev = GfImTime(indices=indices, beta=beta, n_points=n_tau)
    G_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    Sigma_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)

    G_iw << generate_G_0(eps, Delta_iw)
    G_tau_prev << InverseFourier(G_iw)
    for i in range(n_iter_max):
        Sigma_iw << (first_order_Sigma(G_tau_prev, U)
                     + second_order_Sigma(G_tau_prev, U, only_skeleton=self_consistent)
                     + third_order_Sigma(G_tau_prev, U, only_skeleton=self_consistent))
        G_iw << solve_Dyson_for_G(eps, Delta_iw, Sigma_iw)
        G_tau << InverseFourier(G_iw)
        if np.allclose(G_tau_prev.data, G_tau.data, atol=tol) or not self_consistent:
            print("Converged in iteration {}".format(i))
            return G_tau
        else:
            G_tau_prev << 0.8 * G_tau + 0.2 * G_tau_prev
    print("Solution not converged!")
    return G_tau

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

# Function to export

def export_Delta_tau(beta, discrete_bath=None, continuous_bath=None,
                     filename="Delta.txt", only_gf=False,
                     skip_factor=skip_factor_Delta):

    Delta_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    Delta_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau, name=r"$\Delta(\tau)$")

    if(discrete_bath != None):
        e_list, V_list = discrete_bath
        Delta_iw << Delta_from_lists(e_list, V_list)

    if(continuous_bath != None):
        Gamma_func, integrated_Gamma, E_max = continuous_bath
        Delta_iw << Delta_from_func(Gamma_func, integrated_Gamma, E_max, beta)

    Delta_tau << InverseFourier(Delta_iw)
    save_gf(filename, Delta_tau, only_gf=only_gf, skip_factor=skip_factor_Delta)

def weak_coupling_gf(beta, U, eps, discrete_bath=None, continuous_bath=None,
                     Delta=None, delta_tau=None,
                     filenames=("G_0_weak.txt", "G_weak.txt"), plot=False, only_gf=False):

    Delta_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    G_0_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    G_0_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau, name=r"$G_0(\tau)$")

    if(discrete_bath != None):
        e_list, V_list = discrete_bath
        Delta_iw << Delta_from_lists(e_list, V_list)

    if(continuous_bath != None):
        Gamma_func, integrated_Gamma, E_max = continuous_bath
        Delta_iw << Delta_from_func(Gamma_func, integrated_Gamma, E_max, beta)

    if(Delta != None):
        Delta_iw << Delta

    if(delta_tau != None):
        Delta_tau = GfImTime(indices=indices, beta=beta, n_points=len(delta_tau))
        Delta_tau.data[:,0,0] = delta_tau
        Delta_iw << Fourier(Delta_tau)
        Delta_iw.fit_tail(TailGf(1, 1, 0), 4, 8, 28)

    G_0_iw << generate_G_0(eps, Delta_iw)
    G_0_tau << InverseFourier(G_0_iw)
    G_tau = get_G_tau(eps, U, Delta_iw, self_consistent=False)

    # tail = TailGf(1, 1, 3)
    # tail[1][0, 0] = 1
    # print(tail)
    # G_1_tau = GfImTime(indices=indices, beta=beta, n_points=n_tau, name=r"$G_0(\tau)$")
    # G_1_tau.data[:] = G_tau.data
    # G_1_tau.tail[1][0,0] = 1
    # G_1_iw = GfImFreq(indices=indices, beta=beta, n_points=n_iw)
    # G_1_iw << Fourier(G_tau)
    # print(G_1_iw.tail)
    # G_1_iw.fit_tail(tail, 8, 3, 30)
    # print(G_1_iw.tail)
    # print(G_tau.tail)


    # TESTING CONVOLUTION
    # test_convolution(G_0_tau)

    save_gf(filenames[0], G_0_tau, only_gf=only_gf, skip_factor=skip_factor)
    save_gf(filenames[1], G_tau, only_gf=only_gf, skip_factor=skip_factor)

    if plot:
        plot_gf(G_0_tau)
        plot_gf(G_tau)
        plt.legend()
        plt.title("Second-order perturbation theory in $U$")
        plt.show()
        plt.close()

    return G_tau.data[::skip_factor,0,0].real


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

    export_Delta_tau(beta, discrete_bath=[e_list, V_list], filename=Delta_filename)
    weak_coupling_gf(beta, U, eps, discrete_bath=[e_list, V_list], plot=False)
    #weak_coupling_gf(beta, U, eps, continuous_bath=[Gamma_func(D), integrated_Gamma(D), E_max(D)], plot=False)
