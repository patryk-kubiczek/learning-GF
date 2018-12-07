import numpy as np
from subprocess import Popen
from strong_coupling import strong_coupling_gf
from nn_gf import load_nn_model, nn_gf, shorten_gf, interpolate_gf
from learning import max_error, boundary_cond

from params import *

alpha = 0.80
n_tau = 2001
n_tau_NN = 51
n_tau_NCA = 401

def generate_initial_Delta():
    p = Popen("source load_triqs.sh && python2 initial_Delta.py",
              shell=True, executable='/bin/bash')
    p.wait()

def run_weak_coupling_solver(Delta_filename):
    p = Popen("source load_triqs.sh && python2 weak_gf_from_Delta.py {}".format(Delta_filename),
              shell=True, executable='/bin/bash')
    p.wait()

def solve_impurity(nn_model, Delta_filename):
    run_weak_coupling_solver(Delta_filename)

    tau, Delta = np.loadtxt(Delta_filename, unpack=True)
    tau, Delta_NCA = shorten_gf(tau, Delta, n_tau_NCA)

    strong_coupling_gf(beta, U, eps, tau, Delta_NCA)

    tau, strong_gf = np.loadtxt(G_strong_filename, unpack=True)
    tau, strong_gf = shorten_gf(tau, strong_gf, n_tau_NN)

    tau, weak_gf = np.loadtxt(G_weak_filename, unpack=True)
    tau, weak_gf = shorten_gf(tau, weak_gf, n_tau_NN)

    gf = nn_gf(weak_gf, strong_gf, model)
    tau, gf = interpolate_gf(tau, gf, n_tau)

    return tau, gf, Delta


def DMFT_iteration(i, nn_model):
    tau, gf, Delta = solve_impurity(nn_model, DMFT_Delta_filename(i - 1))
    Delta[:] = (1 - alpha) * Delta[:] + alpha * t**2 * gf[:]
    np.savetxt(DMFT_Delta_filename(i), np.array([tau, Delta]).T)
    return Delta


model = load_nn_model('model.h5')
n_iter = 7

Delta_iter = np.zeros((n_iter, n_tau))

generate_initial_Delta()
tau, Delta_iter[0, :] = np.loadtxt(DMFT_Delta_filename(0), unpack=True)

for i in range(1, n_iter):
    Delta_iter[i, :] = DMFT_iteration(i, model)

from matplotlib import pyplot as plt

for i in range(n_iter):
    plt.plot(tau, Delta_iter[i, :], label="{}".format(i))
plt.legend(loc='best')
plt.show()
