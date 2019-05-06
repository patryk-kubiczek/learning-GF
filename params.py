from __future__ import division
from math import sqrt, pi

beta = 1.  # inverse T

# Local
U = 6. # Hubbard U
eps = 1.  # Local energy measured from eps_0 = -U/2 (eps = 0 <=> half-filling)

# Bath
D = 4.  # half-bandwidth
t = D / 2.  # Bethe hopping
Gamma = 1.  # impurity-bath coupling

# Discrete bath
N = 4  # number of bath sites
e_list = [-D + (2 * D) / (N - 1) * i for i in range(N)] if N != 1 else [0]  # bath levels
V_list = [sqrt(2 * D * Gamma / pi / N)] * N if N > 0 else []  # impurity-bath hoppings

# Continuous bath

# Constant
# E_max = D
# def Gamma_func(x):
#     return Gamma if abs(x) <= D else 0
# integrated_Gamma = 2 * D * Gamma

# Semicircular
def E_max(D):
    return D - 0.00001
def Gamma_func(D):
    def func(x):
        Gamma = D / 2
        return Gamma * sqrt(1 - (x / D)**2)
    return func
def integrated_Gamma(D):
    Gamma = D / 2
    return pi / 2 * D * Gamma

# Numerical Delta(tau) 
Delta_filename = "Delta.txt"
G_weak_filename = "G_weak.txt"
G_strong_filename = "G_strong.txt"

def DMFT_Delta_filename(i):
    return "Delta_DMFT_{}.txt".format(i)

