import numpy as np

# Hamiltonian
# H = sum_sigma eps * n_sigma + U * (n_up - 0.5) * (n_down - 0.5) + H_bath + H_hyb

n_tau = 401
skip_factor = 2

# Generating P_0
def generate_P_0(eps, U, tau):
    P_0 = np.zeros((4, len(tau)))
    energies = [0, eps - U / 2, eps - U / 2, 2 * eps]
    for p, e in zip(P_0, energies):
        p[:] = np.exp(-tau * e)
    return P_0

# Propagator self energy

def generate_self_energy(P, Delta):
    n_tau = len(Delta)
    Q = np.zeros((4, n_tau))
    Q[0] = -(P[1] + P[2]) * np.flipud(Delta)
    Q[1] = -P[0] * Delta - P[3] * np.flipud(Delta)
    Q[2] = -P[0] * Delta - P[3] * np.flipud(Delta)
    Q[3] = -(P[1] + P[2]) * Delta
    return Q

# Solving Volterra equation

def second_order_w(n):
    w = np.zeros((n, n))
    # trapezoid rule
    for i in range(1, n):
        w[i, 0] = 0.5
        w[i, 1:i] = 1
        w[i, i] = 0.5
    return w

def fourth_order_w(n):
    w = np.zeros((n, n))
    # i = 1 (trapezoid rule)
    w[1, 0:2] = 1 / 2
    # i = 2 (Simpson's rule)
    w[2, 0] = 1 / 3
    w[2, 1] = 4 / 3
    w[2, 2] = 1 / 3
    # i = 3 (Simpson's 3/8 rule)
    w[3, 0] = 3 / 8
    w[3, 1:3] = 9 / 8
    w[3, 3] = 3 / 8
    # i = 4 (composite Simpson's rule)
    w[4, 0] = 1 / 3
    w[4, 1] = 4 / 3
    w[4, 2] = 2 / 3
    w[4, 3] = 4 / 3
    w[4, 4] = 1 / 3
    # i >= 5 (fourth-order Gregory's rule)
    for i in range(5, n):
        w[i, 0] = 3 / 8
        w[i, 1] = 7 / 6
        w[i, 2] = 23 / 24
        w[i, 3:i-2] = 1
        w[i, i - 2] = 23 / 24
        w[i, i - 1] = 7 / 6
        w[i, i] = 3 / 8
    return w


def solve_Volterra(y_0, f, k, dx, scheme_order=4):
    """ Solve Volterra Integral-Differential Equation:
    y'(x) = f(x) y(x) + int_0_x dx' k(x - x') y(x')
    """
    n_x = len(f)
    if(scheme_order == 2):
        dw = dx * second_order_w(n_x)
    if(scheme_order == 4):
        dw = dx * fourth_order_w(n_x)
    y = np.zeros(n_x)
    yx = np.zeros(n_x)
    # i = 0 (initial condition)
    y[0] = y_0
    yx[0] = f[0] * y_0
    if(scheme_order == 4):
        # i = 1, 2 (Simpson's rule with middle point + quadratic interpolation)
        k12 = 3 / 8 * k[0] + 3 / 4 * k[1] - 1 / 8 * k[2]
        A = np.array([[1, -2 / 3 * dx, 0, dx / 12],
                      [-f[1] - dx * k12 / 2 - dx * k[0] / 6, 1, dx * k12 / 12, 0],
                      [0, -4 / 3 * dx, 1, -dx / 3],
                      [-4 / 3 * dx * k[1], 0, -f[2] - dx * k[0] / 3, 1]])
        B = np.array([y[0] + 5 / 12 * dx * yx[0],
                      (dx * k[1] / 6 + dx * k12 / 4) * y[0],
                      y[0] + dx * yx[0] / 3,
                      dx * k[2] * y[0] / 3])
        y[1], yx[1], y[2], yx[2] = np.linalg.solve(A, B)
    for i in range(3 if scheme_order == 4 else 1, n_x):
        j_start = max(i - (5 if scheme_order == 4 else 1), 0)
        # int_1 = sum((dw[i, j] - dw[i-1, j]) * yx[j] for j in range(j_start, i))
        # int_2 = sum(dw[i, j] * k[i-j] * y[j] for j in range(i))
        int_1 = (dw[i, j_start:i] - dw[i-1, j_start:i]) @ yx[j_start:i]
        int_2 = (dw[i, :i] * np.flipud(k[1:i+1])) @ y[:i]
        y[i] = y[i-1] + int_1 + dw[i, i] * int_2
        y[i] /= 1 - dw[i, i] * f[i] - (dw[i, i])**2 * k[0]
        yx[i] = f[i] * y[i] + int_2 + dw[i, i] * k[0] * y[i]
    return y, yx

# Dyson equation for the NCA propagator

def solve_Dyson_for_P(eps, U, Q, tau, order=4):
    dtau = tau[-1] / (len(tau) - 1)
    P = np.zeros((4, len(tau)))
    dP = np.zeros((4, len(tau)))
    energies = [0, eps - U / 2, eps - U / 2, 2 * eps]
    for p, dp, q, e in zip(P, dP, Q, energies):
        # y(x) = P(tau), y(0) = 1, f(x) = -energy, k(x) = Q(tau)
        f = np.full((len(tau),), -e)
        p[:], dp[:] = solve_Volterra(1, f, q, dtau, scheme_order=order)
    return P, dP


def get_propagator(eps, U, Delta, tau, self_consistent=True, order=4):
    tol, n_iter_max = 1e-5, 40
    P_prev = generate_P_0(eps, U, tau)
    for i in range(n_iter_max):
        Q = generate_self_energy(P_prev, Delta)
        P = solve_Dyson_for_P(eps, U, Q, tau, order=order)[0]
        if np.allclose(P_prev, P, atol=tol) or not self_consistent:
            print("Converged in iteration {}".format(i))
            return P
        else:
            P_prev[:] = 0.8 * P + 0.2 * P_prev
    print("Solution not converged!")
    return P

# Green functions and static observables

def get_gf(P):
    G = np.zeros((2, P.shape[1]))
    Z = np.sum(P[:, -1])
    G[0, :] = -(np.flipud(P[0, :]) * P[1, :] + np.flipud(P[1, :]) * P[3, :]) / Z
    G[1, :] = -(np.flipud(P[0, :]) * P[2, :] + np.flipud(P[2, :]) * P[3, :]) / Z
    return G

def expectation_value(A, P):
    Z = np.sum(P[:, -1])
    return np.sum(P[:, -1] * A[:]) / Z


# Tools

from matplotlib import pyplot as plt

def plot_P(tau, P):
    for n, p in enumerate(P):
        plt.plot(tau, p, '.', label="$P_{}$".format(n))
    plt.legend()
    plt.show()
    plt.close()

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

def strong_coupling_gf(beta, U, eps, tau, Delta,
                       filenames=("G_0_strong.txt", "G_strong.txt"), plot=False, only_gf=False):
    P_0 = generate_P_0(eps, U, tau)
    G_0 = get_gf(P_0)
    P = get_propagator(eps, U, Delta, tau, self_consistent=True, order=4)
    G = get_gf(P)

    save_gf(filenames[0], tau, G_0, only_gf=only_gf, skip_factor=skip_factor)
    save_gf(filenames[1], tau, G, only_gf=only_gf, skip_factor=skip_factor)

    #print(expectation_value([0,0,0,1], P_0), expectation_value([0,0,0,1], P))

    if plot:
        plot_gf(tau, G_0, " (0)")
        plot_gf(tau, G, " (NCA)")
        plt.legend()
        plt.title("NCA")
        plt.show()
        plt.close()

    return G[0, ::skip_factor]


if __name__ == "__main__":
    # Input - example
    # beta = 1.0
    # # Local
    # U = 4.0
    # eps = 0

    from params import *
    # Bath
    tau, Delta = np.loadtxt(Delta_filename, unpack=True)

    strong_coupling_gf(beta, U, eps, tau, Delta, plot=False)

