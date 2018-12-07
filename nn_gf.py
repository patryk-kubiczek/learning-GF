import numpy as np
import keras
from learning import max_error, boundary_cond, transform, back_transform
from scipy import interpolate

preprocessing = "shift_and_rescale"

def shorten_gf(tau, gf, new_n_tau):
    skip_factor = (len(tau) - 1) // (new_n_tau - 1)
    return tau[::skip_factor], gf[::skip_factor]

def load_nn_model(filename):
    return keras.models.load_model(filename, custom_objects={'max_error': max_error,
                                                                'boundary_cond': boundary_cond})

def nn_gf(weak_gf, strong_gf, model):
    n_tau = len(weak_gf)
    X = np.zeros((1, 2 * n_tau))
    X[0, :n_tau] = weak_gf
    X[0, n_tau:] = strong_gf
    X = transform(X, preprocessing)
    Y = model.predict(X)
    Y = back_transform(Y, preprocessing)
    return Y[0]

def interpolate_gf(tau, gf, new_n_tau):
    tck = interpolate.splrep(tau, gf, s=0.000001)
    new_tau = np.linspace(0, tau[-1], num=new_n_tau, endpoint=True)
    new_gf = interpolate.splev(new_tau, tck, der=0)
    return new_tau, new_gf

def save_gf(filename, tau, G) :
    if single_spin == True:
        data = np.asarray([tau, G[0]])
    else:
        data = np.asarray([tau, G[0], G[1]])
    if not only_gf:
        np.savetxt(filename, data.T)
    else:
        with open(filename, 'ab') as f:
            np.savetxt(f, data[1][::skip_factor, np.newaxis].T, delimiter=",", fmt="%1.4f")




if __name__ == "__main__":


    from params import *

    tau_strong, strong_gf = np.loadtxt(G_strong_filename, unpack=True)
    tau_weak, weak_gf = np.loadtxt(G_weak_filename, unpack=True)

    n_tau = 51
    tau, strong_gf = shorten_gf(tau_strong, strong_gf, n_tau)
    tau, weak_gf = shorten_gf(tau_weak, weak_gf, n_tau)

    model = load_nn_model("model.h5")

    gf = nn_gf(weak_gf, strong_gf, model)
    new_tau, new_gf = interpolate_gf(tau, gf, 501)

    from matplotlib import pyplot as plt

    plt.plot(tau, gf, '.')
    plt.plot(new_tau, new_gf)
    plt.show()
