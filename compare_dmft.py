import numpy as np
from matplotlib import pyplot as plt
from params import *

Delta_NN = []
Delta_QMC = []

n_iter = 6

for i in range(n_iter):
    tau_NN, delta = np.loadtxt(DMFT_Delta_filename(i), unpack=True)
    Delta_NN.append(delta)
    tau_QMC, delta = np.loadtxt(DMFT_Delta_filename("QMC_" + str(i)), unpack=True)
    Delta_QMC.append(delta)

for i in range(n_iter):
    plt.plot(tau_QMC, Delta_QMC[i], '-', label="QMC " + str(i))
    plt.plot(tau_NN, Delta_NN[i], '--', label="NN " + str(i))
plt.legend(loc="best")
plt.show()
plt.close()
