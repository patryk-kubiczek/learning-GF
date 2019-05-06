import numpy as np
from matplotlib import pyplot as plt
from params import *

Delta_NN = []
Delta_QMC = []

n_iter = 6

for i in range(n_iter + 1):
    tau_NN, delta = np.loadtxt(DMFT_Delta_filename(i), unpack=True)
    Delta_NN.append(delta / t**2)
    tau_QMC, delta = np.loadtxt(DMFT_Delta_filename("QMC_" + str(i)), unpack=True)
    Delta_QMC.append(delta / t**2)

for i in range(n_iter + 1):
    plt.plot(tau_QMC, Delta_QMC[i], '-', label="QMC " + str(i))
    plt.plot(tau_NN, Delta_NN[i], '--', label="NN " + str(i))

plt.legend(loc="best")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$G(\tau)$")
plt.ylim([-0.7,-0.1])
plt.tight_layout()
plt.savefig("g_dmft_iterations.png", dpi=300)
plt.close()

plt.plot(tau_QMC, Delta_QMC[n_iter], '-', label="QMC")
plt.plot(tau_NN, Delta_NN[n_iter], '--', label="NN")
error = max(np.abs(Delta_QMC[n_iter][::3] - Delta_NN[n_iter]))
bc = Delta_NN[n_iter][0] + Delta_NN[n_iter][-1]
plt.title("Bethe-lattice DMFT \n"
          + r"$U={:.2f},\varepsilon={:.2f},D={:.2f}$".format(
              U, eps, D) +  "\n" +
          "max error = {:.5f}".format(error) + "\n" +
        r"$G(0) + G(\beta) = {:.5f}$".format(bc), fontsize='medium')
plt.legend(loc="best")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$G(\tau)$")
plt.ylim([-0.7,-0.1])
plt.tight_layout()
plt.savefig("g_dmft.png", dpi=300)
#plt.show()
plt.close()




w, A = np.loadtxt("spectral_function.txt", unpack=True)
w, A_QMC = np.loadtxt("spectral_function_QMC.txt", unpack=True)
plt.plot(w, A_QMC, label="QMC")
plt.plot(w, A, label="NN")
plt.legend(loc='best')
plt.xlabel(r"$\omega$")
plt.ylabel(r"$A(\omega)$")
plt.title("Bethe-lattice DMFT \n"
          + r"$U={:.2f},\varepsilon={:.2f},D={:.2f}$".format(
              U, eps, D))

plt.tight_layout()
plt.savefig("spectrum_dmft.png", dpi=300)

plt.close()
