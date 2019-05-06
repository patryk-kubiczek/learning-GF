import numpy as np
from matplotlib import pyplot as plt
import subprocess

generate = True

if generate:
    proc = subprocess.Popen("./generate_single_gf.sh")
    proc.wait()

files_list = [("G_ED.txt", "ED", '-'),
              ("G_strong.txt", "NCA", '-.'), ("G_0_strong.txt", "$\Gamma = 0$", '--'),
              ("G_weak.txt", "3PT", '-.'), ("G_0_weak.txt", "$U=0$", '--')]

for f, name, symbol in files_list:
    tau, G = np.loadtxt(f, unpack=True)
    plt.plot(tau, G, symbol, label=name)
plt.legend()
# plt.yscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$G(\tau)$")
plt.savefig("comparison.png")
plt.show()
plt.close()
