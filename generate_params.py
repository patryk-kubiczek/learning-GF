from math import sqrt, pi
from mpi_tools import *
import random
import csv
import numpy as np

def random_params(beta, U_range, eps_range, D_range, V_range, N_range, filename="params.csv"):
    U = random.uniform(*U_range)
    eps = random.uniform(*eps_range)
    D = random.uniform(*D_range)
    N = random.randint(*N_range)
    V_list = (V_range[1] - V_range[0]) * np.random.random(N) + V_range[0]
    V_list *= sqrt(2 * D / pi / sum(v**2 for v in V_list))
    #print(2 * D / pi, sum(v**2 for v in V_list))
    e_list = np.sort(np.random.random(N))
    e_list -= sum(v**2 * e for v, e in zip(V_list, e_list)) / (2 * D / pi)
    e_list *= 2 * D / (e_list[-1] - e_list[0])
    #print(0, sum(v**2 * e for v, e in zip(V_list, e_list)))
    #print(2 * D, e_list[-1] - e_list[0])
    params = [U, eps, D, N] + e_list.tolist() + V_list.tolist()
    with open(filename, "ab") as f:
        np.savetxt(f, np.asarray(params)[:,np.newaxis].T, delimiter=",", fmt="%1.4f")

def random_params_cont_bath(beta, U_range, eps_range, D_range, filename="params_cont.csv"):
    U = random.uniform(*U_range)
    eps = random.uniform(*eps_range)
    D = random.uniform(*D_range)
    params = [U, eps, D]
    with open(filename, "ab") as f:
        np.savetxt(f, np.asarray(params)[:,np.newaxis].T, delimiter=",", fmt="%1.4f")

def read_params(filename):
    params_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for line in reader:
            params = {}
            params["U"] = float(line[0])
            params["eps"] = float(line[1])
            N = int(float(line[3]))
            params["e_list"] = [float(e) for e in line[4:4+N]]
            params["V_list"] = [float(v) for v in line[4+N:4+2*N]]
            params_list.append(params)
    return params_list

def read_params_cont_bath(filename):
    params_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for line in reader:
            params = {}
            params["U"] = float(line[0])
            params["eps"] = float(line[1])
            params["D"] = float(line[2])
            params_list.append(params)
    return params_list

samples_per_file = 1000
files_per_core = 1 


def name(prefix, beta, n, parent="data/"):
    return parent + prefix + "_beta_" + str(int(beta)) \
            + "_" + str(get_mpi_rank() * files_per_core + n) + ".csv"

beta = 20.



if __name__ == "__main__":

    random.seed(get_mpi_rank())
    np.random.seed(get_mpi_rank())
    for n in range(files_per_core):
        for _ in range(samples_per_file):
            random_params(beta=beta,
                          U_range=[1., 8.],
                          eps_range=[-1., 1.],
                          D_range=[2., 8.],
                          V_range=[0.75, 1.25],
                          N_range=[3, 5],
                          filename=name("params", beta, n))



