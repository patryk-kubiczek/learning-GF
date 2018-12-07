import numpy as np
import argparse
from weak_coupling import weak_coupling_gf

from params import *

parser = argparse.ArgumentParser()
parser.add_argument('Delta_filename', type=str, default=Delta_filename)
args = parser.parse_args()

tau, Delta = np.loadtxt(args.Delta_filename, unpack=True)
weak_coupling_gf(beta, U, eps, delta_tau=Delta)
