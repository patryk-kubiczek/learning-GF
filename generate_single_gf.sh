#!/bin/bash       

source ./load_triqs.sh
pytriqs weak_coupling.py 
source ./unload_triqs.sh

python3 strong_coupling.py
python3 exact_diagonalization.py
