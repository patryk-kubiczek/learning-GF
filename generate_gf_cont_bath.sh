#!/bin/bash       

rm -rf data_cont_bath 
mkdir data_cont_bath

python3 generate_params_cont_bath.py

source ./load_triqs.sh
pytriqs generate_Delta_and_weak_cont_bath.py 
mpirun -np 4 pytriqs generate_QMC_cont_bath.py 
source ./unload_triqs.sh

python3 generate_strong_cont_bath.py

#rm -f Delta*.csv

# Around 5.5 min per parameters' set
