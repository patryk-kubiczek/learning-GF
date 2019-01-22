#!/bin/bash       

#SBATCH -t 03:00:00
#SBATCH -N 1  
#SBATCH --tasks-per-node 40
#SBATCH -p medium 		

N=40

source load_dependencies.sh

rm -rf data_cont_bath 
mkdir data_cont_bath

python3 generate_params_cont_bath.py
python2 generate_Delta_and_weak_cont_bath.py 
mpirun -np $N python2 generate_QMC_cont_bath.py 
python3 generate_strong_cont_bath.py

#rm -f Delta*.csv

# Around 5.5 min per parameters' set
