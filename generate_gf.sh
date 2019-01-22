#!/bin/bash  

#SBATCH -t 03:00:00
#SBATCH -N 1  
#SBATCH --tasks-per-node 40
#SBATCH -p medium 		

N=40

source load_dependencies.sh

rm -rf data 
mkdir data

mpirun -np $N python3 generate_params.py
mpirun -np $N python2 generate_Delta_and_weak.py 
mpirun -np $N python3 generate_strong_and_ED.py

#rm -f Delta*.csv

# Around 5.5 s per parameters' set
