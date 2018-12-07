#!/bin/bash       
rm -rf data 
mkdir data

python3 generate_params.py

source ./load_triqs.sh
pytriqs generate_Delta_and_weak.py 
source ./unload_triqs.sh

python3 generate_strong_and_ED.py

#rm -f Delta*.csv

# Around 5.5 s per parameters' set
