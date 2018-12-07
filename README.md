# learning-GF

## Goal

Predict exact impurity Green functions from approximate ones (NCA and perturbation theory in U) by training a neural network with Green functions obtained for small systems using ED.

## Repository Contents

### Parameters

`params.py`: Parameters for the impurity or lattice model (other scripts import them from here)

### Impurity solvers

`exact_diagonalization.py`: ED solver [dependency: QuSpin]

`strong_coupling.py`: NCA solver

`weak_coupling.py`: Perturbation theory in *U* up to third order [dependency: Python2, TRIQS]

`nn_gf.py`: Neural network based impurity solver, imports the trained model from `model.h5` [dependency: Keras, TensorFlow]

`qmc.py`: CT-HYB-QMC impurity solver, for benchmark purposes [dependency: Python2, TRQIS, TRIQS-CTHYB]

### Machine learning

`learning.py`: Setting up a neural network and the training process [dependency: Keras, TensorFlow]

### Scripts

#### Generating parameters

`generate_params.py`: Generate a set of parameters defining the training set

`generate_params_cont_bath.py`: Generate a set of parameters defining the benchmark set (continuous bath)


#### Generating Green functions

`generate_gf.sh`: Generate a training set of Green functions. Makes use of `generate_Delta_and_weak.py` and `generate_strong_and_ED.py`

`generate_gf_cont_bath.sh`: Generate a benchmark set of Green functions. Makes use of `generate_Delta_and_weak_cont_bath.py`, `generate_strong_cont_bath.py`, `generate_QMC_cont_bath.py`

`generate_single_gf.sh`: Generate a single set of Green functions

#### DMFT

`dmft_loop.py`: Perform a single DMFT calculation using NN as an impurity solver. Makes use of `initial_Delta.py`, `weak_gf_from_Delta.py`

`dmft_qmc.py`: Perform a single DMFT calculation using QMC and an impurity solver

#### Comparing

`compare_gf.py`: Compare a single set of Green functions

`compare_dmft.py`: Compare NN and QMC-based DMFT results

### Other

`mpi_tools.py`: Wrapper for MPI-related functions [dependency: mpi4py]

`load_triqs.sh`, `unload_triqs.sh`: Loading/unloading TRIQS (edit according to your environment)

`model.h5`: Parameters of the trained NN. Can be created by `learning.py`

