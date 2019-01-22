from generate_params import *

n_params = 50 

for _ in range(n_params):
    random_params_cont_bath(beta=beta,
                            U_range=[1., 8.],
                            eps_range=[-1., 1.],
                            D_range=[2. , 8.],
                            filename=name("params_cont_bath", beta, 0,
                                          parent="data_cont_bath/"))

