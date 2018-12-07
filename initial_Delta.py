from weak_coupling import export_Delta_tau
from params import *

export_Delta_tau(beta,
                 continuous_bath=[Gamma_func(D), integrated_Gamma(D), E_max(D)],
                 filename=DMFT_Delta_filename(0), skip_factor=1)
