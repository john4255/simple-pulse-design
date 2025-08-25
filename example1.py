import numpy as np

from model import CGTS

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 140.0, 10)
times = np.r_[rampup_times, flattop_times]

# Load gEQDSK
g_arr_rampup = [f'ramp4/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['Lmode_ex.eqdsk'] * 10
g_arr = np.r_[g_arr_rampup, g_arr_flattop]

# Set heating
eccd_powers = {0: 0, 99: 0, 100: 20.0e6}
nbi_powers = {0: 0, 99: 0, 100: 33e6}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 3.0}
T_e_ped = {0: 0.5, 100: 0.5, 105: 3.0}

mysim = CGTS(150, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_heating(nbi=None, eccd=eccd_powers, eccd_loc=0.35) # TODO: check if ECCD is working
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)

mysim.fly(save_states=True, graph=False)