import numpy as np

from model import CGTS

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 190.0, 10)
rampdown_times = np.linspace(210.0, 290.0, 10)
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
g_arr_rampup = [f'eqdsk/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['eqdsk/Hmode.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set heating
eccd_powers = {0: 0, 99: 0, 100: 20.0e6, 199: 20.0e6, 200: 0}
nbi_powers = {0: 0, 99: 0, 100: 33e6, 199: 33e6, 200: 0}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 3.0, 195: 3.0, 200: 0.5}
T_e_ped = {0: 0.5, 100: 0.5, 105: 3.0, 195: 3.0, 200: 0.5}

mysim = CGTS(300, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_heating(nbi=None, eccd=eccd_powers, eccd_loc=0.35) # TODO: check if ECCD is working
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)

mysim.fly(save_states=True, graph=False)