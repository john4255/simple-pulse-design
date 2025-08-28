import numpy as np

from model import CGTS

# Reproduciton of Capser (2013)

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 490.0, 10)
rampdown_times = np.linspace(510.0, 590.0, 10)
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
g_arr_rampup = [f'ramp5/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['Lmode_ex.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set current
ip = {0: 4.0E6, 10: 4.0E6, 100: 15.0E6, 500: 15.0E6, 590: 4.0E6, 600: 4.0E6}

# Set heating
nbi_powers = {0: 0, 99: 0, 100: 40.0E6, 500: 40.0E6, 501: 0, 600: 0}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 4.5, 495: 4.5, 500: 0.5}
T_e_ped = {0: 0.5, 100: 0.5, 105: 4.5, 495: 4.5, 500: 0.5}

mysim = CGTS(600, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_ip(ip)
mysim.set_heating(nbi=nbi_powers)
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)

mysim.fly(save_states=True, graph=False)