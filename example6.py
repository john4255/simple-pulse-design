import numpy as np

from model import CGTS

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 290.0, 10)
times = np.r_[rampup_times, flattop_times]

# Load gEQDSK
g_arr_rampup = [f'ramp4/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['Lmode_ex.eqdsk'] * 10
g_arr = np.r_[g_arr_rampup, g_arr_flattop]

# Set heating
# eccd = 20.0E6
eccd_powers = {0: 0, 99: 0, 100: 0.0E6}
nbi_powers = {0: 0, 99: 0, 100: 10.0E6}

# Set current
ip = {0: 3.0E6, 10: 3.0E6, 100: 5.0E6}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 6.0}
T_e_ped = {0: 0.5, 100: 0.5, 105: 6.0}

mysim = CGTS(300, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_ip(ip)
mysim.set_heating(nbi=nbi_powers, eccd=eccd_powers, eccd_loc=0.35) # TODO: check if ECCD is working
# mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)

mysim.fly(save_states=True, graph=False)