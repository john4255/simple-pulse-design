import numpy as np

from model import CGTS

# Test 100s rampup

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 140.0, 10)
times = np.r_[rampup_times, flattop_times]

# Load gEQDSK
g_arr_rampup = [f'eqdsk/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['eqdsk/Hmode.eqdsk'] * 10
g_arr = np.r_[g_arr_rampup, g_arr_flattop]

# Set heating
eccd_powers = {0: 0, 99: 0, 100: 20.0e6}
nbi_powers = {0: 0, 99: 0, 100: 33e6}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 3.0}
T_e_ped = {0: 0.5, 100: 0.5, 105: 3.0}

# Set Profiles
T_i = {0.0: {0.0: 6.0, 1.0: 0.2}}
T_e = {0.0: {0.0: 6.0, 1.0: 0.2}}
n_e =  {0: {0.0: 1.3, 1.0: 1.0}}

mysim = CGTS(150, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_heating(nbi=None, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)
mysim.set_density(n_e)
mysim.set_Te(T_e)
mysim.set_Ti(T_i)

mysim.fly(save_states=True, graph=False)