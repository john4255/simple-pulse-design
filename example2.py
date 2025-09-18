import numpy as np

from model import CGTS

# Test increase + decrease aux heating

# Set timesteps
rampup_times = np.linspace(10.0, 90.0, 10)
region_1 = np.linspace(110.0, 140.0, 5)
region_2 = np.linspace(160.0, 190.0, 5)
region_3 = np.linspace(210.0, 240.0, 5)
region_4 = np.linspace(260.0, 290.0, 5)
times = np.r_[rampup_times, region_1, region_2, region_3, region_4]

# Load gEQDSK
g_arr_rampup = [f'ramp7/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['Lmode_ex.eqdsk'] * 20
g_arr = np.r_[g_arr_rampup, g_arr_flattop]

# Set heating
nbi_powers = {0: 0.0, 124: 0.0, 125: 40.0E6, 149: 40.0E6, 150: 10.0E6, 199: 10.0E6, 200: 30.0E6, 249: 30.0E6, 250: 0.0}

# Set current
ip = {0: 3.0E6, 10: 3.0E6, 100: 10.0E6}

# Set pedestals
T_i_ped = {0: 0.5, 100: 0.5, 105: 3.0}
T_e_ped = {0: 0.5, 100: 0.5, 105: 3.0}

# Set Profiles
T_i = {0.0: {0.0: 6.0, 1.0: 0.2}}
T_e = {0.0: {0.0: 6.0, 1.0: 0.2}}
n_e =  {0: {0.0: 1.3, 1.0: 1.0}}

mysim = CGTS(300, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_ip(ip)
mysim.set_heating(nbi=nbi_powers)
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)
mysim.set_density(n_e)
mysim.set_Te(T_e)
mysim.set_Ti(T_i)

mysim.fly(save_states=True, graph=False)