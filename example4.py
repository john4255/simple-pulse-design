import numpy as np

from model import CGTS

# Reproduciton of Capser (2014)

# Set timesteps
rampup_times = np.linspace(5.0, 80.0, 10)
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times = np.linspace(500.0, 600.0, 10)
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
g_arr_rampup = ['ramp8/iter_i=0.eqdsk',
                'ramp8/iter_i=1.eqdsk',
                'ramp8/iter_i=2.eqdsk',
                'ramp8/iter_i=3.eqdsk',
                'ramp8/iter_i=4.eqdsk',
                'ramp8/iter_i=5.eqdsk',
                'ramp8/iter_i=6.eqdsk',
                'ramp8/iter_i=7.eqdsk',
                'ramp8/iter_i=8.eqdsk',
                'ramp8/iter_i=9.eqdsk']
g_arr_flattop = ['ramp8/Hmode.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set current
ip = {0: 3.0E6, 5: 3.0E6, 80: 15.0E6, 500: 15.0E6, 590: 4.0E6, 600: 4.0E6}

# Set heating
powers = {0: 0, 24: 0, 25: 10.0E6, 79: 10.0E6, 80: 52.0E6, 124: 52.0E6, 125: 40.0E6, 500: 40.0E6, 524: 40.0E6, 525: 35.0E6, 549: 35.0E6, 550: 30.0E6}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Set pedestals
T_i_ped = {0: 1.0, 80: 1.0, 85: 4.5, 500: 4.5}
T_e_ped = {0: 1.0, 80: 1.0, 85: 4.5, 500: 4.5}

# Set density profiles
def get_data(fname, mult):
    f = open(fname)
    raw = f.readlines()
    dict = {}
    for line in raw:
        x, y = line.split(',')
        x = float(x.strip())
        y = float(y.strip()) * mult
        dict[x] = y
    dict[1.0] = y

    return dict

n_e_80s = get_data('n_e_80s.txt', 1.0E20)
n_e_300s = get_data('n_e_300s.txt', 1.0E20)

# Set temp profiles
T_e_80s = get_data('T_e_80s.txt', 1.0)
T_i_80s = get_data('T_i_80s.txt', 1.0)

T_e_300s = get_data('T_e_300s.txt', 1.0)
T_i_300s = get_data('T_i_300s.txt', 1.0)

# Run sim
mysim = CGTS(600, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
mysim.set_ip(ip)
# mysim.set_density({80: n_e_80s, 300: n_e_300s})
# mysim.set_Te({80: T_e_80s, 300: T_e_300s})
# mysim.set_Ti({80: T_i_80s, 300: T_i_300s})
mysim.set_z_eff(1.8)
mysim.set_heating(nbi=nbi_powers, eccd=eccd_powers, eccd_loc=0.35) #, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped)

mysim.fly(save_states=True, graph=False)