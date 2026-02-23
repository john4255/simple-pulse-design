import numpy as np

from model import CGTS

# Exact reproduciton of Capser (2014) (no density or temperature evolution)
# https://iopscience.iop.org/article/10.1088/0029-5515/54/1/013005/meta

# Set timesteps
rampup_times = np.linspace(5.0, 80.0, 10)
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times = np.linspace(500.0, 600.0, 10)
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
# g_arr_rampup = [f'eqdsk/iter_i={i}.eqdsk' for i in range(10)]
g_arr_rampup = [f'rampup/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['eqdsk/Hmode.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set current
ip = {0: 3.0E6, 5: 3.0E6, 80: 15.0E6, 500: 15.0E6, 590: 4.0E6, 600: 4.0E6}

# Set heating
powers = {0: 0, 24: 0, 25: 10.0E6, 79: 10.0E6, 80: 52.0E6, 124: 52.0E6, 125: 40.0E6, 500: 40.0E6, 524: 40.0E6, 525: 35.0E6, 549: 35.0E6, 550: 30.0E6}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Set pedestals
T_i_ped = {0: 0.146, 80: 0.146, 85: 3.69, 500: 3.69, 505: 0.146}
T_e_ped = {0: 0.220, 80: 0.220, 85: 3.69, 500: 3.69, 505: 0.220}
n_e_ped = {0: 1.821E19, 79: 1.821E19, 80: 7.482E19}

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

T_e_80s = get_data('T_e_80s.txt', 1.0)
T_e_300s = get_data('T_e_300s.txt', 1.0)
T_i_80s = get_data('T_i_80s.txt', 1.0)
T_i_300s = get_data('T_i_300s.txt', 1.0)

ne_right_bc = {0: 0.157E20, 79: 0.157E20, 80: 0.414E20}
Te_right_bc = 0.01
Ti_right_bc = 0.01

# Run sim
mysim = CGTS(0, 600, times, g_arr)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')

coil_names = ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents)

mysim.set_Ip(ip)
mysim.set_Zeff(1.8)
mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_density({0: n_e_80s, 79: n_e_80s, 80: n_e_300s, 499: n_e_300s, 500: n_e_80s})
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped, n_e_ped=n_e_ped)
mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
mysim.set_nbar({0: 0.326E20, 80: .905E20})

mysim.set_Te({0: T_e_80s, 79: T_e_80s, 80: T_e_300s })
mysim.set_Ti({0: T_i_80s, 79: T_i_80s, 80: T_i_300s })
mysim.set_evolve(density=False, Ti=False, Te=False)

mysim.fly(save_states=True, graph=False)