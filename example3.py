import numpy as np

from model import DISMAL

# Reproduciton of Capser (2014) with density evolution
# https://iopscience.iop.org/article/10.1088/0029-5515/54/1/013005/meta

# Set timesteps
rampup_times = np.linspace(5.0, 80.0, 9) # changed to 9 to ignore i=0 equil
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times = np.linspace(500.0, 600.0, 9) # changed to 9 to match rampup
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
# g_arr_rampup = [f'eqdsk/iter_i={i}.eqdsk' for i in range(10)]
# g_arr_rampup = [f'eqdsk/iter_i={i}.eqdsk' for i in range(10)]
g_arr_rampup = [f'rampup_test/iter_i={i}.eqdsk' for i in range(1,10)] # start at i=1 to skip bad equil at i=0
g_arr_flattop = ['eqdsk/Hmode.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set current
ip = {0: 3.0E6, 5: 3.0E6, 80: 15.0E6, 500: 15.0E6, 590: 4.0E6, 600: 4.0E6}
# ip = {0: 2881088.122, 5: 2881088.122, 80: 15411518.204, 500: 15398677.219, 590: 5255632.946, 600: 5255632.946} # new targets to match output of TM when passing jphi instead of ffp 2026-02-04

# Set heating
powers = {0: 0, 24: 0, 25: 10.0E6, 79: 10.0E6, 80: 52.0E6, 124: 52.0E6, 125: 40.0E6, 500: 40.0E6, 524: 40.0E6, 525: 35.0E6, 549: 35.0E6, 550: 30.0E6}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Set pedestals
T_i_ped = {0: 0.146, 80: 0.146, 85: 3.69, 500: 3.69, 505: 0.146}
T_e_ped = {0: 0.220, 80: 0.220, 85: 3.69, 500: 3.69, 505: 0.220}
n_e_ped = {0: 1.821E19, 79: 1.821E19, 80: 7.482E19, 500: 7.482E19, 505: 1.821E19}

# Set boundary conditions
# ne_right_bc = 1.0E18
ne_right_bc = {0: 0.157E20, 79: 0.157E20, 80: 0.414E20, 500: 0.414E20, 505: 0.157E20}
Te_right_bc = 0.01
Ti_right_bc = 0.01

# Run sim
t_res = np.arange(0.0, 600.0, 20.0)
mysim = DISMAL(0, 600, eqtimes=times, g_eqdsk_arr=g_arr, times=t_res, dt=1.0, last_surface_factor=0.9)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
coil_names = ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents, strict_limit=1.0E8)

mysim.set_Ip(ip)
mysim.set_Zeff(1.8)
mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped, n_e_ped=n_e_ped)
mysim.set_nbar({0: 0.326E20, 80: .905E20})

mysim.fly(save_states=True, graph=False)