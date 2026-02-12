import numpy as np

from model import DISMAL

# new ITER pulse using q95 stable rampup (after q95 has decreased from large value)
# based entirely on guess and check for relative static q95 to get to ITER example ipynb made by Daniel Burgess (13MA)

# Set timesteps
rampup_times = np.linspace(0.0, 80.0, 10)  
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times = np.linspace(500.0, 600.0, 10) 
times = np.r_[rampup_times, flattop_times, rampdown_times]

# Load gEQDSK
g_arr_rampup = [f'2026-02-10_iter_rampup_new/out/iter_new_i={i}.eqdsk' for i in range(0,10)] 
g_arr_flattop = [f'2026-02-10_iter_rampup_new/out/flattop.eqdsk'] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]

# Set current
ip = {0: 2.0E6, 80: 13.0E6, 520: 13.0E6, 600: 2.0E6}
# ip = {0: 2881088.122, 5: 2881088.122, 80: 15411518.204, 500: 15398677.219, 590: 5255632.946, 600: 5255632.946} # new targets to match output of TM when passing jphi instead of ffp 2026-02-04

# Set heating - gradual ramp down during rampdown to avoid temperature collapse
powers = {
    0: 0, 24: 0, 25: 10.0E6,           # Rampup: turn on heating
    79: 10.0E6, 80: 52.0E6,            # Transition to H-mode
    124: 52.0E6, 125: 40.0E6,          # Flattop
    450: 40.0E6,                       # End of flattop
    500: 25.0E6,                       # Start rampdown - reduce heating gradually
    520: 15.0E6,                       # Continue reducing
    540: 10.0E6,                       # Moderate heating to maintain T
    560: 7.0E6,                        # Keep some heating
    580: 5.0E6,                        # Minimal heating to avoid collapse
    600: 3.0E6                         # Small heating at end
}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Set pedestals - gradual ramp up for H-mode transition, then ramp down
T_i_ped = {0: 0.146, 80: 0.146, 82: 1.5, 85: 2.5, 90: 3.69, 450: 3.69, 500: 3.0, 540: 1.0, 580: 0.3, 600: 0.146}
T_e_ped = {0: 0.220, 80: 0.220, 82: 1.5, 85: 2.5, 90: 3.69, 450: 3.69, 500: 3.0, 540: 1.0, 580: 0.4, 600: 0.220}
n_e_ped = {0: 1.821E19, 79: 1.821E19, 80: 3.0E19, 85: 6.0E19, 90: 7.482E19, 450: 7.482E19, 500: 6.0E19, 540: 3.0E19, 580: 2.0E19, 600: 1.821E19}

ped_on = False

# Set boundary conditions - must be at or below pedestal to avoid edge density spike
# ne_right_bc = {0: 0.08E20, 79: 0.08E20, 80: 0.15E20, 82: 0.28E20, 85: 0.35E20, 90: 0.414E20, 450: 0.414E20, 455: 0.30E20, 460: 0.20E20, 500: 0.12E20, 540: 0.10E20, 580: 0.08E20, 600: 0.08E20}
# Te_right_bc = {0: 0.01, 450: 0.01, 500: 0.02, 540: 0.025, 560: 0.025, 580: 0.02, 600: 0.015}
# Ti_right_bc = {0: 0.01, 450: 0.01, 500: 0.02, 540: 0.025, 560: 0.025, 580: 0.02, 600: 0.015}
ne_right_bc = {0: 0.0157E20, 79: 0.157E20, 80: 0.2E20, 100: 0.414E20, 500: 0.414E20, 550: 0.157E20}
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

# Set detailed temperature and density profiles to match pressure targets from gEQDSK
# Must prescribe t=0 to avoid Torax misreading gEQDSK (it reads T incorrectly)

# T_e_profiles = {
#     0:   {0.0: 0.65, 1.0: 0.01},     # 10 kPa target with n_e≈5e19
#     20:  {0.0: 3.0,  1.0: 0.1},      # ~100 kPa
#     40:  {0.0: 6.0,  1.0: 0.15},     # ~250 kPa  
#     60:  {0.0: 9.0,  1.0: 0.20},     # ~450 kPa
#     80:  {0.0: 11.0, 1.0: 0.22},    # Beginning of H-mode transition
#     82:  {0.0: 12.0, 1.0: 0.22},    # Ramping to H-mode
#     85:  {0.0: 13.5, 1.0: 0.22},    # Continuing H-mode transition
#     90:  {0.0: 15.0, 1.0: 0.22},    # Full H-mode
#     100: {0.0: 15.0, 1.0: 0.22},   # Flattop - higher heating
#     450: {0.0: 15.0, 1.0: 0.22},   # End of flattop
#     500: {0.0: 10.0, 1.0: 0.22},   # Start rampdown - keep T above minimum
#     540: {0.0: 5.0,  1.0: 0.15},    # Mid rampdown
#     580: {0.0: 2.5,  1.0: 0.10},    # Late rampdown - avoid collapse
#     600: {0.0: 1.5,  1.0: 0.05},    # End - keep above minimum
# }

# T_i_profiles = {
#     0: {0.0: 0.60, 1.0: 0.01},     # 10 kPa target
#     20: {0.0: 2.0, 1.0: 0.08},
#     40: {0.0: 4.5, 1.0: 0.12},
#     60: {0.0: 7.0, 1.0: 0.16},
#     80: {0.0: 9.0, 1.0: 0.146},    # Beginning of H-mode transition
#     82: {0.0: 10.0, 1.0: 0.146},   # Ramping to H-mode
#     85: {0.0: 11.0, 1.0: 0.146},   # Continuing H-mode transition
#     90: {0.0: 12.0, 1.0: 0.146},   # Full H-mode
#     100: {0.0: 12.0, 1.0: 0.146},
#     450: {0.0: 12.0, 1.0: 0.146},
#     500: {0.0: 8.0, 1.0: 0.146},   # Start rampdown - keep T above minimum
#     540: {0.0: 4.0, 1.0: 0.10},    # Mid rampdown
#     580: {0.0: 2.0, 1.0: 0.08},    # Late rampdown - avoid collapse
#     600: {0.0: 1.2, 1.0: 0.05},    # End - keep above minimum
# }

# n_e_profiles = {
#     0: {0.0: 5.0E19, 1.0: 1.57E19},     # Match gEQDSK - low pressure at t=0
#     20: {0.0: 5.0E19, 1.0: 1.7E19},
#     40: {0.0: 6.5E19, 1.0: 1.75E19},
#     60: {0.0: 7.5E19, 1.0: 1.8E19},
#     80: {0.0: 8.0E19, 1.0: 1.95E19},    # Beginning of H-mode transition
#     82: {0.0: 8.3E19, 1.0: 2.5E19},     # Gradual pedestal increase
#     85: {0.0: 8.6E19, 1.0: 3.0E19},     # Continuing transition
#     90: {0.0: 9.0E19, 1.0: 0.414E20},   # Full H-mode with pedestal
#     100: {0.0: 9.0E19, 1.0: 0.414E20},  # H-mode with pedestal
#     450: {0.0: 9.0E19, 1.0: 0.414E20},
#     500: {0.0: 8.0E19, 1.0: 0.414E20},  # Start rampdown
#     540: {0.0: 6.0E19, 1.0: 2.0E19},    # Pedestal collapses
#     580: {0.0: 3.0E19, 1.0: 1.7E19},
#     600: {0.0: 2.0E19, 1.0: 1.57E19},
# }
# mysim.set_density(n_e_profiles)
mysim.set_Te({0.0: {0.0: 0.5, 1.0: 0.01}}) # initializes Te and Ti to reasonable values (default is 15 keV)
mysim.set_Ti({0.0: {0.0: 0.5, 1.0: 0.01}})

mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
mysim.set_pedestal(set_pedestal=ped_on, T_i_ped=T_i_ped, T_e_ped=T_e_ped, n_e_ped=n_e_ped)
# Ramp up line-averaged density gradually during H-mode transition, then ramp down during rampdown
mysim.set_nbar({0: 0.05E20, 80: 0.55E20, 85: 0.75E20, 90: 0.905E20, 450: 0.905E20, 500: 0.75E20, 540: 0.5E20, 580: 0.35E20, 600: 0.326E20})

mysim.fly(save_states=True, graph=True)