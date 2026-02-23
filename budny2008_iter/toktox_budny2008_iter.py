import numpy as np
from toktox import TokTox
from omfit_classes.utils_fusion import Hmode_profiles, Lmode_profiles

# new ITER pulse using q95 stable rampup (after q95 has decreased from large value)
# based entirely on guess and check for relative static q95 to get to ITER example ipynb made by Daniel Burgess (13MA)

# Set timesteps
rampup_times = np.linspace(0.0, 80.0, 10)  
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times = np.linspace(500.0, 600.0, 10) 
times = np.r_[rampup_times, flattop_times, rampdown_times]
n_rampup_eqdsk = 9

# Load gEQDSK
# Slow the shape progression during rampup: hold at i=0 (circular) for the first
# ~18s, then step through i=1..9 more gradually.  This keeps the GS constraint
# (LCFS shape) compatible with the low pressure TORAX produces for pure-Ohmic rampup.
_ramp_eqdsks = (
    ['budny2008_iter/budny2008_iter_i=0.eqdsk'] * 3         # t=0,8.9,17.8  – circular
    + [f'budny2008_iter/budny2008_iter_i={i}.eqdsk' for i in range(2, 8)]  # t=26.7..71.1
    + ['budny2008_iter/budny2008_iter_i=9.eqdsk']           # t=80
)
g_arr_rampup  = _ramp_eqdsks
g_arr_flattop = [f'budny2008_iter/budny2008_iter_flattop.eqdsk'] * 10
g_arr_rampdown = list(reversed(_ramp_eqdsks))
g_arr = g_arr_rampup + g_arr_flattop + g_arr_rampdown

# Set current
# ip = {0: 2.0E6, 80: 13.0E6, 520: 13.0E6, 600: 2.0E6}

# Set heating
# ITER-like scenario: pure Ohmic rampup (no NBI/ECCD) until just before H-mode
# transition.  Mid-rampup heating only creates a Te spike that collapses later
# because density is still very low (P/n^2 is very large).
powers = {
    0:   0,
    84:  0,  85: 20.0E6,         # Pre-heat 15s before H-mode gate
    99:  20.0E6, 100: 55.0E6,    # H-mode transition
    450: 55.0E6,                 # End of flattop
    500: 30.0E6,                 # Start rampdown
    520: 20.0E6,
    540: 12.0E6,
    560:  7.0E6,
    580:  4.0E6,
    600:  2.0E6,
}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Set pedestals - gradual ramp up for H-mode transition, then ramp down
# T_i_ped = {0: 0.146, 80: 0.146, 82: 1.5, 85: 2.5, 90: 3.69, 450: 3.69, 500: 3.0, 540: 1.0, 580: 0.3, 600: 0.146}
# T_e_ped = {0: 0.220, 80: 0.220, 82: 1.5, 85: 2.5, 90: 3.69, 450: 3.69, 500: 3.0, 540: 1.0, 580: 0.4, 600: 0.220}
# n_e_ped = {0: 1.821E19, 79: 1.821E19, 80: 3.0E19, 85: 6.0E19, 90: 7.482E19, 450: 7.482E19, 500: 6.0E19, 540: 3.0E19, 580: 2.0E19, 600: 1.821E19}


# ped_on = False
ped_on = {0: False, 81: False, 82: True, 549: True, 550: False, 600: False}


# set initial kinetic profiles
n_sample=200
psi_sample = np.linspace(0.0,1.0,n_sample)

### Define ne and Te profiles
# rampup
# ne/Te scaled to give p_ax closer to the seed EQDSK values.
# EQDSK p_ax rises from ~3 kPa (t=0) to ~620 kPa (t=80) — way above what pure-Ohmic
# TORAX can reach.  We target a fraction of that to keep TM GS consistent:
#   p_ax = 2 * n_e * T_e  (e + i, quasineutral); 1 keV at t=0, ~8 keV at t=80.
# This gives p_ax(t=0)≈3 kPa, p_ax(t=80)≈100 kPa — closer but still below EQDSK.
ne_axis = np.linspace(1.5E19, 8.0E19, n_rampup_eqdsk)  # m^-3
Te_axis = np.linspace(1.0,   8.0,    n_rampup_eqdsk)   # keV (Ohmic + moderate confinement)

# Keep edge density low relative to core (Lmode pedestal-free)
ne_edge = np.linspace(0.08E20, 0.25E20, n_rampup_eqdsk)
Te_edge = np.linspace(0.040,   0.080,   n_rampup_eqdsk)  # keV

n_sample = 100
ne_prof_list = []
Te_prof_list = []
psi_sample = np.linspace(0.0, 1.0, n_sample)
for i in range(n_rampup_eqdsk):
    ne_prof = Lmode_profiles(edge=ne_edge[i], core=ne_axis[i], rgrid=n_sample)
    Te_prof = Lmode_profiles(edge=Te_edge[i], core=Te_axis[i], rgrid=n_sample)
    ne_prof_list.append(ne_prof)
    Te_prof_list.append(Te_prof)

# flattop (values from Daniel's ipynb that i used to create H-mode flattop equil)
xphalf = 0.965
widthp_Te = 0.1
widthp_ne = 0.35
ne_ped = 0.6E20
Te_ped = 5.0  # keV
ne_flattop_arr = Hmode_profiles(edge=0.35, ped=ne_ped/1E20, core=1.1, rgrid=n_sample, expin=1.6, expout=1.6, widthp=widthp_ne, xphalf=xphalf) * 1e20
Te_flattop_arr = Hmode_profiles(edge=0.1, ped=Te_ped, core=21.0, rgrid=n_sample, expin=1.3, expout=1.7, widthp=widthp_Te, xphalf=xphalf)  # keV
ni_flattop = ne_flattop_arr.copy() # Assuming quasineutrality
Ti_flattop = Te_flattop_arr.copy() # Assuming isothermal
p_flattop = (1.602e-16 * ne_flattop_arr * Te_flattop_arr) + (1.602e-16 * ni_flattop * Ti_flattop) # 1.602e-16 * [m^-3] * [keV] = [Pa]
pax_flattop = p_flattop[0]

# Helper function to convert 1D array profile to TORAX dict format
def array_to_profile_dict(profile_array, psi_grid=None):
    """Convert 1D profile array to TORAX format dict {psi: value}"""
    if psi_grid is None:
        psi_grid = np.linspace(0.0, 1.0, len(profile_array))
    return {float(psi): float(val) for psi, val in zip(psi_grid, profile_array)}

# Convert all profiles to TORAX format
ne_flattop_dict = array_to_profile_dict(ne_flattop_arr, psi_sample)
Te_flattop_dict = array_to_profile_dict(Te_flattop_arr, psi_sample)

#### set initial kinetic profiles
ne = {
    0: array_to_profile_dict(ne_prof_list[0], psi_sample),
    10: array_to_profile_dict(ne_prof_list[1], psi_sample),
    20: array_to_profile_dict(ne_prof_list[2], psi_sample),
    30: array_to_profile_dict(ne_prof_list[3], psi_sample),
    40: array_to_profile_dict(ne_prof_list[4], psi_sample),
    50: array_to_profile_dict(ne_prof_list[5], psi_sample),
    60: array_to_profile_dict(ne_prof_list[6], psi_sample),
    70: array_to_profile_dict(ne_prof_list[7], psi_sample),
    80: array_to_profile_dict(ne_prof_list[8], psi_sample),
    100: ne_flattop_dict,
    500: ne_flattop_dict,
    510: array_to_profile_dict(ne_prof_list[8], psi_sample),
    520: array_to_profile_dict(ne_prof_list[7], psi_sample),
    530: array_to_profile_dict(ne_prof_list[6], psi_sample),
    540: array_to_profile_dict(ne_prof_list[5], psi_sample),
    550: array_to_profile_dict(ne_prof_list[4], psi_sample),
    560: array_to_profile_dict(ne_prof_list[3], psi_sample),
    570: array_to_profile_dict(ne_prof_list[2], psi_sample),
    580: array_to_profile_dict(ne_prof_list[1], psi_sample),
    600: array_to_profile_dict(ne_prof_list[0], psi_sample)
}
Te = {
    0: array_to_profile_dict(Te_prof_list[0], psi_sample),
    10: array_to_profile_dict(Te_prof_list[1], psi_sample),
    20: array_to_profile_dict(Te_prof_list[2], psi_sample),
    30: array_to_profile_dict(Te_prof_list[3], psi_sample),
    40: array_to_profile_dict(Te_prof_list[4], psi_sample),
    50: array_to_profile_dict(Te_prof_list[5], psi_sample),
    60: array_to_profile_dict(Te_prof_list[6], psi_sample),
    70: array_to_profile_dict(Te_prof_list[7], psi_sample),
    80: array_to_profile_dict(Te_prof_list[8], psi_sample),
    100: Te_flattop_dict,
    500: Te_flattop_dict,
    510: array_to_profile_dict(Te_prof_list[8], psi_sample),
    520: array_to_profile_dict(Te_prof_list[7], psi_sample),
    530: array_to_profile_dict(Te_prof_list[6], psi_sample),
    540: array_to_profile_dict(Te_prof_list[5], psi_sample),
    550: array_to_profile_dict(Te_prof_list[4], psi_sample),
    560: array_to_profile_dict(Te_prof_list[3], psi_sample),
    570: array_to_profile_dict(Te_prof_list[2], psi_sample),
    580: array_to_profile_dict(Te_prof_list[1], psi_sample),
    600: array_to_profile_dict(Te_prof_list[0], psi_sample)
}



# Set boundary conditions - must be at or below pedestal to avoid edge density spike
ne_right_bc = {0: 0.0157E20, 99: 0.157E20, 100: ne_flattop_arr[-1], 500: ne_flattop_arr[-1], 600: 0.157E20}
Te_right_bc = {0: 0.01, 99: 0.1, 100: Te_flattop_arr[-1], 500: Te_flattop_arr[-1], 600: 0.1}
Ti_right_bc = {0: 0.01, 99: 0.1, 100: Te_flattop_arr[-1], 500: Te_flattop_arr[-1], 600: 0.1}

# Run sim
# Start at t=5 to skip the t=0 TORAX Ohmic transient (large dPsi/dt at t=0 from
# initialising current evolution causes anomalously high V_loop and Ip).
# The t=0 seed EQDSK geometry is still available via eqtimes[0]=0 for shape seeding.
t_res = np.concatenate([[5.0], np.arange(20.0, 600.0, 20.0)])
mysim = TokTox(5, 600, eqtimes=times, g_eqdsk_arr=g_arr, times=t_res, dt=1.0, last_surface_factor=0.99)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
coil_names = ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents, strict_limit=1.0E8)

# mysim.set_Ip(ip)
mysim.set_Zeff(1.8)


mysim.set_Te(Te) 
mysim.set_Ti(Te) # Assuming isothermal
mysim.set_ne(ne)

mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)

mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
mysim.set_pedestal(set_pedestal=ped_on, T_i_ped=Te_ped, T_e_ped=Te_ped, n_e_ped=ne_ped) 


mysim.fly(save_states=True, graph=True)