import numpy as np
from model import DISMAL
from omfit_classes.utils_fusion import Hmode_profiles, Lmode_profiles

g_arr = ['/Users/fsheehan/fsheehan_drive/02_areas/Columbia/01_Columbia_projects/2026-01_TokaPULSE/2026-01-08_simple-pulse-design/budny2008_iter_flattop/budny2008_iter_flattop.eqdsk']

powers = {0:   55.E6}

nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}


ped_on = {0: True}


# set initial kinetic profiles
n_sample=200
psi_sample = np.linspace(0.0,1.0,n_sample)

### Define ne and Te profiles

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
ne = {0: ne_flattop_dict}
Te = {0: Te_flattop_dict}

# set boundary conditions
ne_right_bc = {0: ne_flattop_arr[-1]}
Te_right_bc = {0: Te_flattop_arr[-1]}
Ti_right_bc = {0: Te_flattop_arr[-1]}

# Run sim
t_end = 500
n_eqdsk = 10
g_arr = g_arr * n_eqdsk

mysim = DISMAL(0, t_end, eqtimes=np.linspace(0,t_end, n_eqdsk), g_eqdsk_arr=g_arr, times=None, dt=10., last_surface_factor=0.99)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
coil_names = ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents, strict_limit=1.0E8)

# mysim.set_Ip(ip)
mysim.set_Zeff(1.8)

set_gaspuff={0: 5E23}

mysim.set_Te(Te) 
mysim.set_Ti(Te) # Assuming isothermal
mysim.set_ne(ne)

mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)

mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
mysim.set_pedestal(set_pedestal=ped_on, T_i_ped=Te_ped, T_e_ped=Te_ped, n_e_ped=ne_ped) 


mysim.fly(save_states=True, graph=True)