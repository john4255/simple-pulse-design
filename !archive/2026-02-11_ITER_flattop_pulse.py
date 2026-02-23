import numpy as np

from model import DISMAL

# ITER flattop-only pulse - 400 seconds at steady H-mode conditions
# Uses single flattop equilibrium and converges tokamaker every 20 seconds
# SIMPLIFIED: Using minimal known-working values from TORAX ITER examples

# --- Set timesteps ---
flattop_times = np.linspace(0.0, 400.0, 21)  # 21 points for convergence every ~20s
times = flattop_times

# Load gEQDSK - use flattop eqdsk for all timesteps
eqdsk_file = '2026-02-10_iter_rampup_new/out/flattop.eqdsk'
g_arr = [eqdsk_file] * 21

# Set current - constant at 13 MA
ip = {0: 13.0E6, 400: 13.0E6}

# --- SIMPLE flat initial profiles from TORAX ITER hybrid example ---
# These are proven to work - let TORAX evolve to match equilibrium
T_e_profiles = {
    0:   {0.0: 15.0, 1.0: 0.2},  # Simple flat profile: 15 keV core, 0.2 keV edge
    400: {0.0: 15.0, 1.0: 0.2},
}

T_i_profiles = {
    0:   {0.0: 15.0, 1.0: 0.2},  # Same as T_e
    400: {0.0: 15.0, 1.0: 0.2},
}

# Simple density shape (will be normalized to nbar by TORAX)
n_e_profiles = {
    0:   {0.0: 1.5, 1.0: 1.0},   # Relative shape only - TORAX normalizes to nbar
    400: {0.0: 1.5, 1.0: 1.0},
}

# Set heating - use 40 MW total (similar to ITER hybrid scenarios)
powers = {0: 40.0E6, 400: 40.0E6}
nbi_powers = {k: 0.5 * v for k, v in powers.items()}
eccd_powers = {k: 0.5 * v for k, v in powers.items()}

# Simple boundary conditions - use absolute units (ITER example values)
ne_right_bc = {0: 0.25E20, 400: 0.25E20}  # m^-3 (absolute units)
Te_right_bc = {0: 0.2, 400: 0.2}  # keV
Ti_right_bc = {0: 0.2, 400: 0.2}  # keV

# --- Run sim ---
t_res = np.arange(0.0, 400.0, 20.0)
mysim = DISMAL(0, 400, eqtimes=times, g_eqdsk_arr=g_arr, times=t_res, dt=1.0, last_surface_factor=0.9)
mysim.initialize_gs('ITER_mesh.h5', vsc='VS')
coil_names = ['CS3U', 'CS2U', 'CS1U', 'CS1L', 'CS2L', 'CS3L', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents, strict_limit=1.0E8)

mysim.set_Ip(ip)
# Will set Z_eff via config to match TORAX examples

# Set simple profiles directly - no pressure calculation
mysim.set_density(n_e_profiles)
mysim.set_Te(T_e_profiles)
mysim.set_Ti(T_i_profiles)

mysim.set_heating(nbi=nbi_powers, nbi_loc=0.25, eccd=eccd_powers, eccd_loc=0.35)
mysim.set_right_bc(Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc, ne_right_bc=ne_right_bc)
# Disable pedestal for now - just use simple flat profiles
# mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped, n_e_ped=n_e_ped, ped_top=0.9)
mysim.set_nbar({0: 0.85E20, 400: 0.85E20})  # Lower nbar to match ITER hybrid flattop

# Load base config with proper settings - CRITICAL numerics from TORAX examples
from baseconfig import BASE_CONFIG
import copy
config = copy.deepcopy(BASE_CONFIG)

# Plasma composition - use simple Ne impurity like TORAX examples (not W!)
config['plasma_composition'] = {
    'main_ion': {'D': 0.5, 'T': 0.5},
    'impurity': 'Ne',  # Neon only - W causes radiation collapse!
    'Z_eff': 1.6,  # Standard ITER value from TORAX examples
}

# Profile conditions
config['profile_conditions']['normalize_n_e_to_nbar'] = True  # Let TORAX normalize density profile to nbar
config['profile_conditions']['n_e_nbar_is_fGW'] = False  # nbar in absolute units

# CRITICAL NUMERICS - these are essential for stability!
config['numerics'] = {
    't_initial': 0.0,
    't_final': 400.0,
    'fixed_dt': 1.0,
    'resistivity_multiplier': 200,  # CRITICAL! Slows current diffusion to match heat transport
    'evolve_ion_heat': True,
    'evolve_electron_heat': True,
    'evolve_current': True,
    'evolve_density': True,
    'max_dt': 0.5,  # Limit timestep size
    'chi_timestep_prefactor': 50,  # For adaptive timestep
    'dt_reduction_factor': 3,
}

# Disable pedestal - keep it simple
config['pedestal']['set_pedestal'] = False

# CRITICAL: Disable impurity radiation to prevent radiation collapse
config['sources']['impurity_radiation'] = {
    'model_name': 'mavrin_fit',
    'radiation_multiplier': 0.0,  # Turn OFF radiation losses
}

mysim.load_config(config)

mysim.fly(save_states=True, graph=True, max_step=3)
