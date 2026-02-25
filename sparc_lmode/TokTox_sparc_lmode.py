'''SPARC L-mode test pulse from MOSAIC: test_pulse_L_ICRH'''

import torax
import numpy as np
import sys
sys.path.append('/Users/fsheehan/fsheehan_drive/02_areas/Columbia/01_Columbia_projects/2026-01_TokTox/2026-01-08_simple-pulse-design')
from toktox import TokTox


def get_torax_config() -> dict[str, object]:
    """Get the Torax configuration for SPARC pulse."""
    # Radial grid for transport profiles : toroidal normalized grid
    rho = np.linspace(0, 1, 21)

    # Calculate initial temperature profiles [keV]
    electron_temp_axis = 0.3
    ion_temp_axis = 0.3
    temp_width = 0.6
    electron_temp_initial_values = electron_temp_axis * np.exp(-(rho**2) / temp_width**2)
    ion_temp_initial_values = ion_temp_axis * np.exp(-(rho**2) / temp_width**2)
    # Lists for torax_config
    electron_temp_initial = (rho, electron_temp_initial_values)
    ion_temp_initial = (rho, ion_temp_initial_values)

    # Calculate prescribed density profiles as in PPW-Matlab [10^20 m^-3]
    ne_time_grid = np.array([0.0, 3.0, 7.0, 10.0])
    ne_sep =       np.array([1.2, 15.0, 15.0, 10.0]) * 1e19
    ne_0 =         np.array([1.2, 18.0, 18.0, 10.0]) * 1e19
    ne_width =     np.array([1.0, 1.0, 1.0, 1.0])
    ne_profiles =  np.zeros((len(ne_time_grid), len(rho)))
    # Loop through the time grid
    for ii in range(len(ne_time_grid)):
        # Gaussian-like profiles
        ne_shape = np.exp(-(rho**2) / (ne_width[ii] ** 2))
        ne_bc = np.exp(-1.0 / (ne_width[ii] ** 2))  # Value at separatrix
        # Scale to get correct boundary conditions
        ne_scl = ne_sep[ii] + (ne_0[ii] - ne_sep[ii]) * ((ne_shape - ne_bc) / (1.0 - ne_bc))
        ne_profiles[ii, :] = ne_scl  # Fill the column

    ne = (ne_time_grid, rho, ne_profiles)
    nbr_values = ne_profiles[:, -1]
    ne_bound_right = (ne_time_grid, nbr_values)

    return {
        "plasma_composition": {
            "Z_eff": 1.2,
            "main_ion": "D",
            "impurity": "Ne",
        },
        "profile_conditions": {
            "Ip": {0.5: 0.5e6, 1.0: 1.0e6, 2.0: 2.0e6, 3.0: 3.0e6, 6.0: 3.0e6, 7.0: 2.5e6, 8.0: 2.0e6, 9.0: 1.5e6, 10.0: 1.0e6},
            "initial_psi_from_j": True,
            "initial_j_is_total_current": True,
            "current_profile_nu": 1,
            "T_i": ion_temp_initial,  # Initial condition only
            "T_i_right_bc": 0.09,
            "T_e": electron_temp_initial,  # Initial condition only
            "T_e_right_bc": 0.09,
            "n_e_nbar_is_fGW": False,
            "normalize_n_e_to_nbar": False,
            "n_e": ne,
            "n_e_right_bc": ne_bound_right,
            "initial_psi_from_j": True,
            "initial_j_is_total_current": True,
        },
        "numerics": {
            "t_initial": 0.5,
            "t_final": 10.0,
            "exact_t_final": True,
            "fixed_dt": 0.2,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": False,
            "adaptive_T_source_prefactor": 1.0e12,
            "adaptive_n_source_prefactor": 1.0e8,
            "resistivity_multiplier": 1,
        },
        "neoclassical": {
            "bootstrap_current": {
                "bootstrap_multiplier": 1.0,
            }
        },
        "sources": {
            "ecrh": {},
            "ohmic": {},
            "fusion": {},
            "ei_exchange": {},
            "impurity_radiation": {},
            "generic_heat": {  # radial deposition
                # radial deposition
                "gaussian_location": 0.11,
                # Gaussian width in normalized radial coordinate
                "gaussian_width": 0.29,
                # total heating (including accounting for radiation)
                "P_total": (
                    {
                        0: 0.0,
                        3.0: 0.0,
                        3.5: 0.8 * 5.0e6,
                        6.0: 0.8 * 5.0e6,
                        6.5: 0.0,
                        10.0: 0.0,
                    },
                    "PIECEWISE_LINEAR",
                ),
                # electron heating fraction
                "electron_heat_fraction": 0.5,
            },
            "generic_current": {},
        },
        "pedestal": {
            "set_pedestal": False,
        },
        "transport": {
            "model_name": "qlknn",
            "chi_min": 0.1,
            "chi_max": 100.0,
            "D_e_min": 0.05,
            "D_e_max": 100.0,
            "V_e_min": -50.0,
            "V_e_max": 50.0,
            "chi_i_inner": 0.2,
            "chi_e_inner": 0.2,
            "chi_i_outer": 0.2,
            "chi_e_outer": 0.2,
            "rho_inner": 0.2,
            "rho_outer": 0.95,
            "apply_inner_patch": True,
            "apply_outer_patch": False,
            "smoothing_width": 0.1,
            "smooth_everywhere": True,
            "include_ITG": True,
            "include_TEM": True,
            "include_ETG": True,
            "DV_effective": True,
            "An_min": 0.05,
            "avoid_big_negative_s": True,
            "smag_alpha_correction": True,
            "q_sawtooth_proxy": False,
            "ITG_flux_ratio_correction": 1.0,
            "ETG_correction_factor": 1.0,
        },
        "solver": {
            "solver_type": "newton_raphson",
            "theta_implicit": 1.0,
            "use_predictor_corrector": True,
            "n_corrector_steps": 2,
            "convection_dirichlet_mode": "ghost",
            "convection_neumann_mode": "ghost",
            "use_pereverzev": True,
            "chi_pereverzev": 20.0,
            "D_pereverzev": 10.0,
            "log_iterations": False,
        },
        "time_step_calculator": {"calculator_type": "fixed"},
        "geometry": {
            # "geometry_type": "circular",
            # # SPARC geometry: from rBt = 12.2 * 1.85 T-m in MOSAIC gspulse config
            # "R_major": 1.85,   # major radius [m]
            # "a_minor": 0.57,   # minor radius [m] (SPARC V2)
            # "B_0": 12.2,       # toroidal field on axis [T]
            "n_rho": 100,
        },
    }



CONFIG = get_torax_config()


num_eqdsk = 11

geqdsk_arr = [f'sparc_lmode_eqdsk_{i}.eqdsk' for i in range(num_eqdsk)]
eqtimes = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]) # couldn't get 0.5 equil to converge

mysim = TokTox(t_init=0.5, 
               t_final=CONFIG['numerics']['t_final'], \
               dt=CONFIG['numerics']['fixed_dt'], \
               eqtimes=eqtimes,\
               g_eqdsk_arr=geqdsk_arr, \
               last_surface_factor=0.99, \
               n_rho=CONFIG['geometry']['n_rho'])

# mysim.set_evolve(density=False)
mysim.load_config(CONFIG)




mysim.initialize_gs('SPARC_mesh-240613.h5', vsc='VSC')
coil_names = ['VSC', 'CS1U', 'CS1L', 'CS2U', 'CS2L', 'CS3U', 'CS3L', 'PF1U', 'PF1L', 'PF2U', 'PF2L', 'PF3U', 'PF3L', 'PF4U', 'PF4L', 'DV1U', 'DV1L', 'DV2U', 'DV2L']
target_currents = {coil: 0.0 for coil in coil_names}
mysim.set_coil_reg(targets=target_currents, strict_limit=1.e8)

mysim.fly(save_states=False, graph=True, run_name='tmp')