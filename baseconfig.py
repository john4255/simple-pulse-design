"""Base config for Torax simulation.

With Newton-Raphson solver and adaptive timestep (backtracking)
"""

import numpy as np
import copy

_NBI_W_TO_MA = 1/16e6 # rough estimate of NBI heating power to current drive
W_to_Ne_ratio = 0

# No NBI during rampup. Rampup all NBI power between 99-100 seconds
# nbi_times = np.array([0, 99, 100])
# nbi_powers = np.array([0, 0, 33e6])
# nbi_cd = nbi_powers * _NBI_W_TO_MA

# Gaussian prescription of "NBI" deposition profiles and fractional deposition
# r_nbi = 0.25
# w_nbi = 0.4
# el_heat_fraction = 0.66

BASE_CONFIG = {
    'plasma_composition': {
        # 'main_ion': {'D': 0.5, 'T': 0.5},  # (bundled isotope average)
        # 'impurity': {'Ne': 1 - W_to_Ne_ratio, 'W': W_to_Ne_ratio},
        # 'Z_eff': {0.0: {0.0: 2.0, 1.0: 2.0}},  # sets impurity densities
    },
    'profile_conditions': {
        # 'Ip': {0: 3e6, 100: 12.5e6},  # total plasma current in MA
        # 'T_i': {0.0: {0.0: 6.0, 1.0: 0.2}}, # T_i initial condition
        # 'T_i_right_bc': 0.2, # T_i boundary condition
        # 'T_e': {0.0: {0.0: 6.0, 1.0: 0.2}},  # T_e initial condition
        # 'T_e_right_bc': 0.2,  # T_e boundary condition
        # 'n_e_right_bc_is_fGW': False,
        # 'n_e_right_bc': {0: 0.5E18, 100: 0.3585E20}, # n_e boundary condition
        # set initial condition density according to Greenwald fraction.
        # 'nbar': {0: 0.326E20, 80: .905E20},# line average density for initial condition
        # 'n_e': {0: {0.0: 1.3, 1.0: 1.0}},  # Initial electron density profile
        'normalize_n_e_to_nbar': False, # normalize initial n_e to nbar
        'n_e_nbar_is_fGW': False, # nbar is in units for greenwald fraction
        'initial_psi_from_j': False, # initial psi from current formula # TODO: change?
        # 'initial_j_is_total_current': True, # only ohmic current on init
        # 'current_profile_nu': 2, # exponent in initial current formula
    },
    'numerics': {
        # 't_initial': 145,
        # 't_final': 150,  # length of simulation time in seconds
        # 'fixed_dt': 1, # fixed timestep
        # 'evolve_ion_heat': True, # solve ion heat equation
        # 'evolve_electron_heat': True, # solve electron heat equation
        # 'evolve_current': True, # solve current equation
        # 'evolve_density': True, # solve density equation
        'dt_reduction_factor': 3,  # retry failed Newton-Raphson steps with dt/3 (backtracking)
    },
    'geometry': {
        # 'geometry_type': 'eqdsk',
        # 'geometry_directory': '/Users/johnl/Desktop/discharge-model', 
        # 'geometry_file': 'tmp/toraxtest.eqdsk',
        # 'last_surface_factor': 0.95,
        # 'Ip_from_parameters': True,
        # 'R_major': R,  # major radius (R) in meters
        # 'a_minor': a,  # minor radius (a) in meters
        # 'B_0': Bp,  # Toroidal magnetic field on axis [T]
    },
    'sources': {
        # Current sources (for psi equation)
        # 'ecrh': { # ECRH/ECCD (with Lin-Liu)
        #    'gaussian_width': 0.05,
        #    'gaussian_location': 0.35,
        #    'P_total': eccd_power,
        #    },
        # 'generic_heat': { # Proxy for NBI heat source
            # 'mode': 'PRESCRIBED',
            # 'gaussian_location': r_nbi, # Gaussian location in normalized coordinates
            # 'gaussian_width': w_nbi, # Gaussian width in normalized coordinates
            # 'P_total': (nbi_times, nbi_powers), # Total heating power
            # electron heating fraction r
            # 'electron_heat_fraction': el_heat_fraction,
        # },
        # 'generic_current': { # Proxy for NBI current source
            # 'mode': 'PRESCRIBED',
            # 'use_absolute_current': True, # I_generic is total external current
            # 'gaussian_width': w_nbi,
            # 'gaussian_location': r_nbi,
            # 'I_generic': (nbi_times, nbi_cd),
        # },
        # 'fusion': {}, # fusion power
        # 'ei_exchange': {}, # equipartition
        'ohmic': {}, # ohmic heating — always on; TORAX computes it from resistivity and current profile
        # 'cyclotron_radiation': {
        #   'mode': 'ZERO',
        # }, # cyclotron radiation
        'impurity_radiation': { # impurity radiation + bremsstrahlung
            'model_name': 'mavrin_fit', # uses polynomial fit to experimental cooling rates
            'radiation_multiplier': 1.0, # default value for TORAX
        },
        # 'gas_puff': {
        #   'S_total': 2.5e21,
        #   'puff_decay_length': 0.2,
        # },
        # 'pellet': {
        #   'pellet_deposition_location': 0.85,
        #   'pellet_width': 0.1,
        #   'S_total': 2.e22,
        # },
        # 'generic_particle': {
        #   'deposition_location': 0.3,
        #   'S_total': 5e20,
        # },
    },
    'mhd': {'sawtooth': {'redistribution_model': {'model_name': 'simple'},
                        'trigger_model': {'minimum_radius': 0.1,
                                            'model_name': 'simple',
                                            's_critical': 0.4}}},
    'neoclassical': {
        'bootstrap_current': {}, # turns on bootstrap current with default parameters 
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        # use internal boundary condition model (for H-mode and L-mode)
        # 'set_pedestal': True,
        # 'T_i_ped': {0: 0.5, 100: 0.5, 105: 3.0},
        # 'T_e_ped': {0: 0.5, 100: 0.5, 105: 3.0},
        # 'n_e_ped_is_fGW': False,
        # 'n_e_ped': 0.85, # pedestal top n_e in units of fGW
        # 'rho_norm_ped_top': 0.95,  # set ped top location in normalized radius
    },
    # 'transport': { # updated 2026-04-01 using TORAX team recommendations (via Oak, in docs, not used when pedestal is set)
    #     'model_name': 'combined',
    #     'transport_models': [
    #         # Base model: QLKNN applied everywhere (default ADD)
    #         {
    #             'model_name': 'qlknn',
    #             'rho_max': 1.0,
    #         },
    #         # Edge overwrite: Sets D_e and V_e in the edge, ignoring QLKNN there.
    #         # Keeps chi_i/chi_e from QLKNN (because they are disabled here).
    #         {
    #             'model_name': 'constant',
    #             'rho_min': 0.9,
    #             'D_e': 0.5,
    #             'V_e': -1.0,
    #             'merge_mode': 'overwrite',
    #             'disable_chi_i': True,
    #             'disable_chi_e': True,
    #         },
    #     ],
    # },
    'transport': { # old
        'model_name': 'qlknn', 
        # set inner core transport coefficients (ad-hoc MHD/EM transport)
        'apply_inner_patch': True, # TODO: scan chi?
        # 'D_e_inner': 0.1,
        # 'V_e_inner': 0.0,
        # 'chi_i_inner': 0.1,
        # 'chi_e_inner': 0.1,
        'rho_inner': 0.1,  # radius below which patch transport is applied
        # set outer core transport coefficients (L-mode near edge region)
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.95,  # radius above which patch transport is applied
        # allowed chi and diffusivity bounds
        'chi_min': 0.05,  # minimum chi
        'chi_max': 100,  # maximum chi (can be helpful for stability)
        'D_e_min': 0.05,  # minimum electron diffusivity
        'D_e_max': 50,  # maximum electron diffusivity
        'V_e_min': -10,  # minimum electron convection
        'V_e_max': 10,  # minimum electron convection
        'smoothing_width': 0.3,
        'DV_effective': True,
        'include_ITG': True,  # to toggle ITG modes on or off
        'include_TEM': True,  # to toggle TEM modes on or off
        'include_ETG': True,  # to toggle ETG modes on or off
        'avoid_big_negative_s': False,
        # 'rotation_multiplier': 1.0,
        # 'rotation_mode': 'half_radius',
    },
    'solver': {
        'solver_type': 'newton_raphson', # linear solver with picard iteration
        'use_predictor_corrector': True, # for linear solver
        'n_corrector_steps': 10, # for linear solver
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
#        'log_iterations': False,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}

#@title set_LH_transition_time_function
# def set_LH_transition_time(config, LH_transition_time: float):
#   """Modifies the base config by changing the LH transition time which sets the Ip ramp rate and heating switch-on."""

#   config = copy.deepcopy(config)
# #   _validate_input(LH_transition_time, (20.0, 130.0))
#   config['profile_conditions']['Ip'] = {0: 3e6, LH_transition_time: 12.5e6}
#   config['sources']['ecrh']['P_total'] = {0: 0, LH_transition_time-1: 0, LH_transition_time: 20.0e6}
#   config['sources']['generic_heat']['P_total'] = {0: 0, LH_transition_time-1: 0, LH_transition_time: 33.0e6}
#   config['sources']['generic_current']['I_generic'] = {0: 0, LH_transition_time-1: 0, LH_transition_time: 33.0e6 * _NBI_W_TO_MA}
#   config['pedestal']['T_i_ped'] = {0: 0.5, LH_transition_time: 0.5, LH_transition_time+5: 3.0}
#   config['pedestal']['T_e_ped'] = {0: 0.5, LH_transition_time: 0.5, LH_transition_time+5: 3.0}
#   return config
