"""Base config for Torax simulation.

With Newton-Raphson solver and adaptive timestep (backtracking)
"""

BASE_CONFIG = {
    'plasma_composition': {},
    'profile_conditions': {
        'normalize_n_e_to_nbar': False,
        'n_e_nbar_is_fGW': False,
        'initial_psi_from_j': False,
    },
    'numerics': {
        'dt_reduction_factor': 3,
    },
    'geometry': {},
    'sources': {
        'ohmic': {},
        'impurity_radiation': {
            'model_name': 'mavrin_fit',
            'radiation_multiplier': 1.0,
        }
    },
    'mhd': {'sawtooth': {'redistribution_model': {'model_name': 'simple'},
                        'trigger_model': {'minimum_radius': 0.1,
                                            'model_name': 'simple',
                                            's_critical': 0.4}}},
    'neoclassical': {
        'bootstrap_current': {},
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'rho_inner': 0.1,
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.95,
        'chi_min': 0.05,
        'chi_max': 100,
        'D_e_min': 0.05,
        'D_e_max': 50,
        'V_e_min': -10,
        'V_e_max': 10,
        'smoothing_width': 0.3,
        'DV_effective': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'avoid_big_negative_s': False,
    },
    'solver': {
        'solver_type': 'newton_raphson',
        'use_predictor_corrector': True,
        'n_corrector_steps': 10,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
