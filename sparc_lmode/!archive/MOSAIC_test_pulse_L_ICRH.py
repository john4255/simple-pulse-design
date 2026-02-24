"""Test pulse for MOSAIC CI.

- 3MA 3s 5MW Single Null Deuterium
- based on Matlab PPW #71050
"""

from pathlib import Path
from typing import Any

import numpy as np
from gspulse import get_gspulse_root
from gspulse.config import load_shape_evolution
from mosaic import WorkflowConfig

from pop_workflow.workflows.default import DockerImageSource, Runtime
from pop_workflow.workflows.default import get_workflow_config as get_pop_wkfl_config

_HERE_PATH = Path(__file__).resolve().parent
MOSAIC_PATH = _HERE_PATH.parent
GSPULSE_ROOT = get_gspulse_root()


def get_workflow_config(
    iterations: int = 5,
    compute_lin: bool = False,  # noqa: FBT001, FBT002
    runtime: Runtime = Runtime.PYTHON,
    docker_image_source: DockerImageSource = DockerImageSource.GHCR,
) -> WorkflowConfig:
    """MOSAIC Pulse L-mode Test Pulse."""
    torax_config = get_torax_config()
    gspulse_config = get_gspulse_config()
    meq_cfs_config = get_meq_cfs_config(compute_lin=compute_lin)

    return get_pop_wkfl_config(
        pulse_id=100,
        n_iterations=iterations,
        torax_config=torax_config,
        gspulse_config=gspulse_config,
        meq_cfs_config=meq_cfs_config,
        runtime=runtime,
        docker_image_source=docker_image_source,
    )


def get_meq_cfs_config(
    compute_lin: bool = True,  # noqa: FBT001, FBT002
) -> dict[str, Any]:
    """Get MEQ-CFS configuration for SPARC pulse."""
    return {
        "save_debug_output": False,
        "time_idxs_downsample_factor_lin": 10,
        "save_lim_matrices_json": True,
        "compute_lin": compute_lin,
    }


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
    ne_sep = np.array([1.2, 15.0, 15.0, 10.0]) * 1e19
    ne_0 = np.array([1.2, 18.0, 18.0, 10.0]) * 1e19
    ne_width = np.array([1.0, 1.0, 1.0, 1.0])
    ne_profiles = np.zeros((len(ne_time_grid), len(rho)))
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
            # total plasma current in MA
            "Ip": [],
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
        "geometry": {"geometry_type": "circular"},
    }


def get_gspulse_config() -> dict[str, Any]:
    """Get GSPulse configuration for SPARC pulse."""
    # ========
    # Settings
    # ========
    eps = 0.0001
    settings = {
        "tokamak": "sparc",
        "interval_t": [
            np.arange(-20, -2 + eps, 2).tolist() + np.arange(-1.9, 1.9 + eps, 0.1).tolist(),
            np.arange(1, 10 + eps, 0.25).tolist(),
            np.arange(9, 13 + eps, 0.5).tolist(),
        ],
        "plotlevel": 1,
        "niter": 5,
        "verbose": 2,
        "picard_algo": "fbt",
        "do_final_boundary_trace": False,
        "use_spline_basis": False,
        "spline_basis_ratio": 0,
        "specify_psibry_mode": "direct",
        "calc_strike_pts": False,
        "inject_model_offset": False,
        "qpsolver": "quadprog",
        "qpsolver_tol": 1e-3,
        "calc_post_prc_ext": False,
        "vmax": [
            2600,
            2600,
            1000,
            1000,
            650,
            650,
            650,
            650,
            1000,
            1000,
            2126,
            2126,
            2126,
            2126,
            480,
            480,
            480,
            480,
            1000,
        ],
        "vmin": [
            -2600,
            -2600,
            -1000,
            -1000,
            -650,
            -650,
            -650,
            -650,
            -1000,
            -1000,
            -2126,
            -2126,
            -2126,
            -2126,
            -480,
            -480,
            -480,
            -480,
            -1000,
        ],
        "ic_max": [
            48000,
            48000,
            48000,
            48000,
            48000,
            48000,
            45000,
            45000,
            45000,
            45000,
            20000,
            20000,
            0,
            0,
            32000,
            32000,
            32000,
            32000,
            10000,
        ],
        "ic_min": [
            -48000,
            -48000,
            -48000,
            -48000,
            0,
            0,
            0,
            0,
            0,
            0,
            -45000,
            -45000,
            -48000,
            -48000,
            -32000,
            -32000,
            -32000,
            -32000,
            -10000,
        ],
    }

    # ======================
    # L.P overrides for FBT
    # ======================
    P_overrides = {"fbtagcon": ["Ip", "Wk", "ag"]}  # noqa: N806

    # =================
    # Plasma parameters
    # =================

    # pprime and ttprime profile basis functions
    # -------------------------------------------
    time_ = [-np.inf, 0.2, 1, 2, 3, 4, 6, 7, 8, 9, 10, np.inf]
    pprime_coeffs = [1, 1, 0.9, 0.8, 0.7, 0.65, 0.65, 0.65, 0.8, 0.9, 1, 1]
    ttprime1_coeffs = [1.1, 1.1, 1.4, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 1.5, 1.4, 1.4]
    psin = np.linspace(0, 1, 101)

    pprime_data: list[np.ndarray | None] = [None] * len(time_)
    ttprime1_data: list[np.ndarray | None] = [None] * len(time_)

    for i in range(len(time_)):
        pprime_data[i] = ((1 - psin) ** pprime_coeffs[i]) * (psin**0.1)
        ttprime1_data[i] = (1 - psin) ** ttprime1_coeffs[i]

    ttprime2_data: list[np.ndarray | None] = [None] * 2
    ttprime2_data[0] = ttprime2_data[1] = ((1 - psin) ** 0.7) * (psin**0.7)

    pprime = {"time": time_, "data": np.array(pprime_data).tolist()}
    ttprime1 = {"time": time_, "data": np.array(ttprime1_data).tolist()}
    ttprime2 = {"time": [-np.inf, np.inf], "data": np.array(ttprime2_data).tolist()}

    # Define additional plasma scalar parameters
    # -------------------------------------------
    time_ = [-np.inf, 0, 0.2] + list(range(1, 12, 1)) + [np.inf]
    plasma_params = {
        "Ip": {
            "time": time_,
            "data": (
                np.array(
                    [
                        0,
                        0,
                        0.2,
                        1.0,
                        2.0,
                        3.0,
                        3.0,
                        3.0,
                        3.0,
                        2.5,
                        2.0,
                        1.5,
                        1.0,
                        0,
                        0,
                    ]
                )
                * 1e6
            ).tolist(),
        },
        "Wk": {
            "time": time_,
            "data": (
                np.array(
                    [
                        0,
                        0,
                        0.0024,
                        0.072,
                        0.26,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.25,
                        0.12,
                        0.06,
                        0,
                        0,
                    ]
                )
                * 1e6
            ).tolist(),
        },
        "ag": {"time": [-np.inf, np.inf], "data": np.zeros((2, 3)).tolist()},
        "rBt": {
            "time": [-np.inf, np.inf],
            "data": (np.array([1, 1]) * 12.2 * 1.85).tolist(),  # T-m
        },
        "pprime": pprime,
        "ttprime1": ttprime1,
        "ttprime2": ttprime2,
    }

    # ==========
    # Shape data
    # ==========
    shape_dir = GSPULSE_ROOT / "tokamaks" / "sparc" / "shapes" / "3_3_SN_v1"
    shape_time_fns = [
        (-np.inf, "0-2s_v1.json"),
        (0.2, "0-2s_v1.json"),
        (1.0, "1-0s_v1.json"),
        (2.0, "2-0s_v1.json"),
        (3.0, "3-0s_v1.json"),
        (4.0, "4-0s_v1.json"),
        (5.0, "5-0s_v1.json"),
        (6.0, "6-0s_v1.json"),
        (7.0, "7-0s_v1.json"),
        (8.0, "8-0s_v1.json"),
        (9.0, "9-0s_v1.json"),
        (10.0, "10-0s_v1.json"),
        (np.inf, "10-0s_v1.json"),
    ]
    time_ = [t_fn[0] for t_fn in shape_time_fns]
    shape_filepaths = [shape_dir / t_fn[1] for t_fn in shape_time_fns]
    shapes = load_shape_evolution(time_, shape_filepaths)
    # NOTE(mveldhoen): MOSAIC needs to have all config to be serializable.
    shapes = {key: value.model_dump() for key, value in shapes.items()}

    # =================
    # Initial condition
    # =================
    initial_condition = {
        "x1": np.zeros(59).tolist(),  # 19 coils + 40 vessel modes
        "u0": np.zeros(19).tolist(),
        "u1": [np.nan] * 19,
    }

    # ==========================
    # Optimization signals
    # ==========================
    optimization_signals = []

    # Power supply voltages
    # ---------------------
    scale_ = np.array(
        [
            0.0029,
            0.0029,
            0.0098,
            0.0098,
            0.0098,
            0.0098,
            0.0130,
            0.0130,
            0.0031,
            0.0031,
            0.0033,
            0.0033,
            0.0013,
            0.0013,
            0.0620,
            0.0620,
            0.0620,
            0.0620,
            0.9900,
        ]
    )

    optimization_signals.append(
        {
            "name": "voltage",
            "description": "Power supply voltages",
            "calc_type": "voltage",
            "wt": {"time": [-np.inf, np.inf], "data": (np.ones((2, 1)) * scale_ * 1e-4).tolist()},
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 19)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": (np.ones((2, 19)) * 1e-9).tolist()},
        }
    )

    # Coil currents
    # -------------
    optimization_signals.append(
        {
            "name": "ic",
            "description": "Coil currents",
            "calc_type": "coil_currents",
            "target": {"time": [-np.inf, np.inf], "data": np.zeros((2, 19)).tolist()},
            "wt": {
                "time": [-np.inf, 12.9, 13],
                "data": (np.outer(np.array([1, 1, 1e3]), np.array([1] * 18 + [1e3])) * 1e-10).tolist(),
            },
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 19)).tolist()},
            "d2wt": {
                "time": [-np.inf, -0.5, -0.49, 10, 10.01, np.inf],
                "data": (np.outer(np.array([1, 1, 1e-5, 1e-5, 1, 1]), np.ones(19) * 1e-4)).tolist(),
            },
        }
    )

    # Shape control point flux error
    # ------------------------------
    t_lim2div = 3.75
    t_div2lim = 7.0
    time_ = [-np.inf, t_lim2div - 0.25, t_lim2div + 0.25, t_div2lim - 0.25, t_div2lim + 0.25, np.inf]
    ramp_ = np.array([0, 0, 1, 1, 0, 0])
    ramp_ = np.expand_dims(ramp_, axis=1)
    ncp = len(shapes["cp_r"]["data"][0])

    optimization_signals.append(
        {
            "name": "diff_psicp_psimultiref",
            "description": "Flux difference between the control points and the reference touch point / x-points",
            "calc_type": "flux_relative_multipoint",
            "r": shapes["cp_r"],
            "z": shapes["cp_z"],
            "r_multiref": shapes["r_control_pt_ref"],  # contains 3 pts corresponding to up x-pt, lo x-pt, touch pt
            "z_multiref": shapes["z_control_pt_ref"],
            "wt_multiref": {"time": time_, "data": np.hstack([ramp_ * 0.5, ramp_ * 0.5, (1 - ramp_)]).tolist()},
            "target": {"time": [-np.inf, np.inf], "data": np.zeros((2, ncp)).tolist()},
            "wt": {
                "time": [-np.inf, 0, 0.5, 9.5, 10, np.inf],
                "data": (np.outer(np.array([0, 0, 1, 1, 0, 0]), np.ones(ncp)) * 1e4).tolist(),
            },
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, ncp)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros((2, ncp)).tolist()},
        }
    )

    # Strike point flux error
    # ------------------------
    # in/up, in/lo, out/up, out/lo, out2/up, out2/lo strike points
    rstrike_data = np.array(shapes["rstrike"]["data"])[:, 0:6].tolist()
    zstrike_data = np.array(shapes["zstrike"]["data"])[:, 0:6].tolist()

    optimization_signals.append(
        {
            "name": "diff_psisp_psix2",
            "description": "Flux difference between the strike points and the lower x-point",
            "calc_type": "flux_relative",
            "r1": {"time": shapes["rstrike"]["time"], "data": rstrike_data},
            "z1": {"time": shapes["zstrike"]["time"], "data": zstrike_data},
            "r2": shapes["rx2"],  # lower x-point
            "z2": shapes["zx2"],
            "target": {"time": [-np.inf, np.inf], "data": np.zeros((2, 6)).tolist()},
            "wt": {
                "time": time_,
                "data": (ramp_ @ np.array([[0.1, 1, 0.1, 1, 0.1, 1]]) * 100).tolist(),
            },  # lower weight on upper
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 6)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 6)).tolist()},
        }
    )

    # Flux drsep (difference in flux between upper and lower x-points)
    # -----------------------------------------------------------------
    optimization_signals.append(
        {
            "name": "diff_psix1_psix2",
            "description": "Flux difference between the upper and lower x-points",
            "calc_type": "flux_relative",
            "r1": shapes["rx1"],  # upper x-point
            "z1": shapes["zx1"],
            "r2": shapes["rx2"],  # lower x-point
            "z2": shapes["zx2"],
            "target": {"time": [-np.inf, 3, 4, 7, 8, np.inf], "data": [0, 0, 0, 0, 0, 0]},
            "wt": {"time": [-np.inf, np.inf], "data": (np.ones(2) * 100).tolist()},
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros(2).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros(2).tolist()},
        }
    )

    # Radial field at x-points
    # -------------------------
    # slightly different ramp for x-points than for isoflux shape errors,
    # because we need x-points in position before diverting
    time_ = [-np.inf, t_lim2div - 0.5, t_lim2div - 0.25, t_div2lim + 0.25, t_div2lim + 0.5, np.inf]
    ramp_ = np.array([0, 0, 1, 1, 0, 0])
    ramp_ = np.expand_dims(ramp_, axis=1)

    optimization_signals.append(
        {
            "name": "xpt_br",
            "description": "Radial field at x-points",
            "calc_type": "field_absolute_radial",
            "r": {
                "time": shapes["rx1"]["time"],
                "data": np.array([shapes["rx1"]["data"], shapes["rx2"]["data"]]).T.tolist(),
            },
            "z": {
                "time": shapes["zx1"]["time"],
                "data": np.array([shapes["zx1"]["data"], shapes["zx2"]["data"]]).T.tolist(),
            },
            "target": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
            "wt": {"time": time_, "data": (ramp_ @ np.ones((1, 2)) * 100).tolist()},
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
        }
    )

    # Vertical field at x-points
    # -------------------------
    optimization_signals.append(
        {
            "name": "xpt_bz",
            "description": "Vertical field at x-points",
            "calc_type": "field_absolute_vertical",
            "r": {
                "time": shapes["rx1"]["time"],
                "data": np.array([shapes["rx1"]["data"], shapes["rx2"]["data"]]).T.tolist(),
            },
            "z": {
                "time": shapes["zx1"]["time"],
                "data": np.array([shapes["zx1"]["data"], shapes["zx2"]["data"]]).T.tolist(),
            },
            "target": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
            "wt": {"time": time_, "data": (ramp_ @ np.ones((1, 2)) * 100).tolist()},
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 2)).tolist()},
        }
    )

    # Boundary flux
    # -------------
    optimization_signals.append(
        {
            "name": "psibry",
            "description": "Flux value at the plasma boundary",
            "calc_type": "flux_absolute_avg",
            "r": shapes["cp_r"],
            "z": shapes["cp_z"],
            "target": {"time": [-20, 0, 0.2, 8, 10, 13], "data": [0, 9, 8.5, -2, -1, 0]},
            "wt": {"time": [-np.inf, 0, 1, 24, 24.01, np.inf], "data": [0, 0, 1e3, 1e3, 0, 0]},
            "dwt": {"time": [-np.inf, np.inf], "data": (np.ones(2) * 10).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros(2).tolist()},
        }
    )

    # Boundary flux (vacuum, for startup)
    # ----------------------------------
    optimization_signals.append(
        {
            "name": "psibry_vac",
            "description": "Flux value at the plasma boundary",
            "calc_type": "vacuum_flux_absolute_avg",
            "r": shapes["cp_r"],
            "z": shapes["cp_z"],
            "target": {"time": [-20, 0, 0.2, 8, 10, 13], "data": [0, 9, 8.5, -2, -1, 0]},
            "wt": {"time": [-np.inf, -1, -0.5, 0.5, 1, np.inf], "data": [0.1, 0.1, 1e4, 1e4, 0, 0]},
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros(2).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros(2).tolist()},
        }
    )

    # Magnetic field vertical (for plasma startup)
    # --------------------------------------------
    optimization_signals.append(
        {
            "name": "bz_startup",
            "description": "Vertical field at specified location",
            "calc_type": "vacuum_field_absolute_vertical",
            "r": {"time": [-np.inf, np.inf], "data": np.outer(np.ones((2, 1)), np.array([1.5, 1.75, 2, 2.2])).tolist()},
            "z": {"time": [-np.inf, np.inf], "data": np.zeros((2, 4)).tolist()},
            "target": {
                "time": np.array([-20, -0.1 + 0.05, 0.2 + 0.05, np.inf]).tolist(),
                "data": np.outer(np.array([0.04, 0.04, -0.08, -0.08]), np.ones(4)).tolist(),
            },
            "wt": {
                "time": [-20, -0.51, -0.5, 0, 0.01, np.inf],
                "data": np.outer(np.array([0.01, 0.01, 1, 1, 0, 0]) * 1e6, np.ones(4)).tolist(),
            },
            "dwt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 4)).tolist()},
            "d2wt": {"time": [-np.inf, np.inf], "data": np.zeros((2, 4)).tolist()},
        }
    )

    return {
        "gspulse_inputs_pyconfig": {
            "settings": settings,
            "P_overrides": P_overrides,
            "plasma_params": plasma_params,
            "shapes": shapes,
            "initial_condition": initial_condition,
            "optimization_signals": optimization_signals,
        }
    }
