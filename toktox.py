import numpy as np
import pprint
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import copy
import json
import os
import shutil
from datetime import datetime
import time

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, create_power_flux_fun

from baseconfig import BASE_CONFIG

from read_eqdsk_extended import read_eqdsk_extended

LCFS_WEIGHT = 100.0
N_PSI = 1000
_NBI_W_TO_MA = 1/16e6
mu_0 = 4.0 * np.pi * 1e-7

# Setup output re-direct from TORAX to log file, suppressing frivolous warnings.
# Errors will still be output in terminal.
# This is the first step, needs to be given self._log_file once that is configured in self.fly().
import logging
import sys
def log_redirect_setup():
    r'''! Step 1/3 of setup to redirect noisy outputs to log file.
    Performs the initial, minimal logging setup.
    - Removes any handlers pre-configured by libraries.
    - Sets the root logger's level to capture all desired messages.
    - Adds a console handler for critical errors only.
    '''
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Capture INFO level and above
    
    # Remove any pre-existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a handler to show ONLY errors on the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('CONSOLE ERROR: [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

log_redirect_setup()

# Now import "noisy" packages, after running log_redirect_setup:
import torax


from contextlib import contextmanager
@contextmanager
def redirect_outputs_to_log(filename):
    r'''! Step 2/3 of setup to redirect noisy outputs to log file. 
    A context manager to temporarily redirect stdout and stderr to a file.
    @param filename Name of log file (self._log_file)
    '''
    if not filename:
        # If no filename is provided, do nothing.
        yield
        return

    with open(filename, 'a') as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            # Write a separator to the log to show where stdout redirection started
            log_file.write("\n--- [Begin capturing stdout/stderr] ---\n")
            yield
        finally:
            # Write a separator to show where it ended
            log_file.write("\n--- [End capturing stdout/stderr] ---\n")
            # Crucially, restore the original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr

class MyEncoder(json.JSONEncoder):
    '''! JSON Encoder Object to store simulation results.'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle xarray DataArray
        if hasattr(obj, 'to_numpy'):
            print(f'using numpy fix in json coverter for {obj}')
            return obj.to_numpy().tolist()
        return json.JSONEncoder.default(self, obj)

class TokTox:
    '''! TokaMaker + TORAX Coupled Pulse Simulation Code'''


    # ─── Initialization ─────────────────────────────────────────────────────────

    def __init__(self, t_init, t_final, eqtimes, g_eqdsk_arr, dt=0.1, times=None, last_surface_factor=0.95, n_rho=50, prescribed_currents=False, cocos=2, oft_env=None, oft_threads=2):
        r'''! Initialize the Coupled TokaMaker + TORAX object.
        @param t_init Start time (s).
        @param t_final End time (s).
        @param eqtimes Time points of each gEQDSK file.
        @param g_eqdsk_arr Filenames of each gEQDSK file.
        @param dt Time step (s).
        @param times Time points to sample output at.
        @param last_surface_factor Last surface factor for Torax.
        @param prescribed_currents Use prescribed coil currents or solve inverse problem to calculate currents.
        @param oft_env OFT environment if one is already initialized.
        @param oft_threads Number of threads OFT can use.
        '''
        if oft_env is not None:
            self._oftenv = oft_env
        else:
            self._oftenv = OFT_env(nthreads=oft_threads)
        self._tm = TokaMaker(self._oftenv)
        self._cocos = cocos

        self._state = {}
        self._eqtimes = eqtimes
        self._results = {}
        self._init_files = g_eqdsk_arr
        self._t_init = t_init
        self._t_final = t_final
        self._dt = dt # TORAX timestep
        self._prescribed_currents = prescribed_currents
        self._last_surface_factor = last_surface_factor
        self._n_rho = n_rho # resolution of TORAX grid
        self._psi_N = np.linspace(0.0, 1.0, N_PSI) # standardized psi_N grid all values should be mapped onto

        self._current_loop = 0

        if times is None:
            self._times = eqtimes
        else:
            self._times = sorted(times)
        # TODO organize initialization of _state
        self._state['R0_mag'] = np.zeros(len(self._times))
        self._state['Z'] = np.zeros(len(self._times))
        self._state['a'] = np.zeros(len(self._times))
        self._state['kappa'] = np.zeros(len(self._times))
        self._state['delta'] = np.zeros(len(self._times))
        self._state['B0'] = np.zeros(len(self._times))
        self._state['Ip'] = np.zeros(len(self._times))
        self._state['Ip_tm'] = np.zeros(len(self._times))
        self._state['Ip_tx'] = np.zeros(len(self._times))
        self._state['Ip_ni_tx'] = np.zeros(len(self._times))
        self._state['pax'] = np.zeros(len(self._times))
        self._state['pax_tm'] = np.zeros(len(self._times))
        self._state['beta_N_tm'] = np.zeros(len(self._times))
        self._state['beta_N_tx'] = np.zeros(len(self._times))
        self._state['l_i_tm'] = np.zeros(len(self._times))
        self._state['beta_pol'] = np.zeros(len(self._times))
        self._state['vloop_tm'] = np.zeros(len(self._times))
        self._state['vloop_tx'] = np.zeros(len(self._times))
        self._state['q95'] = np.zeros(len(self._times))
        self._state['q0'] = np.zeros(len(self._times))
        self._state['q95_tm'] = np.zeros(len(self._times))
        self._state['q0_tm'] = np.zeros(len(self._times))
        self._state['q_prof_tm'] = {}
        self._state['q_prof_tx'] = {}
        self._state['psi_lcfs_tm'] = np.zeros(len(self._times))
        self._state['psi_axis_tm'] = np.zeros(len(self._times))
        self._state['psi_lcfs_tx'] = np.zeros(len(self._times))
        self._state['psi_axis_tx'] = np.zeros(len(self._times))
        self._state['psi_tx'] = {}  
        self._state['psi_tm'] = {}
        self._state['psi_grid_prev_tm'] = np.zeros(len(self._times))
        self._psi_warm_start = {}  # {timestep_idx: psi_array} — persists across loops for warm-starting

        self._state['lcfs_geo'] = {}
        self._state['ffp_prof'] = {}
        self._state['pp_prof'] = {}
        self._state['eta_prof'] = {}
        self._state['T_e'] = {}
        self._state['T_i'] = {}
        self._state['n_e'] = {}
        self._state['n_i'] = {}
        self._state['f_GW'] = np.zeros(len(self._times))
        self._state['f_GW_vol'] = np.zeros(len(self._times))
        self._state['ptot'] = {}
        self._state['ffp_ni_prof'] = {}
        self._state['ffp_prof_tx'] = {}
        self._state['pp_prof_tx'] = {}
        self._state['ffp_prof_tm'] = {}
        self._state['pp_prof_tm'] = {}
        self._state['p_prof_tm'] = {} 
        self._state['p_prof_tx'] = {}
        self._state['f_prof_tm'] = {}

        # Thermal conductivity profiles
        self._state['chi_neo_e'] = {}
        self._state['chi_neo_i'] = {}
        self._state['chi_etg_e'] = {}
        self._state['chi_itg_e'] = {}
        self._state['chi_itg_i'] = {}
        self._state['chi_tem_e'] = {}
        self._state['chi_tem_i'] = {}
        self._state['chi_turb_e'] = {}
        self._state['chi_turb_i'] = {}

        # Diffusivity profiles
        self._state['D_itg_e'] = {}
        self._state['D_neo_e'] = {}
        self._state['D_tem_e'] = {}
        self._state['D_turb_e'] = {}

        self._state['R_inv_avg_tx'] = {}
        self._state['R_avg_tm'] = {}
        self._state['R_inv_avg_tm'] = {}
        
        # Current density profiles from TORAX
        self._state['j_tot'] = {}
        self._state['j_ohmic'] = {}
        self._state['j_ni'] = {}
        self._state['j_bootstrap'] = {}
        self._state['j_ecrh'] = {}
        self._state['j_external'] = {}
        self._state['j_generic_current'] = {}
        self._state['vol_tm'] = {}
        self._state['vol_tx'] = {}  # volume profile vs psi
        self._state['vol_tx_lcfs'] = np.zeros(len(self._times))  # volume at LCFS (scalar)

        self._results['lcfs_geo'] = {}
        self._results['dpsi_lcfs_dt'] = {}
        self._results['vloop_tm'] = np.zeros([20, len(self._times)])
        # self._results['vloop_tx'] = np.zeros([20, len(self._times)])
        self._results['q'] = {}
        self._results['jtot'] = {}
        self._results['n_e'] = {}
        self._results['T_e'] = {}
        self._results['T_i'] = {}


        R = []
        Z = []  
        a = []
        kappa = []
        delta = []
        B0 = []
        pax = []
        Ip = []
        lcfs = []
        ffp_prof = []
        pp_prof = []
        psi_axis = []
        psi_lcfs = []
        pres_prof = []
        fpol_prof = []

        for i, t in enumerate(self._eqtimes):
            g = read_eqdsk(g_eqdsk_arr[i])
            zmax = np.max(g['rzout'][:,1])
            zmin = np.min(g['rzout'][:,1])
            rmax = np.max(g['rzout'][:,0])
            rmin = np.min(g['rzout'][:,0])
            minor_radius = (rmax - rmin) / 2.0
            rgeo = (rmax + rmin) / 2.0
            highest_pt_idx = np.argmax(g['rzout'][:,1])
            lowest_pt_idx = np.argmin(g['rzout'][:,1])
            rupper = g['rzout'][highest_pt_idx][0]
            rlower = g['rzout'][lowest_pt_idx][0]
            delta_upper = (rgeo - rupper) / minor_radius
            delta_lower = (rgeo - rlower) / minor_radius

            R.append(g['rcentr'])
            Z.append(g['zaxis'])  # magnetic axis Z, not grid midpoint (zmid)
            a.append(minor_radius)
            kappa.append((zmax - zmin) / (2.0 * minor_radius))
            delta.append((delta_upper + delta_lower) / 2.0)
            
            B0.append(g['bcentr'])
            pax.append(g['pres'][0])
            Ip.append(abs(g['ip']))

            psi_axis.append(abs(g['psimag']))
            psi_lcfs.append(abs(g['psibry'])) # EQDSK stored psi in Wb/rad, same as stored in _state

            lcfs.append(g['rzout'])

            psi_eqdsk = np.linspace(0.0, 1.0, g['nr'])            
            ffp = np.interp(self._psi_N, psi_eqdsk, g['ffprim'])
            pp = np.interp(self._psi_N, psi_eqdsk, g['pprime'])
            ffp_prof.append(ffp)
            pp_prof.append(pp)

            pres = np.interp(self._psi_N, psi_eqdsk, g['pres'])
            fpol = np.interp(self._psi_N, psi_eqdsk, g['fpol'])
            pres_prof.append(pres)
            fpol_prof.append(fpol)

        self.lcfs = lcfs

        def interp_prof(profs, time):
            if time <= self._eqtimes[0]:
                return profs[0]
            for i in range(1, len(self._eqtimes)):
                if time == self._eqtimes[i]:
                    return profs[i]
                elif time > self._eqtimes[i-1] and time <= self._eqtimes[i]:
                    dt = self._eqtimes[i] - self._eqtimes[i-1]
                    alpha = (time - self._eqtimes[i-1]) / dt
                    return (1.0 - alpha) * profs[i-1] + alpha * profs[i]
            return profs[-1]

        for i, t in enumerate(self._times):
            # Default Scalars
            self._state['R0_mag'][i] = np.interp(t, self._eqtimes, R)
            self._state['Z'][i] = np.interp(t, self._eqtimes, Z)
            self._state['a'][i] = np.interp(t, self._eqtimes, a)
            self._state['kappa'][i] = np.interp(t, self._eqtimes, kappa)
            self._state['delta'][i] = np.interp(t, self._eqtimes, delta)
            self._state['B0'][i] = np.interp(t, self._eqtimes, B0)
            self._state['pax'][i] = np.interp(t, self._eqtimes, pax)
            self._state['Ip'][i] = np.interp(t, self._eqtimes, Ip)
            self._state['psi_axis_tm'][i] = np.interp(t, self._eqtimes, psi_axis)
            self._state['psi_lcfs_tm'][i] = np.interp(t, self._eqtimes, psi_lcfs)

            # Default Profiles
            self._state['lcfs_geo'][i] = interp_prof(lcfs, t)
            self._state['ffp_prof'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(ffp_prof, t), 'type': 'linterp'}
            self._state['pp_prof'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(pp_prof, t), 'type': 'linterp'}
            self._state['ffp_ni_prof'][i] = {'x': [], 'y': [], 'type': 'linterp'}

            self._state['eta_prof'][i]= {
                'x': self._psi_N.copy(),
                'y': np.zeros(N_PSI),
                'type': 'linterp',
            }
            
        # Save seed values from initial equilibria
        self._psi_axis_seed = self._state['psi_axis_tm'].copy()
        self._psi_lcfs_seed = self._state['psi_lcfs_tm'].copy()
        self._Ip_seed       = self._state['Ip'].copy()
        self._pax_seed      = self._state['pax'].copy()
        self._state['pax_tm'] = self._state['pax'].copy()

        self._psi_init = None
        
        self._Ip = None
        self._Zeff = None

        self._nbi_heating = None
        self._eccd_heating = None
        self._eccd_loc = None
        self._nbi_loc = None

        self._nbar = None
        self._n_e = None
        self._T_i = None
        self._T_e = None

        self._set_pedestal = None
        self._T_i_ped = None
        self._T_e_ped = None
        self._n_e_ped = None

        self._Te_right_bc = None
        self._Ti_right_bc = None
        self._ne_right_bc = None

        self._gp_s = None
        self._gp_dl = None

        # Transport / numerics overrides — None means "use value from
        # loaded config (or base config if no loaded config)".  Only set
        # to a real value when an explicit set_*() call is made AFTER
        # load_config().
        self._normalize_to_nbar = None
        self._evolve_density = None
        self._evolve_current = None
        self._evolve_Ti = None
        self._evolve_Te = None
        self._ped_top = None
        self._chi_min = None
        self._chi_max = None
        self._De_min = None
        self._De_max = None
        self._Ve_min = None
        self._Ve_max = None

        self._main_ion = None
        self._impurity = None
        self._enable_fusion = False
        self._enable_ei_exchange = False

        self._targets = None
        self._loaded_config = None   # set by load_config()
        self._tx_grid_type = None
        self._tx_grid = None

        self._eqdsk_skip = []

        # Psi snapshots for movie generation (populated during _run_tm)
        self._tm_psi_on_nodes = {}  # {loop: {i: psi_array}}

        # Temp/output directory state (set in fly())
        self._eqdsk_dir = None
        self._save_outputs = False
        self._debug_mode = False
        self._diagnostics = False
        self._logging_configured = False
        self._log_file = None

    # ─── Static Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _to_plain_python(obj):
        r'''! Recursively convert numpy scalars/arrays to plain Python types.

        Used before pformat-saving config dicts so the saved .py files are
        loadable without numpy (no ``array([...])`` references).
        '''
        if isinstance(obj, dict):
            return {TokTox._to_plain_python(k): TokTox._to_plain_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [TokTox._to_plain_python(v) for v in obj]
            return type(obj)(converted)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    @staticmethod
    def _config_merge(base, override):
        r'''! Recursively merge override into base TORAX config (in-place).

        For every key in override:
          - If both values are dicts, recurse.
          - Otherwise the override value wins.
        Keys in base that are absent from override are kept as-is.

        @param base     Dict to merge into (modified in-place).
        @param override Dict whose keys take precedence.
        @return base (for convenience).
        '''
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                TokTox._config_merge(base[key], val)
            else:
                base[key] = copy.deepcopy(val)
        return base

    @staticmethod
    def _flatten_time_dependent(config):
        r'''! Recursively flatten time-dependent config values to their initial value only.

        A dict whose keys are ALL numeric (int/float) is treated as time-dependent:
        only the entry with the smallest key is retained.
        Structural dicts (with any string keys) are recursed into.
        Tuples of form (times_array, values_array) are flattened to the first entry.
        '''
        for key in list(config.keys()):
            val = config[key]
            if isinstance(val, dict):
                if val and all(isinstance(k, (int, float, np.integer, np.floating)) for k in val.keys()):
                    # Time-dependent dict: keep only the first (smallest t) entry.
                    # Do NOT recurse — the nested value may be a rho-profile dict.
                    first_key = min(val.keys())
                    config[key] = {first_key: copy.deepcopy(val[first_key])}
                else:
                    # Structural dict with string keys: recurse.
                    TokTox._flatten_time_dependent(val)
            elif isinstance(val, (tuple, list)) and len(val) == 2:
                t_arr, v_arr = val
                try:
                    if (hasattr(t_arr, '__len__') and hasattr(v_arr, '__len__')
                            and not isinstance(t_arr, str) and not isinstance(v_arr, str)
                            and len(t_arr) > 1
                            and max(t_arr) > 1.0):  # rho grids have max==1.0; skip them
                        first_t = t_arr[0]
                        first_v = v_arr[0]
                        if isinstance(t_arr, np.ndarray):
                            config[key] = (np.array([first_t]), np.array([first_v]))
                        elif isinstance(t_arr, tuple):
                            config[key] = ((first_t,), (first_v,))
                        else:
                            config[key] = ([first_t], [first_v])
                except TypeError:
                    pass
            elif isinstance(val, (tuple, list)) and len(val) == 3:
                t_arr, rho_arr, v_arr = val
                try:
                    if (hasattr(t_arr, '__len__') and hasattr(rho_arr, '__len__') and hasattr(v_arr, '__len__')
                            and not isinstance(t_arr, str) and not isinstance(v_arr, str)
                            and len(t_arr) > 1
                            and max(t_arr) > 1.0):
                        first_t = t_arr[0]
                        first_v = v_arr[0]
                        if isinstance(t_arr, np.ndarray):
                            config[key] = (np.array([first_t]), rho_arr, np.array([first_v]))
                        elif isinstance(t_arr, tuple):
                            config[key] = ((first_t,), rho_arr, (first_v,))
                        else:
                            config[key] = ([first_t], rho_arr, [first_v])
                except TypeError:
                    pass


    # ─── Setup & Configuration ──────────────────────────────────────────────────

    def load_config(self, config):
        r'''! Load a TORAX config dict.

        The loaded config is deep-merged on top of BASE_CONFIG when the
        simulation config is built.  Any key present in the loaded config
        will override the corresponding BASE_CONFIG key; keys only in
        BASE_CONFIG are kept as-is.  Geometry is always overwritten by
        TokTox (eqdsk-based).

        Explicit ``set_*()`` calls made AFTER ``load_config()`` will
        override both the base and the loaded config.

        @param config Dictionary (TORAX config format).
        '''
        self._loaded_config = copy.deepcopy(config)

    def set_tx_grid(self, grid_type, grid):
        r'''! Set TORAX grid type and grid points.
        @param grid_type Grid type ('n_rho' or 'face_centers').
        @param grid Grid points (integer or np.array).
        '''
        self._tx_grid_type = grid_type
        self._tx_grid = grid
        if grid_type not in ['n_rho', 'face_centers']:
            raise ValueError(f'Invalid grid type: {type}. Must be "n_rho" or "face_centers".')

    def initialize_tm(self, mesh, R0_geo, weights=None, vsc=None):
        r'''! Initialize GS Solver Object.
        @param mesh Filename of reactor mesh.
        @param R0_geo Major radius of machine geometric center.
        @param vsc Vertical Stability Coil.
        '''
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh(mesh)
        self._tm.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._tm.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._tm.setup(order = 2, F0 = R0_geo*self._state['B0'][0])

        self._tm.settings.maxits = 100

        if vsc is not None:
            self._tm.set_coil_vsc({vsc: 1.0})

    def set_coil_reg(self, targets=None, i=0, coil_bounds=None, updownsym=False,
                     default_weight=1.0E-1, disable_coils=None,
                     disable_weight=1.0E4, symmetry_weight=1.0E3,
                     disable_virtual_vsc=True, vsc_weight=1.0E4):
        r'''! Set coil regularization using the matrix-based TokaMaker input.
        @param targets Dict of {coil_name: target_current} or {coil_name: time_series} for prescribed currents.
        @param i Timestep index (used for prescribed_currents interpolation).
        @param coil_bounds Dict of {coil_name: [min, max]} current bounds. Default ±50 kA.
        @param updownsym Enforce up-down symmetry for coil pairs (U/L naming convention).
        @param default_weight Regularization weight for normal coils (default 0.1).
        @param disable_coils List of coil name prefixes to disable (e.g. ['DV1', 'DV2']).
        @param disable_weight Regularization weight for disabled coils (default 1e4).
        @param symmetry_weight Regularization weight for symmetry constraints (default 1e3).
        @param disable_virtual_vsc Disable the virtual VSC coil (default True).
        @param vsc_weight Regularization weight for disabled VSC (default 1e4).
        '''
        if coil_bounds is None:
            coil_bounds = {key: [-5.0E4, 5.0E4] for key in self._tm.coil_sets}
        self._tm.set_coil_bounds(coil_bounds)
        self._coil_bounds = coil_bounds  # store for re-application after solve

        if self._prescribed_currents and targets:
            self._targets = targets

        # Build matrix-based regularization (same API as tokamaker_runner.py)
        n = self._tm.ncoils + 1  # +1 for virtual VSC coil
        coil_regmat = np.zeros((n, n), dtype=np.float64)
        coil_reg_weights = np.zeros((n,), dtype=np.float64)
        coil_targets = np.zeros((n,), dtype=np.float64)

        if disable_coils is None:
            disable_coils = []

        for name, coil in self._tm.coil_sets.items():
            cid = coil['id']

            # Determine target current for this coil
            if self._prescribed_currents and targets:
                t_current = np.interp(self._times[i], self._targets['time'], self._targets.get(name, [0.0]*len(self._targets.get('time', [0]))))
                coil_targets[cid] = t_current
            elif targets and name in targets:
                coil_targets[cid] = targets[name]

            if updownsym and 'U' in name:
                # Enforce up-down symmetry: I_upper - I_lower = 0
                lower_name = name.replace('U', 'L')
                if lower_name in self._tm.coil_sets:
                    coil_regmat[cid, cid] = 1.0
                    coil_regmat[cid, self._tm.coil_sets[lower_name]['id']] = -1.0
                    coil_reg_weights[cid] = symmetry_weight
                    continue

            # Normal coil regularization
            coil_regmat[cid, cid] = 1.0
            if any(name.startswith(prefix) for prefix in disable_coils):
                coil_reg_weights[cid] = disable_weight
            else:
                coil_reg_weights[cid] = default_weight

        # Virtual VSC coil (last entry in the matrix)
        coil_regmat[-1, -1] = 1.0
        if disable_virtual_vsc:
            coil_reg_weights[-1] = vsc_weight
        else:
            coil_reg_weights[-1] = default_weight

        self._tm.set_coil_reg(coil_regmat, reg_weights=coil_reg_weights, reg_targets=coil_targets)

        # Store config for post-solve re-application
        self._coil_reg_config = {
            'targets': targets, 'coil_bounds': coil_bounds,
            'updownsym': updownsym, 'default_weight': default_weight,
            'disable_coils': disable_coils, 'disable_weight': disable_weight,
            'symmetry_weight': symmetry_weight,
            'disable_virtual_vsc': disable_virtual_vsc, 'vsc_weight': vsc_weight,
        }


    # ─── Property Setters ───────────────────────────────────────────────────────

    def set_Ip(self, Ip):
        r'''! Set plasma current (Amps).
        @param ip Plasma current.
        '''
        self._Ip = Ip

    def set_ne(self, n_e):
        r'''! Set density profiles.
        @param n_e Electron density (m^-3).
        '''
        self._n_e = n_e

    def set_Te(self, T_e):
        r'''! Set electron temperature profiles (keV).
        @param T_e Electron temperature.
        '''
        self._T_e = T_e

    def set_Ti(self, T_i):
        r'''! Set ion temperature profiles (keV).
        @param T_i ion temperature.
        '''
        self._T_i = T_i

    def set_Zeff(self, Zeff):
        r'''! Set plasma effective charge.
        @param z_eff Effective charge.
        '''
        self._Zeff = Zeff

    def set_plasma_composition(self, main_ion=None, impurity=None):
        r'''! Set plasma composition (fuel and impurity species).

        Must be called together with set_Zeff() — set_Zeff() provides the
        Z_eff target which TORAX uses to determine the impurity density for the
        species specified here.

        @param main_ion Main ion species dict, e.g. {'D': 0.5, 'T': 0.5} for DT.
        @param impurity Impurity species string, e.g. 'Ne', 'Ar', 'W'.
        '''
        if main_ion is not None:
            self._main_ion = main_ion
        if impurity is not None:
            self._impurity = impurity

    def set_sources(self, fusion=False, ei_exchange=False):
        r'''! Enable standard TORAX physics sources using TORAX defaults.

        Each flag adds the corresponding source with an empty config dict so
        TORAX uses its built-in defaults.  Only call the ones you need — empty
        dicts pull in TORAX's machine-specific defaults (e.g. ITER geometry for
        fusion), which may not be appropriate for every simulation.
        
        Ohmic heating is enabled by default in the base_config. 
        Currently there is no way to disable ohmic heating, other than changing base_config.

        @param fusion    Enable fusion alpha heating.
        @param ei_exchange Enable electron-ion energy exchange.
        '''
        self._enable_fusion = fusion
        self._enable_ei_exchange = ei_exchange

    def set_nbar(self, nbar, normalize_to_nbar=True):
        r'''! Set line averaged density over time.
        @param nbar Density (m^-3).
        @param normalize_to_nbar Whether to normalize initial n_e profile to match nbar.
        '''
        self._nbar = nbar
        self._normalize_to_nbar = normalize_to_nbar # when True, initial n_e profile will be normalized to match nbar, but time evolution will not be constrained to match nbar

    def set_right_bc(self, ne_right_bc=None, Te_right_bc=None, Ti_right_bc=None):
        if ne_right_bc:
            self._ne_right_bc = ne_right_bc
        if Te_right_bc:
            self._Te_right_bc = Te_right_bc
        if Ti_right_bc:
            self._Ti_right_bc = Ti_right_bc

    def set_heating(self, nbi=None, nbi_loc=None, eccd=None, eccd_loc=None):
        r'''! Set heating sources for Torax.

        Ohmic heating is always enabled (it is on by default in BASE_CONFIG).
        @param nbi NBI heating (dictionary of {time: power_in_watts}).
        @param nbi_loc NBI deposition location (normalized rho).
        @param eccd ECCD heating (dictionary of {time: power_in_watts}).
        @param eccd_loc ECCD deposition location (normalized rho).
        '''
        if nbi is not None and nbi_loc is not None:
            self._nbi_heating = nbi
            self._nbi_loc = nbi_loc
        if eccd is not None and eccd_loc is not None:
            self._eccd_heating = eccd
            self._eccd_loc = eccd_loc

    def set_pedestal(self, set_pedestal=True, T_i_ped=None, T_e_ped=None, n_e_ped=None, ped_top=0.95):
        r'''! Set pedestals for ion and electron temperatures.
        @param T_i_ped Ion temperature pedestal (dictionary of temperature at times).
        @param T_e_ped Electron temperature pedestal (dictionary of temperature at times).
        '''
        self._set_pedestal = set_pedestal
        if T_i_ped:
            self._T_i_ped = T_i_ped
        if T_e_ped:
            self._T_e_ped = T_e_ped
        if n_e_ped:
            self._n_e_ped = n_e_ped

        self._ped_top = ped_top

    def set_evolve(self, density=True, Ti=True, Te=True, current=True):
        r'''! Set variables as either prescribed (False) or evolved (True).
        @param density Evolve density.
        @param Ti Evolve ion temperature.
        @param Te Evolve electron temperature.
        @param current Evolve current.
        '''
        self._evolve_density = density
        self._evolve_current = current
        self._evolve_Ti = Ti
        self._evolve_Te = Te

    # def set_Bp(self, Bp):
    #     Bp_t = sorted(Bp.keys())
    #     Bp_list = [Bp[t] for t in Bp_t]
    #     for i, t in enumerate(self._times):
    #         self._state['beta_pol'][i] = np.interp(t, Bp_t, Bp_list)
    
    # def set_Vloop(self, vloop):
    #     for i in range(len(self._times)):
    #         self._state['vloop_tm'][i] = vloop[i]

    def set_gaspuff(self, s=None, decay_length=None):
        r'''! Set gas puff particle source.
        @param s Particle source (particles/s).
        @param decay_length Decay length from edge (normalized rho coordinates).
        '''
        self._gp_s = s
        self._gp_dl = decay_length

    def set_chi(self, chi_min=None, chi_max=None):
        if chi_min is not None:
            self._chi_min = chi_min
        if chi_max is not None:
            self._chi_max = chi_max

    def set_De(self, De_min=None, De_max=None):
        if De_min is not None:
            self._De_min = De_min
        if De_max is not None:
            self._De_max = De_max

    def set_Ve(self, Ve_min=None, Ve_max=None):
        if Ve_min is not None:
            self._Ve_min = Ve_min
        if Ve_max is not None:
            self._Ve_max = Ve_max


    # ─── TORAX (TX) Methods ───────────────────────────────────────────────────

    def _pull_tx_onto_psi(self, data_tree, var_name, time, load_into_state='state', normalize=False, profile_type='linterp'):
        r'''! Load TORAX variable onto psi_norm grid.

        TORAX normalises its psi so that rho=1 maps to psi_N=1 internally,
        but that boundary corresponds to psi_N = last_surface_factor in the
        real equilibrium.  This method rescales the TORAX psi axis so that
        data are placed correctly in [0, last_surface_factor] on self._psi_N.

        Fill-value policy for the region outside the TORAX domain:
          * Left  (psi_N < first data point): hold first data value (avoids
            spurious zeros on-axis for cell-centred quantities like P_ohmic).
          * Right (psi_N > last_surface_factor):
              - j-profiles (profile_type='jphi-linterp'): fill with 0
                (current density vanishes at the separatrix).
              - All other profiles (T, n, p, FF', p'): hold the edge value.

        @param data_tree TORAX output data tree.
        @param var_name Name of variable (e.g., 'T_i', 'j_ohmic', 'FFprime').
        @param time Time value to extract.
        @param load_into_state If 'state' returns dict to load into '_state'; else returns plain array.
        @param normalize If True, normalize profile by the core value.
        @param profile_type Type key: 'linterp' or 'jphi-linterp'. Default 'linterp'.
        '''

        # Extract variable from profiles
        var = getattr(data_tree.profiles, var_name)
        var_data = var.sel(time=time, method='nearest').to_numpy()

        # Automatically detect which rho coordinate this variable uses
        if 'rho_cell_norm' in var.coords:
            grid = 'rho_cell_norm'
        elif 'rho_face_norm' in var.coords:
            grid = 'rho_face_norm'
        elif 'rho_norm' in var.coords:
            grid = 'rho_norm'
        else:
            raise ValueError(f"Variable {var_name} does not have a recognized rho coordinate")

        # Get psi_norm on rho_face_norm grid and psi on rho_norm grid
        psi_norm_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
        psi_rho_norm = data_tree.profiles.psi.sel(time=time, method='nearest').to_numpy()
        psi_norm_rho_norm = (psi_rho_norm - psi_rho_norm[0]) / (psi_rho_norm[-1] - psi_rho_norm[0])

        # Correct second element to avoid degeneracy from zero-gradient BC at core
        psi_norm_rho_norm[1] = (psi_norm_face[0] + psi_norm_face[1]) / 2.0

        # Convert psi to same grid as variable (TORAX internal: 0 → 1)
        if grid == 'rho_cell_norm':
            psi_on_grid = psi_norm_rho_norm[1:-1]
        elif grid == 'rho_face_norm':
            psi_on_grid = psi_norm_face
        elif grid == 'rho_norm':
            psi_on_grid = psi_norm_rho_norm

        # Rescale to real psi_N: TORAX's domain ceiling is psi_N = last_surface_factor.
        psi_on_grid_real = psi_on_grid * self._last_surface_factor

        # Fill values outside the TORAX domain.
        #   Left  (axis): hold the first available data value.
        #   Right (beyond last_surface_factor): 0 for j-profiles, edge value for all others.
        left_fill  = float(var_data[0])
        right_fill = 0.0 if profile_type == 'jphi-linterp' else float(var_data[-1])

        # Interpolate onto the TokTox psi_N grid
        data_on_psi = interp1d(psi_on_grid_real, var_data, kind='linear',
                               fill_value=(left_fill, right_fill),
                               bounds_error=False)(self._psi_N)

        # Normalize if requested
        if normalize:
            if grid == 'rho_cell_norm':
                # Cell-centred variables don't have a value at psi=0.
                # Find the index in data_on_psi closest to the first cell centre.
                core_idx = np.argmin(np.abs(self._psi_N - psi_on_grid_real[0]))
                data_on_psi /= data_on_psi[core_idx]
                self._log(
                    f"Normalizing {var_name} using value at psi={self._psi_N[core_idx]:.3f}"
                    f" (closest to first cell center at psi_real={psi_on_grid_real[0]:.3f})"
                )
            else:
                # Face or extended grid has actual core value at psi=0
                data_on_psi /= data_on_psi[0]

        if load_into_state == 'state':
            return {'x': self._psi_N.copy(), 'y': data_on_psi.copy(), 'type': profile_type}
        else:
            return data_on_psi

    def _get_tx_config(self):
        r'''! Generate config object for Torax simulation.

        Build order
        -----------
        1. Deep-copy BASE_CONFIG (from baseconfig.py).
        2. Deep-merge the loaded config on top (if load_config() was called).
           Every key in the loaded config overwrites the matching base key;
           keys only in BASE_CONFIG are kept as-is.
        3. Override geometry (always set by TokTox / TokaMaker equilibria).
        4. Override t_initial / t_final / fixed_dt from __init__ params.
        5. Use psi profile from loop 0 (if available) from profile_conditions.
        6. Apply any explicit set_*() overrides (only when the attribute is
           not None, i.e. the user called the setter after load_config).

        @return Torax config object.
        '''

        # ── 1. Start from base config ──────────────────────────────────────
        myconfig = copy.deepcopy(BASE_CONFIG)

        # ── 2. Deep-merge loaded config ────────────────────────────────────
        if self._loaded_config is not None:
            self._config_merge(myconfig, self._loaded_config)

        # ── 3. Geometry (always set by TokTox) ─────────────────────────────
        myconfig['geometry'] = {
            'geometry_type': 'eqdsk',
            'geometry_directory': os.getcwd(),
            'last_surface_factor': self._last_surface_factor,
            'n_surfaces': 50,
            'Ip_from_parameters': True, # True tells TX to pull from config, not from eqdsk, in case eqdsks fail TX retains correct Ip targets
        }
        if self._current_loop == 1:
            eq_safe = []
            t_safe = []
            for i, t in enumerate(self._eqtimes):
                eq = self._init_files[i]
                if self._test_eqdsk(eq):
                    self._log(f'\tTX: Using eqdsk at t={t}')
                    eq_safe.append(eq)
                    t_safe.append(t)
                else:
                    if not self._skip_bad_init_eqdsks:
                        raise ValueError(f'Bad initial gEQDSK at t={t}: {eq}')
                    self._log(f'\tTX: Skipping eqdsk at t={t}')
            myconfig['geometry']['geometry_configs'] = {
                t: {'geometry_file': eq_safe[i], 'cocos': self._cocos} for i, t in enumerate(t_safe)
            }
        else:
            # For times where TM succeeded last loop, use the TM-solved EQDSK.
            # For times where TM failed, fall back to the nearest solved EQDSK and
            eqtimes_arr = np.array(self._eqtimes)
            full_eqdsk_map = {}
            n_tm = 0
            for i, t in enumerate(self._times):
                eqdsk = os.path.join(self._eqdsk_dir, f'{self._current_loop - 1:03d}.{i:03d}.eqdsk')
                tm_ok = (eqdsk not in self._eqdsk_skip) and self._test_eqdsk(eqdsk)
                if tm_ok:
                    full_eqdsk_map[t] = eqdsk
                    n_tm += 1
            if n_tm == 0:
                self._log(f'Warning: Loop {self._current_loop}: no valid TM EQDSKs from loop {self._current_loop-1}, using all seed EQDSKs.')
            else:
                self._log(f'Loop {self._current_loop}: using {n_tm}/{len(self._times)} TM-solved EQDSKs, {len(self._times)-n_tm} seed fallbacks.')
            
            myconfig['geometry']['geometry_configs'] = {
                t: {'geometry_file': eqdsk_f, 'cocos': self._cocos} for t, eqdsk_f in full_eqdsk_map.items()
            }

        if self._tx_grid_type == 'n_rho':
            myconfig['geometry']['n_rho'] = self._tx_grid
        elif self._tx_grid_type == 'face_centers':
            myconfig['geometry']['face_centers'] = self._tx_grid

        # ── 4. Override t_initial / t_final / fixed_dt from __init__ ───────
        myconfig.setdefault('numerics', {})
        myconfig['numerics']['t_initial'] = self._t_init
        myconfig['numerics']['t_final'] = self._t_final
        myconfig['numerics']['fixed_dt'] = self._dt

        # ── 5. Psi profile from loop 0  ──────────
        myconfig.setdefault('profile_conditions', {})
        if self._psi_init is not None:
            myconfig['profile_conditions']['psi'] = self._psi_init
            myconfig['profile_conditions']['initial_psi_mode'] = 'profile_conditions'
            myconfig['profile_conditions']['initial_psi_from_j'] = False
        else:
            myconfig['profile_conditions']['initial_psi_mode'] = 'geometry' # if loop 0 wasn't run, uses psi from initial eqdsk, not ideal

        # ── 6. Explicit set_*() overrides ──────────────────────────────────
        #    Only applied when the attribute is not None (i.e. the user made an explicit setter call, or will be None if relying on the loaded config / base config value).
        if self._Ip is not None:
            myconfig['profile_conditions']['Ip'] = self._Ip

        if self._n_e is not None:
            myconfig['profile_conditions']['n_e'] = self._n_e
        
        if self._T_e is not None:
            myconfig['profile_conditions']['T_e'] = self._T_e
        
        if self._T_i is not None:
            myconfig['profile_conditions']['T_i'] = self._T_i
        
        if self._Zeff is not None:
            myconfig.setdefault('plasma_composition', {})
            myconfig['plasma_composition']['Z_eff'] = self._Zeff

        if self._main_ion is not None:
            myconfig.setdefault('plasma_composition', {})
            myconfig['plasma_composition']['main_ion'] = self._main_ion

        if self._impurity is not None:
            myconfig.setdefault('plasma_composition', {})
            myconfig['plasma_composition']['impurity'] = self._impurity

        if self._enable_fusion:
            myconfig.setdefault('sources', {})
            myconfig['sources'].setdefault('fusion', {})

        if self._enable_ei_exchange:
            myconfig.setdefault('sources', {})
            myconfig['sources'].setdefault('ei_exchange', {})

        if self._eccd_loc is not None:
            myconfig.setdefault('sources', {})
            myconfig['sources'].setdefault('ecrh', {})
            myconfig['sources']['ecrh']['P_total'] = self._eccd_heating
            myconfig['sources']['ecrh']['gaussian_location'] = self._eccd_loc

        if self._nbi_heating is not None:
            nbi_times, nbi_pow = zip(*self._nbi_heating.items())    
            myconfig.setdefault('sources', {})
            myconfig['sources'].setdefault('generic_heat', {})
            myconfig['sources']['generic_heat']['P_total'] = (nbi_times, nbi_pow)
            myconfig['sources']['generic_heat']['gaussian_location'] = self._nbi_loc
            myconfig['sources'].setdefault('generic_current', {})
            myconfig['sources']['generic_current']['I_generic'] = (nbi_times, _NBI_W_TO_MA * np.array(nbi_pow))
            myconfig['sources']['generic_current']['gaussian_location'] = self._nbi_loc

        if self._T_i_ped is not None:
            myconfig.setdefault('pedestal', {})
            myconfig['pedestal']['T_i_ped'] = self._T_i_ped
        if self._T_e_ped is not None:
            myconfig.setdefault('pedestal', {})
            myconfig['pedestal']['T_e_ped'] = self._T_e_ped

        if self._n_e_ped is not None:
            myconfig.setdefault('pedestal', {})
            myconfig['pedestal']['n_e_ped_is_fGW'] = False
            myconfig['pedestal']['n_e_ped'] = self._n_e_ped
        
        if self._set_pedestal is not None:
            myconfig.setdefault('pedestal', {})
            myconfig['pedestal']['set_pedestal'] = self._set_pedestal
        if self._ped_top is not None:
            myconfig.setdefault('pedestal', {})
            myconfig['pedestal']['rho_norm_ped_top'] = self._ped_top
        
        if self._nbar is not None:
            myconfig['profile_conditions']['nbar'] = self._nbar
        if self._normalize_to_nbar is not None:
            myconfig['profile_conditions']['normalize_n_e_to_nbar'] = self._normalize_to_nbar

        if self._ne_right_bc is not None:
            myconfig['profile_conditions']['n_e_right_bc_is_fGW'] = False
            myconfig['profile_conditions']['n_e_right_bc'] = self._ne_right_bc

        if self._Te_right_bc is not None:
            myconfig['profile_conditions']['T_e_right_bc'] = self._Te_right_bc
        if self._Ti_right_bc is not None:
            myconfig['profile_conditions']['T_i_right_bc'] = self._Ti_right_bc

        if self._evolve_density is not None:
            myconfig['numerics']['evolve_density'] = self._evolve_density
        if self._evolve_current is not None:
            myconfig['numerics']['evolve_current'] = self._evolve_current
        if self._evolve_Ti is not None:
            myconfig['numerics']['evolve_ion_heat'] = self._evolve_Ti
        if self._evolve_Te is not None:
            myconfig['numerics']['evolve_electron_heat'] = self._evolve_Te
                
        if self._gp_s is not None and self._gp_dl is not None:
            myconfig.setdefault('sources', {})
            myconfig['sources']['gas_puff'] = {
                'S_total': self._gp_s,
                'puff_decay_length': self._gp_dl,
            }

        myconfig.setdefault('transport', {})
        if self._chi_min is not None:
            myconfig['transport']['chi_min'] = self._chi_min
        if self._chi_max is not None:
            myconfig['transport']['chi_max'] = self._chi_max
        if self._De_min is not None:
            myconfig['transport']['D_e_min'] = self._De_min
        if self._De_max is not None:
            myconfig['transport']['D_e_max'] = self._De_max
        if self._Ve_min is not None:
            myconfig['transport']['V_e_min'] = self._Ve_min
        if self._Ve_max is not None:
            myconfig['transport']['V_e_max'] = self._Ve_max

        if self._save_outputs or self._debug_mode:
            config_filename = os.path.join(self._out_dir, 'results', f'tx_config{self._current_loop}.py')
            with open(config_filename, 'w') as f:
                f.write('# Torax configuration\n')
                f.write(f'# Loop {self._current_loop}\n\n')
                f.write('tx_config = ')
                f.write(pprint.pformat(self._to_plain_python(myconfig), width=100))

        tx_config = torax.ToraxConfig.from_dict(myconfig)
        return tx_config

    def _test_eqdsk(self, eqdsk):
            myconfig = copy.deepcopy(BASE_CONFIG)
            if self._loaded_config is not None:
                self._config_merge(myconfig, self._loaded_config)
            myconfig['geometry'] = {
                'geometry_type': 'eqdsk',
                'geometry_directory': os.getcwd(),
                'last_surface_factor': self._last_surface_factor,
                'Ip_from_parameters': False,
                'geometry_file': eqdsk,
                'cocos': self._cocos,
            }
            try:
                # with redirect_output_to_log(self._log_file):
                    # print('FREDDIE TEST SHOULD BE IN LOG FILE')
                _ = torax.ToraxConfig.from_dict(myconfig)
                return True
            except Exception as e:
                self._log(f"TEST EQDSK FAILED: {repr(e)}")
                return False

    def _run_tx_init(self):
        r'''! Loop 0: Run a short TORAX simulation with eqdsk geometry to equilibrate initial inputs.

        Run TORAX for 1 second with steady-state (time-flattened) inputs lets TORAX
        evolve them to a more physical state.  The relaxed values are then
        injected into the config used by the main simulation (loop 1 onwards).
        '''
        INIT_RUNTIME = 0.5
        self._log('Transport init: building steady-state init config...')

        init_config = copy.deepcopy(BASE_CONFIG)
        if self._loaded_config is not None:
            self._config_merge(init_config, self._loaded_config)

        # eqdsk geometry from the first seed file — same geometry as loop 1, i=0.
        # This ensures the psi evolved here satisfies the same GS metric coefficients
        # as the main sim, so injected psi produces smooth j at t=0 of loop 1.
        init_eqdsk = self._init_files[0]
        if not self._test_eqdsk(init_eqdsk):
            raise ValueError(f'Transport init: first seed eqdsk is not valid: {init_eqdsk}')
        init_config['geometry'] = {
            'geometry_type': 'eqdsk',
            'geometry_directory': os.getcwd(),
            'last_surface_factor': self._last_surface_factor,
            'n_surfaces': 50,
            'Ip_from_parameters': True,
            'geometry_configs': {self._t_init: {'geometry_file': init_eqdsk, 'cocos': 2}},
        }

        if self._tx_grid_type == 'n_rho':
            init_config['geometry']['n_rho'] = self._tx_grid
        elif self._tx_grid_type == 'face_centers':
            init_config['geometry']['face_centers'] = self._tx_grid

        # Numerics: 1-second steady-state init sim
        init_config.setdefault('numerics', {})
        init_config['numerics']['t_initial'] = self._t_init
        init_config['numerics']['t_final'] = self._t_init + INIT_RUNTIME
        init_config['numerics']['fixed_dt'] = 0.01
        
        init_config['numerics']['evolve_current'] = True # Let current evolve to relax to psi profile
        init_config['numerics']['evolve_density'] = False # Fix ne and Te/Ti profiles
        init_config['numerics']['evolve_ion_heat'] = False
        init_config['numerics']['evolve_electron_heat'] = False

        # Explicitly use J mode (nu formula) to initialize psi in the loop-0 sim.
        # This is the same mode the main sim (loop 1+) will use, so T_e/T_i
        # injected from here will be self-consistent with the initial current profile.
        init_config.setdefault('profile_conditions', {})
        init_config['profile_conditions']['initial_psi_mode'] = 'geometry'

        # Propagate Ip from set_Ip() into the Loop 0 config.
        # Without this, TORAX runs without an Ip constraint and may produce
        # psi values with the wrong magnitude or sign (observed: ~10× too large
        # for ITER). The loaded_config Ip (if any) is already merged above;
        # set_Ip() values are stored separately in self._Ip and must be applied here.
        if self._Ip is not None:
            init_config['profile_conditions']['Ip'] = copy.deepcopy(self._Ip)

        # Flatten all time-dependent values to their initial value (steady-state inputs)
        self._flatten_time_dependent(init_config)

        if self._save_outputs or self._debug_mode:
            config_filename = os.path.join(self._out_dir, 'results', 'tx_config0.py')
            with open(config_filename, 'w') as f:
                f.write('# Torax configuration\n# Loop 0 (transport init)\n\n')
                f.write('tx_config = ')
                f.write(pprint.pformat(self._to_plain_python(init_config), width=100))

        self._log(f'Transport init: running ~{INIT_RUNTIME}s steady-state TORAX simulation...')
        tx_config = torax.ToraxConfig.from_dict(init_config)
        data_tree, hist = torax.run_simulation(tx_config, log_timestep_info=False)

        if hist.sim_error != torax.SimError.NO_ERROR:
            raise ValueError(f'Transport init simulation failed: {hist.sim_error}')

        t_final_init = self._t_init + INIT_RUNTIME

        # Propagate psi from init into main simulation config
        main_config = self._loaded_config if self._loaded_config is not None else BASE_CONFIG
        main_config.setdefault('profile_conditions', {})

        # Extract psi directly on its own grid
        psi_xr = data_tree.profiles.psi.sel(time=t_final_init, method='nearest')
        rho_psi_arr = psi_xr.coords['rho_norm'].to_numpy()
        psi_arr = psi_xr.to_numpy()
        self._psi_init = ([self._t_init], rho_psi_arr.tolist(), [psi_arr.tolist()])
        


    def _run_tx(self):
        r'''! Run the TORAX transport simulation.
        @return Tuple (consumed_flux, consumed_flux_integral).
        '''
        self._print(f'  TORAX: running simulation...')
        myconfig = self._get_tx_config()
        try:
            data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)
        except Exception as e:
            self._print(f'  TORAX: config/init FAILED — {e}')
            raise

        if hist.sim_error != torax.SimError.NO_ERROR:
            self._print(f'  TORAX: sim FAILED ({hist.sim_error})')
            raise ValueError(f'TORAX failed to run the simulation: {hist.sim_error}')
        
        v_loops = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._tx_update(i, data_tree)
            v_loops[i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')

        self._res_update(data_tree)

        consumed_flux = 2.0 * np.pi * (self._state['psi_lcfs_tx'][-1] - self._state['psi_lcfs_tx'][0])
        consumed_flux_integral = np.trapezoid(v_loops[0:], self._times[0:])
        self._log(f"Loop {self._current_loop} TORAX: cflux={consumed_flux:.4f} Wb")
        self._print(f'  TORAX: done (cflux={consumed_flux:.4f} Wb)')
        return consumed_flux, consumed_flux_integral

    def _tx_update(self, i, data_tree):
        r'''! Update the simulation state and simulation results based on results of the Torax simulation.
        @param i Timestep of the solve.
        @param data_tree Result object from Torax.
        '''
        t = self._times[i]


        self._state['Ip'][i] =          data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['Ip_tx'][i] =       data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['Ip_ni_tx'][i] =    data_tree.scalars.I_non_inductive.sel(time=t, method='nearest')
        pax_new = data_tree.profiles.pressure_thermal_total.sel(time=t, rho_norm=0.0, method='nearest').values
        pax_old = self._state['pax'][i]
        self._state['pax'][i] = pax_new
        
        self._state['beta_pol'][i] = float(data_tree.scalars.beta_pol.sel(time=t, method='nearest'))
        self._state['beta_N_tx'][i]  = float(data_tree.scalars.beta_N.sel(time=t,  method='nearest'))
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')
        self._state['q0'][i] = data_tree.profiles.q.sel(time=t, rho_face_norm=0.0, method='nearest')


        self._state['ffp_prof'][i] = self._pull_tx_onto_psi(data_tree, 'FFprime', t, load_into_state='state')
        self._state['pp_prof'][i] =  self._pull_tx_onto_psi(data_tree, 'pprime',  t, load_into_state='state')
        
        self._state['ffp_prof_tx'][i] = self._pull_tx_onto_psi(data_tree, 'FFprime', t, load_into_state='state') # temp for calculating j_phi
        self._state['ffp_prof_tx'][i]['y'] *= -2.0*np.pi  # convert from TX units to TM units

        self._state['pp_prof_tx'][i] =  self._pull_tx_onto_psi(data_tree, 'pprime', t, load_into_state='state')
        self._state['pp_prof_tx'][i]['y'] *= -2.0*np.pi  # convert from TX units to TM units

        self._state['p_prof_tx'][i] = self._pull_tx_onto_psi(data_tree, 'pressure_thermal_total', t, load_into_state='state')

        self._state['vloop_tx'][i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
        
        self._state['q_prof_tx'][i] = self._pull_tx_onto_psi(data_tree, 'q', t, load_into_state='state')

        self._state['j_tot'][i] =            self._pull_tx_onto_psi(data_tree, 'j_total',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ohmic'][i] =          self._pull_tx_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ni'][i] =          self._pull_tx_onto_psi(data_tree, 'j_non_inductive',  t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_bootstrap'][i] =      self._pull_tx_onto_psi(data_tree, 'j_bootstrap',      t, load_into_state='state', profile_type='jphi-linterp')
        

        self._state['j_ecrh'][i] = self._pull_tx_onto_psi(data_tree, 'j_ecrh', t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_external'][i] = self._pull_tx_onto_psi(data_tree, 'j_external', t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_generic_current'][i] = self._pull_tx_onto_psi(data_tree, 'j_generic_current', t, load_into_state='state', profile_type='jphi-linterp')

        self._state['R_inv_avg_tx'][i] = self._pull_tx_onto_psi(data_tree, 'gm9', t, load_into_state='state')

        ffp_ni = self._calc_ffp_ni(i, data_tree)

        self._state['ffp_ni_prof'][i] = {'x': self._psi_N.copy(), 'y': ffp_ni.copy(), 'type': 'linterp'}         

        self._state['T_i'][i] = self._pull_tx_onto_psi(data_tree, 'T_i', t, load_into_state='state')
        self._state['T_e'][i] = self._pull_tx_onto_psi(data_tree, 'T_e', t, load_into_state='state')
        self._state['n_i'][i] = self._pull_tx_onto_psi(data_tree, 'n_i', t, load_into_state='state')
        self._state['n_e'][i] = self._pull_tx_onto_psi(data_tree, 'n_e', t, load_into_state='state')
        self._state['f_GW'][i] = data_tree.scalars.fgw_n_e_line_avg.sel(time=t, method='nearest').item()
        self._state['f_GW_vol'][i] = data_tree.scalars.fgw_n_e_volume_avg.sel(time=t, method='nearest').item()

        self._state['ptot'][i] = self._pull_tx_onto_psi(data_tree, 'pressure_thermal_total', t, load_into_state='state')

        # Get conductivity and convert to resistivity (eta = 1/sigma)
        conductivity = self._pull_tx_onto_psi(data_tree, 'sigma_parallel', t, load_into_state=None)
        self._state['eta_prof'][i] = {
            'x': self._psi_N.copy(),
            'y': 1.0 / conductivity,
            'type': 'linterp',
        }

        psi_tx = self._pull_tx_onto_psi(data_tree, 'psi', t, load_into_state=None) / (2.0 * np.pi) # TORAX outputs psi in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        psi_tx = 2.0 * psi_tx[-1] - psi_tx  # reflect over psi_lcfs to convert from TX to TM convention
        self._state['psi_tx'][i] = {'x': self._psi_N.copy(), 'y': psi_tx.copy(), 'type': 'linterp',}
        self._state['psi_lcfs_tx'][i] = self._state['psi_tx'][i]['y'][-1]
        self._state['psi_axis_tx'][i] = self._state['psi_tx'][i]['y'][0]

        # Pull volume and volume derivative from TORAX
        self._state['vol_tx_lcfs'][i] = data_tree.profiles.volume.sel(time=t, rho_norm=1.0, method='nearest').item()
        self._state['vol_tx'][i] = self._pull_tx_onto_psi(data_tree, 'volume', t, load_into_state='state')
        # Pull thermal conductivity (chi) profiles with safety checks
        chi_profiles = [
            'chi_neo_e', 'chi_neo_i', 'chi_etg_e', 'chi_itg_e', 'chi_itg_i',
            'chi_tem_e', 'chi_tem_i', 'chi_turb_e', 'chi_turb_i'
        ]
        for chi_key in chi_profiles:
            try:
                self._state[chi_key][i] = self._pull_tx_onto_psi(data_tree, chi_key, t, load_into_state='state')
            except (KeyError, AttributeError):
                # Variable not available in this data_tree, skip
                pass

        # Pull diffusivity (D) profiles with safety checks
        d_profiles = [
            'D_itg_e', 'D_neo_e', 'D_tem_e', 'D_turb_e'
        ]
        for d_key in d_profiles:
            try:
                self._state[d_key][i] = self._pull_tx_onto_psi(data_tree, d_key, t, load_into_state='state')
            except (KeyError, AttributeError):
                # Variable not available in this data_tree, skip
                pass

    def _calc_ffp_ni(self, i, data_tree):
        r'''! Calculate non-inductive FF' profile from TORAX current densities.
        
        The full GS relation is:
            FF'_total = 2 * mu_0 * (j_tor + p' * <R>) / <1/R>
        
        To avoid double-counting p' when decomposing into inductive/non-inductive:
            FF'_NI = 2 * mu_0 * j_NI / <1/R>
            FF'_I  = 2 * mu_0 * (j_I + p' * <R>) / <1/R>
        
        @param i Time index
        @param data_tree TORAX output data tree
        @return FF'_NI profile array
        '''
        t = self._times[i]
        R_inv_avg = self._state['R_inv_avg_tx'][i]['y']

        j_ni = self._state['j_ni'][i]['y']
        ffp_ni = np.where(R_inv_avg != 0, mu_0 * j_ni / R_inv_avg, 0.0)

        return ffp_ni

    def _res_update(self, data_tree):

        self._results['t_res'] = self._times

        for t in self._times:
            self._results['T_e'][t] = self._pull_tx_onto_psi(data_tree, 'T_e', t, load_into_state='state', normalize=False)
            self._results['T_i'][t] = self._pull_tx_onto_psi(data_tree, 'T_i', t, load_into_state='state', normalize=False)
            self._results['n_e'][t] = self._pull_tx_onto_psi(data_tree, 'n_e', t, load_into_state='state', normalize=False)
            self._results['q'][t] =   self._pull_tx_onto_psi(data_tree, 'q', t, load_into_state='state', normalize=False)

        self._results['E_fusion'] = {
            'x': list(data_tree.scalars.E_fusion.coords['time'].values),
            'y': data_tree.scalars.E_fusion.to_numpy()
        }

        self._results['Q'] = {
            'x': list(data_tree.scalars.Q_fusion.coords['time'].values),
            'y': data_tree.scalars.Q_fusion.to_numpy(),
        }

        self._results['Ip'] = {
            'x': list(data_tree.scalars.Ip.coords['time'].values),
            'y': data_tree.scalars.Ip.to_numpy(),
        }

        self._results['B0'] = {
            'x': list(data_tree.scalars.B_0.coords['time'].values),
            'y': data_tree.scalars.B_0.to_numpy(),
        }

        self._results['n_e_line_avg'] = {
            'x': list(data_tree.scalars.n_e_line_avg.coords['time'].values),
            'y': data_tree.scalars.n_e_line_avg.to_numpy(),
        }

        self._results['n_i_line_avg'] = {
            'x': list(data_tree.scalars.n_i_line_avg.coords['time'].values),
            'y': data_tree.scalars.n_i_line_avg.to_numpy(),
        }

        my_times = list(data_tree.profiles.T_e.coords['time'].values)
        T_e_line_avg = np.array([])
        for my_t in my_times:
            T_e_line_avg = np.append(T_e_line_avg, data_tree.profiles.T_e.sel(time=my_t).mean(dim='rho_norm'))
        self._results['T_e_line_avg'] = {
            'x': my_times,
            'y': T_e_line_avg,
        }

        my_times = list(data_tree.profiles.T_i.coords['time'].values)
        T_i_line_avg = np.array([])
        for my_t in my_times:
            T_i_line_avg = np.append(T_i_line_avg, data_tree.profiles.T_i.sel(time=my_t).mean(dim='rho_norm'))
        self._results['T_i_line_avg'] = {
            'x': my_times,
            'y': T_i_line_avg,
        }
        
        n_e_core = data_tree.profiles.n_e.sel(rho_norm=0.0)
        self._results['n_e_core'] = {
            'x': list(n_e_core.coords['time'].values),
            'y': n_e_core.to_numpy(),
        }

        n_i_core = data_tree.profiles.n_i.sel(rho_norm=0.0)
        self._results['n_i_core'] = {
            'x': list(n_i_core.coords['time'].values),
            'y': n_i_core.to_numpy(),
        }

        T_e_core = data_tree.profiles.T_e.sel(rho_norm=0.0)
        self._results['T_e_core'] = {
            'x': list(T_e_core.coords['time'].values),
            'y': T_e_core.to_numpy(),
        }

        T_i_core = data_tree.profiles.T_i.sel(rho_norm=0.0)
        self._results['T_i_core'] = {
            'x': list(T_i_core.coords['time'].values),
            'y': T_i_core.to_numpy(),
        }

        self._results['beta_N'] = {
            'x': list(data_tree.scalars.beta_N.coords['time'].values),
            'y': data_tree.scalars.beta_N.to_numpy(),
        }

        self._results['q95'] = {
            'x': list(data_tree.scalars.q95.coords['time'].values),
            'y': data_tree.scalars.q95.to_numpy(),
        }

        self._results['H98'] = {
            'x': list(data_tree.scalars.H98.coords['time'].values),
            'y': data_tree.scalars.H98.to_numpy(),
        }

        self._results['v_loop_lcfs'] = {
            'x': list(data_tree.scalars.v_loop_lcfs.coords['time'].values),
            'y': data_tree.scalars.v_loop_lcfs.to_numpy(),
        }

        psi_lcfs = data_tree.profiles.psi.sel(rho_norm = 1.0)
        self._results['psi_lcfs_tx'] = {
            'x': list(psi_lcfs.coords['time'].values),
            'y': psi_lcfs.to_numpy() / (2.0 * np.pi), # TORAX outputs in Wb, stored in Wb/rad
        }

        psi_axis = data_tree.profiles.psi.sel(rho_norm = 0.0)
        self._results['psi_axis_tx'] = {
            'x': list(psi_axis.coords['time'].values),
            'y': psi_axis.to_numpy() / (2.0 * np.pi), # TORAX outputs in Wb, stored in Wb/rad
        }

        self._results['li3'] = {
            'x': list(data_tree.scalars.li3.coords['time'].values),
            'y': data_tree.scalars.li3.to_numpy(),
        }

        self._results['P_alpha_total'] = {
            'x': list(data_tree.scalars.P_alpha_total.coords['time'].values),
            'y': data_tree.scalars.P_alpha_total.to_numpy(),
        }

        self._results['P_aux_total'] = {
            'x': list(data_tree.scalars.P_aux_total.coords['time'].values),
            'y': data_tree.scalars.P_aux_total.to_numpy(),
        }

        self._results['P_ohmic_e'] = {
            'x': list(data_tree.scalars.P_ohmic_e.coords['time'].values),
            'y': data_tree.scalars.P_ohmic_e.to_numpy(),
        }

        self._results['P_radiation_e'] = {
            'x': list(data_tree.scalars.P_radiation_e.coords['time'].values),
            'y': -1.0 * data_tree.scalars.P_radiation_e.to_numpy(),
        }

        self._results['P_SOL_total'] = {
            'x': list(data_tree.scalars.P_SOL_total.coords['time'].values),
            'y': data_tree.scalars.P_SOL_total.to_numpy(),
        }

        # ── Additional comprehensive physics outputs ──

        # Stored energy
        try:
            self._results['W_thermal'] = {
                'x': list(data_tree.scalars.W_thermal.coords['time'].values),
                'y': data_tree.scalars.W_thermal.to_numpy(),
            }
        except AttributeError:
            pass

        # Energy confinement time
        try:
            self._results['tau_E'] = {
                'x': list(data_tree.scalars.tau_E.coords['time'].values),
                'y': data_tree.scalars.tau_E.to_numpy(),
            }
        except AttributeError:
            pass

        # Bootstrap fraction
        try:
            self._results['f_bootstrap'] = {
                'x': list(data_tree.scalars.f_bootstrap.coords['time'].values),
                'y': data_tree.scalars.f_bootstrap.to_numpy(),
            }
        except AttributeError:
            pass

        # Non-inductive current fraction and current
        self._results['f_ni'] = {
            'x': list(data_tree.scalars.f_non_inductive.coords['time'].values),
            'y': data_tree.scalars.f_non_inductive.to_numpy(),
        }
        self._results['I_ni'] = {
            'x': list(data_tree.scalars.I_non_inductive.coords['time'].values),
            'y': data_tree.scalars.I_non_inductive.to_numpy(),
        }

        # Beta poloidal
        self._results['beta_pol'] = {
            'x': list(data_tree.scalars.beta_pol.coords['time'].values),
            'y': data_tree.scalars.beta_pol.to_numpy(),
        }

        # Greenwald fraction
        self._results['f_GW'] = {
            'x': list(self._times),
            'y': np.array(self._state['f_GW']),
        }

        # Peak values for quick access
        Q_arr = self._results.get('Q', {}).get('y', np.array([0]))
        self._results['Q_max'] = float(np.nanmax(Q_arr)) if len(Q_arr) > 0 else 0.0
        self._results['Q_avg_flattop'] = float(np.nanmean(Q_arr[self._flattop])) if np.any(self._flattop) and len(Q_arr) == len(self._flattop) else 0.0

        # TokaMaker state arrays (for visualization)
        self._results['Ip_tm'] = {'x': list(self._times), 'y': np.array(self._state['Ip_tm'])}
        self._results['Ip_tx'] = {'x': list(self._times), 'y': np.array(self._state['Ip_tx'])}
        self._results['Ip_ni_tx'] = {'x': list(self._times), 'y': np.array(self._state['Ip_ni_tx'])}
        self._results['psi_lcfs_tm'] = {'x': list(self._times), 'y': np.array(self._state['psi_lcfs_tm'])}
        self._results['psi_axis_tm'] = {'x': list(self._times), 'y': np.array(self._state['psi_axis_tm'])}
        self._results['psi_lcfs_tx'] = {'x': list(self._times), 'y': np.array(self._state['psi_lcfs_tx'])}
        self._results['psi_axis_tx'] = {'x': list(self._times), 'y': np.array(self._state['psi_axis_tx'])}
        self._results['vloop_tm'] = {'x': list(self._times), 'y': np.array(self._state['vloop_tm'])}
        self._results['vloop_tx'] = {'x': list(self._times), 'y': np.array(self._state['vloop_tx'])}
        self._results['beta_N_tm'] = {'x': list(self._times), 'y': np.array(self._state['beta_N_tm'])}
        self._results['l_i_tm'] = {'x': list(self._times), 'y': np.array(self._state['l_i_tm'])}
        self._results['q95_tm'] = {'x': list(self._times), 'y': np.array(self._state['q95_tm'])}
        self._results['q0_tm'] = {'x': list(self._times), 'y': np.array(self._state['q0_tm'])}
        self._results['pax'] = {'x': list(self._times), 'y': np.array(self._state['pax'])}
        self._results['pax_tm'] = {'x': list(self._times), 'y': np.array(self._state['pax_tm'])}


    # ─── TokaMaker (TM) Methods ─────────────────────────────────────────────────

    def _run_tm(self):
        r'''! Run the GS solve across n timesteps using TokaMaker.
        @return Tuple (consumed_flux, consumed_flux_integral).
        '''
        from tqdm import tqdm
        self._print(f'  TokaMaker: solving {len(self._times)} equilibria...')
        self._log(f"Loop {self._current_loop} TokaMaker:")

        self._eqdsk_skip = []
        _loop_level_log = []

        # ── Per-loop initialization (before timestep sweep) ──────────────────
        self._state['psi_grid_prev_tm'] = {}

        # Reset coil regularization to i=0 targets so stale end-of-loop targets
        # from the previous loop don't carry over.
        cfg = getattr(self, '_coil_reg_config', {})
        if cfg:
            if self._prescribed_currents:
                self.set_coil_reg(i=0, **{k: v for k, v in cfg.items() if k != 'targets'})
            else:
                self.set_coil_reg(targets=None, **{k: v for k, v in cfg.items() if k != 'targets'})

        # Warm-start psi at t=0: set psi_dt so eddy-current contribution is negligible.
        if 0 in self._psi_warm_start and self._psi_warm_start[0] is not None:
            self._tm.set_psi_dt(psi0=self._psi_warm_start[0], dt=1.0e10)

        _pbar = tqdm(enumerate(self._times), total=len(self._times),
                    desc=f'  TM loop {self._current_loop}', unit='eq',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]{postfix}')
        for i, t in _pbar:
            # Clear isoflux, flux, and saddle targets from previous timepoint
            self._tm.set_isoflux(None)
            self._tm.set_flux(None,None)
            self._tm.set_saddles(None)



            Ip_target = abs(self._state['Ip'][i])
            P0_target = abs(self._state['pax'][i])
            
            self._tm.set_targets(Ip=Ip_target, pax=P0_target) # using pax target with j_phi inputs 
            self._tm.set_resistivity(eta_prof=self._state['eta_prof'][i])
            

            ffp_prof = {'x': self._state['ffp_prof'][i]['x'].copy(),
                           'y': self._state['ffp_prof'][i]['y'].copy(),
                           'type': self._state['ffp_prof'][i]['type']}
            pp_prof = {'x': self._state['pp_prof'][i]['x'].copy(),
                          'y': self._state['pp_prof'][i]['y'].copy(),
                          'type': self._state['pp_prof'][i]['type']}
            
            # Initialize psi from geometry parameters # TODO this is probably not doing anything, remove (and test)
            # Using the seed EQDSK geometry should give a good initial guess
            self._tm.init_psi(self._state['R0_mag'][i],
                                         self._state['Z'][i],
                                         self._state['a'][i],
                                         self._state['kappa'][i],
                                         self._state['delta'][i])

            # Warm-start: prefer previous loop's converged solution for this
            # timestep; fall back to the previous timestep's solution within
            # the current loop (adjacent-time warm-start).
            if i in self._psi_warm_start and self._psi_warm_start[i] is not None:
                # self._log(f'\tTM: Warm-starting psi at t={t} from previous loop converged solution.')
                self._tm.set_psi(self._psi_warm_start[i])
            elif i > 0 and (i-1) in self._state.get('psi_grid_prev_tm', {}) and self._state['psi_grid_prev_tm'][i-1] is not None:
                # self._log(f'\tTM: Warm-starting psi at t={t} from adjacent timestep (i-1) within current loop.')
                self._tm.set_psi(self._state['psi_grid_prev_tm'][i-1])



            lcfs = self._state['lcfs_geo'][i]

            # Set saddle-point (X-point) constraints during diverted phase
            if self._diverted_times[i] and self._x_point_targets is not None:
                saddle_weights = self._x_point_weight * np.ones(self._x_point_targets.shape[0])
                self._tm.set_saddles(self._x_point_targets, saddle_weights)

                # trims lcfs targets near X-point(s)
                perc_limit = 0.60       # LCFS points above percentage limit* max(abs(Z)) are removed from isoflux targets
                Z_max_abs = np.max(np.abs(lcfs[:, 1]))
                Z_lim = perc_limit * Z_max_abs
                if np.shape(self._x_point_targets)[0] == 1 and self._x_point_targets[0][1] > 0: # upper single null
                    lcfs = lcfs[lcfs[:, 1] <= Z_lim]   
                elif np.shape(self._x_point_targets)[0] == 1 and self._x_point_targets[0][1] < 0: # lower single null
                    lcfs = lcfs[lcfs[:, 1] >= -Z_lim]
                elif np.shape(self._x_point_targets)[0] == 2: # double null
                    lcfs = lcfs[np.abs(lcfs[:, 1]) <= Z_lim]

            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            lcfs_psi_target = self._state['psi_lcfs_tx'][i] # _state in Wb/rad, TM expects Wb/rad (AKA Wb-rad)

            # Shape control: set_isoflux on all LCFS points for lcfs shape targets.
            self._tm.set_isoflux(lcfs, isoflux_weights*10) # shape targets

            # Pick outboard midplane point (largest R at approx Z = Z_axis)
            z_axis = self._state['Z'][i]
            omp_idx = np.argmax(lcfs[:, 0] * np.exp(-0.5 * ((lcfs[:, 1] - z_axis) / (0.3 * self._state['a'][i]))**2))
            omp_point = lcfs[omp_idx:omp_idx+1, :]  # shape (1, 2)
            # Set lcfs psi value target (from torax) only at midplane outboard side of lcfs.
            self._tm.set_flux(omp_point, targets=np.array([lcfs_psi_target]),
                              weights=np.array([LCFS_WEIGHT * 100])) # psi value target

            
            
            self._tm.update_settings() # TODO what does this do?

            
            if i>0:
                if self._state['psi_grid_prev_tm'][i-1] is not None:
                    self._tm.set_psi_dt(psi0=self._state['psi_grid_prev_tm'][i-1], dt=self._times[i]-self._times[i-1])
            
            skip_coil_update = False
            eq_name = os.path.join(self._eqdsk_dir, f'{self._current_loop:03d}.{i:03d}.eqdsk')

            solve_succeeded = False
            level_attempts = []

            ffp_prof_raw = copy.deepcopy(ffp_prof)
            pp_prof_raw  = copy.deepcopy(pp_prof)
            
            # Pre-calculate all level profiles
            level_profiles = []
            
            # Level 0: raw
            ffp_0, pp_0 = self._level0_raw(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw))
            level_profiles.append({'ffp': ffp_0, 'pp': pp_0, 'name': 'lv0: raw'})
            
            # Level 1: sign flip
            ffp_1, pp_1 = self._level1_sign_flip(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw))
            level_profiles.append({'ffp': ffp_1, 'pp': pp_1, 'name': 'lv1: sign_flip'})
            
            # Level 2: pedestal smoothing (takes p_profile as input) # TODO: read in actual n_rho_ped_top, have to add to state first
            ffp_2, pp_2 = self._level2_pedestal_smoothing(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw), copy.deepcopy(self._state['p_prof_tx'][i])) 
            level_profiles.append({'ffp': ffp_2, 'pp': pp_2, 'name': 'lv2: pedestal_smoothing'})
            
            # Level 3: power flux
            ffp_3, pp_3 = self._level3_power_flux(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw))
            level_profiles.append({'ffp': ffp_3, 'pp': pp_3, 'name': 'lv3: power_flux'})

            # Level 4: power flux + pax from initial eqdsk
            ffp_4, pp_4 = self._level3_power_flux(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw))
            level_profiles.append({'ffp': ffp_4, 'pp': pp_4, 'name': 'lv4: power_flux + pax'})

            # Try each level
            for level_idx, level_prof in enumerate(level_profiles):
                level_name = level_prof['name']
                ffp_level = level_prof['ffp']
                pp_level = level_prof['pp']

                try:
                    self._tm.set_profiles(ffp_prof=ffp_level, pp_prof=pp_level,
                                          ffp_NI_prof=self._state['ffp_ni_prof'][i])
                    with self._quiet_tm():
                        self._tm.solve()
                    self._log(f'\tTM: Solve succeeded at t={t} (level {level_idx}: {level_name}).')

                    level_attempts.append({'level': level_idx, 'name': level_name,
                                          'ffp': ffp_level, 'pp': pp_level,
                                          'succeeded': True, 'error': None})
                    ffp_prof, pp_prof = ffp_level, pp_level
                    solve_succeeded = True
                    break
                except Exception as e:
                    self._log(f'\tTM: level {level_idx} solve failed: {e}')
                    level_attempts.append({'level': level_idx, 'name': level_name,
                                          'ffp': ffp_level, 'pp': pp_level,
                                          'succeeded': False, 'error': str(e)})

            if not solve_succeeded:
                self._eqdsk_skip.append(eq_name)
                skip_coil_update = True
                self._log(f'\tTM: Solve failed at t={t} (all levels attempted).')
                self._state['psi_grid_prev_tm'][i] = None  # if solve failed, set psi grid to None
            
            if solve_succeeded:
                with self._quiet_tm():
                    self._tm.save_eqdsk(eq_name,
                        lcfs_pad=0.001, run_info='TokaMaker EQDSK',
                        cocos=2, nr=300, nz=300, truncate_eq=False)
                self._tm_update(i)

                # Store diverted/limited flag for this timestep
                if not hasattr(self, '_diverted_flags'):
                    self._diverted_flags = {}
                self._diverted_flags[i] = self._tm.diverted

                # Store psi on nodes for later movie generation
                self._tm_psi_on_nodes.setdefault(self._current_loop, {})[i] = self._tm.get_psi(normalized=False)

            _winning = next((a for a in level_attempts if a['succeeded']), None)
            _last_attempt = level_attempts[-1] if level_attempts else {}
            _loop_level_log.append({
                'i': i, 't': t,
                'succeeded': solve_succeeded,
                'level': _winning['level'] if _winning else None,
                'level_name': _winning['name'] if _winning else None,
                'error': _last_attempt.get('error') if not solve_succeeded else None,
            })

            if self._debug_mode:
                from toktox_visualization import tm_diagnostic_plot, profile_plot
                _diag_path = os.path.join(self._out_dir, 'tm_plots',
                                          f'{self._current_loop:03d}.{i:03d}_tm_diag.png')
                try:
                    tm_diagnostic_plot(self, i, t, level_attempts, solve_succeeded,
                                       save_path=_diag_path, display=False)
                except Exception as _e:
                    self._log(f'tm_diagnostic_plot failed at i={i}: {_e}')
                if solve_succeeded:
                    _prof_path = os.path.join(self._out_dir, 'plots',
                                              f'{self._current_loop:03d}.{i:03d}_profile.png')
                    try:
                        profile_plot(self, i, t, save_path=_prof_path, display=False)
                    except Exception as _e:
                        self._log(f'profile_plot failed at i={i}: {_e}')

            # Update progress bar postfix; print FAIL messages above the bar
            if solve_succeeded:
                lvl = _winning['level']
                _pbar.set_postfix_str(f't={t:.2f}s OK(L{lvl})', refresh=False)
            else:
                err_short = (_last_attempt.get('error') or 'unknown')[:60]
                tqdm.write(f'    WARNING: TM FAIL at t={t:.2f}s — {err_short}')
                self._log(f'    TM FAIL at t={t:.2f}s — {err_short}')
                _pbar.set_postfix_str(f't={t:.2f}s FAIL', refresh=False)

            if self._prescribed_currents:
                if i < len(self._times):
                    cfg = getattr(self, '_coil_reg_config', {})
                    self.set_coil_reg(i=i+1, **{k: v for k, v in cfg.items() if k != 'targets'})
            elif not skip_coil_update:
                coil_targets, _ = self._tm.get_coil_currents()
                cfg = getattr(self, '_coil_reg_config', {})
                self.set_coil_reg(targets=coil_targets, **{k: v for k, v in cfg.items() if k != 'targets'})

        consumed_flux = (self._state['psi_lcfs_tm'][-1] - self._state['psi_lcfs_tm'][0]) * 2.0 * np.pi
        consumed_flux_integral = np.trapezoid(self._state['vloop_tm'][0:], self._times[0:])

        n_ok = sum(1 for e in _loop_level_log if e['succeeded'])
        self._print(f'  TokaMaker: {n_ok}/{len(self._times)} solved (cflux={consumed_flux:.4f} Wb)')

        if self._debug_mode:
            from toktox_visualization import tm_loop_summary_plot
            _summary_path = os.path.join(self._out_dir, 'tm_plots',
                                         f'{self._current_loop:03d}_tm_summary.png')
            try:
                tm_loop_summary_plot(self, _loop_level_log, save_path=_summary_path, display=False)
            except Exception as _e:
                self._log(f'tm_loop_summary_plot failed: {_e}')

        return consumed_flux, consumed_flux_integral
        
    # ── Profile level functions ──────────────────────────────────────────
    # Each level takes (self, ffp_prof, pp_prof, i) and returns (ffp_prof, pp_prof).
    # All levels receive deep copies of the raw TORAX profiles (not cumulative).
    # Level 0 is always identity. Add new levels by appending to self._profile_levels.

    def _level0_raw(self, ffp_prof, pp_prof):
        r'''! Raw TORAX profiles passed through unchanged.'''
        return ffp_prof, pp_prof

    def _level1_sign_flip(self, ffp_prof, pp_prof):
        r'''! Sign-flip clipping: clip each profile to its dominant sign.'''
        def _clip(prof):
            y = prof['y']
            sign = 1 if np.sum(y > 0) >= np.sum(y < 0) else -1
            y_new = np.clip(y, 0, None) if sign > 0 else np.clip(y, None, 0)
            return {**prof, 'y': y_new}
        return _clip(ffp_prof), _clip(pp_prof)

    def _level2_pedestal_smoothing(self, ffp_prof, pp_prof, p_prof, transition_psi_N = 0.6, gauss_sigma=8, blend_width=0.02, sav_window=41, sav_order=3):
        r'''! Edge smoothing with Gaussian filter: smooth p profile and take derivative for pp_prof.'''
        
        # Extract pressure 'y' values and ensure they're 1D
        p = np.atleast_1d(p_prof['y'])
        
        # Handle case where input is empty or scalar
        if p.size == 0:
            return ffp_prof, pp_prof
        
        # First smooth entire profile
        p_smooth = gaussian_filter1d(p, gauss_sigma, mode='nearest')

        # Sigmoid blend weight: 0 = pure original, 1 = pure smoothed
        # Centered at edge_psi, width controlled by blend_width
        blend = 0.5 * (1 + np.tanh((self._psi_N - transition_psi_N) / blend_width))

        # blend original and smoothed profiles so the value and slope are continuous across transition
        p_new = (1 - blend) * p + blend * p_smooth

        pp_new = np.gradient(p_new, self._psi_N)
        pp_new_smooth = savgol_filter(pp_new, sav_window, sav_order)

        # Return modified pp_prof with smoothed values, ffp_prof unchanged
        return ffp_prof, {**pp_prof, 'y': pp_new_smooth}

    def _level3_power_flux(self, ffp_prof, pp_prof):
        r'''! Generic power-flux shape, sign matched to raw profile means.'''
        # ffp_sign = float(np.sign(np.nanmean(ffp_prof['y']))) or 1.0
        # pp_sign  = float(np.sign(np.nanmean(pp_prof['y'])))  or 1.0
        ffp_out = create_power_flux_fun(N_PSI, 1.5, 2.0)
        pp_out  = create_power_flux_fun(N_PSI, 4.0, 1.0)
        ffp_out = {**ffp_out, 'y': ffp_out['y']}
        pp_out  = {**pp_out,  'y': pp_out['y']}
        return ffp_out, pp_out

    def _tm_update(self, i):
        r'''! Update internal state and coil current results based on results of GS solver.
        @param i Timestep of the solve.
        '''
        eq_stats = self._tm.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']
        self._state['Ip_tm'][i] = eq_stats['Ip']
        self._state['pax_tm'][i] = eq_stats['P_ax']
        self._state['beta_N_tm'][i] = eq_stats['beta_n']
        self._state['l_i_tm'][i] = eq_stats['l_i']
        
        eq_read_extended = read_eqdsk_extended(os.path.join(self._eqdsk_dir, f'{self._current_loop:03d}.{i:03d}.eqdsk'))
        vol_tm = np.interp(self._psi_N, eq_read_extended['psi_n'], eq_read_extended['vol'])
        self._state['vol_tm'][i] = {'x': self._psi_N.copy(), 'y': vol_tm, 'type': 'linterp'}
        self._state['psi_lcfs_tm'][i] = self._tm.psi_bounds[0] # TM outputs in Wb/rad (AKA Wb-rad) which is how psi_lcfs is stored
        self._state['psi_axis_tm'][i] = self._tm.psi_bounds[1] 
        self._state['psi_tm'][i] = {'x': self._psi_N.copy(), 'y': self._state['psi_axis_tm'][i] + (self._state['psi_lcfs_tm'][i] - self._state['psi_axis_tm'][i]) * self._psi_N, 'type': 'linterp'}

        try:
            # self._log(f'Ip_ni = {self._state["Ip_ni_tx"][i]:.3f} A, vloop = {self._state["vloop_tx"][i]:.3f} V')
            self._state['vloop_tm'][i] = self._tm.calc_loopvoltage()
        except ValueError:
            self._log(f'WARNING: calc_loopvoltage failed at t-idx {i} '
                      f'(likely Ip_ni > Ip); using TORAX vloop as fallback.')
            self._state['vloop_tm'][i] = float(self._state['vloop_tx'][i])
        
        # store TokaMaker pressure profile from get_profiles()
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._tm.get_profiles(npsi=N_PSI)

        self._state['ffp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_fp_prof*tm_f_prof), 'type': 'linterp'}
        self._state['pp_prof_tm'][i] =  {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_pp_prof), 'type': 'linterp'}
        self._state['p_prof_tm'][i] =   {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_p_prof), 'type': 'linterp'}
        self._state['f_prof_tm'][i] =   {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_f_prof), 'type': 'linterp'}

        # pull geo profiles
        psi_geo, q_tm, geo, _, _, _ = self._tm.get_q(npsi=N_PSI, psi_pad=0.02)
        
        self._state['q0_tm'][i] = q_tm[0] if len(q_tm) > 0 else np.nan
        self._state['q95_tm'][i] = np.interp(0.95, psi_geo, q_tm) if len(psi_geo) > 0 and len(q_tm) > 0 else np.nan
        self._state['q_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, q_tm), 'type': 'linterp'}

        self._state['R_avg_tm'][i] =     {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[0])), 'type': 'linterp'}
        self._state['R_inv_avg_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[1])), 'type': 'linterp'}
        
        # Update Results
        coils, _ = self._tm.get_coil_currents()
        if 'COIL' not in self._results:
            self._results['COIL'] = {coil: {} for coil in coils}
        for coil, current in coils.items():
            if coil not in self._results['COIL']:
                self._results['COIL'][coil] = {}
            self._results['COIL'][coil][self._times[i]] = current * 1.0 # TODO: handle nturns > 1

        # get psi to use in next timestep
        self._state['psi_grid_prev_tm'][i] = self._tm.get_psi(normalized=False)
        self._psi_warm_start[i] = self._tm.get_psi(normalized=False)  # persist across steps


        # TODO: pull LCFS geometry from gs solve (trace_surf often silently fails)


    # ─── I/O & Logging ──────────────────────────────────────────────────────────

    def save_state(self, fname):
        r'''! Save intermediate simulation state to JSON.
        @param fname Filename to save to.
        '''
        with open(fname, 'w') as f:
            json.dump(self._state, f, cls=MyEncoder)

    def save_res(self):
        r'''! Save simulation results to JSON.'''
        if self._fname_out is not None:
            with open(self._fname_out, 'w') as f:
                json.dump(self._results, f, cls=MyEncoder)

    def _log(self, msg):
        r'''! Write message to log file only.'''
        if hasattr(self, '_log_file') and self._log_file is not None:
            with open(self._log_file, 'a') as f:
                print(msg, file=f)

    def _print(self, msg):
        r'''! Write message to both stdout and log file.'''
        print(msg)
        self._log(msg)

    def _quiet_tm(self):
        r'''! Context manager: redirect C/Fortran-level stdout+stderr.

        In debug mode, redirects to the log file so nothing is lost.
        Otherwise, redirects to /dev/null to suppress noise.
        '''
        import contextlib, os, sys
        @contextlib.contextmanager
        def _cm():
            if getattr(self, '_debug_mode', False) and getattr(self, '_log_file', None):
                target_fd = os.open(self._log_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
            else:
                target_fd = os.open(os.devnull, os.O_WRONLY)
            saved_out = os.dup(1)
            saved_err = os.dup(2)
            os.dup2(target_fd, 1)
            os.dup2(target_fd, 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
                os.close(target_fd)
        return _cm()

    def configure_redirect_to_log(self):
        r'''! Step 3/3 of setup to divert noisy outputs to log file.
        In debug mode, captures DEBUG and above. Otherwise captures INFO and above.
        '''
        if self._logging_configured or not self._log_file:
            return

        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self._log_file, mode='a')

        if getattr(self, '_debug_mode', False):
            root_logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s [%(name)-12s:%(levelname)-8s] %(message)s')
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        self._logging_configured = True
        logging.info(f"File logging configured. All logs will be written to {self._log_file}")

    # =========================================================================
    #  fly — main simulation loop
    # =========================================================================


    # ─── Main Simulation Loop ───────────────────────────────────────────────────

    def fly(self, convergence_threshold=-1.0, max_loop=3, run_name='tmp', diverted_times=None, x_point_targets=None,
            save_outputs=False, debug=False, x_point_weight=100.0, skip_bad_init_eqdsks=False):
        r'''! Run TokaMaker-TORAX coupled simulation loop.

        @param convergence_threshold Max fractional change in consumed flux between loops for convergence.
        @param max_loop Maximum number of coupling iterations.
        @param run_name Name tag for this run (used in output directory and log file).
        @param save_outputs If True, create persistent output directory with eqdsks, configs, and results JSON.
        @param debug If True, redirect all outputs (including TM/TX noise) to the log file,
               save intermediate states, and save diagnostic plots into the output directory.
        @param x_point_targets X-point target locations, shape (n_xpoints, 2) with [R, Z] pairs.
        @param x_point_weight Weight for saddle-point constraints (default 100).
        @param diverted_times Tuple (t_start, t_end) defining the diverted plasma window.
        @param skip_bad_init_eqdsks If True, skip broken initial gEQDSK files instead of raising.
        '''
        import tempfile  # local import: only needed when fly() is called

        # Disable JAX's persistent XLA compilation cache before any TORAX/JAX JIT
        # compilation occurs.  Since JAX 0.4.x the cache stores serialized XLA
        # executables keyed by a hash that does NOT include the XLA/JAX version.
        # After a JAX upgrade the old entries are loaded anyway (triggering the
        # "Assume version compatibility. PjRt-IFRT does not track XLA executable
        # versions" warnings) and can produce silently wrong numerical results or
        # semaphore leaks.  Disabling the persistent cache here means JAX still
        # JIT-compiles in memory for the duration of the session (fast after the
        # first call) but never reads or writes stale on-disk entries.
        try:
            import jax
            jax.config.update('jax_enable_compilation_cache', False)
        except Exception:
            pass  # non-fatal: older JAX versions may not have this config key

        self._save_outputs = save_outputs
        self._debug_mode = debug
        self._diagnostics = debug
        self._skip_bad_init_eqdsks = skip_bad_init_eqdsks
        self._run_name = run_name

        dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        _sim_start_time = time.time()

        # ── Log file: same directory as toktox_outputs (i.e. cwd / './') ──
        if run_name == 'tmp':
            self._log_file = os.path.abspath('toktox_log_tmp.log')
        else:
            self._log_file = os.path.abspath(f'toktox_log_{run_name}_{dt_str}.log')
        with open(self._log_file, 'w'):
            pass
        print(f'  Log file: {self._log_file}', flush=True)
        self._log(f'Log file: {self._log_file}')

        # In debug mode, attach file handler to Python logging so library
        # messages (TORAX, JAX, etc.) are captured in the log file.
        if debug:
            self._logging_configured = False
            self.configure_redirect_to_log()

        # ── Output directory ──
        # debug=True forces output directory creation even if save_outputs=False
        _needs_out_dir = save_outputs or debug
        if _needs_out_dir:
            if run_name == 'tmp':
                self._out_dir = os.path.join('./toktox_outputs', 'tmp')
                if os.path.exists(self._out_dir):
                    shutil.rmtree(self._out_dir)
            else:
                dir_name = f'{run_name}_{dt_str}'
                self._out_dir = os.path.join('./toktox_outputs', dir_name)
            os.makedirs(os.path.join(self._out_dir, 'results'), exist_ok=True)
            if debug:
                os.makedirs(os.path.join(self._out_dir, 'plots'), exist_ok=True)
                os.makedirs(os.path.join(self._out_dir, 'tm_plots'), exist_ok=True)
                os.makedirs(os.path.join(self._out_dir, 'equil'), exist_ok=True)
            self._fname_out = os.path.join(self._out_dir, 'results', 'results.json')
        else:
            # Lightweight mode: temp directory for transient eqdsks, no persistent output
            self._out_dir = tempfile.mkdtemp(prefix='toktox_')
            os.makedirs(os.path.join(self._out_dir, 'results'), exist_ok=True)
            self._fname_out = None

        # ── EQDSK directory: persistent only when save_outputs, temp otherwise ──
        if save_outputs:
            self._eqdsk_dir = os.path.join(self._out_dir, 'equil')
            os.makedirs(self._eqdsk_dir, exist_ok=True)
            self._eqdsk_dir_is_temp = False
        else:
            self._eqdsk_dir = tempfile.mkdtemp(prefix='toktox_equil_')
            self._eqdsk_dir_is_temp = True

        # ── Diverted / saddle-point configuration ──
        if diverted_times is not None and x_point_targets is not None:
            x_point_targets = np.atleast_2d(x_point_targets)
            t_div_start, t_div_end = diverted_times
            self._diverted_times = np.array([(t >= t_div_start and t <= t_div_end) for t in self._times])
            self._x_point_targets = x_point_targets
            self._x_point_weight  = x_point_weight
            self._log(f'Diverted window: t=[{t_div_start}, {t_div_end}] s '
                      f'({int(self._diverted_times.sum())}/{len(self._times)} timesteps)')
        else:
            self._diverted_times  = np.zeros(len(self._times), dtype=bool)
            self._x_point_targets = None
            self._x_point_weight  = x_point_weight

        # ── Flattop detection ──
        Ip_arr = np.array(self._state['Ip'])
        Ip_max = np.max(Ip_arr)
        flattop_threshold = 0.95 * Ip_max
        above = Ip_arr >= flattop_threshold
        if np.any(above):
            ft_start = self._times[np.argmax(above)]
            ft_end   = self._times[len(above) - 1 - np.argmax(above[::-1])]
            self._flattop = np.array([(t >= ft_start and t <= ft_end) for t in self._times])
        else:
            self._flattop = np.zeros(len(self._times), dtype=bool)

        # ── Header ──
        self._print(f'\n{"="*60}\n TokaMaker + TORAX (TokTox) \n run_name = {run_name} | t=[{self._t_init:.1f}, {self._t_final:.1f}] s '
                      f'| {len(self._times)} timepoints | dt={self._dt} s | max_loop={max_loop}')

        err = convergence_threshold + 1.0
        cflux_tx_prev = 0.0
        tm_cflux_psi = []
        tm_cflux_vloop = []
        tx_cflux_psi = []
        tx_cflux_vloop = []

        try:
            # ── Loop 0: Transport initialization ──
            self._print(f'\n{"="*60}\n  Loop 0: Transport Initialization\n{"="*60}')
            self._run_tx_init()

            self._current_loop = 1

            # ── Main coupling loop ──
            while err > convergence_threshold and self._current_loop <= max_loop:
                self._print(f'\n{"="*60}\n  Loop {self._current_loop}\n{"="*60}')

                cflux_tx, cflux_tx_vloop = self._run_tx()
                # if debug:
                #     self.save_state(os.path.join(self._out_dir, 'results', f'ts_state{self._current_loop}.json'))

                cflux_tm, cflux_tm_vloop = self._run_tm()
                # if debug:
                    # self.save_state(os.path.join(self._out_dir, 'results', f'tm_state{self._current_loop}.json'))
                    # self.save_res()

                if debug:
                    from toktox_visualization import _render_equil_frames
                    try:
                        _render_equil_frames(self, self._current_loop, os.path.join(self._out_dir, 'equil'))
                    except Exception as _e:
                        self._log(f'_render_equil_frames failed at loop {self._current_loop}: {_e}')

                tm_cflux_psi.append(cflux_tm)
                tm_cflux_vloop.append(cflux_tm_vloop)
                tx_cflux_psi.append(cflux_tx)
                tx_cflux_vloop.append(cflux_tx_vloop)

                err = np.abs(cflux_tx - cflux_tx_prev) / cflux_tx_prev if cflux_tx_prev != 0 else convergence_threshold + 1.0
                cflux_diff = np.abs(cflux_tx - cflux_tm) / cflux_tm * 100.0 if cflux_tm != 0 else np.inf

                self._print(f'  Loop {self._current_loop} result: conv_err={err*100:.3f}% | '
                              f'TX-TM diff={cflux_diff:.4f}% | '
                              f'cflux_TX={cflux_tx:.4f} Wb | cflux_TM={cflux_tm:.4f} Wb')
                self._log(f'TX Convergence error = {err*100.0:.3f} %')
                self._log(f'Difference Convergence error = {cflux_diff:.4f} %')

                if debug:
                    from toktox_visualization import plot_scalars
                    _scalars_path = os.path.join(self._out_dir, 'plots',
                                                  f'scalars_loop{self._current_loop:03d}.png')
                    try:
                        plot_scalars(self, save_path=_scalars_path, display=False)
                    except Exception as _e:
                        self._log(f'plot_scalars failed at loop {self._current_loop}: {_e}')

                cflux_tx_prev = cflux_tx
                self._current_loop += 1

        finally:
            # ── Cleanup temp directories ──
            if not _needs_out_dir and hasattr(self, '_out_dir') and os.path.exists(self._out_dir):
                try:
                    shutil.rmtree(self._out_dir)
                except OSError:
                    pass
            if getattr(self, '_eqdsk_dir_is_temp', False) and hasattr(self, '_eqdsk_dir') and os.path.exists(self._eqdsk_dir):
                try:
                    shutil.rmtree(self._eqdsk_dir)
                except OSError:
                    pass

        # ── Summary table ──
        _sim_elapsed = time.time() - _sim_start_time
        n_loops = self._current_loop - 1
        converged = err <= convergence_threshold
        self._print(f'\n{"="*60}')
        if converged:
            self._print(f'  CONVERGED in {n_loops} loops (err={err*100:.3f}%)')
        else:
            self._print(f'  Max loops ({max_loop}) reached (err={err*100:.3f}%)')

        # Print convergence history
        self._print(f'\n  {"Loop":<6} {"cflux TX [Wb]":<16} {"cflux TM [Wb]":<16} {"TX-TM diff %":<14}')
        self._print(f'  {"-"*52}')
        for s in range(len(tx_cflux_psi)):
            diff_pct = np.abs(tx_cflux_psi[s] - tm_cflux_psi[s]) / tm_cflux_psi[s] * 100 if tm_cflux_psi[s] != 0 else np.inf
            self._print(f'  {s+1:<6} {tx_cflux_psi[s]:<16.4f} {tm_cflux_psi[s]:<16.4f} {diff_pct:<14.4f}')
        self._print(f'{"="*60}')

        # ── Elapsed time ──
        _mins, _secs = divmod(_sim_elapsed, 60)
        self._print(f'  Total sim time: {int(_mins)}m {_secs:.1f}s')

        if save_outputs or debug:
            self.save_res()
            self._print(f'  Outputs saved to: {self._out_dir}')
        self._print(f'  Log file: {self._log_file}')

    # =========================================================================
    # ─── Results & Visualization ────────────────────────────────────────────────

    @property
    def results(self):
        r'''! Access simulation results dict.'''
        return self._results
    
    @property
    def state(self):
        r'''! Access simulation state dict.'''
        return self._state

    # =========================================================================
    #  Visualization wrapper methods (lazy-import from toktox_visualization)
    # =========================================================================

    def make_movie(self, save_bool=False, save_path=None, **kwargs):
        r'''! Generate pulse movie from stored psi snapshots.
        @param save_path Path to save MP4 file. If None, uses default naming.
        '''
        from toktox_visualization import make_movie
        if save_bool:
            if save_path is None:
                save_path = os.path.join(self._out_dir, f'toktox_pulse_loop{self._current_loop:03d}.mp4')
        else: # if save_bool == True, but save_path has contents, for some reason, set save_path to None so nothing is saved.
            save_path = None
        return make_movie(self,save_path=save_path, **kwargs)

    def plot_scalars(self, save_bool=False, save_path=None, display=True, **kwargs):
        r'''! Plot scalar time traces (Ip, Q, Te, ne, power channels, etc.).
        @param save_path Path to save figure. If None, displays inline.
        @param display Whether to show the plot (for Jupyter).
        '''
        from toktox_visualization import plot_scalars
        if save_bool:
            if save_path is None:
                save_path = os.path.join(self._out_dir, 'plots', f'toktox_pulse_loop{self._current_loop:03d}.mp4')
        else:
            save_path = None
        return plot_scalars(self, save_path=save_path, display=display, **kwargs)

    def plot_profiles(self, **kwargs):
        r'''! Interactive profile viewer (ipywidgets slider in Jupyter, static otherwise).'''
        from toktox_visualization import plot_profiles_interactive
        return plot_profiles_interactive(self, **kwargs)

    def plot_equil(self, **kwargs):
        r'''! Interactive equilibrium viewer (ipywidgets slider in Jupyter, static otherwise).'''
        from toktox_visualization import plot_equil_interactive
        return plot_equil_interactive(self, **kwargs)

    def plot_coils(self, save_bool=False, save_path=None, display=True, **kwargs):
        r'''! Plot coil current traces over the pulse.
        @param save_path Path to save figure. If None, displays inline.
        @param display Whether to show the plot.
        '''
        from toktox_visualization import plot_coils
        if save_bool:
            if save_path is None:
                save_path = os.path.join(self._out_dir, 'plots', f'toktox_pulse_loop{self._current_loop:03d}.mp4')
        else:
            save_path = None
        return plot_coils(self, save_path=save_path, display=display, **kwargs)

    def summary(self, **kwargs):
        r'''! Print/display a physics summary of the simulation.'''
        from toktox_visualization import summary
        return summary(self, **kwargs)

