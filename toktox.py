import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import torax
import copy
import json
import os
import shutil
from datetime import datetime

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

    def __init__(self, t_init, t_final, eqtimes, g_eqdsk_arr, dt=0.1, times=None, last_surface_factor=0.95, n_rho=50, prescribed_currents=False):
        r'''! Initialize the Coupled TokaMaker + TORAX object.
        @param t_init Start time (s).
        @param t_final End time (s).
        @param eqtimes Time points of each gEQDSK file.
        @param g_eqdsk_arr Filenames of each gEQDSK file.
        @param dt Time step (s).
        @param times Time points to sample output at.
        @param last_surface_factor Last surface factor for Torax.
        @param prescribed_currents Use prescribed coil currents or solve inverse problem to calculate currents.
        '''
        self._oftenv = OFT_env(nthreads=6)
        self._gs = TokaMaker(self._oftenv)

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

        self._current_step = 0

        if times is None:
            self._times = eqtimes
        else:
            self._times = sorted(times)
        # TODO organize initialization of _state
        self._state['R'] = np.zeros(len(self._times))
        self._state['Z'] = np.zeros(len(self._times))
        self._state['a'] = np.zeros(len(self._times))
        self._state['kappa'] = np.zeros(len(self._times))
        self._state['delta'] = np.zeros(len(self._times))    
        self._state['deltaU'] = np.zeros(len(self._times))    
        self._state['deltaL'] = np.zeros(len(self._times))    
        self._state['B0'] = np.zeros(len(self._times))
        self._state['V0'] = np.zeros(len(self._times))
        self._state['Ip'] = np.zeros(len(self._times))
        self._state['Ip_tm'] = np.zeros(len(self._times))
        self._state['Ip_tx'] = np.zeros(len(self._times))
        self._state['Ip_NI_tx'] = np.zeros(len(self._times))
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

        self._state['lcfs_geo'] = {}
        self._state['ffp_prof'] = {}
        self._state['pp_prof'] = {}
        self._state['ffp_prof_save'] = {}
        self._state['pp_prof_save'] = {}
        self._state['pres'] = {}
        self._state['fpol'] = {}
        self._state['eta_prof'] = {}
        self._state['T_e'] = {}
        self._state['T_i'] = {}
        self._state['n_e'] = {}
        self._state['n_i'] = {}
        self._state['f_GW'] = np.zeros(len(self._times))
        self._state['f_GW_vol'] = np.zeros(len(self._times))
        self._state['ptot'] = {}
        self._state['ffpni_prof'] = {}
        self._state['ffpni_sub_prof'] = {}

        self._state['ffp_prof_tx'] = {}
        self._state['pp_prof_tx'] = {}
        self._state['ffp_prof_tm'] = {}
        self._state['pp_prof_tm'] = {}
        self._state['p_prof_tm'] = {} 
        self._state['p_prof_tx'] = {}
        self._state['f_prof_tm'] = {}
        
        self._state['test'] = {}

        # self._state['R_avg_tx'] = {}
        self._state['R_inv_avg_tx'] = {}
        self._state['R_sr_inv_avg_tx'] = {}
        self._state['R_avg_tm'] = {}
        self._state['R_inv_avg_tm'] = {}
        
        # Current density profiles from TORAX
        self._state['j_tot'] = {}
        self._state['j_parallel_total'] = {}
        self._state['j_ohmic'] = {}
        self._state['j_ni'] = {}
        self._state['j_bootstrap'] = {}
        self._state['j_ohmic_tx'] = {}
        # self._state['j_ni_tx'] = {}
        self._state['f_NI'] = np.zeros(len(self._times))

        self._state['vol_tm'] = {}
        self._state['vol_tx'] = {}  # volume profile vs psi
        self._state['vol_tx_lcfs'] = np.zeros(len(self._times))  # volume at LCFS (scalar)
        self._state['vpr_tm'] = {}
        self._state['vpr_tx'] = {}

        self._results['lcfs_geo'] = {}
        self._results['dpsi_lcfs_dt'] = {}
        self._results['vloop_tm'] = np.zeros([20, len(self._times)])
        self._results['vloop_torax'] = np.zeros([20, len(self._times)])
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
            Z.append(g['zmid'])
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
                if time > self._eqtimes[i-1] and time <= self._eqtimes[i]:
                    dt = self._eqtimes[i] - self._eqtimes[i-1]
                    alpha = (time - self._eqtimes[i-1]) / dt
                    return (1.0 - alpha) * profs[i-1] + alpha * profs[i]
            return profs[-1]

        for i, t in enumerate(self._times):
            # Default Scalars
            self._state['R'][i] = np.interp(t, self._eqtimes, R)
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
            # self._state['psi'][i] = np.linspace(g['psimag'], g['psibry'], N_PSI)
            self._state['ffpni_prof'][i] = {'x': [], 'y': [], 'type': 'linterp'}

            # Normalize profiles
            self._state['ffp_prof'][i]['y'] /= self._state['ffp_prof'][i]['y'][0]
            self._state['pp_prof'][i]['y'] /= self._state['pp_prof'][i]['y'][0]

            self._state['eta_prof'][i]= {
                'x': self._psi_N.copy(),
                'y': np.zeros(N_PSI),
                'type': 'linterp',
            }
            
            self._state['pres'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(pres_prof, t), 'type': 'linterp'}
            self._state['fpol'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(fpol_prof, t), 'type': 'linterp'}

        # Save seed values from initial equilibria
        self._psi_axis_seed = self._state['psi_axis_tm'].copy()
        self._psi_lcfs_seed = self._state['psi_lcfs_tm'].copy()
        self._Ip_seed       = self._state['Ip'].copy()
        self._pax_seed      = self._state['pax'].copy()

        self._Ip = None
        self._Zeff = None

        self._normalize_to_nbar = False

        self._nbi_heating = None
        self._eccd_heating = None
        self._eccd_loc = None
        self._nbi_loc = None
        self._ohmic_power = None

        self._evolve_density = True
        self._evolve_current = True
        self._evolve_Ti = True
        self._evolve_Te = True

        self._nbar = None
        self._n_e = None
        self._T_i = None
        self._T_e = None

        self._set_pedestal = None
        self._T_i_ped = None
        self._T_e_ped = None
        self._n_e_ped = None
        self._ped_top = 0.95

        self._Te_right_bc = None
        self._Ti_right_bc = None
        self._ne_right_bc = None

        self._ohmic = None

        self._gp_s = None
        self._gp_dl = None

        self._chi_min = 0.05
        self._chi_max = 100.0
        self._De_min = 0.05
        self._De_max = 50.0
        self._Ve_min = -10.0
        self._Ve_max = 10.0

        self._targets = None
        self._baseconfig = None
        self._tx_grid_type = None
        self._tx_grid = None

        self._eqdsk_skip = []
    
    def load_config(self, config):
        r'''! Load a base config for torax.
        
        When a config is loaded, all settings in it override TokTox defaults.
        Geometry is excluded (set dynamically by TokaMaker equilibria).
        
        @param config Dictionary object to be converted to torax config.
        '''
        self._baseconfig = config
        self._extract_config_to_attributes(config)
        
    def _extract_config_to_attributes(self, config):
        r'''! Extract settings from a loaded config and set internal attributes.
        
        This ensures that loaded config values override TokTox defaults for all
        settings except geometry (which is set dynamically by TokaMaker).
        
        @param config TORAX config dictionary
        '''
        # --- Profile conditions ---
        pc = config.get('profile_conditions', {})
        if 'n_e' in pc:
            self._n_e = pc['n_e']
        if 'T_i' in pc:
            self._T_i = pc['T_i']
        if 'T_e' in pc:
            self._T_e = pc['T_e']
        if 'normalize_n_e_to_nbar' in pc:
            self._normalize_to_nbar = pc['normalize_n_e_to_nbar']
        if 'nbar' in pc:
            self._nbar = pc['nbar']
        if 'n_e_right_bc' in pc:
            self._ne_right_bc = pc['n_e_right_bc']
        if 'T_i_right_bc' in pc:
            self._Ti_right_bc = pc['T_i_right_bc']
        if 'T_e_right_bc' in pc:
            self._Te_right_bc = pc['T_e_right_bc']
        
        # --- Numerics (evolve flags) ---
        num = config.get('numerics', {})
        if 'evolve_density' in num:
            self._evolve_density = num['evolve_density']
        if 'evolve_current' in num:
            self._evolve_current = num['evolve_current']
        if 'evolve_ion_heat' in num:
            self._evolve_Ti = num['evolve_ion_heat']
        if 'evolve_electron_heat' in num:
            self._evolve_Te = num['evolve_electron_heat']
        
        # --- Pedestal ---
        ped = config.get('pedestal', {})
        if 'set_pedestal' in ped:
            self._set_pedestal = ped['set_pedestal']
        if 'T_i_ped' in ped:
            self._T_i_ped = ped['T_i_ped']
        if 'T_e_ped' in ped:
            self._T_e_ped = ped['T_e_ped']
        if 'n_e_ped' in ped:
            self._n_e_ped = ped['n_e_ped']
        if 'rho_norm_ped_top' in ped:
            self._ped_top = ped['rho_norm_ped_top']
        
        # --- Sources ---
        src = config.get('sources', {})
        
        # ECRH/ECCD
        ecrh = src.get('ecrh', {})
        if 'P_total' in ecrh:
            p_total = ecrh['P_total']
            # Handle both dict and tuple (dict, mode) formats from MOSAIC
            if isinstance(p_total, tuple) and len(p_total) >= 1:
                p_total = p_total[0]
            if isinstance(p_total, dict):
                self._eccd_heating = p_total
        if 'gaussian_location' in ecrh:
            self._eccd_loc = ecrh['gaussian_location']
        
        # Generic heat (NBI proxy)
        gen_heat = src.get('generic_heat', {})
        if 'P_total' in gen_heat:
            p_total = gen_heat['P_total']
            # Handle both dict and tuple (dict, mode) formats from MOSAIC
            if isinstance(p_total, tuple) and len(p_total) >= 1:
                p_total = p_total[0]
            if isinstance(p_total, dict):
                self._nbi_heating = p_total
        if 'gaussian_location' in gen_heat:
            self._nbi_loc = gen_heat['gaussian_location']
        
        # Ohmic
        ohmic = src.get('ohmic', {})
        if 'prescribed_values' in ohmic:
            pv = ohmic['prescribed_values']
            # Handle both dict and tuple formats
            if isinstance(pv, tuple) and len(pv) >= 1:
                pv = pv[0]
            if isinstance(pv, dict):
                self._ohmic_power = pv
        
        # Gas puff
        gp = src.get('gas_puff', {})
        if 'S_total' in gp:
            self._gp_s = gp['S_total']
        if 'puff_decay_length' in gp:
            self._gp_dl = gp['puff_decay_length']
        
        # --- Transport ---
        trn = config.get('transport', {})
        if 'chi_min' in trn:
            self._chi_min = trn['chi_min']
        if 'chi_max' in trn:
            self._chi_max = trn['chi_max']
        if 'D_e_min' in trn:
            self._De_min = trn['D_e_min']
        if 'D_e_max' in trn:
            self._De_max = trn['D_e_max']
        if 'V_e_min' in trn:
            self._Ve_min = trn['V_e_min']
        if 'V_e_max' in trn:
            self._Ve_max = trn['V_e_max']
        
        # --- Plasma composition ---
        pc_comp = config.get('plasma_composition', {})
        if 'Z_eff' in pc_comp:
            self._Zeff = pc_comp['Z_eff']
        
    def set_tx_grid(self, type, grid):
        r'''! Set TORAX grid type and grid points.
        @param type Grid type ('n_rho' or 'face_centers').
        @param grid Grid points (integer or np.array).
        '''
        self._tx_grid_type = type
        self._tx_grid = grid
        if type not in ['n_rho', 'face_centers']:
            raise ValueError(f'Invalid grid type: {type}. Must be "n_rho" or "face_centers".')



    def initialize_gs(self, mesh, weights=None, vsc=None):
        r'''! Initialize GS Solver Object.
        @param mesh Filename of reactor mesh.
        @param vsc Vertical Stability Coil.
        '''
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh(mesh)
        self._gs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._gs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._gs.setup(order = 2, F0 = self._state['R'][0]*self._state['B0'][0])

        self._gs.settings.maxits = 100
        # self._gs.settings.pm = False

        if vsc is not None:
            self._gs.set_coil_vsc({vsc: 1.0})
        # self.set_coil_reg(targets, weights=weights, weight_mult=0.1)

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

    # def set_pressure(self, p_profiles, n_e_profiles, Ti_Te_ratio=1.0):
    #     r'''! Set initial T_e and T_i profiles from pressure and density profiles.
        
    #     Computes temperature from P = n_e * (T_e + T_i) = n_e * T_e * (1 + Ti_Te_ratio).
    #     Sets both T_e and T_i as time-varying-array dicts for TORAX.
        
    #     @param p_profiles Pressure profiles in Pa. Dict of {time: {rho: P_Pa, ...}, ...}.
    #     @param n_e_profiles Density profiles in m^-3. Dict of {time: {rho: n_e, ...}, ...}.
    #     @param Ti_Te_ratio Ratio of T_i to T_e (default 1.0, i.e. T_i = T_e).
    #     '''
    #     eV_to_J = 1.602e-19  # 1 eV in Joules
    #     keV_to_J = 1.602e-16  # 1 keV in Joules
        
    #     T_e_profiles = {}
    #     T_i_profiles = {}
        
    #     for t in sorted(p_profiles.keys()):
    #         p_dict = p_profiles[t]
    #         n_dict = n_e_profiles[t] if t in n_e_profiles else n_e_profiles[max(k for k in n_e_profiles.keys() if k <= t)]
            
    #         T_e_prof = {}
    #         T_i_prof = {}
            
    #         # Get sorted rho values from pressure profile
    #         rho_vals = sorted(p_dict.keys())
    #         for rho in rho_vals:
    #             P_Pa = p_dict[rho]
    #             # Interpolate n_e at this rho from the density profile
    #             n_rho_vals = sorted(n_dict.keys())
    #             n_vals = [n_dict[r] for r in n_rho_vals]
    #             n_e_at_rho = np.interp(rho, n_rho_vals, n_vals)
                
    #             if n_e_at_rho > 0 and P_Pa > 0:
    #                 # P = n_e * T_e * (1 + Ti_Te_ratio) in Joules
    #                 # T_e [keV] = P / (n_e * (1 + ratio) * keV_to_J)
    #                 T_e_keV = P_Pa / (n_e_at_rho * (1.0 + Ti_Te_ratio) * keV_to_J)
    #                 T_i_keV = T_e_keV * Ti_Te_ratio
    #             else:
    #                 T_e_keV = 0.01  # minimum temperature
    #                 T_i_keV = 0.01
                
    #             T_e_prof[rho] = T_e_keV
    #             T_i_prof[rho] = T_i_keV
            
    #         T_e_profiles[t] = T_e_prof
    #         T_i_profiles[t] = T_i_prof
        
    #     self._T_e = T_e_profiles
    #     self._T_i = T_i_profiles
        
    #     self._print_out(f'set_pressure: Computed T_e and T_i from pressure and density at times {sorted(p_profiles.keys())}')
    #     for t in sorted(p_profiles.keys()):
    #         rho_0 = min(T_e_profiles[t].keys())
    #         rho_1 = max(T_e_profiles[t].keys())
    #         self._print_out(f'  t={t}: T_e(0)={T_e_profiles[t][rho_0]:.2f} keV, T_e(1)={T_e_profiles[t][rho_1]:.4f} keV, '
    #                       f'T_i(0)={T_i_profiles[t][rho_0]:.2f} keV, T_i(1)={T_i_profiles[t][rho_1]:.4f} keV')

    def set_Zeff(self, Zeff):
        r'''! Set plasma effective charge.
        @param z_eff Effective charge.
        '''
        self._Zeff = Zeff
    
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

    def set_heating(self, nbi=None, nbi_loc=None, eccd=None, eccd_loc=None, ohmic=None):
        r'''! Set heating sources for Torax.
        @param nbi NBI heating (dictionary of heating at times).
        @param eccd ECCD heating (dictionary of heating at times).
        @param eccd_loc Location of ECCD heating.
        '''
        if nbi is not None and nbi_loc is not None:
            self._nbi_heating = nbi
            self._nbi_loc = nbi_loc
        if eccd is not None and eccd_loc is not None:
            self._eccd_heating = eccd
            self._eccd_loc = eccd_loc
        if ohmic is not None:
            self._ohmic_power = ohmic

    def set_pedestal(self, set_pedestal=True, T_i_ped=None, T_e_ped=None, n_e_ped=None, ped_top=0.95):
        r'''! Set pedestals for ion and electron temperatures.
        @pararm T_i_ped Ion temperature pedestal (dictionary of temperature at times).
        @pararm T_e_ped Electron temperature pedestal (dictionary of temperature at times).
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
    
    def set_Bp(self, Bp):
        Bp_t = sorted(Bp.keys())
        Bp_list = [Bp[t] for t in Bp_t]
        for i, t in enumerate(self._times):
            self._state['beta_pol'][i] = np.interp(t, Bp_t, Bp_list)

    def set_Vloop(self, vloop):
        for i in range(len(self._times)):
            self._state['vloop_tm'][i] = vloop[i]
    
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

    def set_ohmic(self, times, rho, values):
        self._ohmic = ((times), (rho), (values))
    
    def set_validation_density(self, ne):
        self._validation_ne = ne

    def _pull_torax_onto_psi(self, data_tree, var_name, time, load_into_state='state', normalize=False, profile_type='linterp'): # TODO adjust interpolation to account for last_surface_factor<1
        r'''! Load TORAX variable onto psi_norm grid.
        @param data_tree TORAX output data tree.
        @param var_name Name of variable (e.g., 'T_i', 'j_ohmic', 'FFprime').
        @param time Time value to extract.
        @param load_into_state If 'state' returns dict to load right into '_state' into '_state', elif None, return interpolated data array.
        @param normalize If True, normalize profile: subtract edge value, divide by core value (for FFprime, pprime).
        @param profile_type Type key for returned dict: 'linterp' or 'jphi-linterp'. Default is 'linterp'.
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
        
        # Get rho_tor coordinate for this variable
        rho_tor = var.coords[grid].values
        
        # Get psi_norm on rho_face_norm grid and psi on rho_norm grid
        psi_norm_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
        psi_rho_norm = data_tree.profiles.psi.sel(time=time, method='nearest').to_numpy()
        psi_norm_rho_norm = (psi_rho_norm - psi_rho_norm[0]) / (psi_rho_norm[-1] - psi_rho_norm[0])
        
        # Correct second element to avoid degeneracy from zero-gradient BC at core
        psi_norm_rho_norm[1] = (psi_norm_face[0] + psi_norm_face[1]) / 2.0

        # Convert psi to same grid as variable
        if grid == 'rho_cell_norm':
            psi_on_grid = psi_norm_rho_norm[1:-1]
        elif grid == 'rho_face_norm':
            psi_on_grid = psi_norm_face
        elif grid == 'rho_norm':
            psi_on_grid = psi_norm_rho_norm
          
        # Interpolate onto uniform psi grid
        data_on_psi = interp1d(psi_on_grid, var_data, kind='linear',
                            fill_value=0, bounds_error=False)(self._psi_N)

        
        # Normalize if requested
        if normalize:
            if grid == 'rho_cell_norm':
                # Cell-centered variables don't have a value at psi=0
                # Find the index in data_on_psi closest to the first cell center
                core_idx = np.argmin(np.abs(self._psi_N - psi_on_grid[0]))
                data_on_psi /= data_on_psi[core_idx]
                self._print_out(f"Normalizing {var_name} using value at psi={self._psi_N[core_idx]:.3f} (closest to first cell center at {psi_on_grid[0]:.3f})")
            else:
                # Face or extended grid has actual core value at psi=0
                data_on_psi /= data_on_psi[0]
        
        if load_into_state == 'state':
            return {'x': self._psi_N.copy(), 'y': data_on_psi.copy(), 'type': profile_type}
        else:
            return data_on_psi
        
    def _get_torax_config(self):
        r'''! Generate config object for Torax simulation. Modifies BASE_CONFIG based on current simulation state.
        @return Torax config object.
        '''

        myconfig = copy.deepcopy(BASE_CONFIG)
        if self._baseconfig:
            myconfig = self._baseconfig.copy()
        myconfig['numerics'] = {
            't_initial': self._t_init,
            't_final': self._t_final,  # length of simulation time in seconds
            'fixed_dt': self._dt, # fixed timestep
            'evolve_ion_heat': self._evolve_Ti, # solve ion heat equation
            'evolve_electron_heat': self._evolve_Te, # solve electron heat equation
            'evolve_current': self._evolve_current, # solve current equation
            'evolve_density': self._evolve_density, # solve density equation
        }
        myconfig['geometry'] = {
            'geometry_type': 'eqdsk',
            'geometry_directory': os.getcwd(),
            'last_surface_factor': self._last_surface_factor,
            'n_surfaces': 50,
            'Ip_from_parameters': False, # tells TX to pull Ip from eqdsk
        }
        if self._current_step == 1:
            eq_safe = []
            t_safe = []
            for i, t in enumerate(self._eqtimes):
                eq = self._init_files[i]
                if self._test_eqdsk(eq):
                    self._print_out(f'\tTX: Using eqdsk at t={t}')
                    eq_safe.append(eq)
                    t_safe.append(t)
                else:
                    if not self._skip_bad_init_eqdsks:
                        raise ValueError(f'Bad initial gEQDSK at t={t}: {eq}')
                    self._print_out(f'\tTX: Skipping eqdsk at t={t}')
            myconfig['geometry']['geometry_configs'] = {
                t: {'geometry_file': eq_safe[i], 'cocos': 2} for i, t in enumerate(t_safe)
            }
        else:
            # For times where TM succeeded last step, use the TM-solved EQDSK.
            # For times where TM failed, fall back to the nearest seed EQDSK and
            eqtimes_arr = np.array(self._eqtimes)
            full_eqdsk_map = {}
            n_tm = 0
            for i, t in enumerate(self._times):
                eqdsk = os.path.join(self._out_dir, 'equil', '{:03}.{:03}.eqdsk'.format(self._current_step - 1, i))
                tm_ok = (eqdsk not in self._eqdsk_skip) and self._test_eqdsk(eqdsk)
                if tm_ok:
                    full_eqdsk_map[t] = eqdsk
                    n_tm += 1
                else:
                    # TM failed: use nearest seed EQDSK and reset psi to seed values
                    seed_idx = int(np.argmin(np.abs(eqtimes_arr - t)))
                    full_eqdsk_map[t] = self._init_files[seed_idx]
                    self._state['psi_axis_tm'][i] = self._psi_axis_seed[i] 
                    self._state['psi_lcfs_tm'][i] = self._psi_lcfs_seed[i]
            if n_tm == 0:
                print(f'Warning: Step {self._current_step}: no valid TM EQDSKs from step {self._current_step-1}, using all seed EQDSKs.')
            else:
                print(f'Step {self._current_step}: using {n_tm}/{len(self._times)} TM-solved EQDSKs, {len(self._times)-n_tm} seed fallbacks.')
            myconfig['geometry']['geometry_configs'] = {
                t: {'geometry_file': eqdsk_f, 'cocos': 2} for t, eqdsk_f in full_eqdsk_map.items()
            }

        if self._tx_grid_type == 'n_rho':
            myconfig['geometry']['n_rho'] = self._tx_grid
        elif self._tx_grid_type == 'face_centers':
            myconfig['geometry']['face_centers'] = self._tx_grid


        myconfig['profile_conditions']['psi'] = { # TORAX takes in Wb, psi_lcfs stored as Wb/rad (AKA Wb-rad) so needs *2pi factor
            # TX and TM have different Ip sign conventions, meaning they expect psi profile differently
            # TM expects psi to increase, TX expects it to decrease. They match psi_lcfs and they have the same abs(psi(0) - psi(1)).
            # But to correctly pass psi_axis to TX, we have to reflect it over psi_lcfs: psi_axis_tx = 2*psi_axis_tm - psi_lcfs_tm
            t: {0.0: (2.0 * self._state['psi_lcfs_tm'][i] - self._state['psi_axis_tm'][i]) * 2.0 * np.pi, 1.0: self._state['psi_lcfs_tm'][i]* 2.0 * np.pi} for i, t in enumerate(self._times)}
        if self._n_e:
            myconfig['profile_conditions']['n_e'] = self._n_e
        
        if self._T_e:
            myconfig['profile_conditions']['T_e'] = self._T_e
        
        if self._T_i:
            myconfig['profile_conditions']['T_i'] = self._T_i
        
        if self._Zeff:
            myconfig['plasma_composition']['Z_eff'] = self._Zeff

        if self._eccd_loc:
            myconfig['sources'].setdefault('ecrh', {})
            myconfig['sources']['ecrh']['P_total'] = self._eccd_heating
            myconfig['sources']['ecrh']['gaussian_location'] = self._eccd_loc

        if self._ohmic_power:
            myconfig['sources'].setdefault('ohmic', {})
            myconfig['sources']['ohmic']['mode'] = 'PRESCRIBED'
            myconfig['sources']['ohmic']['prescribed_values'] = self._ohmic_power

        if self._nbi_heating:
            nbi_times, nbi_pow = zip(*self._nbi_heating.items())    
            myconfig['sources'].setdefault('generic_heat', {})
            myconfig['sources']['generic_heat']['P_total'] = (nbi_times, nbi_pow)
            myconfig['sources']['generic_heat']['gaussian_location'] = self._nbi_loc
            myconfig['sources'].setdefault('generic_current', {})
            myconfig['sources']['generic_current']['I_generic'] = (nbi_times, _NBI_W_TO_MA * np.array(nbi_pow))
            myconfig['sources']['generic_current']['gaussian_location'] = self._nbi_loc

        if self._T_i_ped:
            myconfig['pedestal']['T_i_ped'] = self._T_i_ped
        if self._T_e_ped:
            myconfig['pedestal']['T_e_ped'] = self._T_e_ped
        

        if self._n_e_ped:
            myconfig['pedestal']['n_e_ped_is_fGW'] = False
            myconfig['pedestal']['n_e_ped'] = self._n_e_ped
        
        if self._set_pedestal:
            myconfig['pedestal']['set_pedestal'] = self._set_pedestal
        myconfig['pedestal']['rho_norm_ped_top'] = self._ped_top
        
        myconfig['profile_conditions']['normalize_n_e_to_nbar'] = self._normalize_to_nbar # if on, normalizes ne profile to make nbar = input nbar
        if self._nbar:
            myconfig['profile_conditions']['nbar'] = self._nbar

        if self._ne_right_bc:
            myconfig['profile_conditions']['n_e_right_bc_is_fGW'] = False
            myconfig['profile_conditions']['n_e_right_bc'] = self._ne_right_bc

        if self._Te_right_bc:
            myconfig['profile_conditions']['T_e_right_bc'] = self._Te_right_bc
        if self._Ti_right_bc:
            myconfig['profile_conditions']['T_i_right_bc'] = self._Ti_right_bc
                
        if self._gp_s and self._gp_dl:
            myconfig['sources']['gas_puff'] = {
                'S_total': self._gp_s,
                'puff_decay_length': self._gp_dl,
            }

        myconfig['transport']['chi_min'] = self._chi_min
        myconfig['transport']['chi_max'] = self._chi_max
 
        myconfig['transport']['D_e_min'] = self._De_min
        myconfig['transport']['D_e_max'] = self._De_max

        myconfig['transport']['V_e_min'] = self._Ve_min
        myconfig['transport']['V_e_max'] = self._Ve_max

        # with open('torax_config.json', 'w') as json_file:
        #     json.dump(myconfig, json_file, indent=4, cls=MyEncoder)
        torax_config = torax.ToraxConfig.from_dict(myconfig)
        return torax_config

    def _test_eqdsk(self, eqdsk):
            myconfig = copy.deepcopy(BASE_CONFIG)
            myconfig['geometry'] = {
                'geometry_type': 'eqdsk',
                'geometry_directory': os.getcwd(),
                'last_surface_factor': self._last_surface_factor,
                'Ip_from_parameters': False,
                'geometry_file': eqdsk,
                'cocos': 2,
            }
            try:
                _ = torax.ToraxConfig.from_dict(myconfig)
                return True
            except Exception as e:
                # self._print_out(e)
                return False

    def _run_transport(self, graph=False):
        r'''! Run the Torax simulation.
        @param graph Whether to display profiles at each iteration (for testing).
        @return Consumed flux.
        '''
        myconfig = self._get_torax_config()
        data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)

        # save data_tree object
        # data_tree_name = 'tmp/test.nc'
        # data_tree.to_netcdf(data_tree_name)

        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        v_loops = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._transport_update(i, data_tree)
            v_loops[i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
        # self._print_out(f'Step {self._current_step}: TX output (w/ /2pi): psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._res_update(data_tree)

        consumed_flux = 2.0 * np.pi * (self._state['psi_lcfs_tx'][-1] - self._state['psi_lcfs_tx'][0]) # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need *2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(v_loops[0:], self._times[0:]) 
        self._print_out(f"Step {self._current_step} TORAX:")
        # self._print_out(f"\tTX: vloop: min={v_loops.min():.3f}, max={v_loops.max():.3f}, mean={v_loops.mean():.3f} V")
        # self._print_out(f"\tTX: psi_lcfs: start={self._state['psi_lcfs'][0]:.3f}, end={self._state['psi_lcfs'][-1]:.3f} Wb/rad")
        # self._print_out(f'\tTX: psi_bound consumed flux={consumed_flux:.3f} Wb')
        # self._print_out(f'\tTX: int v_loop consumed flux w/o t=0 ={consumed_flux_integral:.3f} Wb')
        return consumed_flux, consumed_flux_integral

    def _transport_update(self, i, data_tree, smooth=True):
        r'''! Update the simulation state and simulation results based on results of the Torax simulation.
        @param i Timestep of the solve.
        @param data_tree Result object from Torax.
        @smooth Whether to smooth profiles generated by Torax.
        '''
        t = self._times[i]

        # temp 2026-02-03
        self._state['test'][i] = self._pull_torax_onto_psi(data_tree, 'FFprime', t, load_into_state='state', normalize=True)


        self._state['Ip'][i] =          data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['Ip_tx'][i] =       data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['Ip_NI_tx'][i] =    data_tree.scalars.I_non_inductive.sel(time=t, method='nearest')
        self._state['f_NI'][i] =        data_tree.scalars.f_non_inductive.sel(time=t, method='nearest')
        
        pax_new = data_tree.profiles.pressure_thermal_total.sel(time=t, rho_norm=0.0, method='nearest').values
        pax_old = self._state['pax'][i]
        self._state['pax'][i] = pax_new
        
        self._state['beta_pol'][i] = float(data_tree.scalars.beta_pol.sel(time=t, method='nearest'))
        self._state['beta_N_tx'][i]  = float(data_tree.scalars.beta_N.sel(time=t,  method='nearest'))
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')
        self._state['q0'][i] = data_tree.profiles.q.sel(time=t, rho_face_norm=0.0, method='nearest')

        # deep copy to prevent mutation when normalizing ffp_prof/pp_prof later
        # these are normalized already from the previous step
        self._state['ffp_prof_save'][i] = {
            'x': self._state['ffp_prof'][i]['x'].copy(),
            'y': self._state['ffp_prof'][i]['y'].copy(),
            'type': self._state['ffp_prof'][i]['type'],
        }
        self._state['pp_prof_save'][i] = {
            'x': self._state['pp_prof'][i]['x'].copy(),
            'y': self._state['pp_prof'][i]['y'].copy(),
            'type': self._state['pp_prof'][i]['type'],
        }


        self._state['ffp_prof'][i] = self._pull_torax_onto_psi(data_tree, 'FFprime', t, load_into_state='state', normalize=True)
        self._state['pp_prof'][i] =  self._pull_torax_onto_psi(data_tree, 'pprime',  t, load_into_state='state', normalize=True)
        
        self._state['ffp_prof_tx'][i] = self._pull_torax_onto_psi(data_tree, 'FFprime', t, load_into_state='state', normalize=False) # temp for calculating j_phi
        self._state['ffp_prof_tx'][i]['y'] *= -2.0*np.pi  # convert from TX units to TM units

        self._state['pp_prof_tx'][i] =  self._pull_torax_onto_psi(data_tree, 'pprime', t, load_into_state='state', normalize=False)
        self._state['pp_prof_tx'][i]['y'] *= -2.0*np.pi  # convert from TX units to TM units

        self._state['p_prof_tx'][i] = self._pull_torax_onto_psi(data_tree, 'pressure_thermal_total', t, load_into_state='state', normalize=False)

        self._state['vloop_tx'][i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
        
        self._state['q_prof_tx'][i] = self._pull_torax_onto_psi(data_tree, 'q', t, load_into_state='state', normalize=False)

        self._state['j_tot'][i] =            self._pull_torax_onto_psi(data_tree, 'j_total',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ohmic'][i] =          self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ni'][i] =          self._pull_torax_onto_psi(data_tree, 'j_non_inductive',  t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_bootstrap'][i] =      self._pull_torax_onto_psi(data_tree, 'j_bootstrap',      t, load_into_state='state', profile_type='jphi-linterp')
        

        self._state['R_inv_avg_tx'][i] = self._pull_torax_onto_psi(data_tree, 'gm9', t, load_into_state='state', normalize=False)

        ffp_ni = self._calc_ffp_ni(i, data_tree)

        self._state['ffpni_prof'][i] = {'x': self._psi_N.copy(), 'y': ffp_ni.copy(), 'type': 'linterp'}         

        self._state['T_i'][i] = self._pull_torax_onto_psi(data_tree, 'T_i', t, load_into_state='state', normalize=False)
        self._state['T_e'][i] = self._pull_torax_onto_psi(data_tree, 'T_e', t, load_into_state='state', normalize=False)
        self._state['n_i'][i] = self._pull_torax_onto_psi(data_tree, 'n_i', t, load_into_state='state', normalize=False)
        self._state['n_e'][i] = self._pull_torax_onto_psi(data_tree, 'n_e', t, load_into_state='state', normalize=False)
        # ne_bar = data_tree.scalars.n_e_line_avg.sel(time=t, method='nearest').item()
        # n_GW = self._state['Ip'][i] / (np.pi * self._state['a'][i]**2)  # GW density based on line-averaged density
        # self._state['f_GW'][i] = ne_bar / n_GW if n_GW > 0 else 0.0
        self._state['f_GW'][i] = data_tree.scalars.fgw_n_e_line_avg.sel(time=t, method='nearest').item()
        self._state['f_GW_vol'][i] = data_tree.scalars.fgw_n_e_volume_avg.sel(time=t, method='nearest').item()

        self._state['ptot'][i] = self._pull_torax_onto_psi(data_tree, 'pressure_thermal_total', t, load_into_state='state', normalize=False)

        # Get conductivity and convert to resistivity (eta = 1/sigma)
        conductivity = self._pull_torax_onto_psi(data_tree, 'sigma_parallel', t, load_into_state=None, normalize=False)
        self._state['eta_prof'][i] = {
            'x': self._psi_N.copy(),
            'y': 1.0 / conductivity,
            'type': 'linterp',
        }

        psi_tx = self._pull_torax_onto_psi(data_tree, 'psi', t, load_into_state=None, normalize=False) / (2.0 * np.pi) # TORAX outputs psi in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        psi_tx = 2.0 * psi_tx[-1] - psi_tx  # reflect over psi_lcfs to convert from TX to TM convention
        self._state['psi_tx'][i] = {'x': self._psi_N.copy(), 'y': psi_tx.copy(), 'type': 'linterp',}
        self._state['psi_lcfs_tx'][i] = self._state['psi_tx'][i]['y'][-1]  # update psi_lcfs based on reflected psi profile
        self._state['psi_axis_tx'][i] = self._state['psi_tx'][i]['y'][0]  # update psi_axis based on reflected psi profile


        # self._state['psi_lcfs'][i] = data_tree.profiles.psi.sel(time=t, rho_norm=1.0, method='nearest').item() / (2.0 * np.pi) # TORAX outputs psi_lcfs in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        # self._state['psi_axis'][i] = data_tree.profiles.psi.sel(time=t, rho_norm=0.0, method='nearest').item() / (2.0 * np.pi) # TORAX outputs psi_lcfs in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        # self._state['psi_tx'][i]   = self._pull_torax_onto_psi(data_tree, 'psi', t, load_into_state='state', normalize=False)
        # self._state['psi_tx'][i]['y'] /= (2.0 * np.pi) # TORAX outputs psi in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        # 2.0 * self._state['psi_lcfs'][i] - self._state['psi_axis'][i]) * 2.0 * np.pi, 1.0: self._state['psi_lcfs'][i]* 2.0 * np.pi
        
        # Pull volume and volume derivative from TORAX
        self._state['vol_tx_lcfs'][i] = data_tree.profiles.volume.sel(time=t, rho_norm=1.0, method='nearest').item()
        self._state['vol_tx'][i] = self._pull_torax_onto_psi(data_tree, 'volume', t, load_into_state='state', normalize=False)
        self._state['vpr_tx'][i] = self._pull_torax_onto_psi(data_tree, 'vpr', t, load_into_state='state', normalize=False)


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

        j_tot = self._state['j_tot'][i]['y']
        j_ohmic = self._state['j_ohmic'][i]['y']
        j_ni_sub = j_tot - j_ohmic
        ffpni_sub = mu_0 * j_ni_sub / R_inv_avg

        self._state['ffpni_sub_prof'][i] = {
            'x': self._psi_N.copy(),
            'y': ffpni_sub,
            'type': 'linterp',
        }

        j_ni = self._state['j_ni'][i]['y']  
        
        ffp_ni = mu_0 * j_ni / R_inv_avg

        return ffp_ni

    def set_coil_reg(self, targets=None, i=0, updownsym=False, weights=None, strict_limit=50.0E6, disable_virtual_vsc=True, weight_mult=1.0):
        r'''! Set coil regularization terms.
        @param targets Target values for each coil.
        @param weights Default weight for each coil.
        @param strict_limit Strict limit for coil currents.
        @param disable_virtual_vsc Disable VSC virtual coil. 
        @param weight_mult Factor by which to multiply target weights (reduce to allow for more flexibility).
        '''
        # Set regularization weights
        # coil_bounds = {key: [-strict_limit, strict_limit] for key in self._gs.coil_sets}
        # self._gs.set_coil_bounds(coil_bounds)

        coil_bounds = {key: [-strict_limit, strict_limit] for key in self._gs.coil_sets}
        # for key in [x for x in self._gs.coil_sets if 'DIV' in x]:   
        #     coil_bounds[key] = [0, 0] # turn off div coils, for now
        self._gs.set_coil_bounds(coil_bounds)

        if self._prescribed_currents and targets:
            self._targets = targets
        
        regularization_terms = []
        if self._prescribed_currents:
            for name, currents in self._targets.items():
                if name == 'time':
                    continue
                t_current = np.interp(self._times[i], self._targets['time'], currents)
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=t_current,weight=1.0E-3))
        else:
            for name, target_current in targets.items():
                if name == 'time':
                    continue
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=target_current,weight=1.0E-3))

        # Pass regularization terms to TokaMaker
        self._gs.set_coil_reg(reg_terms=regularization_terms)

    def _run_gs(self, graph=False):
        r'''! Run the GS solve across n timesteps using TokaMaker.
        @param graph Whether to display psi graphs at each iteration (for testing).
        @return Consumed flux.
        '''
        self._print_out(f"Step {self._current_step} TokaMaker:")

        self._eqdsk_skip = []
        _step_level_log = []
        for i, t in enumerate(self._times):
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)

            Ip_target = abs(self._state['Ip'][i])
            P0_target = abs(self._state['pax'][i])
            
            self._gs.set_targets(Ip=Ip_target, pax=P0_target) # using pax target with j_phi inputs 
            self._gs.set_resistivity(eta_prof=self._state['eta_prof'][i])
            

            ffp_prof = {'x': self._state['ffp_prof'][i]['x'].copy(),
                           'y': self._state['ffp_prof'][i]['y'].copy(),
                           'type': self._state['ffp_prof'][i]['type']}
            pp_prof = {'x': self._state['pp_prof'][i]['x'].copy(),
                          'y': self._state['pp_prof'][i]['y'].copy(),
                          'type': self._state['pp_prof'][i]['type']}
            
            # Normalize profiles
            ffp_prof['y'] /= ffp_prof['y'][0]
            pp_prof['y'] /= pp_prof['y'][0]

            lcfs = self._state['lcfs_geo'][i]
            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            lcfs_psi_target = self._state['psi_lcfs_tx'][i] # _state in Wb/rad, TM expects Wb/rad (AKA Wb-rad) 

            self._gs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)

            # Initialize psi from geometry parameters
            # For step 0, using the seed EQDSK geometry should give a good initial guess
            err_flag = self._gs.init_psi(self._state['R'][i],
                                         self._state['Z'][i],
                                         self._state['a'][i],
                                         self._state['kappa'][i],
                                         self._state['delta'][i])
            if err_flag:
                print("Error initializing psi.")



            self._gs.update_settings()

            skip_coil_update = False
            eq_name = os.path.join(self._out_dir, 'equil', '{:03}.{:03}.eqdsk'.format(self._current_step, i))

            equals = '='*50
            solve_succeeded = False
            level_attempts = []

            ffp_prof_raw = copy.deepcopy(ffp_prof)
            pp_prof_raw  = copy.deepcopy(pp_prof)
            
            # Pre-calculate all level profiles
            level_profiles = []
            
            # Level 0: raw
            ffp_0, pp_0 = self._level0_raw(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw), i)
            level_profiles.append({'ffp': ffp_0, 'pp': pp_0, 'name': 'lv0: raw'})
            
            # Level 1: sign flip
            ffp_1, pp_1 = self._level1_sign_flip(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw), i)
            level_profiles.append({'ffp': ffp_1, 'pp': pp_1, 'name': 'lv1: sign_flip'})
            
            # Level 2: pedestal smoothing (takes p_profile as input)
            ffp_2, pp_2 = self._level2_pedestal_smoothing(copy.deepcopy(ffp_prof_raw), copy.deepcopy(self._state['p_prof_tx'][i]), i)
            level_profiles.append({'ffp': ffp_2, 'pp': pp_2, 'name': 'lv2: pedestal_smoothing'})
            
            # Level 3: power flux
            # ffp_3, pp_3 = self._level3_power_flux(copy.deepcopy(ffp_prof_raw), copy.deepcopy(pp_prof_raw), i)
            # level_profiles.append({'ffp': ffp_3, 'pp': pp_3, 'name': 'lv3: power_flux'})

            # Try each level
            for level_idx, level_prof in enumerate(level_profiles):
                level_name = level_prof['name']
                ffp_level = level_prof['ffp']
                pp_level = level_prof['pp']
                try:
                    print(f'{equals} trying level {level_idx} solve ({level_name})')
                    self._gs.set_profiles(ffp_prof=ffp_level, pp_prof=pp_level,
                                          ffp_NI_prof=self._state['ffpni_prof'][i])
                    err_flag = self._gs.solve()
                    print(f'{equals} level {level_idx} solve succeeded!')
                    self._print_out(f'\tTM: Solve succeeded at t={t} (level {level_idx}: {level_name}).')

                    level_attempts.append({'level': level_idx, 'name': level_name,
                                          'ffp': ffp_level, 'pp': pp_level,
                                          'succeeded': True, 'error': None})
                    ffp_prof, pp_prof = ffp_level, pp_level
                    solve_succeeded = True
                    break
                except Exception as e:
                    print(f'\t{equals} level {level_idx} solve failed: {e}')
                    level_attempts.append({'level': level_idx, 'name': level_name,
                                          'ffp': ffp_level, 'pp': pp_level,
                                          'succeeded': False, 'error': str(e)})

            if not solve_succeeded:
                self._eqdsk_skip.append(eq_name)
                skip_coil_update = True
                self._print_out(f'\tTM: Solve failed at t={t} (all levels attempted).')
            
            # self._tm_diagnostic_plot(step, i, t, ffp_prof, pp_prof, solve_succeeded, fail_msg=fail_msg)

            if solve_succeeded:
                self._gs.save_eqdsk(eq_name,
                    lcfs_pad=0.001,run_info='TokaMaker EQDSK',
                    cocos=2, nr=200, nz=200, truncate_eq=False)
                self._gs_update(i)
                self._profile_plot(i, t)

                if graph:
                    fig, ax = plt.subplots(1,1)
                    self._gs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
                    self._gs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
                    ax.plot(self._state['lcfs_geo'][i][:, 0], self._state['lcfs_geo'][i][:, 1], color='r')
                    ax.set_title(f't={self._times[i]}')
                    plt.savefig(os.path.join(self._out_dir, 'equil', 'equil_{:03}.{:03}.png'.format(self._current_step, i)))
                    plt.close(fig)
                
            self._tm_diagnostic_plot(i, t, level_attempts, solve_succeeded)

            _winning = next((a for a in level_attempts if a['succeeded']), None)
            _last_attempt = level_attempts[-1] if level_attempts else {}
            _step_level_log.append({
                'i': i, 't': t,
                'succeeded': solve_succeeded,
                'level': _winning['level'] if _winning else None,
                'level_name': _winning['name'] if _winning else None,
                'error': _last_attempt.get('error') if not solve_succeeded else None,
            })

            if self._prescribed_currents:
                if i < len(self._times):
                    self.set_coil_reg(i=i+1)
            elif not skip_coil_update:
                coil_targets, _ = self._gs.get_coil_currents()
                self.set_coil_reg(targets=coil_targets)

        consumed_flux = (self._state['psi_lcfs_tm'][-1] - self._state['psi_lcfs_tm'][0]) * 2.0 * np.pi # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need 2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(self._state['vloop_tm'][0:], self._times[0:])

        self._gs_step_summary_plot(_step_level_log)

        return consumed_flux, consumed_flux_integral
        
    # ── Profile level functions ──────────────────────────────────────────
    # Each level takes (self, ffp_prof, pp_prof, i) and returns (ffp_prof, pp_prof).
    # All levels receive deep copies of the raw TORAX profiles (not cumulative).
    # Level 0 is always identity. Add new levels by appending to self._profile_levels.

    def _level0_raw(self, ffp_prof, pp_prof, i):
        r'''! Raw TORAX profiles passed through unchanged.'''
        return ffp_prof, pp_prof

    def _level1_sign_flip(self, ffp_prof, pp_prof, i):
        r'''! Sign-flip clipping: clip each profile to its dominant sign.'''
        def _clip(prof):
            y = prof['y']
            sign = 1 if np.sum(y > 0) >= np.sum(y < 0) else -1
            y_new = np.clip(y, 0, None) if sign > 0 else np.clip(y, None, 0)
            return {**prof, 'y': y_new}
        return _clip(ffp_prof), _clip(pp_prof)

    def _level2_pedestal_smoothing(self, ffp_prof, p_prof, i):
        r'''! Pedestal smoothing with Gaussian filter: smooth p profile and take derivative for pp_prof.'''
        # Use provided p_profile, normalize it
        p_prof_y = p_prof['y'].copy()
        if p_prof_y[0] != 0:
            p_prof_y = p_prof_y / p_prof_y[0]
        
        # Apply Gaussian filter for smoothing
        sigma = 2.0  # smoothing parameter
        p_smooth = gaussian_filter1d(p_prof_y, sigma=sigma)
        
        # Ensure pressure is 0 at edge (psi_N = 1)
        p_smooth[-1] = 0.0
        
        # Take derivative to get pp_prof (pressure gradient)
        pp_smooth = np.gradient(p_smooth, p_prof['x'])
        
        # Create output profiles
        pp_out = {**p_prof, 'y': pp_smooth}
        
        # Return ffp_prof unchanged
        return ffp_prof, pp_out

    def _level3_power_flux(self, ffp_prof, pp_prof, i):
        r'''! Generic power-flux shape, sign matched to raw profile means.'''
        ffp_sign = float(np.sign(np.nanmean(ffp_prof['y']))) or 1.0
        pp_sign  = float(np.sign(np.nanmean(pp_prof['y'])))  or 1.0
        ffp_out = create_power_flux_fun(N_PSI, 1.5, 2.0)
        pp_out  = create_power_flux_fun(N_PSI, 4.0, 1.0)
        ffp_out = {**ffp_out, 'y': ffp_out['y'] * ffp_sign}
        pp_out  = {**pp_out,  'y': pp_out['y']  * pp_sign}
        return ffp_out, pp_out

    def _gs_update(self, i):
        r'''! Update internal state and coil current results based on results of GS solver.
        @param i Timestep of the solve.
        '''
        eq_stats = self._gs.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']
        self._state['Ip_tm'][i] = eq_stats['Ip']
        self._state['pax_tm'][i] = eq_stats['P_ax']
        self._state['beta_N_tm'][i] = eq_stats['beta_n']
        self._state['l_i_tm'][i] = eq_stats['l_i']
        
        eq_read_extended = read_eqdsk_extended(os.path.join(self._out_dir, 'equil', '{:03}.{:03}.eqdsk'.format(self._current_step, i)))
        vol_tm = np.interp(self._psi_N, eq_read_extended['psi_n'], eq_read_extended['vol'])
        vpr_tm = np.interp(self._psi_N, eq_read_extended['psi_n'], eq_read_extended['vpr'])
        self._state['vol_tm'][i] = {'x': self._psi_N.copy(), 'y': vol_tm, 'type': 'linterp'}
        self._state['vpr_tm'][i] = {'x': self._psi_N.copy(), 'y': vpr_tm, 'type': 'linterp'}


        self._state['psi_lcfs_tm'][i] = self._gs.psi_bounds[0] # TM outputs in Wb/rad (AKA Wb-rad) which is how psi_lcfs is stored
        self._state['psi_axis_tm'][i] = self._gs.psi_bounds[1] 
        self._state['psi_tm'][i] = {'x': self._psi_N.copy(), 'y': self._state['psi_axis_tm'][i] + (self._state['psi_lcfs_tm'][i] - self._state['psi_axis_tm'][i]) * self._psi_N, 'type': 'linterp'}

        try:
            self._state['vloop_tm'][i] = self._gs.calc_loopvoltage()
        except ValueError:
            # TokaMaker wrapper raises ValueError for any negative Vloop, including the
            # physically valid case where Ip_NI > Ip (NI current exceeds total plasma current),
            # which makes the denominator (itor - Ip_NI) negative.  Fall back to TORAX vloop.
            print(f'\tWARNING: calc_loopvoltage failed at t-idx {i} '
                  f'(likely Ip_NI > Ip); using TORAX vloop as fallback.')
            self._state['vloop_tm'][i] = float(self._state['vloop_tx'][i])
        
        # store TokaMaker pressure profile from get_profiles()
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)

        self._state['ffp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_fp_prof*tm_f_prof), 'type': 'linterp'}
        self._state['pp_prof_tm'][i] =  {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_pp_prof), 'type': 'linterp'}
        self._state['p_prof_tm'][i] =   {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_p_prof), 'type': 'linterp'}
        self._state['f_prof_tm'][i] =   {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_f_prof), 'type': 'linterp'}

        # pull geo profiles
        psi_geo, q_tm, geo, _, _, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02)
        
        self._state['q0_tm'][i] = q_tm[0] if len(q_tm) > 0 else np.nan
        self._state['q95_tm'][i] = np.interp(0.95, psi_geo, q_tm) if len(psi_geo) > 0 and len(q_tm) > 0 else np.nan
        self._state['q_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, q_tm), 'type': 'linterp'}

        self._state['R_avg_tm'][i] =     {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[0])), 'type': 'linterp'}
        self._state['R_inv_avg_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[1])), 'type': 'linterp'}
        
        # Update Results
        coils, _ = self._gs.get_coil_currents()
        if 'COIL' not in self._results:
            self._results['COIL'] = {coil: {} for coil in coils}
        for coil, current in coils.items():
            if coil not in self._results['COIL']:
                self._results['COIL'][coil] = {}
            self._results['COIL'][coil][self._times[i]] = current * 1.0 # TODO: handle nturns > 1


    def _res_update(self, data_tree):

        self._results['t_res'] = self._times

        for t in self._times:
            self._results['T_e'][t] = self._pull_torax_onto_psi(data_tree, 'T_e', t, load_into_state='state', normalize=False)
            self._results['T_i'][t] = self._pull_torax_onto_psi(data_tree, 'T_i', t, load_into_state='state', normalize=False)
            self._results['n_e'][t] = self._pull_torax_onto_psi(data_tree, 'n_e', t, load_into_state='state', normalize=False)
            self._results['q'][t] =   self._pull_torax_onto_psi(data_tree, 'q', t, load_into_state='state', normalize=False)

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
        self._results['psi_lcfs_torax'] = {
            'x': list(psi_lcfs.coords['time'].values),
            'y': psi_lcfs.to_numpy() / (2.0 * np.pi), # TORAX outputs in Wb, stored in Wb/rad
        }

        psi_axis = data_tree.profiles.psi.sel(rho_norm = 0.0)
        self._results['psi_axis_torax'] = {
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


    def save_state(self, fname):
        r'''! Save intermediate simulation state to JSON.
        @param fname Filename to save to.
        '''
        with open(fname, 'w') as f:
            json.dump(self._state, f, cls=MyEncoder)
    
    def save_res(self):
        r'''! Save simulation results to JSON.'''
        with open(self._fname_out, 'w') as f:
            json.dump(self._results, f, cls=MyEncoder)
        
    def _print_out(self, str):
        with open(self._log_file, 'a') as f:
            print(str, file=f)
    

    def _profile_plot(self, i, t): 
        # plot and save profiles at each time step and iteration
        # called in _run_gs after successful GS solve

        # Get TokaMaker output profiles
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)
        tm_ffp_prof = tm_f_prof * tm_fp_prof

        # Helper to normalize an array: subtract boundary, divide by axis value
        def normalize_arr(y):
            y_norm = copy.deepcopy(y)
            # y_norm -= y_norm[-1]
            axis_val = y_norm[0]
            y_norm /= axis_val if axis_val != 0 else 1.0
            return y_norm

        # Normalized versions
        tm_p_norm = normalize_arr(tm_p_prof)
        tm_pp_norm = normalize_arr(tm_pp_prof)
        tm_f_norm = normalize_arr(tm_f_prof)
        tm_ffp_norm = normalize_arr(tm_ffp_prof)

        # FF'_NI is NOT normalized - it's in physical units
        ffpni_real = self._state['ffpni_prof'][i]['y'].copy()
        
        # FF'_inductive = FF'_total - FF'_NI (on TM psi grid, in physical units)
        ffpni_on_tm = np.interp(tm_psi, self._state['ffpni_prof'][i]['x'], ffpni_real)
        ffp_inductive = tm_ffp_prof - ffpni_on_tm

        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        plt.suptitle(f'Step {self._current_step} - Time index {i}/{len(self._times)-1} - t = {t:.1f} s', fontsize=14)

        # =======================
        # ROW 0: p' and FF' comparisons
        # =======================
        
        # (0,0): p' and p comparison (real units) with dual y-axes
        ax_pp_real = axes[0,0]
        ax_pp_real.set_title("p' and p comparison (real units)")
        ax_pp_real.plot(self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y'], 'b-', label="p' TX", linewidth=2)
        ax_pp_real.plot(self._psi_N, self._state['pp_prof_tm'][i]['y'], 'b--', label="p' TM", linewidth=2)
        ax_pp_real.set_ylabel("p' [Pa/Wb]", color='b')
        ax_pp_real.set_xlabel(r'$\hat{\psi}$')
        ax_pp_real.tick_params(axis='y', labelcolor='b')
        ax_pp_real.legend(fontsize=9, loc='upper left')
        # Secondary y-axis for p
        ax2_pp_real = ax_pp_real.twinx()
        ax2_pp_real.plot(self._state['p_prof_tm'][i]['x'], self._state['p_prof_tm'][i]['y'], 'r-', label='p TM', linewidth=2)
        ax2_pp_real.plot(self._state['ptot'][i]['x'], self._state['ptot'][i]['y'], 'r--', label='p TX', linewidth=2)
        ax2_pp_real.set_ylabel('p [Pa]', color='r')
        ax2_pp_real.tick_params(axis='y', labelcolor='r')
        ax2_pp_real.legend(fontsize=9, loc='upper right')

        # (0,1): p' and p comparison (normalized) with dual y-axes
        ax_pp_norm = axes[0,1]
        ptot_torax_norm = normalize_arr(self._state['ptot'][i]['y'])
        ax_pp_norm.set_title("p' and p comparison (normalized)")
        ax_pp_norm.plot(self._state['pp_prof'][i]['x'], self._state['pp_prof'][i]['y'], 'b-', label="p' TX (norm)", linewidth=2)
        ax_pp_norm.plot(tm_psi, tm_pp_norm, 'b--', label="p' TM (norm)", linewidth=2)
        ax_pp_norm.set_ylabel("p' (norm)", color='b')
        ax_pp_norm.set_xlabel(r'$\hat{\psi}$')
        ax_pp_norm.tick_params(axis='y', labelcolor='b')
        ax_pp_norm.legend(fontsize=9, loc='upper left')
        # Secondary y-axis for p normalized
        ax2_pp_norm = ax_pp_norm.twinx()
        ax2_pp_norm.plot(tm_psi, tm_p_norm, 'r-', label='p TM (norm)', linewidth=2)
        ax2_pp_norm.plot(self._state['ptot'][i]['x'], ptot_torax_norm, 'r--', label='p TX (norm)', linewidth=2)
        ax2_pp_norm.set_ylabel('p (norm)', color='r')
        ax2_pp_norm.tick_params(axis='y', labelcolor='r')
        ax2_pp_norm.legend(fontsize=9, loc='upper right')

        # (0,2): FF' comparison (normalized)
        ax_ffp_norm = axes[0,2]
        ax_ffp_norm.set_title("FF' comparison (normalized)")
        ax_ffp_norm.plot(self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'], 'b-', label='TX (norm)', linewidth=2)
        ax_ffp_norm.plot(tm_psi, tm_ffp_norm, 'r--', label='TM (norm)', linewidth=2)
        ax_ffp_norm.set_ylabel("FF' (norm)")
        ax_ffp_norm.set_xlabel(r'$\hat{\psi}$')
        ax_ffp_norm.legend(fontsize=9)

        # =======================
        # ROW 1: Current densities, resistivity, FF' real units
        # =======================
        
        # (1,0): jPhi plot with j_tot, j_ohmic, j_ni, j_bootstrap
        ax_jphi = axes[1,0]
        ax_jphi.set_title('Current densities')
        ax_jphi.plot(self._state['j_tot'][i]['x'], self._state['j_tot'][i]['y'] / 1e6, 'k-', label=r'$j_{tot}$', linewidth=2)
        ax_jphi.plot(self._state['j_ohmic'][i]['x'], self._state['j_ohmic'][i]['y'] / 1e6, 'r-', label=r'$j_{ohmic}$', linewidth=1.5)
        ax_jphi.plot(self._state['j_ni'][i]['x'], self._state['j_ni'][i]['y'] / 1e6, 'b-', label=r'$j_{NI}$', linewidth=1.5)
        if i in self._state['j_bootstrap']:
            ax_jphi.plot(self._state['j_bootstrap'][i]['x'], self._state['j_bootstrap'][i]['y'] / 1e6, 'g-', label=r'$j_{bootstrap}$', linewidth=1.5)
        ax_jphi.set_ylabel(r'$j$ [MA/m²]')
        ax_jphi.set_xlabel(r'$\hat{\psi}$')
        ax_jphi.legend(fontsize=9)

        # (1,1): Resistivity profile
        ax_eta = axes[1,1]
        ax_eta.set_title('Resistivity')
        ax_eta.plot(self._state['eta_prof'][i]['x'], self._state['eta_prof'][i]['y'], 'r-', label='TX', linewidth=2)
        ax_eta.set_yscale('log')
        ax_eta.set_ylabel(r'$\eta$ [Ohm m]')
        ax_eta.set_xlabel(r'$\hat{\psi}$')
        ax_eta.legend(fontsize=9)

        # (1,2): FF' comparison (real units)
        ax_ffp_real = axes[1,2]
        ax_ffp_real.set_title("FF' comparison (real units)")
        ax_ffp_real.plot(self._psi_N, self._state['ffp_prof_tx'][i]['y'], 'k-', label="FF' total TX", linewidth=2)
        ax_ffp_real.plot(self._state['ffpni_prof'][i]['x'], self._state['ffpni_prof'][i]['y'], 'b-', label="FF' NI")
        ax_ffp_real.plot(self._psi_N, self._state['ffp_prof_tx'][i]['y'] - self._state['ffpni_prof'][i]['y'], 'r--', label="FF' inductive")
        ax_ffp_real.plot(self._psi_N, self._state['ffp_prof_tm'][i]['y'], 'g--', label="FF' total TM", linewidth=1)
        ax_ffp_real.set_ylabel("FF'")
        ax_ffp_real.set_xlabel(r'$\hat{\psi}$')
        ax_ffp_real.legend(fontsize=9)

        # =======================
        # ROW 2: Shaping parameters and volume
        # =======================
        
        # (2,0): psi profile comparison
        ax_psi = axes[2,0]
        ax_psi.set_title('Psi profile comparison')
        ax_psi.plot(self._state['psi_tx'][i]['x'], self._state['psi_tx'][i]['y'], 'b-', label='Psi TX', linewidth=2)
        ax_psi.plot(self._state['psi_tm'][i]['x'], self._state['psi_tm'][i]['y'], 'r--', label='Psi TM', linewidth=2)
        ax_psi.set_xlabel(r'$\hat{\psi}$')
        ax_psi.set_ylabel(r'$\psi$ [Wb/rad]')
        ax_psi.legend(fontsize=9)   


        # # (2,0): R_avg comparison
        # ax_r_avg = axes[2,0]
        # ax_r_avg.set_title('<R> comparison')
        # if i in self._state['R_avg_tm']:
        #     ax_r_avg.plot(self._state['R_avg_tm'][i]['x'], self._state['R_avg_tm'][i]['y'], 'r-', label='R_avg TM', linewidth=2)
        # if i in self._state.get('R_avg_tx', {}):
        #     ax_r_avg.plot(self._state['R_avg_tx'][i]['x'], self._state['R_avg_tx'][i]['y'], 'b--', label='R_avg TX', linewidth=2)
        # ax_r_avg.set_xlabel(r'$\hat{\psi}$')
        # ax_r_avg.set_ylabel('<R> [m]')
        # ax_r_avg.legend(fontsize=9)
        # ax_r_avg.grid(True, alpha=0.3)

        # (2,1): <1/R> comparison
        ax_r_inv_avg = axes[2,1]
        ax_r_inv_avg.set_title('<1/R> comparison')
        if i in self._state['R_inv_avg_tm']:
            ax_r_inv_avg.plot(self._state['R_inv_avg_tm'][i]['x'], self._state['R_inv_avg_tm'][i]['y'], 'r-', label='<1/R> TM', linewidth=2)
        if i in self._state['R_inv_avg_tx']:
            ax_r_inv_avg.plot(self._state['R_inv_avg_tx'][i]['x'], self._state['R_inv_avg_tx'][i]['y'], 'b--', label='<1/R> TX', linewidth=2)
        ax_r_inv_avg.set_xlabel(r'$\hat{\psi}$')
        ax_r_inv_avg.set_ylabel('<1/R> [1/m]')
        ax_r_inv_avg.legend(fontsize=9)
        ax_r_inv_avg.grid(True, alpha=0.3)

        # (2,2): Volume comparison
        ax_vol = axes[2,2]
        ax_vol.set_title('Volume comparison')
        vol_tm_lcfs = self._state['vol_tm'][i]['y'][-1] if i in self._state['vol_tm'] else np.nan
        vol_tx_lcfs = self._state['vol_tx_lcfs'][i]
        if i in self._state['vol_tm']:
            ax_vol.plot(self._state['vol_tm'][i]['x'], self._state['vol_tm'][i]['y'], 'r-', label='Vol TM', linewidth=2)
        if i in self._state['vol_tx']:
            ax_vol.plot(self._state['vol_tx'][i]['x'], self._state['vol_tx'][i]['y'], 'b-', label='Vol TX', linewidth=2)
            ax_vol.set_xlabel(r'$\hat{\psi}$')
            ax_vol.set_ylabel('Volume [m³]')
            ax_vol.grid(True, alpha=0.3)
            # Overlay volume comparison at LCFS as text in upper corner
            ax_vol.text(0.98, 0.95, f'Vol TM LCFS: {vol_tm_lcfs:.2f} m³\nVol TX LCFS: {vol_tx_lcfs:.2f} m³\nΔ: {abs(vol_tm_lcfs-vol_tx_lcfs)/vol_tm_lcfs*100:.1f}%', 
                       transform=ax_vol.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax_vol.legend(fontsize=9)
        else:
            ax_vol.text(0.5, 0.5, f'Vol TM: {vol_tm_lcfs:.3f} m³\nVol TX: {vol_tx_lcfs:.3f} m³\nDiff: {abs(vol_tm_lcfs-vol_tx_lcfs)/vol_tm_lcfs*100:.2f}%',
                       ha='center', va='center', fontsize=10)
            ax_vol.axis('off')

        # =======================
        # ROW 3: Profiles (q, T, n)
        # =======================

        # (3,0): q-profile panel (TORAX q if available)
        ax_q = axes[3,0]
        ax_q.set_title('q profile')
        ax_q.plot(self._state['q_prof_tx'][i]['x'], self._state['q_prof_tx'][i]['y'], 'b--', label='TX', linewidth=1)
        ax_q.plot(self._state['q_prof_tm'][i]['x'], self._state['q_prof_tm'][i]['y'], 'r--', label='TM', linewidth=2)  
        ax_q.set_xlabel(r'$\hat{\psi}$')
        ax_q.set_ylabel('q')
        ax_q.legend(fontsize=9)  


        # (3,1): Ti and Te profiles (same panel)
        ax_temp = axes[3,1]
        ax_temp.set_title('T_e and T_i')
        if i in self._state.get('T_e', {}) and i in self._state.get('T_i', {}):
            ax_temp.plot(self._state['T_e'][i]['x'], self._state['T_e'][i]['y'], 'r-', label=r'$T_e$')
            ax_temp.plot(self._state['T_i'][i]['x'], self._state['T_i'][i]['y'], 'm--', label=r'$T_i$')
            ax_temp.set_xlabel(r'$\hat{\psi}$')
            ax_temp.set_ylabel('T [keV]')
            ax_temp.legend(fontsize=9)
        else:
            ax_temp.text(0.5, 0.5, 'No T profiles', ha='center', va='center')
            ax_temp.set_xticks([])
            ax_temp.set_yticks([])

        # (3,2): n_e and n_i profiles (same panel)
        ax_dens = axes[3,2]
        ax_dens.set_title('n_e and n_i')
        if i in self._state.get('n_e', {}) and i in self._state.get('n_i', {}):
            ax_dens.plot(self._state['n_e'][i]['x'], self._state['n_e'][i]['y'], 'b-', label=r'$n_e$')
            ax_dens.plot(self._state['n_i'][i]['x'], self._state['n_i'][i]['y'], 'c--', label=r'$n_i$')
            ax_dens.set_xlabel(r'$\hat{\psi}$')
            ax_dens.set_ylabel(r'$n$ [m$^{-3}$]')
            ax_dens.legend(fontsize=9)
        else:
            ax_dens.text(0.5, 0.5, 'No n profiles', ha='center', va='center')
            ax_dens.set_xticks([])
            ax_dens.set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.35)
        plt.savefig(os.path.join(self._out_dir, 'plots', f'profile_plot_{self._current_step:03}.{i:03}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _tm_diagnostic_plot(self, i, t, level_attempts, solve_succeeded):
        r'''! Create and save a compact TokaMaker input/output diagnostic plot.

        Plots all attempted tier profiles, highlighting which succeeded and which failed.
        Both success and failure saves go to tm_plots/. Filename prefix tm_OK_ vs tm_FAIL_
        and suptitle color (green/red) distinguish the two cases.

        @param i             Time index within self._times.
        @param t             Physical time value (s).
        @param level_attempts List of dicts from the level loop: {level, name, ffp, pp, succeeded, error}.
        @param solve_succeeded Whether any level succeeded.
        '''

        # Extract winning / last-attempted profiles for scalar tables and TM output panels
        _winning = next((a for a in level_attempts if a['succeeded']), None)
        _last    = level_attempts[-1] if level_attempts else {}
        ffp_prof = _winning['ffp'] if _winning else _last.get('ffp')
        pp_prof  = _winning['pp']  if _winning else _last.get('pp')
        fail_msg = _last.get('error') if not solve_succeeded else None

        # Color/style helper: plots every attempted level on ax for profile key 'ffp' or 'pp'
        # Normalizes profiles by dividing by core value (first element)
        _level_colors = plt.cm.tab10.colors
        def _plot_levels(ax, key, seed_x=None, seed_y=None, seed_label=None):
            for attempt in level_attempts:
                color = _level_colors[attempt['level'] % len(_level_colors)]
                # Normalize profile by dividing by core value (first element)
                y_data = attempt[key]['y'].copy()
                if y_data[0] != 0:
                    y_data = y_data / y_data[0]
                if attempt['succeeded']:
                    ax.plot(attempt[key]['x'], y_data,
                            color='forestgreen', linewidth=2.5, zorder=5,
                            label=f"Level {attempt['level']}: {attempt['name']} \u2713")
                else:
                    ax.plot(attempt[key]['x'], y_data,
                            color=color, linewidth=1.2, linestyle='--', alpha=0.6,
                            label=f"Level {attempt['level']}: {attempt['name']} \u2717")
            if seed_x is not None:
                ax.plot(seed_x, seed_y, 'k--', linewidth=1.5, alpha=0.7, label=seed_label)

        def render_table(ax, rows, title):
            """Render a table on a matplotlib axis. rows[0] is the header row."""
            ax.axis('off')
            ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
            tbl = ax.table(
                cellText=rows[1:],
                colLabels=rows[0],
                loc='center', cellLoc='left',
                bbox=[0.0, 0.0, 1.0, 0.92],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.5)
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#d0e4f7')
                elif row % 2 == 0:
                    cell.set_facecolor('#f5f5f5')

        # --- Scalar values ---
        # Seed / initial EQDSK values
        Ip_seed      = abs(float(self._Ip_seed[i]))
        pax_seed     = abs(float(self._pax_seed[i]))
        psi_lcfs_seed = float(self._psi_lcfs_seed[i])
        # TORAX values at this timestep
        Ip_tx        = abs(float(self._state['Ip'][i]))
        pax_tx       = abs(float(self._state['pax'][i]))
        psi_lcfs_tx  = float(self._state['psi_lcfs_tx'][i])
        q95_tx       = float(self._state['q95'][i])
        q0_tx        = float(self._state['q0'][i])
        vloop_tx     = float(self._state['vloop_tx'][i])
        beta_pol_tx  = float(self._state['beta_pol'][i])
        beta_n_tx    = float(self._state['beta_N_tx'][i])

        # --- Load nearest seed EQDSK for reference profiles and q scalars ---
        _eqtimes_arr = np.array(self._eqtimes)
        _seed_idx = int(np.argmin(np.abs(_eqtimes_arr - t)))
        _seed_g = read_eqdsk_extended(self._init_files[_seed_idx])
        _seed_psi_n = np.linspace(0.0, 1.0, len(_seed_g['ffprim']))
        _seed_ffp_norm = _seed_g['ffprim'].copy() / _seed_g['ffprim'][0]
        _seed_pp_norm  = _seed_g['pprime'].copy()  / _seed_g['pprime'][0]
        q0_seed_eq  = _seed_g['q0']
        q95_seed_eq = _seed_g['q95']

        if solve_succeeded:
            input_rows = [
                ['Parameter', 'Init EQDSK', 'TORAX', 'TokaMaker'],
                ['Ip',
                 f'{Ip_seed/1e6:.3f} MA',
                 f'{Ip_tx/1e6:.3f} MA',
                 f'{float(self._state["Ip_tm"][i])/1e6:.3f} MA'],
                ['pax',
                 f'{pax_seed/1e3:.2f} kPa',
                 f'{pax_tx/1e3:.2f} kPa',
                 f'{float(self._state["pax_tm"][i])/1e3:.2f} kPa'],
                ['psi_lcfs',
                 f'{psi_lcfs_seed:.4f} Wb/rad',
                 f'{psi_lcfs_tx:.4f} Wb/rad',
                 f'{float(self._state["psi_lcfs_tm"][i]):.4f} Wb/rad'],
            ]
            diag_rows = [
                ['Parameter', 'Init EQDSK', 'TORAX', 'TokaMaker'],
                ['q95',      f'{q95_seed_eq:.3f}', f'{q95_tx:.3f}',  f'{float(self._state["q95_tm"][i]):.3f}'],
                ['q0',       f'{q0_seed_eq:.3f}',  f'{q0_tx:.3f}',   f'{float(self._state["q0_tm"][i]):.3f}'],

                ['v_loop',   '\u2014', f'{vloop_tx:.3f} V',   f'{float(self._state["vloop_tm"][i]):.3f} V'],
                ['beta_pol', '\u2014', f'{beta_pol_tx:.4f}',  '\u2014'],
                ['beta_N',   '\u2014', f'{beta_n_tx:.4f}',    f'{float(self._state["beta_N_tm"][i]):.4f}'],
                ['l_i',      '\u2014', '\u2014',              f'{float(self._state["l_i_tm"][i]):.4f}'],
            ]

            # Layout: 3 rows x 6 cols  (TX inputs | TM outputs | tables)
            fig = plt.figure(figsize=(22, 12))
            gs_layout = fig.add_gridspec(3, 6, hspace=0.48, wspace=0.55)

            ax_ffp_tx = fig.add_subplot(gs_layout[0, 0:2])
            ax_pp_tx  = fig.add_subplot(gs_layout[1, 0:2])
            ax_eta    = fig.add_subplot(gs_layout[2, 0:2])
            ax_ffp_tm = fig.add_subplot(gs_layout[0, 2:4])
            ax_pp_tm  = fig.add_subplot(gs_layout[1, 2:4])
            ax_q_tm   = fig.add_subplot(gs_layout[2, 2:4])
            ax_tbl1   = fig.add_subplot(gs_layout[0, 4:6])
            ax_tbl2   = fig.add_subplot(gs_layout[1:3, 4:6])

            # TX input plots — all attempted levels
            _plot_levels(ax_ffp_tx, 'ffp', seed_x=_seed_psi_n, seed_y=_seed_ffp_norm, seed_label="FF\' seed EQDSK (norm)")
            ax_ffp_tx.plot(self._state['ffpni_prof'][i]['x'], self._state['ffpni_prof'][i]['y'],
                           'g--', linewidth=1.5, label="FF\'_NI (real)")
            ax_ffp_tx.set_title("FF\' tried levels (normalized)", fontsize=10)
            ax_ffp_tx.set_xlabel(r'$\hat{\psi}$')
            ax_ffp_tx.set_ylabel("FF\' (norm / real)")
            ax_ffp_tx.set_ylim([0, 1])
            ax_ffp_tx.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
            ax_ffp_tx.grid(True, alpha=0.3)
            ax_ffp_tx.axhline(0, color='k', linewidth=0.5)

            _plot_levels(ax_pp_tx, 'pp', seed_x=_seed_psi_n, seed_y=_seed_pp_norm, seed_label="p\' seed EQDSK (norm)")
            ax_pp_tx.set_title("p\' tried levels (normalized)", fontsize=10)
            ax_pp_tx.set_xlabel(r'$\hat{\psi}$')
            ax_pp_tx.set_ylabel("p\' (norm)")
            ax_pp_tx.set_ylim([0, 1])
            ax_pp_tx.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
            ax_pp_tx.grid(True, alpha=0.3)

            ax_eta.plot(self._state['eta_prof'][i]['x'], self._state['eta_prof'][i]['y'], 'r-', linewidth=2)
            ax_eta.set_yscale('log')
            ax_eta.set_title(r'Resistivity $\eta$ (input to TM)', fontsize=10)
            ax_eta.set_xlabel(r'$\hat{\psi}$')
            ax_eta.set_ylabel(r'$\eta$ [$\Omega\cdot$m]')
            ax_eta.grid(True, alpha=0.3)

            # TM output plots
            ax_ffp_tm.plot(self._state['ffp_prof_tx'][i]['x'], self._state['ffp_prof_tx'][i]['y'],
                           'b--', linewidth=1.5, label="FF\' TX (real)")
            ax_ffp_tm.plot(self._state['ffp_prof_tm'][i]['x'], self._state['ffp_prof_tm'][i]['y'],
                           'r-', linewidth=2, label="FF\' TM (real)")
            ax_ffp_tm.set_title("FF\' TM output vs TX input", fontsize=10)
            ax_ffp_tm.set_xlabel(r'$\hat{\psi}$')
            ax_ffp_tm.set_ylabel("FF\'")
            ax_ffp_tm.legend(fontsize=8)
            ax_ffp_tm.grid(True, alpha=0.3)
            ax_ffp_tm.axhline(0, color='k', linewidth=0.5)

            ax_pp_tm.plot(self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y'],
                          'b--', linewidth=1.5, label="p\' TX (real)")
            ax_pp_tm.plot(self._state['pp_prof_tm'][i]['x'], self._state['pp_prof_tm'][i]['y'],
                          'r-', linewidth=2, label="p\' TM (real)")
            ax_pp_tm.set_title("p\' TM output vs TX input", fontsize=10)
            ax_pp_tm.set_xlabel(r'$\hat{\psi}$')
            ax_pp_tm.set_ylabel("p\'")
            ax_pp_tm.legend(fontsize=8)
            ax_pp_tm.grid(True, alpha=0.3)

            # q profile: TM vs TORAX
            try:
                psi_geo, q_tm_vals, _, _, _, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02)
                ax_q_tm.plot(psi_geo, q_tm_vals, 'r--', linewidth=2, label='TokaMaker')
            except Exception:
                pass

            ax_q_tm.plot(self._state['q_prof_tx'][i]['x'], self._state['q_prof_tx'][i]['y'], 'b-', linewidth=2, label='TORAX')
            ax_q_tm.set_title('q profile (TM vs TORAX)', fontsize=10)
            ax_q_tm.set_xlabel(r'$\hat{\psi}$')
            ax_q_tm.set_ylabel('q')
            ax_q_tm.legend(fontsize=8)
            ax_q_tm.grid(True, alpha=0.3)

            render_table(ax_tbl1, input_rows, 'Scalar Inputs vs TokaMaker Outputs')
            render_table(ax_tbl2, diag_rows,  'TORAX Diagnostics vs TokaMaker')

            plt.suptitle(
                f'TM Diagnostic \u2014 Step {self._current_step}, t-idx {i}/{len(self._times)-1}, t = {t:.2f} s'
                f'  |  TokaMaker: SUCCESS',
                fontsize=13, color='darkgreen',
            )
            out_path = os.path.join(self._out_dir, 'tm_plots', f'tm_{self._current_step:03}.{i:03}_OK.png')

        else:
            input_rows = [
                ['Parameter', 'Init EQDSK', 'TORAX'],
                ['Ip',       f'{Ip_seed/1e6:.3f} MA',       f'{Ip_tx/1e6:.3f} MA'],
                ['pax',      f'{pax_seed/1e3:.2f} kPa',     f'{pax_tx/1e3:.2f} kPa'],
                ['psi_lcfs', f'{psi_lcfs_seed:.4f} Wb/rad', f'{psi_lcfs_tx:.4f} Wb/rad'],
            ]
            diag_rows = [
                ['Parameter', 'Init EQDSK', 'TORAX'],
                ['q95',      f'{q95_seed_eq:.3f}', f'{q95_tx:.3f}'],
                ['q0',       f'{q0_seed_eq:.3f}',  f'{q0_tx:.3f}'],
                ['v_loop',   '\u2014', f'{vloop_tx:.3f} V'],
                ['beta_pol', '\u2014', f'{beta_pol_tx:.4f}'],
                ['beta_N',   '\u2014', f'{beta_n_tx:.4f}'],
            ]

            fig = plt.figure(figsize=(16, 12))
            gs_layout = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.55)

            ax_ffp  = fig.add_subplot(gs_layout[0, 0:2])
            ax_pp   = fig.add_subplot(gs_layout[1, 0:2])
            ax_eta  = fig.add_subplot(gs_layout[2, 0:2])
            ax_tbl1 = fig.add_subplot(gs_layout[0, 2:4])
            ax_tbl2 = fig.add_subplot(gs_layout[1, 2:4])
            ax_fail = fig.add_subplot(gs_layout[2, 2:4])

            _plot_levels(ax_ffp, 'ffp', seed_x=_seed_psi_n, seed_y=_seed_ffp_norm, seed_label="FF\' seed EQDSK (norm)")
            ax_ffp.plot(self._state['ffpni_prof'][i]['x'], self._state['ffpni_prof'][i]['y'],
                        'g--', linewidth=1.5, label="FF\'_NI (real)")
            ax_ffp.set_title("FF\' tried levels (normalized)", fontsize=10)
            ax_ffp.set_xlabel(r'$\hat{\psi}$')
            ax_ffp.set_ylabel("FF\' (norm / real)")
            ax_ffp.set_ylim([0, 1])
            ax_ffp.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
            ax_ffp.grid(True, alpha=0.3)
            ax_ffp.axhline(0, color='k', linewidth=0.5)

            _plot_levels(ax_pp, 'pp', seed_x=_seed_psi_n, seed_y=_seed_pp_norm, seed_label="p\' seed EQDSK (norm)")
            ax_pp.set_title("p\' tried levels (normalized)", fontsize=10)
            ax_pp.set_xlabel(r'$\hat{\psi}$')
            ax_pp.set_ylabel("p\' (norm)")
            ax_pp.set_ylim([0, 1])
            ax_pp.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
            ax_pp.grid(True, alpha=0.3)

            ax_eta.plot(self._state['eta_prof'][i]['x'], self._state['eta_prof'][i]['y'], 'r-', linewidth=2)
            ax_eta.set_yscale('log')
            ax_eta.set_title(r'Resistivity $\eta$ (input to TM)', fontsize=10)
            ax_eta.set_xlabel(r'$\hat{\psi}$')
            ax_eta.set_ylabel(r'$\eta$ [$\Omega\cdot$m]')
            ax_eta.grid(True, alpha=0.3)

            render_table(ax_tbl1, input_rows, 'Scalar Inputs  (Init EQDSK vs TORAX)')
            render_table(ax_tbl2, diag_rows,  'TORAX Diagnostics')

            # Fail message panel
            ax_fail.axis('off')
            fail_text = f'{fail_msg}' if fail_msg else 'TokaMaker failed (unknown reason)'
            ax_fail.text(
                0.5, 0.5, fail_text,
                ha='center', va='center', fontsize=9, color='darkred', fontweight='bold',
                transform=ax_fail.transAxes,
                wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='darkred', linewidth=1.5),
            )
            ax_fail.set_title('Failure Reason', fontsize=10, fontweight='bold', pad=4, color='darkred')

            plt.suptitle(
                f'TM Diagnostic \u2014 Step {self._current_step}, t-idx {i}/{len(self._times)-1}, t = {t:.2f} s'
                f'  |  TokaMaker: FAILED',
                fontsize=13, color='darkred', fontweight='bold',
            )
            out_path = os.path.join(self._out_dir, 'tm_plots', f'tm_{self._current_step:03}.{i:03}_FAIL.png')

        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _gs_step_summary_plot(self, step_level_log):
        r'''! Save a summary figure for the completed GS step showing per-timestep solve outcomes.

        Each row is a timestep. Columns: time index, time (s), outcome (SUCCESS Level N / FAILED).
        Color-coded green for success, red for failure. Includes a legend of available levels.

        @param step_level_log List of dicts {i, t, succeeded, level, level_name} from _run_gs.
        '''
        n = len(step_level_log)
        if n == 0:
            return

        fig = plt.figure(figsize=(10, max(5, 0.4 * n + 3.5)))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.4)
        
        ax_table = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])
        ax_table.axis('off')
        ax_legend.axis('off')

        col_labels = ['t-idx', 't (s)', 'Result']
        rows = []
        cell_colors = []
        for entry in step_level_log:
            if entry['succeeded']:
                result = f"Lvl {entry['level']}"
                row_color = ['#d4edda'] * 3
            else:
                # Abbreviate error message
                error_msg = entry['error'] if entry['error'] else 'Unknown error'
                if len(error_msg) > 50:
                    result = error_msg[:47] + '...'
                else:
                    result = error_msg
                row_color = ['#f8d7da'] * 3
            rows.append([str(entry['i']), f"{entry['t']:.3f}", result])
            cell_colors.append(row_color)

        tbl = ax_table.table(
            cellText=rows,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc='center', cellLoc='center',
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor('#d0e4f7')
                cell.set_text_props(fontweight='bold')

        # Create legend of available levels
        level_descriptions = [
            "Level 0: Raw TORAX profiles",
            "Level 1: Sign-flip clipping",
            "Level 2: Pedestal smoothing",
            "Level 3: Power-flux shape",
        ]
        
        legend_text = "Levels: " + " | ".join(level_descriptions)
        ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', fontsize=9,
                      transform=ax_legend.transAxes,
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', edgecolor='gray', linewidth=1))

        n_ok   = sum(1 for e in step_level_log if e['succeeded'])
        n_fail = n - n_ok
        plt.suptitle(
            f'GS Step {self._current_step} Summary \u2014 {n_ok}/{n} timesteps succeeded, {n_fail} failed',
            fontsize=12, fontweight='bold',
            color='darkgreen' if n_fail == 0 else ('darkred' if n_ok == 0 else 'darkorange'),
        )

        os.makedirs(os.path.join(self._out_dir, 'tm_plots'), exist_ok=True)
        out_path = os.path.join(self._out_dir, 'tm_plots', f'step_{self._current_step:03}_summary.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _profile_evolution_plot(self):
        r'''! Plot multiple profiles over time with color representing time.
        Creates a single figure with subplots for ne, Te, ni, Ti, p (TORAX), p (TokaMaker), q.
        Each subplot shows profiles at all time points with color mapped to time using plasma colormap.
        '''
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        times = self._times
        n_times = len(times)
        
        # Set up colormap and normalization
        cmap = cm.plasma
        norm = Normalize(vmin=times[0], vmax=times[-1])
        
        # Create figure with 7 subplots (2 rows x 4 cols, last one empty or use 3+4 layout)
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        fig.suptitle(f'Profile Evolution Over Time (Step {self._current_step})', fontsize=14)
        
        # =======================
        # ROW 0
        # =======================
        
        # (0,0): n_e evolution
        ax_00 = axes[0,0]
        ax_00.set_title('n_e')
        ax_00.set_xlabel(r'$\hat{\psi}$')
        ax_00.set_ylabel(r'$n_e$ [m$^{-3}$]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('n_e', {}):
                x = self._state['n_e'][i]['x']
                y = self._state['n_e'][i]['y']
                ax_00.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_00.set_xlim([0, 1])

        # (0,1): T_e evolution
        ax_01 = axes[0,1]
        ax_01.set_title('T_e')
        ax_01.set_xlabel(r'$\hat{\psi}$')
        ax_01.set_ylabel(r'$T_e$ [keV]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('T_e', {}):
                x = self._state['T_e'][i]['x']
                y = self._state['T_e'][i]['y']
                ax_01.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_01.set_xlim([0, 1])

        # (0,2): n_i evolution
        ax_02 = axes[0,2]
        ax_02.set_title('n_i')
        ax_02.set_xlabel(r'$\hat{\psi}$')
        ax_02.set_ylabel(r'$n_i$ [m$^{-3}$]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('n_i', {}):
                x = self._state['n_i'][i]['x']
                y = self._state['n_i'][i]['y']
                ax_02.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_02.set_xlim([0, 1])

        # (0,3): T_i evolution
        ax_03 = axes[0,3]
        ax_03.set_title('T_i')
        ax_03.set_xlabel(r'$\hat{\psi}$')
        ax_03.set_ylabel(r'$T_i$ [keV]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('T_i', {}):
                x = self._state['T_i'][i]['x']
                y = self._state['T_i'][i]['y']
                ax_03.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_03.set_xlim([0, 1])

        # =======================
        # ROW 1
        # =======================
        
        # (1,0): p (TORAX) evolution
        ax_10 = axes[1,0]
        ax_10.set_title('p (TORAX)')
        ax_10.set_xlabel(r'$\hat{\psi}$')
        ax_10.set_ylabel(r'$p$ (TORAX) [Pa]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('ptot', {}):
                x = self._state['ptot'][i]['x']
                y = self._state['ptot'][i]['y']
                ax_10.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_10.set_xlim([0, 1])

        # (1,1): p (TokaMaker) evolution
        ax_11 = axes[1,1]
        ax_11.set_title('p (TokaMaker)')
        ax_11.set_xlabel(r'$\hat{\psi}$')
        ax_11.set_ylabel(r'$p$ (TokaMaker) [Pa]')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if i in self._state.get('p_prof_tm', {}):
                x = self._state['p_prof_tm'][i]['x']
                y = self._state['p_prof_tm'][i]['y']
                ax_11.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_11.set_xlim([0, 1])

        # (1,2): q evolution
        ax_12 = axes[1,2]
        ax_12.set_title('q')
        ax_12.set_xlabel(r'$\hat{\psi}$')
        ax_12.set_ylabel(r'$q$')
        for i, t in enumerate(times):
            color = cmap(norm(t))
            if t in self._results.get('q', {}):
                x = self._results['q'][t]['x']
                y = self._results['q'][t]['y']
                ax_12.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
        ax_12.set_xlim([0, 1])

        # (1,3): EMPTY - hide this unused subplot
        axes[1,3].axis('off')
        
        # Add a single colorbar for time on the right side
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Add colorbar to the figure
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('Time [s]', fontsize=12)
        
        plt.savefig(os.path.join(self._out_dir, 'plots', f'profile_evolution_step{self._current_step}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _scalar_plot(self):
        r'''! Plot a grid of time-series scalars across the entire pulse for a given step.
        Produces a 4x3 grid with diagnostics including Ip, psi (TM/TORAX), V_loop comparison,
        Q, n_e_line_avg, T_e_line_avg, P_ohmic, beta_N, l_i, q95, pax.
        '''
        fig, axes = plt.subplots(4, 3, figsize=(16, 12))

        # =======================
        # ROW 0
        # =======================
        
        # (0,0): Ip comparison (TM vs TX) with Ip_NI
        ax_00 = axes[0,0]
        ax_00.set_title('Ip [A]')
        ax_00.plot(self._times, self._state['Ip_tm'], '-o', markersize=3, label='Ip TM')
        ax_00.plot(self._times, self._state['Ip_tx'], '-o', markersize=3, label='Ip TX')
        ax_00.plot(self._times, self._state['Ip_NI_tx'], '--', markersize=3, label='Ip NI TX')
        ax_00.set_xlabel('Time [s]')
        ax_00.set_ylabel('Ip [A]')
        ax_00.grid(True, alpha=0.3)
        ax_00.legend(fontsize=8)

        # (0,1): psi_lcfs and psi_axis (TM & TX) 
        ax_01 = axes[0,1]
        ax_01.set_title(r'$\psi_{lcfs}$ & $\psi_{axis}$ (TM & TX)')
        ax_01.plot(self._times, self._state['psi_lcfs_tm'], '-', color='tab:blue', label=r'$\psi_{lcfs}$ TM')
        ax_01.plot(self._times, self._state['psi_lcfs_tx'], '--', color='tab:blue', label=r'$\psi_{lcfs}$ TX')
        ax_01.plot(self._times, self._state['psi_axis_tm'], '-', color='tab:orange', label=r'$\psi_{axis}$ TM')
        ax_01.plot(self._times, self._state['psi_axis_tx'], '--', color='tab:orange', label=r'$\psi_{axis}$ TX')
        ax_01.set_ylabel(r'$\psi_{axis}$ [Wb/rad]', color='tab:orange')
        ax_01.tick_params(axis='y', labelcolor='tab:orange')
        ax_01.legend(fontsize=8, loc='upper right')
        ax_01.set_xlabel('Time [s]')
        ax_01.set_ylabel(r'$\psi_{lcfs}$ [Wb/rad]', color='tab:blue')
        ax_01.tick_params(axis='y', labelcolor='tab:blue')
        ax_01.legend(fontsize=8, loc='upper left')
        ax_01.grid(True, alpha=0.3)


        # (0,2): V_loop comparison with ratio
        ax_02 = axes[0,2]
        ax_02.set_title('V_loop (TM vs TX) [V]')
        ax_02.plot(self._times, self._state['vloop_tm'], '-o', markersize=3, label='TokaMaker')

        rx = self._times
        ry = self._state['vloop_tx']
        ax_02.plot(rx, ry, '--o', markersize=3, label='TORAX')
        # Secondary axis for vloop ratio (TokaMaker / TORAX)
        tm_vloop = np.array(self._state['vloop_tm'])
        tx_vloop = np.array(ry)
        # Interpolate TokaMaker vloop to TORAX time points if needed
        if len(tm_vloop) == len(tx_vloop):
            ratio = tm_vloop / tx_vloop
            ratio_times = np.array(self._times)
        else:
            interp_tm = interp1d(self._times, tm_vloop, bounds_error=False, fill_value=np.nan)
            ratio = interp_tm(rx) / tx_vloop
            ratio_times = np.array(rx)
        ax2_02 = ax_02.twinx()
        ax2_02.plot(ratio_times, ratio, 'g-s', markersize=3, label='TM/TX ratio')
        ax2_02.set_ylim(0,30)
        ax2_02.set_ylabel('V_loop ratio (TM/TX)', color='g')
        ax2_02.tick_params(axis='y', labelcolor='g')
        ax2_02.legend(fontsize=8, loc='upper right')
        # Print average vloop and ratio between 150 and 200 seconds
        mask = (ratio_times >= 150) & (ratio_times <= 200)
        if np.any(mask):
            avg_ratio = np.nanmean(ratio[mask])
            # Get averages for both codes
            mask_tm = (np.array(self._times) >= 150) & (np.array(self._times) <= 200)
            mask_tx = (np.array(rx) >= 150) & (np.array(rx) <= 200)
            avg_vloop_tm = np.mean(tm_vloop[mask_tm]) if np.any(mask_tm) else np.nan
            avg_vloop_tx = np.mean(tx_vloop[mask_tx]) if np.any(mask_tx) else np.nan
            self._print_out(f"V_loop 150-200: TokaMaker avg={avg_vloop_tm:.3f} V, TORAX avg={avg_vloop_tx:.3f} V, ratio={avg_ratio:.4f}")
            ax2_02.text(0.5, 0.9, f'Avg ratio (150-200s): {avg_ratio:.4f}', transform=ax2_02.transAxes, color='g', fontsize=8, ha='center')
        ax_02.set_xlabel('Time [s]')
        ax_02.grid(True, alpha=0.3)
        ax_02.legend(fontsize=8, loc='upper left')
        ax_02.set_ylim(0,3)

        # =======================
        # ROW 1
        # =======================
        
        # (1,0): Q_fusion with E_fusion on secondary axis
        ax_10 = axes[1,0]
        ax_10.set_title('Q_fusion')
        if 'Q' in self._results:
            ax_10.plot(self._results['Q']['x'], self._results['Q']['y'], '-o', markersize=3, label='Q')
        ax_10.set_xlabel('Time [s]')
        ax_10.grid(True, alpha=0.3)
        # E_fusion on secondary axis
        if 'E_fusion' in self._results:
            ax2_10 = ax_10.twinx()
            ax2_10.plot(self._results['E_fusion']['x'], self._results['E_fusion']['y'], '--', color='crimson', markersize=3, label='E_fusion')
            ax2_10.set_ylabel('E_fusion')
            ax2_10.legend(fontsize=8, loc='upper right')
        ax_10.legend(fontsize=8, loc='upper left')

        # (1,1): n_e_line_avg with n_e_core and n_e_edge
        ax_11 = axes[1,1]
        ax_11.set_title(r'$\bar{n}_e$ line avg [m$^{-3}$]')
        if 'n_e_line_avg' in self._results:
            ax_11.plot(self._results['n_e_line_avg']['x'], self._results['n_e_line_avg']['y'], '-o', markersize=3, label='n_e line avg')
        # plot core from results if present
        if 'n_e_core' in self._results:
            ax_11.plot(self._results['n_e_core']['x'], self._results['n_e_core']['y'], '--', markersize=3, label='n_e core')
        # edge from state profiles
        ne_edge_x = self._times
        ne_edge_y = [self._state['n_e'][ii]['y'][-1] if ii in self._state.get('n_e', {}) else np.nan for ii in range(len(self._times))]
        ax_11.plot(ne_edge_x, ne_edge_y, ':', marker='s', markersize=3, label='n_e edge')
        ax_11.set_xlabel('Time [s]')
        ax_11_02 = ax_11.twinx()
        ax_11_02.plot(self._times, self._state['f_GW'], 'm--', markersize=3, label='f_GW_line') # f_GW using line averaged ne
        ax_11_02.plot(self._times, self._state['f_GW_vol'], 'c--', markersize=3, label='f_GW_vol') # f_GW using volume averaged ne
        ax_11_02.set_ylabel('f_GW')
        ax_11_02.set_ylim(0,1)
        ax_11_02.legend(fontsize=8)
        ax_11.legend(fontsize=8)
        ax_11.grid(True, alpha=0.3)

        # (1,2): T_e_line_avg with T_e_core and T_e_edge
        ax_12 = axes[1,2]
        ax_12.set_title(r'$T_e$ line avg [keV]')
        if 'T_e_line_avg' in self._results:
            ax_12.plot(self._results['T_e_line_avg']['x'], self._results['T_e_line_avg']['y'], '-o', markersize=3, label='T_e line avg')
        if 'T_e_core' in self._results:
            ax_12.plot(self._results['T_e_core']['x'], self._results['T_e_core']['y'], '--', markersize=3, label='T_e core')
        te_edge_x = self._times
        te_edge_y = [self._state['T_e'][ii]['y'][-1] if ii in self._state.get('T_e', {}) else np.nan for ii in range(len(self._times))]
        ax_12.plot(te_edge_x, te_edge_y, ':', marker='s', markersize=3, label='T_e edge')
        ax_12.set_xlabel('Time [s]')
        ax_12.legend(fontsize=8)
        ax_12.grid(True, alpha=0.3)

        # =======================
        # ROW 2
        # =======================
        
        # (2,0): Power channels (P_ohmic_e, P_radiation_e, P_SOL_total, P_alpha_total, P_aux_total)
        ax_20 = axes[2,0]
        ax_20.set_title('Power channels [W]')
        if 'P_ohmic_e' in self._results:
            ax_20.plot(self._results['P_ohmic_e']['x'], self._results['P_ohmic_e']['y'], 'r-o', markersize=3, label='P_ohmic_e')
        if 'P_radiation_e' in self._results:
            ax_20.plot(self._results['P_radiation_e']['x'], self._results['P_radiation_e']['y'], 'm--', markersize=3, label='P_radiation_e')
        if 'P_SOL_total' in self._results:
            ax_20.plot(self._results['P_SOL_total']['x'], self._results['P_SOL_total']['y'], 'c--', markersize=3, label='P_SOL_total')
        if 'P_alpha_total' in self._results:
            ax_20.plot(self._results['P_alpha_total']['x'], self._results['P_alpha_total']['y'], 'g-.', markersize=3, label='P_alpha_total')
        if 'P_aux_total' in self._results:
            ax_20.plot(self._results['P_aux_total']['x'], self._results['P_aux_total']['y'], 'y-.', markersize=3, label='P_aux_total')
        ax_20.set_xlabel('Time [s]')
        ax_20.legend(fontsize=8)
        ax_20.grid(True, alpha=0.3)
        # ax_20.set_ylim(0,1E8)

        # (2,1): beta_N (TX and TM)
        ax_21 = axes[2,1]
        ax_21.set_title('beta_N')
        # Plot beta_N from results (TORAX)
        if 'beta_N' in self._results:
            ax_21.plot(self._results['beta_N']['x'], self._results['beta_N']['y'], '-o', markersize=3, label='beta_N TX')
        # Plot beta_N_tm from state (TokaMaker)
        ax_21.plot(self._times, self._state['beta_N_tm'], '--o', markersize=3, label='beta_N TM')
        ax_21.set_xlabel('Time [s]')
        ax_21.set_ylabel('beta_N')
        ax_21.legend(fontsize=8)
        ax_21.grid(True, alpha=0.3)

        # (2,2): l_i (TX and TM)
        ax_22 = axes[2,2]
        ax_22.set_title('l_i (li3)')
        # Plot li3 from results (TORAX)
        if 'li3' in self._results:
            ax_22.plot(self._results['li3']['x'], self._results['li3']['y'], '-o', markersize=3, label='l_i TX')
        # Plot l_i_tm from state (TokaMaker)
        ax_22.plot(self._times, self._state['l_i_tm'], '--o', markersize=3, label='l_i TM')
        ax_22.set_xlabel('Time [s]')
        ax_22.set_ylabel('l_i')
        ax_22.legend(fontsize=8)
        ax_22.grid(True, alpha=0.3)

        # =======================
        # ROW 3
        # =======================
        
        # (3,0): q95 and q0 (TX and TM)
        ax_30 = axes[3,0]
        ax_30.set_title('Safety Factor q')
        ax_30.plot(self._times, self._state['q95'], 'b-', markersize=3, label='q95 TX')
        ax_30.plot(self._times, self._state['q0'], 'r-', markersize=3, label='q0 TX')
        ax_30.plot(self._times, self._state['q95_tm'], 'b--', markersize=3, label='q95 TM')
        ax_30.plot(self._times, self._state['q0_tm'], 'r--', markersize=3, label='q0 TM')
        ax_30.set_xlabel('Time [s]')
        ax_30.set_ylabel('Safety Factor')
        ax_30.legend(fontsize=8)
        ax_30.grid(True, alpha=0.3)

        # (3,1): pax (TX and TM)
        ax_31 = axes[3,1]
        ax_31.set_title('pax [Pa]')
        # Plot pax from state (TORAX)
        ax_31.plot(self._times, self._state['pax'], '-o', markersize=3, label='pax TX')
        # Plot pax_tm from state (TokaMaker)
        ax_31.plot(self._times, self._state['pax_tm'], '--o', markersize=3, label='pax TM')
        ax_31.set_xlabel('Time [s]')
        ax_31.set_ylabel('pax [Pa]')
        ax_31.legend(fontsize=8)
        ax_31.grid(True, alpha=0.3)

        # (3,2): Flux Consumption (psi_lcfs change * 2pi)
        ax_32 = axes[3,2]
        ax_32.set_title('Flux Consumption [Wb]')
        # Calculate flux consumption: (psi_lcfs(t) - psi_lcfs(t=0))*2pi
        psi_lcfs_tm_arr = np.array(self._state['psi_lcfs_tm'])
        psi_lcfs_tx_arr = np.array(self._state['psi_lcfs_tx'])
        flux_consumption_tm = (psi_lcfs_tm_arr - psi_lcfs_tm_arr[0]) * 2 * np.pi
        flux_consumption_tx = (psi_lcfs_tx_arr - psi_lcfs_tx_arr[0]) * 2 * np.pi
        ax_32.plot(self._times, flux_consumption_tm, '-o', markersize=3, label='Flux consumption TM')
        ax_32.plot(self._times, flux_consumption_tx, '--o', markersize=3, label='Flux consumption TX')
        ax_32.set_xlabel('Time [s]')
        ax_32.set_ylabel('Flux Consumption [Wb]')
        ax_32.legend(fontsize=8)
        ax_32.grid(True, alpha=0.3)

        plt.suptitle(f'Scalars Over Pulse (Step {self._current_step})', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(self._out_dir, 'plots', f'scalars_step{self._current_step}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)





    def fly(self, convergence_threshold=-1.0, save_states=False, graph=False, max_step=11, out='results.json', run_name = 'tmp', skip_bad_init_eqdsks=False):
        r'''! Run Tokamaker-Torax simulation loop until convergence or max_step reached. Saves results to JSON object.
        @pararm convergence_threshold Maximum percent difference between iterations allowed for convergence.
        @param save_states Save intermediate simulation states (for testing).
        @param graph Whether to display psi and profile graphs at each iteration (for testing).
        @param max_step Maximum number of simulation iterations allowed.
        @param skip_bad_init_eqdsks If True, silently skip broken initial gEQDSK files; if False, raise an error when one is found.
        '''

        self._skip_bad_init_eqdsks = skip_bad_init_eqdsks

        dt_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if run_name == 'tmp':
            self._out_dir = os.path.join('./toktox_outputs', 'tmp')
            if os.path.exists(self._out_dir):
                shutil.rmtree(self._out_dir)
            self._log_file = os.path.join(self._out_dir, 'log.txt')
        else:
            dir_name = f'{run_name}_{dt_str}'
            self._out_dir = os.path.join('./toktox_outputs', dir_name)
            self._log_file = os.path.join(self._out_dir, f'{dir_name}_log.txt')
        os.makedirs(os.path.join(self._out_dir, 'equil'), exist_ok=True)
        os.makedirs(os.path.join(self._out_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self._out_dir, 'tm_plots'), exist_ok=True)
        os.makedirs(os.path.join(self._out_dir, 'results'), exist_ok=True)
        with open(self._log_file, 'w'):
            pass

        self._fname_out = os.path.join(self._out_dir, 'results', out)

        err = convergence_threshold + 1.0
        cflux_tx_prev = 0.0
        # records of flux consumption throughout steps
        tm_cflux_psi = []
        tm_cflux_vloop = []
        tx_cflux_psi = []
        tx_cflux_vloop = []

        self._print_out(f'---------------------------------------')
        self._current_step = 1

        while err > convergence_threshold and self._current_step < max_step:
            self._print_out(f'---- Step {self._current_step} ---- \n')
            cflux_tx, cflux_tx_vloop = self._run_transport(graph=graph)
            if save_states:
                self.save_state(os.path.join(self._out_dir, 'results', 'ts_state{}.json'.format(self._current_step)))

            cflux_gs, cflux_gs_vloop = self._run_gs(graph=graph)
            if save_states:
                self.save_state(os.path.join(self._out_dir, 'results', 'gs_state{}.json'.format(self._current_step)))

            self.save_res()

            # record convergence history
            tm_cflux_psi.append(cflux_gs)
            tm_cflux_vloop.append(cflux_gs_vloop)
            tx_cflux_psi.append(cflux_tx)
            tx_cflux_vloop.append(cflux_tx_vloop)

            self._print_out(f'\n ---- Step {self._current_step} results ---- ')

            err = np.abs(cflux_tx - cflux_tx_prev) / cflux_tx_prev
            self._print_out(f"\t(original) TX Convergence error = {err*100.0:.3f} %")
            self._print_out(f'\tDifference Convergence error = {np.abs(cflux_tx - cflux_gs) / (cflux_gs)*100.0:.4f} %')
            self._print_out(f'---------------------------------------\n')

            cflux_tx_prev = cflux_tx

            self._profile_evolution_plot()
            self._scalar_plot()

            self._current_step += 1
        

        if err < convergence_threshold:
            self._print_out(f'Convergence achieved in {self._current_step-1} steps with error = {err*100.0:.3f} %')
        elif self._current_step >= max_step:
            self._print_out(f'Maximum steps {max_step} reached without convergence (last error = {err*100.0:.3f} %)')
