import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import interp1d
import torax
import copy
import json
import os
import shutil

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

from baseconfig import BASE_CONFIG

LCFS_WEIGHT = 100.0
N_PSI = 100
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

class DISMAL:
    '''! Discharge Modeling Algorithm.'''

    def __init__(self, t_init, t_final, eqtimes, g_eqdsk_arr, dt=0.1, times=None, last_surface_factor=0.95, prescribed_currents=False):
        r'''! Initialize the Coupled Grad-Shafranov/Transport Solver Object.
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
        self._dt = dt
        self._prescribed_currents = prescribed_currents
        self._last_surface_factor = last_surface_factor
        self._psi_N = np.linspace(0.0, 1.0, N_PSI) # standardized psi_N grid all values should be mapped onto

        if times is None:
            self._times = eqtimes
        else:
            self._times = sorted(times)

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
        self._state['Ip_tm'] = np.zeros(len(self._times)) # for testing
        self._state['Ip_tx'] = np.zeros(len(self._times)) # for testing
        self._state['Ip_NI_tx'] = np.zeros(len(self._times)) # for testing
        self._state['pax'] = np.zeros(len(self._times))
        self._state['pax_tm'] = np.zeros(len(self._times)) # for testing
        self._state['beta_N_tm'] = np.zeros(len(self._times)) # for testing
        self._state['l_i_tm'] = np.zeros(len(self._times)) # for testing
        self._state['beta_pol'] = np.zeros(len(self._times))
        self._state['vloop_tm'] = np.zeros(len(self._times))
        self._state['vloop_tx'] = np.zeros(len(self._times)) # for testing
        self._state['q95'] = np.zeros(len(self._times))
        self._state['q0'] = np.zeros(len(self._times))
        self._state['q95_tm'] = np.zeros(len(self._times))
        self._state['q0_tm'] = np.zeros(len(self._times))
        self._state['psi_lcfs'] = np.zeros(len(self._times))
        self._state['psi_axis'] = np.zeros(len(self._times))

        self._state['lcfs'] = {}
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
        self._state['ptot'] = {}
        self._state['ffpni_prof'] = {}
        self._state['ffpni_sub_prof'] = {}

        # Outputs from TORAX (already normalized)
        self._state['ffp_prof_tx'] = {}
        self._state['pp_prof_tx'] = {}
        self._state['ffp_prof_tm'] = {}
        self._state['pp_prof_tm'] = {}
        self._state['p_prof_tm'] = {} 
        self._state['f_prof_tm'] = {}
        self._state['test'] = {}

        # geo factors from both codes
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

        # Volume and shape comparisons
        self._state['vol_tm'] = np.zeros(len(self._times))
        self._state['vol_tx'] = {}  # volume profile vs psi
        self._state['vol_tx_lcfs'] = np.zeros(len(self._times))  # volume at LCFS (scalar)
        self._state['vpr_tm'] = {}
        self._state['vpr_tx'] = {}


        self._results['lcfs'] = {}
        self._results['dpsi_lcfs_dt'] = {}
        self._results['vloop_tmaker'] = np.zeros([20, len(self._times)])
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
            self._state['psi_axis'][i] = np.interp(t, self._eqtimes, psi_axis)
            self._state['psi_lcfs'][i] = np.interp(t, self._eqtimes, psi_lcfs)

            # Default Profiles
            self._state['lcfs'][i] = interp_prof(lcfs, t)
            self._state['ffp_prof'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(ffp_prof, t), 'type': 'linterp'}
            self._state['pp_prof'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(pp_prof, t), 'type': 'linterp'}
            # self._state['psi'][i] = np.linspace(g['psimag'], g['psibry'], N_PSI)
            self._state['ffpni_prof'][i] = {'x': [], 'y': [], 'type': 'linterp'}

            # Normalize profiles
            # self._state['ffp_prof'][i]['y'] -= self._state['ffp_prof'][i]['y'][-1]
            # self._state['pp_prof'][i]['y'] -= self._state['pp_prof'][i]['y'][-1]
            self._state['ffp_prof'][i]['y'] /= self._state['ffp_prof'][i]['y'][0]
            self._state['pp_prof'][i]['y'] /= self._state['pp_prof'][i]['y'][0]

            self._state['eta_prof'][i]= {
                'x': self._psi_N.copy(),
                'y': np.zeros(N_PSI),
                'type': 'linterp',
            }
            
            self._state['pres'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(pres_prof, t), 'type': 'linterp'}
            self._state['fpol'][i] = {'x': self._psi_N.copy(), 'y': interp_prof(fpol_prof, t), 'type': 'linterp'}
        
        self._Ip = None
        self._Zeff = None

        self._normalize_to_nbar = False

        self._nbi_heating = {t_init: 0, t_final: 0}
        self._eccd_heating = {t_init: 0, t_final: 0}
        self._eccd_loc = 0.1
        self._nbi_loc = 0.25
        self._ohmic_power = None

        self._evolve_density = True
        self._evolve_current = True
        self._evolve_Ti = True
        self._evolve_Te = True

        self._nbar = None
        self._n_e = None
        self._T_i = None
        self._T_e = None

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
        self._prof_mix_ratio = 1.0
        self._prof_smoothing = False
        self._eqdsk_skip = []
    
    def load_config(self, config):
        r'''! Load a base config for torax.
        @param config Dictionary object to be converted to torax config.
        '''
        self._baseconfig = config
        
    def initialize_gs(self, mesh, weights=None, vsc=None):
        r'''! Initialize GS Solver Object.
        @param mesh Filename of reactor mesh.
        @param vsc Vertical Stability Coil.
        '''
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh(mesh)
        self._gs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._gs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._gs.setup(order = 2, F0 = self._state['R'][0]*self._state['B0'][0])

        self._gs.settings.maxits = 500
        # self._gs.settings.pm = False

        if vsc is not None:
            self._gs.set_coil_vsc({vsc: 1.0})
        # self.set_coil_reg(targets, weights=weights, weight_mult=0.1)

    def set_Ip(self, Ip):
        r'''! Set plasma current (Amps).
        @param ip Plasma current.
        '''
        self._Ip = Ip
    
    def set_density(self, n_e):
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

    def set_pressure(self, p_profiles, n_e_profiles, Ti_Te_ratio=1.0):
        r'''! Set initial T_e and T_i profiles from pressure and density profiles.
        
        Computes temperature from P = n_e * (T_e + T_i) = n_e * T_e * (1 + Ti_Te_ratio).
        Sets both T_e and T_i as time-varying-array dicts for TORAX.
        
        @param p_profiles Pressure profiles in Pa. Dict of {time: {rho: P_Pa, ...}, ...}.
        @param n_e_profiles Density profiles in m^-3. Dict of {time: {rho: n_e, ...}, ...}.
        @param Ti_Te_ratio Ratio of T_i to T_e (default 1.0, i.e. T_i = T_e).
        '''
        eV_to_J = 1.602e-19  # 1 eV in Joules
        keV_to_J = 1.602e-16  # 1 keV in Joules
        
        T_e_profiles = {}
        T_i_profiles = {}
        
        for t in sorted(p_profiles.keys()):
            p_dict = p_profiles[t]
            n_dict = n_e_profiles[t] if t in n_e_profiles else n_e_profiles[max(k for k in n_e_profiles.keys() if k <= t)]
            
            T_e_prof = {}
            T_i_prof = {}
            
            # Get sorted rho values from pressure profile
            rho_vals = sorted(p_dict.keys())
            for rho in rho_vals:
                P_Pa = p_dict[rho]
                # Interpolate n_e at this rho from the density profile
                n_rho_vals = sorted(n_dict.keys())
                n_vals = [n_dict[r] for r in n_rho_vals]
                n_e_at_rho = np.interp(rho, n_rho_vals, n_vals)
                
                if n_e_at_rho > 0 and P_Pa > 0:
                    # P = n_e * T_e * (1 + Ti_Te_ratio) in Joules
                    # T_e [keV] = P / (n_e * (1 + ratio) * keV_to_J)
                    T_e_keV = P_Pa / (n_e_at_rho * (1.0 + Ti_Te_ratio) * keV_to_J)
                    T_i_keV = T_e_keV * Ti_Te_ratio
                else:
                    T_e_keV = 0.01  # minimum temperature
                    T_i_keV = 0.01
                
                T_e_prof[rho] = T_e_keV
                T_i_prof[rho] = T_i_keV
            
            T_e_profiles[t] = T_e_prof
            T_i_profiles[t] = T_i_prof
        
        self._T_e = T_e_profiles
        self._T_i = T_i_profiles
        
        self._print_out(f'set_pressure: Computed T_e and T_i from pressure and density at times {sorted(p_profiles.keys())}')
        for t in sorted(p_profiles.keys()):
            rho_0 = min(T_e_profiles[t].keys())
            rho_1 = max(T_e_profiles[t].keys())
            self._print_out(f'  t={t}: T_e(0)={T_e_profiles[t][rho_0]:.2f} keV, T_e(1)={T_e_profiles[t][rho_1]:.4f} keV, '
                          f'T_i(0)={T_i_profiles[t][rho_0]:.2f} keV, T_i(1)={T_i_profiles[t][rho_1]:.4f} keV')

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
    
    def set_prof_mix_ratio(self, ratio):
        self._prof_mix_ratio = ratio
    
    def set_prof_smoothing(self, smoothing):
        self._prof_smoothing = smoothing
            
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

    def _run_gs(self, step, graph=False):
        r'''! Run the GS solve across n timesteps using TokaMaker.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @param graph Whether to display psi graphs at each iteration (for testing).
        @return Consumed flux.
        '''
        self._print_out(f"Step {step} TokaMaker:")
        # self._print_out(f'\tStep {step}: TM input: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._eqdsk_skip = []
        for i, t in enumerate(self._times):
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)

            Ip_target = abs(self._state['Ip'][i])
            P0_target = abs(self._state['pax'][i])
            
            self._gs.set_targets(Ip=Ip_target, pax=P0_target) # using pax target with j_phi inputs 


            def mix_profiles(prev, curr, ratio=1.0):
                my_prof = {'x': np.zeros(len(curr['x'])), 'y': np.zeros(len(curr['x'])), 'type': 'linterp'}
                for i, x in enumerate(curr['x']):
                    my_prof['x'][i] = x
                    my_prof['y'][i] = (1.0 - ratio) * prev['y'][i] + ratio * curr['y'][i]
                return my_prof

            def make_smooth(x, y):
                spline = make_smoothing_spline(x, y)
                smoothed = spline(x)
                return smoothed

            # For step 1, use original profiles directly (no mixing since _prof_save doesn't exist yet)
            # profile_strategy = "original gEQDSK"
            # if step == 1:
            #     ffp_prof = {'x': self._state['ffp_prof'][i]['x'].copy(), 
            #                'y': self._state['ffp_prof'][i]['y'].copy(), 
            #                'type': self._state['ffp_prof'][i]['type']}
            #     pp_prof = {'x': self._state['pp_prof'][i]['x'].copy(), 
            #               'y': self._state['pp_prof'][i]['y'].copy(), 
            #               'type': self._state['pp_prof'][i]['type']}
            # else:
            #     profile_strategy = f"mixed (ratio={self._prof_mix_ratio:.2f})"
            ffp_prof=mix_profiles(self._state['ffp_prof_save'][i], self._state['ffp_prof'][i], ratio=self._prof_mix_ratio)
            pp_prof=mix_profiles(self._state['pp_prof_save'][i], self._state['pp_prof'][i], ratio=self._prof_mix_ratio)
            
            
            if self._prof_smoothing:
                ffp_prof['y'] = make_smooth(ffp_prof['x'], ffp_prof['y'])
                pp_prof['y'] = make_smooth(pp_prof['x'], pp_prof['y'])

            # Normalize profiles
            # ffp_prof['y'] -= ffp_prof['y'][-1]
            ffp_prof['y'] /= ffp_prof['y'][0]

            # pp_prof['y'] -= pp_prof['y'][-1]
            pp_prof['y'] /= pp_prof['y'][0]

            # ffpni = self._state['ffpni_prof'][i]
            # self._print_out(f"  t={t}: ffpni_prof min={np.min(ffpni['y']):.3e}, max={np.max(ffpni['y']):.3e}")

            self._gs.set_profiles(
                ffp_prof=ffp_prof,
                pp_prof=pp_prof,
                # ffp_prof=self._state['ffp_prof'][i],
                # ffp_prof=self._state['j_tot'][i], 
                # pp_prof=self._state['pp_prof'][i],
                ffp_NI_prof=self._state['ffpni_prof'][i], 
            )

            self._gs.set_resistivity(eta_prof=self._state['eta_prof'][i])

            lcfs = self._state['lcfs'][i]
            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            lcfs_psi_target = self._state['psi_lcfs'][i] # _state in Wb/rad, TM expects Wb/rad (AKA Wb-rad)

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
            eq_name = 'tmp/{:03}.{:03}.eqdsk'.format(step, i)
            
            try:
                err_flag = self._gs.solve()
                print(f'Ip_NI from TX = {self._state["Ip_NI_tx"][i]:.3f} A')
                self._gs_update(i)
                self._gs.save_eqdsk(eq_name,
                                    lcfs_pad=0.01,run_info='TokaMaker EQDSK',
                                    cocos=2)

                solve_succeeded = True
            except Exception as e:
                print(f'\tGS solve failed: {e}')
                self._eqdsk_skip.append(eq_name)
                skip_coil_update = True
                self._print_out(f'TM: Solve failed at t={t}.')
                self._print_tokamaker_inputs(step, i, t, ffp_prof, pp_prof)
                solve_succeeded = False
            
            if solve_succeeded:
                self._profile_plot(step, i, t)

                if graph:
                    fig, ax = plt.subplots(1,1)
                    self._gs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
                    self._gs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
                    ax.plot(self._state['lcfs'][i][:, 0], self._state['lcfs'][i][:, 1], color='r')
                    ax.set_title(f't={self._times[i]}')
                    # plt.savefig(f'tmp/step_{step}_rampdown_equil_{i}.png')
                    plt.savefig('tmp/equil_{:03}.{:03}.png'.format(step, i))
                    plt.close(fig)
                    # plt.show()

            if self._prescribed_currents:
                if i < len(self._times):
                    self.set_coil_reg(i=i+1)
            elif not skip_coil_update:
                coil_targets, _ = self._gs.get_coil_currents()
                self.set_coil_reg(targets=coil_targets)


        # self._print_out(f'Step {step}: TM out: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        consumed_flux = (self._state['psi_lcfs'][-1] - self._state['psi_lcfs'][0]) * 2.0 * np.pi # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need 2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(self._state['vloop_tm'][0:], self._times[0:]) 

        # Diagnostic: print vloop statistics
        vloop_arr = np.array(self._state['vloop_tm'])
        # self._print_out(f"\tTM: vloop: min={vloop_arr.min():.3f}, max={vloop_arr.max():.3f}, mean={vloop_arr.mean():.3f} V")
        # self._print_out(f"\tTM: psi_lcfs: start={self._state['psi_lcfs'][0]:.3f}, end={self._state['psi_lcfs'][-1]:.3f} Wb")
        # self._print_out(f'\tTM: psi_bound consumed flux={consumed_flux:.3f} Wb')
        # self._print_out(f'\tTM: int v_loop consumed flux w/o t=0 ={consumed_flux_integral:.3f} Wb')

        return consumed_flux, consumed_flux_integral
        
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
        
        # Store volume from TokaMaker
        self._state['vol_tm'][i] = eq_stats.get('vol', 0.0)  # will use 0 if not available



        self._state['psi_lcfs'][i] = self._gs.psi_bounds[0] # TM outputs in Wb/rad (AKA Wb-rad) which is how psi_lcfs is stored
        self._state['psi_axis'][i] = self._gs.psi_bounds[1] 

        if 'psi_lcfs_tmaker' not in self._results:
            self._results['psi_lcfs_tmaker'] = {'x': np.zeros(len(self._times)), 'y': np.zeros(len(self._times))}
        self._results['psi_lcfs_tmaker']['x'][i] = self._times[i]
        self._results['psi_lcfs_tmaker']['y'][i] = self._state['psi_lcfs'][i] # stored as Wb/rad

        self._state['vloop_tm'][i] = self._gs.calc_loopvoltage()
        
        # Store TokaMaker pressure profile from get_profiles()
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)

        self._state['ffp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_fp_prof*tm_f_prof), 'type': 'linterp'}
        self._state['pp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_pp_prof), 'type': 'linterp'}
        self._state['p_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_p_prof), 'type': 'linterp'}
        self._state['f_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_f_prof), 'type': 'linterp'}

        # pull geo profiles
        psi_geo, q_tm, geo, vol_geo, vpr_geo, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02, compute_geo=False)
        
        # Extract q0 and q95 from TokaMaker q profile
        self._state['q0_tm'][i] = q_tm[0] if len(q_tm) > 0 else np.nan
        self._state['q95_tm'][i] = np.interp(0.95, psi_geo, q_tm) if len(psi_geo) > 0 and len(q_tm) > 0 else np.nan
        
        # R_avg = np.interp(self._psi_N, psi_geo, np.array(geo[0]))
        # R_inv_avg = np.interp(self._psi_N, psi_geo, np.array(geo[1]))

        self._state['R_avg_tm'][i] =     {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[0])), 'type': 'linterp'}
        self._state['R_inv_avg_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(geo[1])), 'type': 'linterp'}
        
        # Store vpr_tm (volume derivative) from get_q return values
        if vpr_geo is not None and len(vpr_geo) > 0:
            self._state['vpr_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, psi_geo, np.array(vpr_geo)), 'type': 'linterp'}
        
        # Update Results
        coils, _ = self._gs.get_coil_currents()
        if 'COIL' not in self._results:
            self._results['COIL'] = {coil: {} for coil in coils}
        for coil, current in coils.items():
            if coil not in self._results['COIL']:
                self._results['COIL'][coil] = {}
            self._results['COIL'][coil][self._times[i]] = current * 1.0 # TODO: handle nturns > 1

    def _test_eqdsk(self, eqdsk):
            myconfig = copy.deepcopy(BASE_CONFIG)
            myconfig['geometry'] = {
                'geometry_type': 'eqdsk',
                # 'geometry_directory': '/Users/johnl/Desktop/discharge-model', 
                'geometry_directory': os.getcwd(),
                'last_surface_factor': self._last_surface_factor,
                'Ip_from_parameters': False,
                'geometry_file': eqdsk,
                'cocos': 2,
            }
            try:
                _ = torax.ToraxConfig.from_dict(myconfig)
                return True
            except:
                return False

    # def _pull_torax_onto_psi(self, data_tree, var_name, time, load_into_state='state', normalize=False, profile_type='linterp'):
    #     r'''! Load TORAX variable onto psi_norm grid.
    #     @param data_tree TORAX output data tree.
    #     @param var_name Name of variable (e.g., 'T_i', 'j_ohmic', 'FFprime').
    #     @param time Time value to extract.
    #     @param load_into_state If 'state' loads into '_state', elif 'results' loads into '_results', elif None, return (psi, data).
    #     @param normalize If True, normalize profile: subtract edge value, divide by core value (for FFprime, pprime).
    #     @param profile_type Type key for returned dict: 'linterp' or 'jphi-linterp'. Default is 'linterp'.
    #     '''
        
    #     # Extract variable from profiles
    #     var = getattr(data_tree.profiles, var_name)
    #     var_data = var.sel(time=time, method='nearest').to_numpy()
        
    #     # Automatically detect which rho coordinate this variable uses
    #     if 'rho_cell_norm' in var.coords:
    #         grid = 'rho_cell_norm'
    #     elif 'rho_face_norm' in var.coords:
    #         grid = 'rho_face_norm'
    #     elif 'rho_norm' in var.coords:
    #         grid = 'rho_norm'
    #     else:
    #         raise ValueError(f"Variable {var_name} does not have a recognized rho coordinate")

    #     psi_norm_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
    #     psi_rho_norm = data_tree.profiles.psi.sel(time=time, method='nearest').to_numpy()
    #     psi_norm_rho_norm=(psi_rho_norm - psi_rho_norm[0])/(psi_rho_norm[-1] - psi_rho_norm[0])
    #     psi_norm_rho_norm[1] = (psi_norm_face[0]  + psi_norm_face[1])/2.0


    #     # Convert psi to same grid as variable
    #     if grid == 'rho_cell_norm':     # only cell centers (excludes rho=0 and rho=1)
    #         psi_on_grid = psi_norm_rho_norm[1:-1]
    #     elif grid == 'rho_face_norm':   # only cell faces (includes rho=0 and rho=1)
    #         psi_on_grid = psi_norm_face
    #     elif grid == 'rho_norm':        # includes both cell centers and faces (includes rho=0 and rho=1)
    #         psi_on_grid = psi_norm_rho_norm
        
    #     # Interpolate onto uniform psi grid
    #     data_on_psi = interp1d(psi_on_grid, var_data, kind='linear',
    #                         fill_value='extrapolate', bounds_error=False)(self._psi_N)
        
    #     # Normalize if requested
    #     if normalize:
    #         # data_on_psi -= data_on_psi[-1]  # Subtract edge value
    #         data_on_psi /= data_on_psi[0]   # Divide by core value
        
    #     if load_into_state == 'state':
    #         return {'x': self._psi_N.copy(), 'y': data_on_psi.copy(), 'type': profile_type}
    #     elif load_into_state == 'results':
    #         return {'x': self._psi_N.copy(), 'y': data_on_psi.copy(), 'type': profile_type}
    #     else:
    #         return data_on_psi

    # 2026-02-09 
    def _pull_torax_onto_psi(self, data_tree, var_name, time, load_into_state='state', normalize=False, profile_type='linterp'):
        r'''! Load TORAX variable onto psi_norm grid.
        @param data_tree TORAX output data tree.
        @param var_name Name of variable (e.g., 'T_i', 'j_ohmic', 'FFprime').
        @param time Time value to extract.
        @param load_into_state If 'state' loads into '_state', elif 'results' loads into '_results', elif None, return (psi, data).
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
                            fill_value='extrapolate', bounds_error=False)(self._psi_N)

        
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
        elif load_into_state == 'results':
            return {'x': self._psi_N.copy(), 'y': data_on_psi.copy(), 'type': profile_type}
        else:
            return data_on_psi
        
    def _get_torax_config(self, step):
        r'''! Generate config object for Torax simulation. Modifies BASE_CONFIG based on current simulation state.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @return Torax config object.
        '''
        # self._print_out(f'Step {step}: TX input: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')
        # self._print_out(f'\tTX input: pax: min = {np.min(self._state["pax"]):.1f}, max = {np.max(self._state["pax"]):.1f} Pa')
        # self._print_out(f'\tTX input: pax at t=0: {self._state["pax"][0]:.1f} Pa, t=end: {self._state["pax"][-1]:.1f} Pa')

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
            # 'geometry_directory': '/Users/johnl/Desktop/discharge-model', 
            'geometry_directory': os.getcwd(),
            'last_surface_factor': self._last_surface_factor,
            'n_surfaces': 100,
            'Ip_from_parameters': True,
            'geometry_configs': {
                t: {'geometry_file': self._init_files[i], 'cocos': 2} for i, t in enumerate(self._eqtimes)
            },
            'n_rho': 25, 
        }
        if step > 1:
            safe_times = []
            safe_eqdsk = []
            for i, t in enumerate(self._times):
                eqdsk = 'tmp/{:03}.{:03}.eqdsk'.format(step - 1, i)
                if eqdsk in self._eqdsk_skip:
                    print('Skipping failed solver step.')
                    continue
                if self._test_eqdsk(eqdsk):
                    safe_times.append(t)
                    safe_eqdsk.append(eqdsk)
                else:
                    print('Deleting invalid eqdsk file.')
            myconfig['geometry']['geometry_configs'] = {
                t: {'geometry_file': safe_eqdsk[i], 'cocos': 2} for i, t in enumerate(safe_times)
            }

        myconfig['profile_conditions']['Ip'] = {
            t: abs(self._state['Ip'][i]) for i, t in enumerate(self._times)
        }
        myconfig['profile_conditions']['psi'] = { # TORAX takes in Wb, psi_lcfs stored as Wb/rad (AKA Wb-rad) so needs *2pi factor
            t: {0.0: self._state['psi_axis'][i] * 2.0 * np.pi, 1.0: self._state['psi_lcfs'][i]* 2.0 * np.pi} for i, t in enumerate(self._times) 
        }

        if self._Ip:
            myconfig['profile_conditions']['Ip'] = self._Ip
        
        if self._n_e:
            myconfig['profile_conditions']['n_e'] = self._n_e
        
        if self._T_e:
            myconfig['profile_conditions']['T_e'] = self._T_e
        
        if self._T_i:
            myconfig['profile_conditions']['T_i'] = self._T_i
        
        if self._Zeff:
            myconfig['plasma_composition']['Z_eff'] = self._Zeff
        
        myconfig['sources']['ecrh']['P_total'] = self._eccd_heating
        myconfig['sources']['ecrh']['gaussian_location'] = self._eccd_loc

        if self._ohmic_power:
            myconfig['sources']['ohmic']['mode'] = 'PRESCRIBED'
            myconfig['sources']['ohmic']['prescribed_values'] = self._ohmic_power

        nbi_times, nbi_pow = zip(*self._nbi_heating.items())
        myconfig['sources']['generic_heat']['P_total'] = (nbi_times, nbi_pow)
        myconfig['sources']['generic_heat']['gaussian_location'] = self._nbi_loc
        myconfig['sources']['generic_current']['I_generic'] = (nbi_times, _NBI_W_TO_MA * np.array(nbi_pow))
        myconfig['sources']['generic_current']['gaussian_location'] = self._nbi_loc

        if self._T_i_ped:
            myconfig['pedestal']['T_i_ped'] = self._T_i_ped
        if self._T_e_ped:
            myconfig['pedestal']['T_e_ped'] = self._T_e_ped
        

        if self._n_e_ped:
            myconfig['pedestal']['n_e_ped_is_fGW'] = False
            myconfig['pedestal']['n_e_ped'] = self._n_e_ped
        
        myconfig['pedestal']['set_pedestal'] = self._set_pedestal # TODO not working
        myconfig['pedestal']['rho_norm_ped_top'] = self._ped_top
        
        myconfig['profile_conditions']['normalize_n_e_to_nbar'] = self._normalize_to_nbar
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

        with open('torax_config.json', 'w') as json_file:
            json.dump(myconfig, json_file, indent=4, cls=MyEncoder)
        torax_config = torax.ToraxConfig.from_dict(myconfig)
        return torax_config

    def _run_transport(self, step, graph=False):
        r'''! Run the Torax simulation.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @param graph Whether to display profiles at each iteration (for testing).
        @return Consumed flux.
        '''
        myconfig = self._get_torax_config(step)
        data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)

        # save data_tree object
        # data_tree_name = 'tmp/test.nc'
        # data_tree.to_netcdf(data_tree_name)

        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        v_loops = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._transport_update(step, i, data_tree)
            v_loops[i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
        # self._print_out(f'Step {step}: TX output (w/ /2pi): psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._res_update(data_tree)

        consumed_flux = 2.0 * np.pi * (self._state['psi_lcfs'][-1] - self._state['psi_lcfs'][0]) # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need *2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(v_loops[0:], self._times[0:]) 
        self._print_out(f"Step {step} TORAX:")
        # self._print_out(f"\tTX: vloop: min={v_loops.min():.3f}, max={v_loops.max():.3f}, mean={v_loops.mean():.3f} V")
        # self._print_out(f"\tTX: psi_lcfs: start={self._state['psi_lcfs'][0]:.3f}, end={self._state['psi_lcfs'][-1]:.3f} Wb/rad")
        # self._print_out(f'\tTX: psi_bound consumed flux={consumed_flux:.3f} Wb')
        # self._print_out(f'\tTX: int v_loop consumed flux w/o t=0 ={consumed_flux_integral:.3f} Wb')
        return consumed_flux, consumed_flux_integral

    def _transport_update(self, step, i, data_tree, smooth=True):
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
        
        self._state['beta_pol'][i] = data_tree.scalars.beta_pol.sel(time=t, method='nearest')
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

        self._state['vloop_tx'][i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
        


        # calculate ffp_ni profile
        self._state['j_tot'][i] =            self._pull_torax_onto_psi(data_tree, 'j_total',          t, load_into_state='state', profile_type='jphi-linterp')
        # self._state['j_parallel_total'][i] = self._pull_torax_onto_psi(data_tree, 'j_parallel_total', t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ohmic'][i] =          self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        # self._state['j_ohmic_tx'][i] =       self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ni'][i] =          self._pull_torax_onto_psi(data_tree, 'j_non_inductive',  t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_bootstrap'][i] =      self._pull_torax_onto_psi(data_tree, 'j_bootstrap',      t, load_into_state='state', profile_type='jphi-linterp')
        
        # DIAGNOSTIC: Compare j_ni from TORAX vs subtraction method
        j_tot_vals = self._state['j_tot'][i]['y']
        j_ohmic_vals = self._state['j_ohmic'][i]['y']
        j_ni_vals = self._state['j_ni'][i]['y']
        j_ni_subtraction = j_tot_vals - j_ohmic_vals
        


        # if step == 1:
        #     ffp_ni = np.zeros_like(self._psi_N)
        # else:

        self._state['R_inv_avg_tx'][i] = self._pull_torax_onto_psi(data_tree, 'gm9', t, load_into_state='state', normalize=False)

        ffp_ni = self._calc_ffp_ni(i, data_tree)

        self._state['ffpni_prof'][i] = {'x': self._psi_N.copy(), 'y': ffp_ni.copy(), 'type': 'linterp'} 
        # self._state['ffpni_prof'][i]['y'] *= -2.0 * np.pi  # im calculating ffp_ni, don't need to convert from torax units
        

        self._state['T_i'][i] = self._pull_torax_onto_psi(data_tree, 'T_i', t, load_into_state='state', normalize=False)
        self._state['T_e'][i] = self._pull_torax_onto_psi(data_tree, 'T_e', t, load_into_state='state', normalize=False)
        self._state['n_i'][i] = self._pull_torax_onto_psi(data_tree, 'n_i', t, load_into_state='state', normalize=False)
        self._state['n_e'][i] = self._pull_torax_onto_psi(data_tree, 'n_e', t, load_into_state='state', normalize=False)

        self._state['ptot'][i] = self._pull_torax_onto_psi(data_tree, 'pressure_thermal_total', t, load_into_state='state', normalize=False)

        # Get conductivity and convert to resistivity (eta = 1/sigma)
        conductivity = self._pull_torax_onto_psi(data_tree, 'sigma_parallel', t, load_into_state=None, normalize=False)
        self._state['eta_prof'][i] = {
            'x': self._psi_N.copy(),
            'y': 1.0 / conductivity,
            'type': 'linterp',
        }

        self._state['psi_lcfs'][i] = data_tree.profiles.psi.sel(time=t, rho_norm=1.0, method='nearest').item() / (2.0 * np.pi) # TORAX outputs psi_lcfs in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi
        self._state['psi_axis'][i] = data_tree.profiles.psi.sel(time=t, rho_norm=0.0, method='nearest').item() / (2.0 * np.pi) # TORAX outputs psi_lcfs in units of Wb, stored as Wb/rad (AKA Wb-rad), so needs 1/2pi

        # Pull volume and volume derivative from TORAX
        self._state['vol_tx_lcfs'][i] = data_tree.profiles.volume.sel(time=t, rho_norm=1.0, method='nearest').item()
        self._state['vol_tx'][i] = self._pull_torax_onto_psi(data_tree, 'volume', t, load_into_state='state', normalize=False)
        self._state['vpr_tx'][i] = self._pull_torax_onto_psi(data_tree, 'vpr', t, load_into_state='state', normalize=False)



    def _res_update(self, data_tree):

        self._results['t_res'] = self._times

        for t in self._times:
            self._results['T_e'][t] = self._pull_torax_onto_psi(data_tree, 'T_e', t, load_into_state='results', normalize=False)
            self._results['T_i'][t] = self._pull_torax_onto_psi(data_tree, 'T_i', t, load_into_state='results', normalize=False)
            self._results['n_e'][t] = self._pull_torax_onto_psi(data_tree, 'n_e', t, load_into_state='results', normalize=False)
            self._results['q'][t] =   self._pull_torax_onto_psi(data_tree, 'q', t, load_into_state='results', normalize=False)

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
        with open('convergence_history.txt', 'a') as f:
            print(str, file=f)
    
    def _print_tokamaker_inputs(self, step, i, t, ffp_prof, pp_prof):
        r'''! Print all TokaMaker inputs when solver fails.
        @param step Current iteration step
        @param i Time index
        @param t Current time
        @param ffp_prof FF' profile used (after mixing/normalization)
        @param pp_prof p' profile used (after mixing/normalization)
        '''

        g = read_eqdsk(self._init_files[i])
        self._print_out(f"\n===== TM SOLVER FAILURE: Step {step}, t={t:.3f}s (idx {i}/{len(self._times)-1}) =====")
        
        # TARGETS & GEOMETRY
        self._print_out(f"TARGETS: Ip={abs(self._state['Ip'][i]):.4e}A, pax={abs(self._state['pax'][i]):.4e}Pa" + 
                       (f", Ip_NI(TX)={self._state['Ip_NI_tx'][i]:.4e}A" if step > 1 else ""))
        self._print_out(f"Init EQDSK: Ip={abs(g['ip']):.4e}A, pax={g['pres'][0]:.4e}Pa")
        self._print_out(f"GEOMETRY: R={self._state['R'][i]:.3f}m, Z={self._state['Z'][i]:.3f}m, a={self._state['a'][i]:.3f}m, κ={self._state['kappa'][i]:.3f}, δ={self._state['delta'][i]:.3f}, B0={self._state['B0'][i]:.2f}T")
        
        # FLUX CONSTRAINTS
        lcfs_R = self._state['lcfs'][i][:, 0]
        lcfs_Z = self._state['lcfs'][i][:, 1]
        self._print_out(f"TORAX psi: psi_lcfs={self._state['psi_lcfs'][i]:.4e}Wb/rad, psi_axis={self._state['psi_axis'][i]:.4e}Wb/rad")
        self._print_out(f'Init EQDSK psi: psi_lcfs={abs(g["psibry"]):.4e}Wb/rad, psi_axis={abs(g["psimag"]):.4e}Wb/rad')
        
        # SCALAR QUANTITIES FROM TORAX (if step > 1)
        if step > 1:
            self._print_out(f"TORAX scalars: beta_pol={self._state['beta_pol'][i]:.4f}, q95={self._state['q95'][i]:.3f}, q0={self._state['q0'][i]:.3f}, vloop={self._state['vloop_tx'][i]:.3f}V")
            if self._state['vol_tx_lcfs'][i] > 0:
                self._print_out(f"TORAX volume: vol_lcfs={self._state['vol_tx_lcfs'][i]:.3f}m³")
        # self._print_out(f"LCFS: {len(self._state['lcfs'][i])} pts, R=[{lcfs_R.min():.3f},{lcfs_R.max():.3f}]m, Z=[{lcfs_Z.min():.3f},{lcfs_Z.max():.3f}]m, weight={LCFS_WEIGHT}")
        
        # PROFILES (compact format)
        # self._print_out(f"PROFILES (mix_ratio={self._prof_mix_ratio:.2f}, smoothing={self._prof_smoothing}):")
        # self._print_out(f"  ffp_norm: [{np.min(ffp_prof['y']):.3e}, {np.max(ffp_prof['y']):.3e}], ψ=0:{ffp_prof['y'][0]:.3e}, ψ=1:{ffp_prof['y'][-1]:.3e}")
        # self._print_out(f"  pp_norm:  [{np.min(pp_prof['y']):.3e}, {np.max(pp_prof['y']):.3e}], ψ=0:{pp_prof['y'][0]:.3e}, ψ=1:{pp_prof['y'][-1]:.3e}")
        
        # ffpni = self._state['ffpni_prof'][i]['y']
        # eta = self._state['eta_prof'][i]['y']
        # self._print_out(f"  ffpni:    [{np.min(ffpni):.3e}, {np.max(ffpni):.3e}], ψ=0:{ffpni[0]:.3e}, ψ=1:{ffpni[-1]:.3e}")
        # self._print_out(f"  eta(Ωm):  [{np.min(eta):.3e}, {np.max(eta):.3e}], ψ=0:{eta[0]:.3e}, ψ=1:{eta[-1]:.3e}")
        
        # Current densities (if available)
        # if step > 1:
        #     j_tot = self._state['j_tot'][i]['y']
        #     j_ohm = self._state['j_ohmic'][i]['y']
        #     j_ni = self._state['j_ni'][i]['y']
        #     self._print_out(f"CURRENTS: j_tot=[{np.min(j_tot):.3e},{np.max(j_tot):.3e}], j_ohm=[{np.min(j_ohm):.3e},{np.max(j_ohm):.3e}], j_ni=[{np.min(j_ni):.3e},{np.max(j_ni):.3e}] A/m²")
        
        # Previous step comparison
        # if step > 1 and i in self._state['ffp_prof_save']:
        #     ffp_save = self._state['ffp_prof_save'][i]['y']
        #     pp_save = self._state['pp_prof_save'][i]['y']
        #     self._print_out(f"PREV STEP: ffp=[{np.min(ffp_save):.3e},{np.max(ffp_save):.3e}], pp=[{np.min(pp_save):.3e},{np.max(pp_save):.3e}]")
        
        # Initial EQDSK info
        # if step == 1:
        #     eqdsk_idx = min(i, len(self._init_files) - 1)
        #     self._print_out(f"SEED EQDSK: {self._init_files[eqdsk_idx]}")
        
        self._print_out(f"===============================================================================\n")

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

    def _normalize_profile(self, profile):
        r'''! Normalize a profile to range [0, 1].
        @param profile Profile dictionary with 'x' and 'y' keys.
        @return Normalized profile dictionary.
        '''
        y = copy.deepcopy(profile['y'])
        # y -= y[-1]
        y /= y[0]
        return {'x': profile['x'], 'y': y, 'type': profile['type']}

    def _profile_plot(self, step, i, t): 
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
        plt.suptitle(f'Step {step} - Time index {i}/{len(self._times)-1} - t = {t:.1f} s', fontsize=14)

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
        
        # (2,0): R_avg comparison
        ax_r_avg = axes[2,0]
        ax_r_avg.set_title('<R> comparison')
        if i in self._state['R_avg_tm']:
            ax_r_avg.plot(self._state['R_avg_tm'][i]['x'], self._state['R_avg_tm'][i]['y'], 'r-', label='R_avg TM', linewidth=2)
        if i in self._state.get('R_avg_tx', {}):
            ax_r_avg.plot(self._state['R_avg_tx'][i]['x'], self._state['R_avg_tx'][i]['y'], 'b--', label='R_avg TX', linewidth=2)
        ax_r_avg.set_xlabel(r'$\hat{\psi}$')
        ax_r_avg.set_ylabel('<R> [m]')
        ax_r_avg.legend(fontsize=9)
        ax_r_avg.grid(True, alpha=0.3)

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
        vol_tm_lcfs = self._state['vol_tm'][i]
        vol_tx_lcfs = self._state['vol_tx_lcfs'][i]
        # Plot volume profile from TORAX
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
        
        psi_geo, q_tm, geo, vol_geo, vpr_geo, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02, compute_geo=False)

        # (3,0): q-profile panel (TORAX q if available)
        ax_q = axes[3,0]
        ax_q.set_title('q profile')
        if 'q' in self._results and self._times[i] in self._results['q']:
            q_prof = self._results['q'][self._times[i]]
            ax_q.plot(q_prof['x'], q_prof['y'], 'b-', linewidth=2, label = 'TX')
            ax_q.plot(psi_geo, q_tm, 'r--', label='TM', linewidth=2)  
            ax_q.set_xlabel(r'$\hat{\psi}$')
            ax_q.set_ylabel('q')
            ax_q.legend(fontsize=9)  
        else:
            ax_q.text(0.5, 0.5, 'No q profile', ha='center', va='center')
            ax_q.set_xticks([])
            ax_q.set_yticks([])

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
        plt.savefig(f'tmp/profile_plot_{step:03}.{i:03}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _profile_evolution_plot(self, step):
        r'''! Plot multiple profiles over time with color representing time.
        Creates a single figure with subplots for ne, Te, ni, Ti, p (TORAX), p (TokaMaker), q.
        Each subplot shows profiles at all time points with color mapped to time using plasma colormap.
        @param step The step number (for labeling the saved figure).
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
        fig.suptitle(f'Profile Evolution Over Time (Step {step})', fontsize=14)
        
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
        
        plt.savefig(f'tmp/profile_evolution_step{step}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _scalar_plot(self, step):
        r'''! Plot a grid of time-series scalars across the entire pulse for a given step.
        Produces a 4x3 grid with diagnostics including Ip, psi (TM/TORAX), V_loop comparison,
        Q, n_e_line_avg, T_e_line_avg, P_ohmic, beta_N, l_i, q95, pax.
        @param step Step number used for filename/label.
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

        # (0,1): psi_lcfs (TM & TX)
        ax_01 = axes[0,1]
        ax_01.set_title('psi_lcfs (TM & TX)')
        if 'psi_lcfs_tmaker' in self._results:
            ax_01.plot(self._results['psi_lcfs_tmaker']['x'], self._results['psi_lcfs_tmaker']['y'], '-', label='TokaMaker')
        if 'psi_lcfs_torax' in self._results:
            ax_01.plot(self._results['psi_lcfs_torax']['x'], self._results['psi_lcfs_torax']['y'], '--', label='TORAX')
        ax_01.set_xlabel('Time [s]')
        ax_01.set_ylabel(r'$\psi_{lcfs}$ [Wb/rad]')
        ax_01.legend(fontsize=8)
        ax_01.grid(True, alpha=0.3)

        # (0,2): V_loop comparison with ratio
        ax_02 = axes[0,2]
        ax_02.set_title('V_loop (TM vs TX) [V]')
        ax_02.plot(self._times, self._state['vloop_tm'], '-o', markersize=3, label='TokaMaker')
        if 'v_loop_lcfs' in self._results:
            rx = self._results['v_loop_lcfs']['x']
            ry = self._results['v_loop_lcfs']['y']
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
            # Print average vloop and ratio between 300 and 400 seconds
            mask = (ratio_times >= 300) & (ratio_times <= 400)
            if np.any(mask):
                avg_ratio = np.nanmean(ratio[mask])
                # Get averages for both codes
                mask_tm = (np.array(self._times) >= 300) & (np.array(self._times) <= 400)
                mask_tx = (np.array(rx) >= 300) & (np.array(rx) <= 400)
                avg_vloop_tm = np.mean(tm_vloop[mask_tm]) if np.any(mask_tm) else np.nan
                avg_vloop_tx = np.mean(tx_vloop[mask_tx]) if np.any(mask_tx) else np.nan
                self._print_out(f"V_loop (300-400s): TokaMaker avg={avg_vloop_tm:.3f} V, TORAX avg={avg_vloop_tx:.3f} V, ratio={avg_ratio:.4f}")
                ax2_02.text(0.5, 0.9, f'Avg ratio (300-400s): {avg_ratio:.4f}', transform=ax2_02.transAxes, color='g', fontsize=8, ha='center')
        ax_02.set_xlabel('Time [s]')
        ax_02.grid(True, alpha=0.3)
        ax_02.legend(fontsize=8, loc='upper left')

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

        # (3,2): B0
        ax_32 = axes[3,2]
        ax_32.set_title('B0 [T]')
        ax_32.plot(self._times, self._state['B0'], '-o', markersize=3, label='B0')
        ax_32.set_xlabel('Time [s]')
        ax_32.set_ylabel('B0 [T]')
        ax_32.legend(fontsize=8)
        ax_32.grid(True, alpha=0.3)

        plt.suptitle(f'Scalars Over Pulse (Step {step})', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'tmp/scalars_step{step}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)





    def fly(self, convergence_threshold=-1.0, save_states=False, graph=False, max_step=5, out='res.json'):
        r'''! Run Tokamaker-Torax simulation loop until convergence or max_step reached. Saves results to JSON object.
        @pararm convergence_threshold Maximum percent difference between iterations allowed for convergence.
        @param save_states Save intermediate simulation states (for testing).
        @param graph Whether to display psi and profile graphs at each iteration (for testing).
        @param max_step Maximum number of simulation iterations allowed.
        '''
        # del_tmp = input('Delete temporary storage? [y/n] ')
        # if del_tmp != 'y':
            # quit()
        with open('convergence_history.txt', 'w'):
            pass
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')

        self._fname_out = out

        err = convergence_threshold + 1.0
        step = 0
        cflux_tx_prev = 0.0
        # records of flux consumption throughout steps
        tm_cflux_psi = []
        tm_cflux_vloop = []
        tx_cflux_psi = []
        tx_cflux_vloop = []

        # Step 0: Initialize state from seed equilibria without running TokaMaker
        # This populates psi_lcfs, psi_axis, Ip, vloop from EQDSK files so that we can calculate non inductive FF' using <R> and <1/R> from tokamaker using torax step 1 profiles for gs step 1
        # self._print_out("Step 0: Loading seed equilibria")
        # for i, t in enumerate(self._times):
        #     eqdsk_idx = min(i, len(self._init_files) - 1)
        #     g = read_eqdsk(self._init_files[eqdsk_idx])
            
        #     # Update state from EQDSK (vloop=0 for static equilibrium)
        #     self._state['psi_lcfs'][i] = abs(g['psibry']) # EQDSK psi is in Wb/rad, _state is also Wb/rad
        #     self._print_out(f'Time {t:.3f} s: Loaded EQDSK {self._init_files[eqdsk_idx]} with psi_lcfs = {self._state["psi_lcfs"][i]:.6f} Wb/rad')
        #     self._state['psi_axis'][i] = abs(g['psimag'])
        #     self._state['Ip'][i] = abs(g['ip'])
        #     self._state['vloop_tm'][i] = 0.0  # No vloop from static EQDSK

        # self._print_out(f'Step {step}: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._print_out(f'---------------------------------------')
        step = 1

        while err > convergence_threshold and step < max_step:
            self._print_out(f'---- Step {step} ---- \n')
            cflux_tx, cflux_tx_vloop = self._run_transport(step, graph=graph)
            if save_states:
                self.save_state('tmp/ts_state{}.json'.format(step))

            cflux_gs, cflux_gs_vloop = self._run_gs(step, graph=graph)
            if save_states:
                self.save_state('tmp/gs_state{}.json'.format(step))

            self.save_res()

            # record convergence history
            tm_cflux_psi.append(cflux_gs)
            tm_cflux_vloop.append(cflux_gs_vloop)
            tx_cflux_psi.append(cflux_tx)
            tx_cflux_vloop.append(cflux_tx_vloop)
            vloop_tm = self._state['vloop_tm'].copy()
            vloop_tx = self._results['v_loop_lcfs']['y'].copy()

            self._print_out(f'\n ---- Step {step} results ---- ')

            err = np.abs(cflux_tx - cflux_tx_prev) / cflux_tx_prev
            self._print_out(f"\t(original) TX Convergence error = {err*100.0:.3f} %")
            self._print_out(f'\tDifference Convergence error = {np.abs(cflux_tx - cflux_gs) / (cflux_gs)*100.0:.4f} %')
            self._print_out(f'---------------------------------------\n')

            cflux_tx_prev = cflux_tx


            # self._print_out(f'self._times =')
            # temp_text = 'yesffpni'

            # self._print_out(f'tm_time_{temp_text}={repr(self._times)}')
            # # self._print_out(f'self._state[\'vloop\'] =')
            # self._print_out(f'tm_vloop_{temp_text}=np.{repr(self._state["vloop"])}')

            # # self._print_out(f'step {step} self._results["v_loop_lcfs"]["x"] =')
            # self._print_out(f'tx_time_{temp_text}={repr(self._results["v_loop_lcfs"]["x"])}')
            # # self._print_out(f'self._results["v_loop_lcfs"]["y"] =')
            # self._print_out(f'tx_vloop_{temp_text}=np.{repr(self._results["v_loop_lcfs"]["y"])}')
            # plt.close('all')



            self._profile_evolution_plot(step)
            self._scalar_plot(step)



            step += 1
        

        # from save_tokamaker_inputs import save_tokamaker_inputs
        # save_tokamaker_inputs(self, step=step-1, fname='2026-02-09_tokamaker_test_inputs.npz')


        if err < convergence_threshold:
            self._print_out(f'Convergence achieved in {step-1} steps with error = {err*100.0:.3f} %')
        elif step >= max_step:
            self._print_out(f'Maximum steps {max_step} reached without convergence (last error = {err*100.0:.3f} %)')

        # plot final convergence history
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].set_title('Flux consumption convergence history (psi difference)')
        axes[0].plot(range(1, len(tm_cflux_psi)+1), tm_cflux_psi, 'r-o', label='TokaMaker psi')
        axes[0].plot(range(1, len(tx_cflux_psi)+1), tx_cflux_psi, 'b-o', label='TORAX psi')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Flux consumption [Wb]')
        axes[0].legend(fontsize=8)
        axes[1].set_title('Flux consumption convergence history (V_loop integral)')
        axes[1].plot(range(1, len(tm_cflux_vloop)+1), tm_cflux_vloop, 'r-o', label='TokaMaker V_loop')
        axes[1].plot(range(1, len(tx_cflux_vloop)+1), tx_cflux_vloop, 'b-o', label='TORAX V_loop')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Flux consumption [Wb]')
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('tmp/A_convergence_history.png', dpi=150, bbox_inches='tight')
        plt.close('all')

        # Plot profile evolution over time for the last step


