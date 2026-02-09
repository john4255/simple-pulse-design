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
        self._state['beta_pol'] = np.zeros(len(self._times))
        self._state['vloop'] = np.zeros(len(self._times))
        self._state['q95'] = np.zeros(len(self._times))
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
        self._state['j_ohmic_tx'] = {}
        # self._state['j_ni_tx'] = {}
        self._state['f_NI'] = np.zeros(len(self._times))


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

        self._gs.settings.maxits = 1000
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

    def set_Zeff(self, Zeff):
        r'''! Set plasma effective charge.
        @param z_eff Effective charge.
        '''
        self._Zeff = Zeff
    
    def set_nbar(self, nbar):
        r'''! Set line averaged density over time.
        @param nbar Density (m^-3).
        '''
        self._nbar = nbar
    
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

    def set_pedestal(self, T_i_ped=None, T_e_ped=None, n_e_ped=None, ped_top=0.95):
        r'''! Set pedestals for ion and electron temperatures.
        @pararm T_i_ped Ion temperature pedestal (dictionary of temperature at times).
        @pararm T_e_ped Electron temperature pedestal (dictionary of temperature at times).
        '''
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
            self._state['vloop'][i] = vloop[i]
    
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
        self._print_out(f'\tStep {step}: TM input: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._eqdsk_skip = []
        for i, t in enumerate(self._times):
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)

            Ip_target = abs(self._state['Ip'][i])
            P0_target = abs(self._state['pax'][i])
            if self._state['beta_pol'][i] != 0: # always true because TX runs first now
                print('Using beta_p...')
                Ip_ratio=(1.0/self._state['beta_pol'][i] - 1.0)
                self._gs.set_targets(Ip=Ip_target, Ip_ratio=Ip_ratio, pax=P0_target)
            else:
                self._gs.set_targets(Ip=Ip_target, pax=P0_target)
            
            self._gs.set_targets(Ip=Ip_target*(15/15.4), pax=P0_target) # using pax target with j_phi inputs, error code told me # TODO remove 15/15.4 factor


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

            # For step 1, use original profiles directly (no mixing since _prof_save doesn't exist yet) #TODO might not be necessary anymore
            if step == 1:
                ffp_prof = {'x': self._state['ffp_prof'][i]['x'].copy(), 
                           'y': self._state['ffp_prof'][i]['y'].copy(), 
                           'type': self._state['ffp_prof'][i]['type']}
                pp_prof = {'x': self._state['pp_prof'][i]['x'].copy(), 
                          'y': self._state['pp_prof'][i]['y'].copy(), 
                          'type': self._state['pp_prof'][i]['type']}
            else:
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
                # ffp_prof=self._state['ffp_prof'][i],
                ffp_prof=self._state['j_tot'][i], 
                pp_prof=self._state['pp_prof'][i],
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
                # self._print_out(f'Solve completed at t={t}.')
                solve_succeeded = True
            except Exception as e:
                print(f'\tGS solve failed: {e}')
                self._eqdsk_skip.append(eq_name)
                skip_coil_update = True
                self._print_out(f'TM: Solve failed at t={t}.')
                solve_succeeded = False
                
                # Save ff' and p' for inspection
                # fig, ax = plt.subplots(1, 2)
                # ax[0].plot(ffp_prof['x'], ffp_prof['y'])
                # ax[1].plot(pp_prof['x'], pp_prof['y'])
                # plt.title(f't={t}')
                # plt.savefig(f'err/t={t}.png')
                # plt.close(fig)
            
            if solve_succeeded:
                self._profile_plot(step, i, t)

            if self._prescribed_currents:
                if i < len(self._times):
                    self.set_coil_reg(i=i+1)
            elif not skip_coil_update:
                coil_targets, _ = self._gs.get_coil_currents()
                self.set_coil_reg(targets=coil_targets)


        self._print_out(f'Step {step}: TM out: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        consumed_flux = (self._state['psi_lcfs'][-1] - self._state['psi_lcfs'][0]) * 2.0 * np.pi # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need 2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(self._state['vloop'][1:], self._times[1:]) # ignore t=0 vloop

        # Diagnostic: print vloop statistics
        vloop_arr = np.array(self._state['vloop'])
        self._print_out(f"\tTM: vloop: min={vloop_arr.min():.3f}, max={vloop_arr.max():.3f}, mean={vloop_arr.mean():.3f} V")
        self._print_out(f"\tTM: psi_lcfs: start={self._state['psi_lcfs'][0]:.3f}, end={self._state['psi_lcfs'][-1]:.3f} Wb")
        self._print_out(f'\tTM: psi_bound consumed flux={consumed_flux:.3f} Wb')
        self._print_out(f'\tTM: int v_loop consumed flux w/o t=0 ={consumed_flux_integral:.3f} Wb')

        return consumed_flux, consumed_flux_integral
        
    def _gs_update(self, i):
        r'''! Update internal state and coil current results based on results of GS solver.
        @param i Timestep of the solve.
        '''
        eq_stats = self._gs.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']
        self._state['Ip_tm'][i] = eq_stats['Ip']

        self._state['psi_lcfs'][i] = self._gs.psi_bounds[0] # TM outputs in Wb/rad (AKA Wb-rad) which is how psi_lcfs is stored
        self._state['psi_axis'][i] = self._gs.psi_bounds[1] 

        if 'psi_lcfs_tmaker' not in self._results:
            self._results['psi_lcfs_tmaker'] = {'x': np.zeros(len(self._times)), 'y': np.zeros(len(self._times))}
        self._results['psi_lcfs_tmaker']['x'][i] = self._times[i]
        self._results['psi_lcfs_tmaker']['y'][i] = self._state['psi_lcfs'][i] # stored as Wb/rad

        self._state['vloop'][i] = self._gs.calc_loopvoltage()
        
        # Store TokaMaker pressure profile from get_profiles()
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)

        self._state['ffp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_fp_prof*tm_f_prof), 'type': 'linterp'}
        self._state['pp_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_pp_prof), 'type': 'linterp'}
        self._state['p_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_p_prof), 'type': 'linterp'}
        self._state['f_prof_tm'][i] = {'x': self._psi_N.copy(), 'y': np.interp(self._psi_N, tm_psi, tm_f_prof), 'type': 'linterp'}

        # pull geo profiles
        psi_geo, _, geo, _, _, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02, compute_geo=False)
        
        # R_avg = np.interp(self._psi_N, psi_geo, np.array(geo[0]))
        # R_inv_avg = np.interp(self._psi_N, psi_geo, np.array(geo[1]))

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
        
        # Get psi_norm on rho_face_norm grid
        # psi_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
        
        # # Convert psi to same grid as variable
        # if grid == 'rho_cell_norm':
        #     psi_on_grid = (psi_face[:-1] + psi_face[1:]) / 2
        # elif grid == 'rho_face_norm':
        #     psi_on_grid = psi_face
        # elif grid == 'rho_norm':
        #     psi_on_grid = np.concatenate([[psi_face[0]], 
        #                                 (psi_face[:-1] + psi_face[1:]) / 2, 
        #                                 [psi_face[-1]]])

        psi_norm_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
        psi_rho_norm = data_tree.profiles.psi.sel(time=time, method='nearest').to_numpy()
        psi_norm_rho_norm=(psi_rho_norm - psi_rho_norm[0])/(psi_rho_norm[-1] - psi_rho_norm[0])
        psi_norm_rho_norm[1] = (psi_norm_face[0]  + psi_norm_face[1])/2.0

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
            # data_on_psi -= data_on_psi[-1]  # Subtract edge value
            data_on_psi /= data_on_psi[0]   # Divide by core value
        
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
        self._print_out(f'Step {step}: TX input: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

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
        
        myconfig['pedestal']['rho_norm_ped_top'] = self._ped_top
        
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
        self._print_out(f'Step {step}: TX output (w/ /2pi): psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')

        self._res_update(data_tree)

        consumed_flux = 2.0 * np.pi * (self._state['psi_lcfs'][-1] - self._state['psi_lcfs'][0]) # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need *2pi factor to get Wb to calculate consumed flux
        consumed_flux_integral = np.trapezoid(v_loops[1:], self._times[1:])  # ignore t=0 vloop # TODO fix
        self._print_out(f"Step {step} TORAX:")
        self._print_out(f"\tTX: vloop: min={v_loops.min():.3f}, max={v_loops.max():.3f}, mean={v_loops.mean():.3f} V")
        self._print_out(f"\tTX: psi_lcfs: start={self._state['psi_lcfs'][0]:.3f}, end={self._state['psi_lcfs'][-1]:.3f} Wb/rad")
        self._print_out(f'\tTX: psi_bound consumed flux={consumed_flux:.3f} Wb')
        self._print_out(f'\tTX: int v_loop consumed flux w/o t=0 ={consumed_flux_integral:.3f} Wb')
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

        self._state['pax'][i] = data_tree.profiles.pressure_thermal_total.sel(time=t, rho_norm=0.0, method='nearest')
        self._state['beta_pol'][i] = data_tree.scalars.beta_pol.sel(time=t, method='nearest')
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')

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

        # calculate ffp_ni profile
        self._state['j_tot'][i] =            self._pull_torax_onto_psi(data_tree, 'j_total',          t, load_into_state='state', profile_type='jphi-linterp')
        # self._state['j_parallel_total'][i] = self._pull_torax_onto_psi(data_tree, 'j_parallel_total', t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ohmic'][i] =          self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        # self._state['j_ohmic_tx'][i] =       self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ni'][i] =          self._pull_torax_onto_psi(data_tree, 'j_non_inductive',  t, load_into_state='state', profile_type='jphi-linterp')

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
        
        j_tot = self._state['j_tot'][i]['y']
        j_ohmic = self._state['j_ohmic'][i]['y']
        j_ni = j_tot - j_ohmic

        self._state['R_inv_avg_tx'][i] = self._pull_torax_onto_psi(data_tree, 'gm9', t, load_into_state='state', normalize=False)
        
        ffp_ni = mu_0 * j_ni / self._state['R_inv_avg_tx'][i]['y']

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

        fig, axes = plt.subplots(4, 4, figsize=(18, 16))
        plt.suptitle(f'Step {step} - Time index {i}/{len(self._times)-1} - t = {t:.1f} s', fontsize=14)

        # p (real units)
        axes[0,0].set_title('p (real)')
        # axes[0,0].plot(tm_psi, tm_p_prof, 'r--', label='TM get_profiles() (input?)')
        axes[0,0].plot(self._state['p_prof_tm'][i]['x'], self._state['p_prof_tm'][i]['y'], 'g-', label='TM get_prof')
        axes[0,0].plot(self._state['ptot'][i]['x'], self._state['ptot'][i]['y'], 'b-', label='TORAX output')
        axes[0,0].set_ylabel('p [Pa]')
        axes[0,0].set_xlabel(r'$\hat{\psi}$')
        axes[0,0].legend(fontsize=8)

        # p (normalized)
        ptot_torax_norm = normalize_arr(self._state['ptot'][i]['y'])
        axes[0,1].set_title('p (normalized)')
        axes[0,1].plot(tm_psi, tm_p_norm, 'r--', label='TM get_profiles (norm)')
        axes[0,1].plot(self._state['ptot'][i]['x'], ptot_torax_norm, 'b-', label='TORAX output (norm)')
        axes[0,1].set_ylabel('p (norm)')
        axes[0,1].set_xlabel(r'$\hat{\psi}$')
        axes[0,1].legend(fontsize=8)

        # p' comparison
        axes[0,2].set_title("p' comparison (normalized)")
        axes[0,2].plot(self._state['pp_prof'][i]['x'], self._state['pp_prof'][i]['y'], 'b-', label='TORAX output (norm)')
        # axes[0,2].plot(self._psi_N, self._state['pp_prof_tx'][i]['y'], 'g-', label='TM input (norm)')
        axes[0,2].plot(tm_psi, tm_pp_norm, 'r--', label='TM get_profiles (norm)')
        axes[0,2].set_ylabel("p'")
        axes[0,2].set_xlabel(r'$\hat{\psi}$')
        axes[0,2].legend(fontsize=8)

        # p' comparison (real units)
        ax_pp_comp = axes[0,3]
        ax_pp_comp.set_title("p' comparison (real units)")
        # ax_pp_comp.plot(self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y'], 'b-', label='TORAX output *-1 (real)', linewidth=2)
        ax_pp_comp.plot(self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y'], 'b-', label='TORAX output *-2pi (real)', linewidth=2)
        ax_pp_comp.plot(self._psi_N, self._state['pp_prof_tm'][i]['y'], 'g-', label='TM get_profiles (real)', linewidth=2)
        ax_pp_comp.set_ylabel("p' [Pa/Wb]")
        ax_pp_comp.set_xlabel(r'$\hat{\psi}$')
        ax_pp_comp.legend(fontsize=8)


        # FF' comparison (normalized)
        axes[1,1].set_title("FF' comparison (normalized)")
        axes[1,1].plot(self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'], 'b-', label='TORAX output (normalized)', linewidth=2)
        # axes[1,1].plot(ffp_prof_in['x'], ffp_prof_in['y'], 'g-', label='TM input (norm)')
        axes[1,1].plot(tm_psi, tm_ffp_norm, 'r--', label='TM get_profiles (normalized)', linewidth=2)
        # axes[1,1].plot(self._psi_N, ffp_calc_norm, 'g--', label="FF' from eq (normalized)", linewidth=2)
        # axes[1,1].plot(self._psi_N, ffp_calc_norm_parallel, 'm-.', label="FF' from eq w/ j_parallel (normalized)", linewidth=2)
        axes[1,1].set_ylabel("FF'")
        axes[1,1].set_xlabel(r'$\hat{\psi}$')
        axes[1,1].legend(fontsize=8)


        # FF' comparison (real units)
        axes[1,2].set_title("FF' comparison (real units)")
        # axes[1,2].plot(self._psi_N, -1*self._state['ffp_prof_tx'][i]['y'], 'k-', label="FF' total TX * -1.0", linewidth=2)
        # axes[1,2].plot(self._psi_N, -np.pi*self._state['ffp_prof_tx'][i]['y'], 'k--', label="FF' total TX * -pi", linewidth=2)
        # axes[1,2].plot(self._psi_N, -2*np.pi*self._state['ffp_prof_tx'][i]['y'], 'k-.', label="FF' total TX * -2pi", linewidth=2)
        axes[1,2].plot(self._psi_N, self._state['ffp_prof_tx'][i]['y'], 'k-', label="FF' total TX (converted)", linewidth=2)
        axes[1,2].plot(self._state['ffpni_prof'][i]['x'], self._state['ffpni_prof'][i]['y'], 'b-', label="FF' NI")
        axes[1,2].plot(self._psi_N, self._state['ffp_prof_tx'][i]['y'] - self._state['ffpni_prof'][i]['y'], 'r--', label="FF' inductive")
        axes[1,2].plot(self._psi_N, self._state['ffp_prof_tm'][i]['y'], 'g--', label="FF' total TM", linewidth=1)

        axes[1,2].set_ylabel("FF'")
        axes[1,2].set_xlabel(r'$\hat{\psi}$')
        axes[1,2].legend(fontsize=8)




        # resistivity profile
        axes[2,0].set_title('Resistivity')
        axes[2,0].plot(self._state['eta_prof'][i]['x'], self._state['eta_prof'][i]['y'], 'r-', label='TORAX output', linewidth=2)
        axes[2,0].set_yscale('log')
        axes[2,0].set_ylabel(r'$\eta$ [Ohm m]')
        axes[2,0].set_xlabel(r'$\hat{\psi}$')
        axes[2,0].legend(fontsize=8)

        # j comparison

        # compute j using basic equation        
        # j_tx_geo = -1 * (self._state['ffp_prof_tx'][i]['y'] * self._state['R_inv_avg_tx'][i]['y'] / mu_0 + self._state['pp_prof_tx'][i]['y'] * self._state['R_avg_tx'][i]['y']) * 2.0 * np.pi
        
        # j_tm_geo = -2.0*np.pi * self._state['ffp_prof_tx'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / mu_0 + self._state['pp_prof_tx'][i]['y'] * self._state['R_avg_tm'][i]['y']

        
        # j_tm_tm_geo_tx_eq = (self._state['ffp_prof_tm'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / 2*mu_0 - self._state['pp_prof_tm'][i]['y'] * self._state['R_avg_tm'][i]['y']) *2*np.pi
        # j_tm_tm_geo = (-1 * self._state['ffp_prof_tm'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / mu_0 + self._state['pp_prof_tm'][i]['y'] * self._state['R_avg_tm'][i]['y']) * 2.0 * np.pi
        # j_tm_tm_geo_tm_eq = 0.5 * self._state['ffp_prof_tm'][i]['y'] * (self._state['R_inv_avg_tm'][i]['y'] / mu_0) + self._state['pp_prof_tm'][i]['y'] * self._state['R_avg_tm'][i]['y']
        # j_tm_tm_geo_tm_eq2 = self._state['ffp_prof_tm'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / (2.0*mu_0)   - self._state['pp_prof_tm'][i]['y'] * self._state['R_avg_tm'][i]['y']
        
        j_tx_prof_tm_eq = self._state['ffp_prof_tx'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / mu_0 + self._state['pp_prof_tx'][i]['y'] * self._state['R_avg_tm'][i]['y']
        j_tm_new =        self._state['ffp_prof_tm'][i]['y'] * self._state['R_inv_avg_tm'][i]['y'] / mu_0 + self._state['pp_prof_tm'][i]['y'] * self._state['R_avg_tm'][i]['y']

        axes[2,1].plot(self._state['j_tot'][i]['x'], self._state['j_tot'][i]['y'] / 1e6, 'k-', label=r'$j_{tot}$', linewidth=2)
        # axes[2,1].plot(self._state['j_parallel_total'][i]['x'], self._state['j_parallel_total'][i]['y'] / 1e6, 'm-', label=r'$j_{parallel\_total}$', linewidth=2)
        axes[2,1].plot(self._psi_N, j_tx_prof_tm_eq / 1e6,          'b--', label='TX profs, tm geo', linewidth=2)
        # axes[2,1].plot(self._psi_N, j_tm_geo / 1e6,          'r--', label='tx eq, TX profs, tm geo', linewidth=2)
        # axes[2,1].plot(self._psi_N, j_tm_tm_geo / 1e6,       'g--', label='tx eq, tm profs, tm geo', linewidth=2)
        # axes[2,1].plot(self._psi_N, j_tm_tm_geo_tm_eq / 1e6, 'c--', label='tm eq, tm profs, tm geo', linewidth=2)
        axes[2,1].plot(self._psi_N, j_tm_new / 1e6, 'g--', label='TM profs, tm geo', linewidth=2)
        # axes[2,1].plot(self._psi_N, j_tm_tm_geo_tx_eq / 1e6, 'c--', label='tx eq (mod), tm profs, tm geo', linewidth=2)

        axes[2,1].set_title('j_phi comp')
        axes[2,1].set_ylabel(r'$j$ [MA/m²]')
        axes[2,1].set_xlabel(r'$\hat{\psi}$')
        axes[2,1].legend(fontsize=8)



        # j_phi profiles
        axes[2,2].set_title('Current densities')
        axes[2,2].plot(self._state['j_tot'][i]['x'], self._state['j_tot'][i]['y'] / 1e6, 'k-', label=r'$j_{tot}$', linewidth=2)
        axes[2,2].plot(self._state['j_ohmic'][i]['x'], self._state['j_ohmic'][i]['y'] / 1e6, 'r-', label=r'$j_{ohmic}$', linewidth=1.5)
        axes[2,2].plot(self._state['j_ni'][i]['x'], self._state['j_ni'][i]['y'] / 1e6, 'b-', label=r'$j_{NI}$', linewidth=1.5)
        axes[2,2].set_ylabel(r'$j$ [MA/m²]')
        axes[2,2].set_xlabel(r'$\hat{\psi}$')
        axes[2,2].legend(fontsize=8)



        # q-profile panel (TORAX q if available)
        axes[3,0].set_title('q profile (TORAX)')
        if 'q' in self._results and self._times[i] in self._results['q']:
            q_prof = self._results['q'][self._times[i]]
            axes[3,0].plot(q_prof['x'], q_prof['y'], 'b-', linewidth=2)
            axes[3,0].set_xlabel(r'$\hat{\psi}$')
            axes[3,0].set_ylabel('q')
        else:
            axes[3,0].text(0.5, 0.5, 'No q profile', horizontalalignment='center', verticalalignment='center')
            axes[3,0].set_xticks([])
            axes[3,0].set_yticks([])

        # Ti and Te profiles (same panel)
        axes[3,1].set_title('T_e and T_i')
        if i in self._state.get('T_e', {}) and i in self._state.get('T_i', {}):
            axes[3,1].plot(self._state['T_e'][i]['x'], self._state['T_e'][i]['y'], 'r-', label=r'$T_e$')
            axes[3,1].plot(self._state['T_i'][i]['x'], self._state['T_i'][i]['y'], 'm--', label=r'$T_i$')
            axes[3,1].set_xlabel(r'$\hat{\psi}$')
            axes[3,1].set_ylabel('T [keV]')
            axes[3,1].legend(fontsize=8)
        else:
            axes[3,1].text(0.5, 0.5, 'No T profiles', horizontalalignment='center', verticalalignment='center')
            axes[3,1].set_xticks([])
            axes[3,1].set_yticks([])

        # n_e and n_i profiles (same panel)
        axes[3,2].set_title('n_e and n_i')
        if i in self._state.get('n_e', {}) and i in self._state.get('n_i', {}):
            axes[3,2].plot(self._state['n_e'][i]['x'], self._state['n_e'][i]['y'], 'b-', label=r'$n_e$')
            axes[3,2].plot(self._state['n_i'][i]['x'], self._state['n_i'][i]['y'], 'c--', label=r'$n_i$')
            axes[3,2].set_xlabel(r'$\hat{\psi}$')
            axes[3,2].set_ylabel(r'$n$ [m$^{-3}$]')
            axes[3,2].legend(fontsize=8)
        else:
            axes[3,2].text(0.5, 0.5, 'No n profiles', horizontalalignment='center', verticalalignment='center')
            axes[3,2].set_xticks([])
            axes[3,2].set_yticks([])

        # # Add R_avg (TM vs TORAX) and <1/R> comparison in row=1, last column (axes[1,3])
        # ax_03 = axes[0,3]
        # ax_03.set_title('<R> and <1/R> TM vs TORAX')
        # ax_03.plot(self._state['R_avg_tm'][i]['x'], self._state['R_avg_tm'][i]['y'], 'r-', label='R_avg TM')
        # ax_03.plot(self._state['R_avg_tx'][i]['x'], self._state['R_avg_tx'][i]['y'], 'b-', label='R_avg TORAX')
        # ax_03.set_xlabel(r'$\hat{\psi}$')
        # ax_03.set_ylabel('R_avg [m]')
        # ax_03.grid(True, alpha=0.3)
        # ax_03.legend(fontsize=8)
        # # secondary axis for <1/R>
        # ax2_03 = ax_03.twinx()
        # ax2_03.plot(self._state['R_inv_avg_tm'][i]['x'], self._state['R_inv_avg_tm'][i]['y'], 'r--', label='<1/R> TM')
        # ax2_03.plot(self._state['R_inv_avg_tx'][i]['x'], self._state['R_inv_avg_tx'][i]['y'], 'b--', label='<1/R> TORAX')
        # ax2_03.set_ylabel('<1/R> [1/m]')
        # # combine legends
        # handles1, labels1 = ax_03.get_legend_handles_labels()
        # handles2, labels2 = ax2_03.get_legend_handles_labels()
        # ax_03.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc='upper right')


        axes[2,3].set_title('j_ni comparison')
        axes[2,3].plot(self._psi_N, self._state['j_ni'][i]['y'] / 1e6, 'b-', label='j_NI from TX', linewidth=2)
        axes[2,3].set_ylabel(r'$j_{NI}$ [MA/m²]')
        axes[2,3].set_xlabel(r'$\hat{\psi}$')
        axes[2,3].legend(fontsize=8)


        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
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
        
        # Flatten axes for easier indexing
        ax_flat = axes.flatten()
        
        # Profile configurations: (name, data_source, y_label, x_label)
        profiles = [
            ('n_e', 'state', r'$n_e$ [m$^{-3}$]', r'$\hat{\psi}$'),
            ('T_e', 'state', r'$T_e$ [keV]', r'$\hat{\psi}$'),
            ('n_i', 'state', r'$n_i$ [m$^{-3}$]', r'$\hat{\psi}$'),
            ('T_i', 'state', r'$T_i$ [keV]', r'$\hat{\psi}$'),
            ('ptot', 'state', r'$p$ (TORAX) [Pa]', r'$\hat{\psi}$'),
            ('p_prof_tm', 'state', r'$p$ (TokaMaker) [Pa]', r'$\hat{\psi}$'),
            ('q', 'results', r'$q$', r'$\hat{\psi}$'),
        ]
        
        # Plot each profile type
        for idx, (prof_name, source, ylabel, xlabel) in enumerate(profiles):
            ax = ax_flat[idx]
            ax.set_title(prof_name.replace('_', ' '))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            for i, t in enumerate(times):
                color = cmap(norm(t))
                
                if source == 'state':
                    if i in self._state[prof_name]:
                        x = self._state[prof_name][i]['x']
                        y = self._state[prof_name][i]['y']
                        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
                elif source == 'results':
                    if t in self._results[prof_name]:
                        x = self._results[prof_name][t]['x']
                        y = self._results[prof_name][t]['y']
                        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
            ax.set_xlim([0, 1])
        
        # Hide the last (unused) subplot
        ax_flat[-1].axis('off')
        
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
        ax_flat = axes.flatten()

        def get_series(key):
            # special-case derived/state variables
            if key == 'Ip':
                return (self._times, self._state['Ip'])
            if key == 'vloop_tm':
                return (self._times, self._state['vloop'])
            if key == 'pax':
                return (self._times, self._state['pax'])
            if key == 'q95_state':
                return (self._times, self._state['q95'])
            if key == 'B0_state':
                return (self._times, self._state['B0'])

            # results dict entries
            if key in self._results:
                entry = self._results[key]
                return (entry['x'], entry['y'])

            # fallback: if present in _state as time-indexed dict
            if key in self._state and isinstance(self._state[key], dict):
                x = []
                y = []
                for i, t in enumerate(self._times):
                    if i in self._state[key]:
                        x.append(t)
                        # attempt to extract scalar value
                        val = self._state[key][i]
                        if isinstance(val, dict) and 'y' in val and np.isscalar(val['y']):
                            y.append(val['y'])
                        else:
                            try:
                                y.append(float(val))
                            except:
                                y.append(np.nan)
                    else:
                        x.append(t)
                        y.append(np.nan)
                return (np.array(x), np.array(y))

            return (None, None)

        plots = [
            ('Ip', 'Ip [A]'),
            ('psi_both', r'$\psi_{lcfs}$ TM & TORAX [Wb/rad]'),
            ('vloops_combined', 'V_loop (TM vs TORAX) [V]'),
            ('Q', 'Q_fusion'),
            ('n_e_line_avg', r'$\bar{n}_e$ line avg [m$^{-3}$]'),
            ('T_e_line_avg', r'$T_e$ line avg [keV]'),
            ('P_ohmic_e', 'Power channels [W]'),
            ('beta_N', 'beta_N'),
            ('li3', 'l_i (li3)'),
            ('q95_state', 'q95'),
            ('pax', 'pax [Pa]'),
            ('B0_state', 'B0 [T]'),
        ]

        for idx, (key, title) in enumerate(plots):
            ax = ax_flat[idx]
            # handle special panels
            if key == 'Ip':
                ax.set_title(title)
                ax.plot(self._times, self._state['Ip_tm'], 'r-o', markersize=3, label='Ip TM')
                ax.plot(self._times, self._state['Ip_tx'], 'b-o', markersize=3, label='Ip TORAX')
                ax.plot(self._times, self._state['Ip_NI_tx'], 'g--', markersize=3, label='Ip NI TORAX')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Ip [A]')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                # B0 on secondary axis if available
                # if 'B0' in self._results:
                #     ax2 = ax.twinx()
                #     bx, by = self._results['B0']['x'], self._results['B0']['y']
                #     ax2.plot(bx, by, 'g--', markersize=3, label='B0')
                #     ax2.set_ylabel('B0 [T]')
                #     ax2.legend(fontsize=8)
                continue

            if key == 'psi_both':
                ax.set_title('psi_lcfs (TM & TORAX)')
                if 'psi_lcfs_tmaker' in self._results:
                    ax.plot(self._results['psi_lcfs_tmaker']['x'], self._results['psi_lcfs_tmaker']['y'], 'r-', label='TokaMaker')
                if 'psi_lcfs_torax' in self._results:
                    ax.plot(self._results['psi_lcfs_torax']['x'], self._results['psi_lcfs_torax']['y'], 'b--', label='TORAX')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(r'$\psi_{lcfs}$ [Wb/rad]')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                continue

            if key == 'vloops_combined':
                ax.set_title(title)
                ax.plot(self._times, self._state['vloop'], 'r-o', markersize=3, label='TokaMaker')
                if 'v_loop_lcfs' in self._results:
                    rx = self._results['v_loop_lcfs']['x']
                    ry = self._results['v_loop_lcfs']['y']
                    ax.plot(rx, ry, 'b--o', markersize=3, label='TORAX')
                    # Secondary axis for vloop ratio (TokaMaker / TORAX)
                    tm_vloop = np.array(self._state['vloop'])
                    tx_vloop = np.array(ry)
                    # Interpolate TokaMaker vloop to TORAX time points if needed
                    if len(tm_vloop) == len(tx_vloop):
                        ratio = tm_vloop / tx_vloop
                        ratio_times = np.array(self._times)
                    else:
                        interp_tm = interp1d(self._times, tm_vloop, bounds_error=False, fill_value=np.nan)
                        ratio = interp_tm(rx) / tx_vloop
                        ratio_times = np.array(rx)
                    ax2 = ax.twinx()
                    ax2.plot(ratio_times, ratio, 'g-s', markersize=3, label='TM/TX ratio')
                    ax2.set_ylim(0,30)
                    ax2.set_ylabel('V_loop ratio (TM/TX)', color='g')
                    ax2.tick_params(axis='y', labelcolor='g')
                    ax2.legend(fontsize=8, loc='upper right')
                    # Print average ratio between 300 and 400 seconds
                    mask = (ratio_times >= 300) & (ratio_times <= 400)
                    if np.any(mask):
                        avg_ratio = np.nanmean(ratio[mask])
                        self._print_out(f"Average V_loop ratio (TokaMaker/TORAX) between 300-400s: {avg_ratio:.4f}")
                        ax2.text(0.5, 0.9, f'Avg ratio (300-400s): {avg_ratio:.4f}', transform=ax2.transAxes, color='g', fontsize=8, horizontalalignment='center')
                ax.set_xlabel('Time [s]')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='upper left')
                continue

            if key == 'Q':
                ax.set_title(title)
                if 'Q' in self._results:
                    ax.plot(self._results['Q']['x'], self._results['Q']['y'], 'b-o', markersize=3, label='Q')
                ax.set_xlabel('Time [s]')
                ax.grid(True, alpha=0.3)
                # E_fusion on secondary axis
                if 'E_fusion' in self._results:
                    ax2 = ax.twinx()
                    ax2.plot(self._results['E_fusion']['x'], self._results['E_fusion']['y'], 'g--', markersize=3, label='E_fusion')
                    ax2.set_ylabel('E_fusion')
                continue

            if key == 'n_e_line_avg':
                ax.set_title(title)
                if 'n_e_line_avg' in self._results:
                    ax.plot(self._results['n_e_line_avg']['x'], self._results['n_e_line_avg']['y'], 'b-o', markersize=3, label='n_e line avg')
                # plot core from results if present
                if 'n_e_core' in self._results:
                    ax.plot(self._results['n_e_core']['x'], self._results['n_e_core']['y'], 'k--', markersize=3, label='n_e core')
                # edge from state profiles
                ne_edge_x = self._times
                ne_edge_y = [self._state['n_e'][ii]['y'][-1] if ii in self._state.get('n_e', {}) else np.nan for ii in range(len(self._times))]
                ax.plot(ne_edge_x, ne_edge_y, 'c:', marker='s', markersize=3, label='n_e edge')
                ax.set_xlabel('Time [s]')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                continue

            if key == 'T_e_line_avg':
                ax.set_title(title)
                if 'T_e_line_avg' in self._results:
                    ax.plot(self._results['T_e_line_avg']['x'], self._results['T_e_line_avg']['y'], 'r-o', markersize=3, label='T_e line avg')
                if 'T_e_core' in self._results:
                    ax.plot(self._results['T_e_core']['x'], self._results['T_e_core']['y'], 'k--', markersize=3, label='T_e core')
                te_edge_x = self._times
                te_edge_y = [self._state['T_e'][ii]['y'][-1] if ii in self._state.get('T_e', {}) else np.nan for ii in range(len(self._times))]
                ax.plot(te_edge_x, te_edge_y, 'm:', marker='s', markersize=3, label='T_e edge')
                ax.set_xlabel('Time [s]')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                continue

            if key == 'P_ohmic_e':
                ax.set_title(title)
                # Plot P_ohmic_e and related channels if available
                if 'P_ohmic_e' in self._results:
                    ax.plot(self._results['P_ohmic_e']['x'], self._results['P_ohmic_e']['y'], 'r-o', markersize=3, label='P_ohmic_e')
                if 'P_radiation_e' in self._results:
                    ax.plot(self._results['P_radiation_e']['x'], self._results['P_radiation_e']['y'], 'm--', markersize=3, label='P_radiation_e')
                if 'P_SOL_total' in self._results:
                    ax.plot(self._results['P_SOL_total']['x'], self._results['P_SOL_total']['y'], 'c--', markersize=3, label='P_SOL_total')
                if 'P_alpha_total' in self._results:
                    ax.plot(self._results['P_alpha_total']['x'], self._results['P_alpha_total']['y'], 'g-.', markersize=3, label='P_alpha_total')
                if 'P_aux_total' in self._results:
                    ax.plot(self._results['P_aux_total']['x'], self._results['P_aux_total']['y'], 'y-.', markersize=3, label='P_aux_total')
                ax.set_xlabel('Time [s]')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                continue

            x, y = get_series(key)
            ax.set_title(title)
            if x is None or y is None:
                ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            try:
                ax.plot(x, y, '-o', markersize=3)
            except Exception:
                ax.plot(x, np.array(y, dtype=float), '-o', markersize=3)

            ax.set_xlabel('Time [s]')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Scalars Over Pulse (Step {step})', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'tmp/scalars_step{step}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)





    def fly(self, convergence_threshold=-1.0, save_states=False, graph=False, max_step=10, out='res.json'):
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
        #     self._state['vloop'][i] = 0.0  # No vloop from static EQDSK

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
            vloop_tm = self._state['vloop'].copy()
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
        

        from save_tokamaker_inputs import save_tokamaker_inputs
        save_tokamaker_inputs(self, step=step-1, fname='2026-02-09_tokamaker_test_inputs.npz')
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


