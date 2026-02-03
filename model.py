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

class MyEncoder(json.JSONEncoder):
    '''! JSON Encoder Object to store simulation results.'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
        self._oftenv = OFT_env(nthreads=2)
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
        self._state['p_tmaker'] = {}  # TokaMaker pressure profile from get_profiles()
        self._state['ffpni_prof'] = {}

        # Outputs from TORAX (already normalized)
        self._state['ffp_prof_tx'] = {}
        self._state['pp_prof_tx'] = {}
        self._state['test'] = {}
        
        # Current density profiles from TORAX
        self._state['j_tot'] = {}
        self._state['j_parallel_total'] = {}
        self._state['j_ohmic'] = {}
        self._state['j_ni'] = {}

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
            self._state['ffp_prof'][i]['y'] -= self._state['ffp_prof'][i]['y'][-1]
            self._state['pp_prof'][i]['y'] -= self._state['pp_prof'][i]['y'][-1]
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
        self._prof_smoothing = True
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
            if self._state['beta_pol'][i] != 0: # always the case because TX runs first
                print('Using beta_p...')
                Ip_ratio=(1.0/self._state['beta_pol'][i] - 1.0)
                self._gs.set_targets(Ip=Ip_target, Ip_ratio=Ip_ratio, pax=P0_target)
            else:
                self._gs.set_targets(Ip=Ip_target, pax=P0_target)
            
            # self._gs.set_targets(Ip=Ip_target, pax=P0_target) # using pax target with j_phi inputs, error code told me


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
            ffp_prof['y'] -= ffp_prof['y'][-1]
            ffp_prof['y'] /= ffp_prof['y'][0]

            pp_prof['y'] -= pp_prof['y'][-1]
            pp_prof['y'] /= pp_prof['y'][0]
 
            # ffpni = self._state['ffpni_prof'][i]
            # self._print_out(f"  t={t}: ffpni_prof min={np.min(ffpni['y']):.3e}, max={np.max(ffpni['y']):.3e}")

            self._gs.set_profiles(
                ffp_prof=ffp_prof,
                pp_prof=pp_prof,
                ffp_NI_prof=self._state['ffpni_prof'][i], # TM wants ffp_ni without the factor of 2 present in the normal equation, per Chris 2026-01-27
            )
            
            # pass j_phi instead of ffp
            # self._gs.set_profiles(
            #     ffp_prof=self._state['j_tot'][i],
            #     pp_prof=pp_prof,
            #     ffp_NI_prof=self._state['j_ni'][i]
            # )

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
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(ffp_prof['x'], ffp_prof['y'])
                ax[1].plot(pp_prof['x'], pp_prof['y'])
                plt.title(f't={t}')
                plt.savefig(f'err/t={t}.png')
                plt.close(fig)
            
            if solve_succeeded:
                self._big_plot(step, i, t, pp_prof, ffp_prof)

            if self._prescribed_currents:
                if i < len(self._times):
                    self.set_coil_reg(i=i+1)
            elif not skip_coil_update:
                coil_targets, _ = self._gs.get_coil_currents()
                self.set_coil_reg(targets=coil_targets)


        self._print_out(f'Step {step}: TM out: psi_lcfs: min = {np.min(self._state["psi_lcfs"]):.6f}, max = {np.max(self._state["psi_lcfs"]):.6f}, swing = {(self._state["psi_lcfs"][-1] - self._state["psi_lcfs"][0]):.6f} Wb/rad')


        consumed_flux = (self._state['psi_lcfs'][-1] - self._state['psi_lcfs'][0]) * 2.0 * np.pi # psi_lcfs stored as Wb/rad (AKA Wb-rad), so need 2pi factor to get Wb to calculate consumed flux
        # consumed_flux = np.trapezoid(self._state['vloop'], self._times)
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

        self._state['psi_lcfs'][i] = self._gs.psi_bounds[0] # TM outputs in Wb/rad (AKA Wb-rad) which is how psi_lcfs is stored
        self._state['psi_axis'][i] = self._gs.psi_bounds[1] 

        if 'psi_lcfs_tmaker' not in self._results:
            self._results['psi_lcfs_tmaker'] = {'x': np.zeros(len(self._times)), 'y': np.zeros(len(self._times))}
        self._results['psi_lcfs_tmaker']['x'][i] = self._times[i]
        self._results['psi_lcfs_tmaker']['y'][i] = self._state['psi_lcfs'][i] # stored as Wb/rad

        self._state['vloop'][i] = self._gs.calc_loopvoltage()
        
        # Store TokaMaker pressure profile from get_profiles()
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)
        self._state['p_tmaker'][i] = {'x': tm_psi.copy(), 'y': tm_p_prof.copy()}
        
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
        psi_face = data_tree.profiles.psi_norm.sel(time=time, method='nearest').to_numpy()
        
        # Convert psi to same grid as variable
        if grid == 'rho_cell_norm':
            psi_on_grid = (psi_face[:-1] + psi_face[1:]) / 2
        elif grid == 'rho_face_norm':
            psi_on_grid = psi_face
        elif grid == 'rho_norm':
            psi_on_grid = np.concatenate([[psi_face[0]], 
                                        (psi_face[:-1] + psi_face[1:]) / 2, 
                                        [psi_face[-1]]])
        
        # Interpolate onto uniform psi grid
        data_on_psi = interp1d(psi_on_grid, var_data, kind='linear',
                            fill_value='extrapolate', bounds_error=False)(self._psi_N)
        
        # Normalize if requested
        if normalize:
            data_on_psi -= data_on_psi[-1]  # Subtract edge value
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
        myconfig['profile_conditions']['psi'] = {
            t: {0.0: self._state['psi_axis'][i], 1.0: self._state['psi_lcfs'][i]* 2.0 * np.pi} for i, t in enumerate(self._times) # TORAX takes in Wb, psi_lcfs stores as Wb/rad (AKA Wb-rad) so needs *2pi factor
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
        data_tree_name = 'tmp/test.nc'
        data_tree.to_netcdf(data_tree_name)

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
        consumed_flux_integral = np.trapezoid(v_loops[1:], self._times[1:])  # ignore t=0 vloop
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


        self._state['Ip'][i] = data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['pax'][i] = data_tree.profiles.pressure_thermal_total.sel(time=t, rho_norm=0.0, method='nearest')
        self._state['beta_pol'][i] = data_tree.scalars.beta_pol.sel(time=t, method='nearest')
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')

        # Deep copy to prevent mutation when normalizing ffp_prof/pp_prof later
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
        self._state['pp_prof'][i] = self._pull_torax_onto_psi(data_tree, 'pprime', t, load_into_state='state', normalize=True)
        
        # Store profiles from TORAX for plotting
        ffp_sample = self._state['ffp_prof'][i]['y'].copy()
        pp_sample = self._state['pp_prof'][i]['y'].copy()
        self._state['ffp_prof_tx'][i] = {'x': self._psi_N.copy(), 'y': ffp_sample.copy()}
        self._state['pp_prof_tx'][i] = {'x': self._psi_N.copy(), 'y': pp_sample.copy()}
        

        # calculate ffp_ni profile
        self._state['j_tot'][i] =            self._pull_torax_onto_psi(data_tree, 'j_total',          t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_parallel_total'][i] = self._pull_torax_onto_psi(data_tree, 'j_parallel_total', t, load_into_state='state', profile_type='jphi-linterp')
        self._state['j_ohmic'][i] =          self._pull_torax_onto_psi(data_tree, 'j_ohmic',          t, load_into_state='state', profile_type='jphi-linterp')
    
        ffp_ni, j_ni = self._calc_ffp_ni(i, data_tree)

        self._state['ffpni_prof'][i] = {'x': self._psi_N.copy(), 'y': ffp_ni.copy(), 'type': 'linterp'} 
        self._state['j_ni'][i] = {'x': self._psi_N.copy(), 'y': j_ni.copy(), 'type': 'jphi-linterp'}
        

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
            self._results['q'][t] = self._pull_torax_onto_psi(data_tree, 'q', t, load_into_state='results', normalize=False)

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
            'y': -1.0 * data_tree.scalars.P_SOL_total.to_numpy(),
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
        @return Dictionary with FF'_NI profile in TokaMaker format
        '''
        t = self._times[i]
        
        j_tot = self._state['j_tot'][i]['y']
        j_ohmic = self._state['j_ohmic'][i]['y']
        
        j_ni = j_tot - j_ohmic

        R_inv_avg = self._pull_torax_onto_psi(data_tree, 'gm9', t, load_into_state=None, normalize=False)
        
        # FF'_NI = 2 * mu_0 * j_NI / <1/R>  (no p' term to avoid double-counting)
        mu_0 = 4.0 * np.pi * 1e-7
        ffp_ni =  mu_0 * j_ni / R_inv_avg # *2 removed out because of tokamaker convention, Chris might change it to expect the factor of 2 later
        

        # # TORAX calculation - same as above but smaller by 2pi because TORAX works with psi in units of [Wb] not [Wb/rad]
        # Rsr_inv_avg_rho = data_tree.profiles.g3.sel(time=t, method='nearest').to_numpy()  # <1/R^2> on rho_N grid
        # Rsr_inv_coords = np.pow(data_tree.profiles.g3.sel(time=t, method='nearest').coords['rho_norm'].values, 2)
        # Rsr_inv_avg = np.interp(self._psi_N, Rsr_inv_coords, Rsr_inv_avg_rho)

        # j_ni_over_R = j_ni * R_inv_avg

        # ffp_ni_torax = j_ni_over_R / (2.0 * np.pi) * mu_0 / Rsr_inv_avg

        return ffp_ni, j_ni

    def _normalize_profile(self, profile):
        r'''! Normalize a profile to range [0, 1].
        @param profile Profile dictionary with 'x' and 'y' keys.
        @return Normalized profile dictionary.
        '''
        y = copy.deepcopy(profile['y'])
        y -= y[-1]
        y /= y[0]
        return {'x': profile['x'], 'y': y, 'type': profile['type']}

    def _big_plot(self, step, i, t, pp_prof_in, ffp_prof_in): 
        # plot and save profiles at each time step and iteration
        # called in _run_gs after successful GS solve

        # Get TokaMaker output profiles
        tm_psi, tm_f_prof, tm_fp_prof, tm_p_prof, tm_pp_prof = self._gs.get_profiles(npsi=N_PSI)
        tm_ffp_prof = tm_f_prof * tm_fp_prof

        # Helper to normalize an array: subtract boundary, divide by axis value
        def normalize_arr(y):
            y_norm = copy.deepcopy(y)
            y_norm -= y_norm[-1]
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

        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        plt.suptitle(f'Step {step} - Time index {i}/{len(self._times)-1} - t = {t:.1f} s', fontsize=14)

        # p (real units)
        axes[0,0].set_title('p (real)')
        # axes[0,0].plot(tm_psi, tm_p_prof, 'r--', label='TM get_profiles() (input?)')
        axes[0,0].plot(self._state['p_tmaker'][i]['x'], self._state['p_tmaker'][i]['y'], 'g-', label='TM get_prof')
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
        axes[0,2].set_title("p' comparison")
        axes[0,2].plot(self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y'], 'b-', label='TORAX output')
        axes[0,2].plot(pp_prof_in['x'], pp_prof_in['y'], 'g-', label='TM input (norm)')
        axes[0,2].plot(tm_psi, tm_pp_norm, 'r--', label='TM get_profiles (norm)')
        axes[0,2].set_ylabel("p'")
        axes[0,2].set_xlabel(r'$\hat{\psi}$')
        axes[0,2].legend(fontsize=8)

        # f from TM
        # axes[1,0].set_title('F')
        # axes[1,0].plot(tm_psi, tm_f_prof, 'r--', label='TM get_profiles() (input?)')
        # axes[1,0].plot(self._state['fpol'][i]['x'], self._state['fpol'][i]['y'], 'b-', label='Initial EQDSK')
        # axes[1,0].set_ylabel(r'$F [T \cdot m]$')
        # axes[1,0].set_xlabel(r'$\hat{\psi}$')
        # axes[1,0].legend(fontsize=8)


        # FF' comparison: TORAX output vs calculated from basic equation
        # Basic equation: FF'_total = 2.0 * mu_0 * (j_tot + p' * <R>) / <1/R>
        j_tot_arr = self._state['j_tot'][i]['y']
        pp_real = np.interp(self._psi_N, self._state['pp_prof_tx'][i]['x'], self._state['pp_prof_tx'][i]['y']) 
        
        # Get geometry factors
        psi_geo, _, geo, _, _, _ = self._gs.get_q(npsi=N_PSI, psi_pad=0.02, compute_geo=False)
        R_avg = np.interp(self._psi_N, psi_geo, np.array(geo[0]))
        R_inv_avg = np.interp(self._psi_N, psi_geo, np.array(geo[1]))
        
        mu_0 = 4.0 * np.pi * 1e-7
        ffp_calc = 2.0 *  mu_0 * (j_tot_arr + pp_real * R_avg) / R_inv_avg

        ffp_calc_norm = self._normalize_profile({'x': self._psi_N, 'y': ffp_calc, 'type': 'linterp'})['y']
        
        axes[1,0].set_title("FF' comp (basic eq)")
        axes[1,0].plot(self._state['ffp_prof_tx'][i]['x'], self._state['ffp_prof_tx'][i]['y'], 'b-', label="TORAX")
        axes[1,0].plot(self._psi_N, ffp_calc_norm, 'g--', label="FF' from eq (norm)", linewidth=2)
        axes[1,0].plot(self._state['test'][i]['x'], self._state['test'][i]['y'], 'r--', label="FF' pull_torax_onto_psi", linewidth=1)
        axes[1,0].set_ylabel("FF'")
        axes[1,0].set_xlabel(r'$\hat{\psi}$')
        axes[1,0].legend(fontsize=8)


        # FF' comparison
        axes[1,1].set_title("FF' comparison")
        axes[1,1].plot(self._state['ffp_prof_tx'][i]['x'], self._state['ffp_prof_tx'][i]['y'], 'b-', label='TORAX')
        axes[1,1].plot(ffp_prof_in['x'], ffp_prof_in['y'], 'g-', label='TM input (norm)')
        axes[1,1].plot(tm_psi, tm_ffp_norm, 'r--', label='TM get_profiles (norm)')
        axes[1,1].set_ylabel("FF'")
        axes[1,1].set_xlabel(r'$\hat{\psi}$')
        axes[1,1].legend(fontsize=8)


        # FF' decomposition (real units): total, NI, inductive
        axes[1,2].set_title("FF' decomposition")
        axes[1,2].plot(tm_psi, tm_ffp_prof, 'k-', label="FF' total", linewidth=2)
        axes[1,2].plot(self._state['ffpni_prof'][i]['x'], ffpni_real, 'b-', label="FF' NI")
        axes[1,2].plot(tm_psi, ffp_inductive, 'r--', label="FF' inductive")

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

        # TORAX j_total vs j_parallel_total
        axes[2,1].plot(self._state['j_tot'][i]['x'], self._state['j_tot'][i]['y'] / 1e6, 'k-', label=r'$j_{tot}$', linewidth=2)
        axes[2,1].plot(self._state['j_parallel_total'][i]['x'], self._state['j_parallel_total'][i]['y'] / 1e6, 'm-', label=r'$j_{parallel\_total}$', linewidth=1.5)
        axes[2,1].set_title('Total and parallel current densities from torax')
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



        # V_loop comparison (TokaMaker vs TORAX)
        axes[3,0].set_title('V_loop comparison')
        axes[3,0].plot(self._times, self._state['vloop'], 'r-', label='TokaMaker', linewidth=2)
        if 'v_loop_lcfs' in self._results:
            axes[3,0].plot(self._results['v_loop_lcfs']['x'], self._results['v_loop_lcfs']['y'], 'b--', label='TORAX', linewidth=2)
        axes[3,0].axvline(t, color='gray', linestyle=':', alpha=0.7, label=f't={t:.1f}s')
        axes[3,0].set_ylabel('V_loop [V]')
        axes[3,0].set_xlabel('Time [s]')
        axes[3,0].legend(fontsize=8)


        # unused plot (row 3, col 1)


        # psi_boundary evolution
        axes[3,2].set_title(r'$\psi_{boundary}$')
        if 'psi_lcfs_tmaker' in self._results:
            axes[3,2].plot(self._results['psi_lcfs_tmaker']['x'], self._results['psi_lcfs_tmaker']['y'], 'r-', linewidth=2, label = 'TokaMaker')
        if 'psi_lcfs_torax' in self._results:
            axes[3,2].plot(self._results['psi_lcfs_torax']['x'], self._results['psi_lcfs_torax']['y'], 'b--', linewidth=2, label = 'TORAX')
        axes[3,2].axvline(t, color='gray', linestyle=':', alpha=0.7, label=f't={t:.1f}s')
        axes[3,2].set_ylabel(r'$\psi_{lcfs}$ [Wb/rad]')
        axes[3,2].set_xlabel('Time [s]')
        axes[3,2].legend(fontsize=8)


        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(f'tmp/big_plot_{step:03}.{i:03}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


    def profile_color_plot(self, step):
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
            ('p_tmaker', 'state', r'$p$ (TokaMaker) [Pa]', r'$\hat{\psi}$'),
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





    def fly(self, convergence_threshold=-1.0, save_states=False, graph=False, max_step=3, out='res.json'):
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

            fig, axes = plt.subplots()
            axes.set_title(f'V_loop step {step}')
            axes.plot(self._times, self._state['vloop'], 'r-o', label='TokaMaker')
            axes.plot(self._results['v_loop_lcfs']['x'], self._results['v_loop_lcfs']['y'], 'b-o', label='TORAX')
            axes.set_xlabel('Time [s]')
            axes.set_ylabel('V_loop [V]')
            axes.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(f'tmp/vloop_comparison_step{step}.png', dpi=150, bbox_inches='tight')
            plt.close('all')

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



            self.profile_color_plot(step)



            step += 1
        
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


