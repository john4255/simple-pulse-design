import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
import torax
import copy
import json
import os
import shutil

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

from baseconfig import BASE_CONFIG, set_LH_transition_time

LCFS_WEIGHT = 100.0
N_PSI = 100

class MyEncoder(json.JSONEncoder):
    '''! JSON Encoder Object to store simulation results.'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CGTS:
    '''! Coupled Grad-Shafranov/Transport Solver Object.'''

    def __init__(self, times, g_eqdsk_arr, config_overrides={}):
        r'''! Initialize the Coupled Grad-Shafranov/Transport Solver Object.
        @param times Time points of each gEQDSK file.
        @param g_eqdsk_arr Filenames of each gEQDSK file.
        @param config_overrides Dictionary of values to override values from default torax config.
        '''
        self._oftenv = OFT_env(nthreads=2)
        self._gs = TokaMaker(self._oftenv)
        self._state = {}
        self._times = times
        self._boundary = {}
        self._results = {}

        self._config_overrides = config_overrides

        self._state['R'] = np.zeros(len(times))
        self._state['Z'] = np.zeros(len(times))
        self._state['a'] = np.zeros(len(times))
        self._state['kappa'] = np.zeros(len(times))
        self._state['delta'] = np.zeros(len(times))    
        self._state['deltaU'] = np.zeros(len(times))    
        self._state['deltaL'] = np.zeros(len(times))    
        self._state['B0'] = np.zeros(len(times))
        self._state['V0'] = np.zeros(len(times))
        self._state['Ip'] = np.zeros(len(times))
        self._state['pax'] = np.zeros(len(times))
        self._state['beta_pol'] = np.zeros(len(times))
        self._state['vloop'] = np.zeros(len(times))
        self._state['q95'] = np.zeros(len(times))

        self._state['ffp_prof'] = {}
        self._state['pp_prof'] = {}
        self._state['eta_prof'] = {}
        self._state['psi_prof'] = {}
        self._state['T_e'] = {}
        self._state['T_i'] = {}
        self._state['n_e'] = {}
        self._state['n_i'] = {}
        self._state['Ptot'] = {}

        self._results['lcfs'] = {}

        for i, _ in enumerate(times):
            # Calculate geometry
            g = read_eqdsk(g_eqdsk_arr[i])

            self._boundary[i] = g['rzout'].copy()
            self._results['lcfs'][i] = g['rzout'].copy()
            zmax = np.max(self._boundary[i][:,1])
            zmin = np.min(self._boundary[i][:,1])
            rmax = np.max(self._boundary[i][:,0])
            rmin = np.min(self._boundary[i][:,0])
            minor_radius = (rmax - rmin) / 2.0
            rgeo = (rmax + rmin) / 2.0
            highest_pt_idx = np.argmax(self._boundary[i][:,1])
            lowest_pt_idx = np.argmin(self._boundary[i][:,1])
            rupper = self._boundary[i][highest_pt_idx][0]
            rlower = self._boundary[i][lowest_pt_idx][0]
            delta_upper = (rgeo - rupper) / minor_radius
            delta_lower = (rgeo - rlower) / minor_radius

            # Default Scalars
            self._state['R'][i] = g['rcentr']
            self._state['Z'][i] = g['zmid']
            self._state['a'][i] = minor_radius
            self._state['kappa'][i] = (zmax - zmin) / (2.0 * minor_radius)
            self._state['delta'][i] = (delta_upper + delta_lower) / 2.0
            self._state['deltaU'][i] = delta_upper
            self._state['deltaL'][i] = delta_lower
            self._state['B0'][i] = g['bcentr']
            self._state['V0'][i] = g['zaxis']
            self._state['pax'][i] = g['pres'][0]
            self._state['q95'][i] = np.percentile(g['qpsi'], 95)
            self._state['Ip'][i] = abs(g['ip'])
            psi_sample = np.linspace(0.0, 1.0, N_PSI)
            psi_eqdsk = np.linspace(0.0, 1.0, g['nr'])
            ffp_prof = np.interp(psi_sample, psi_eqdsk, g['ffprim'])
            pp_prof = np.interp(psi_sample, psi_eqdsk, g['pprime'])
            self._state['ffp_prof'][i] = {'x': psi_sample, 'y': ffp_prof, 'type': 'linterp'}
            self._state['pp_prof'][i] = {'x': psi_sample, 'y': pp_prof, 'type': 'linterp'}

            # Normalize profiles
            self._state['ffp_prof'][i]['y'] -= self._state['ffp_prof'][i]['y'][-1]
            self._state['pp_prof'][i]['y'] -= self._state['pp_prof'][i]['y'][-1]
            self._state['ffp_prof'][i]['y'] /= self._state['ffp_prof'][i]['y'][0]
            self._state['pp_prof'][i]['y'] /= self._state['pp_prof'][i]['y'][0]

            self._state['eta_prof'][i]= {
                'x': np.linspace(0.0, 1.0, N_PSI),
                'y': np.zeros(N_PSI),
                'type': 'linterp',
            }
            self._state['psi_prof'][i] = {
                'x': np.linspace(0.0, 1.0, N_PSI),
                'y': -50.0 + 2.0 * np.pi * np.linspace(0.0, abs(g['psibry']), N_PSI),
            }
        
    def initialize_gs(self, mesh, vsc=None):
        r'''! Initialize GS Solver Object.
        @param mesh Filename of reactor mesh.
        @param vsc Vertical Stability Coil.
        '''
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh(mesh)
        self._gs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._gs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._gs.setup(order = 2, F0 = self._state['R'][0]*self._state['B0'][0])

        self._gs.settings.maxits = 500

        # print(coil_dict.keys())
        targets = {coil_name: 0.0 for coil_name in self._gs.coil_sets}
        print(targets)

        if vsc is not None:
            self._gs.set_coil_vsc({vsc: 1.0})
        self.set_coil_reg(targets, weight_mult=0.1)

    def set_coil_reg(self, targets, weights=None, strict_limit=50.0E6, disable_virtual_vsc=True, weight_mult=1.0):
        r'''! Set coil regularization terms.
        @param targets Target values for each coil.
        @param weights Default weight for each coil.
        @param strict_limit Strict limit for coil currents.
        @param disable_virtual_vsc Disable VSC virtual coil. 
        @param weight_mult Factor by which to multiply target weights (reduce to allow for more flexibility).
        '''
        # Set regularization weights
        coil_bounds = {key: [-strict_limit, strict_limit] for key in self._gs.coil_sets}
        self._gs.set_coil_bounds(coil_bounds)

        regularization_terms = []
        if weights is None:
            weights = {}
            for name, coil in self._gs.coil_sets.items():
                if name.startswith('CS'):
                    weights[name] = 2.0E-2
                else:
                    weights[name] = 1.0E-2

        print(targets)
        for name, coil in self._gs.coil_sets.items():
            regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=targets[name],weight=weights[name] * weight_mult))

        # Disable VSC virtual coil
        if disable_virtual_vsc:
            regularization_terms.append(self._gs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E2))
        
        # Pass regularization terms to TokaMaker
        self._gs.set_coil_reg(reg_terms=regularization_terms)
        self._gs.update_settings()

    def _run_gs(self, step, graph=False):
        r'''! Run the GS solve across n timesteps using TokaMaker.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @param graph Whether to display psi graphs at each iteration (for testing).
        @return Consumed flux.
        '''
        v_loops = []

        for i, _ in enumerate(self._times):
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)

            Ip_target = abs(self._state['Ip'][i])
            # P0_target = abs(self._state['pax'][i])
            # V0_target = self._state['V0'][i]
            self._gs.set_targets(Ip=Ip_target, Ip_ratio=2.0)

            ffp_prof = self._state['ffp_prof'][i]
            pp_prof = self._state['pp_prof'][i]

            self._gs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

            if step:
                self._gs.set_resistivity(eta_prof=self._state['eta_prof'][i])

            lcfs = self._boundary[i]
            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            lcfs_psi_target = self._state['psi_prof'][i]['y'][-1] / (2.0 * np.pi)
            self._gs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)

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
                ax.plot(self._boundary[i][:, 0], self._boundary[i][:, 1], color='r')
                plt.show()

            self._gs.update_settings()
            err_flag = self._gs.solve()

            if step:
                v_loops.append(self._gs.calc_loopvoltage())

            self._gs.print_info()
            self._gs_update(i, calc_vloop=step)
            self._gs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, i),lcfs_pad=0.001,run_info='TokaMaker EQDSK', cocos=2)

            coils, _ = self._gs.get_coil_currents()
            self.set_coil_reg(coils, weight_mult=0.1)

        consumed_flux = 0.0
        if step:
            consumed_flux = np.trapz(v_loops, self._times)
        return consumed_flux
        
    def _gs_update(self, i, calc_vloop=False):
        r'''! Update internal state and coil current results based on results of GS solver.
        @param i Timestep of the solve.
        @param calc_vloop Whether to calculate loop voltage.
        '''
        eq_stats = self._gs.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']

        psi = self._gs.get_psi(normalized=False)
        default_space = np.linspace(0.0, 1.0, len(psi))
        psi_space = np.linspace(0.0, 1.0, N_PSI)
        psi_sample = np.interp(psi_space, default_space, psi)
        self._state['psi_prof'][i] = {'x': psi_space, 'y': psi_sample}

        if 'flux_lcfs_tmaker' not in self._results:
            self._results['flux_lcfs_tmaker'] = {'x': np.zeros(len(self._times)), 'y': np.zeros(len(self._times))}
        self._results['flux_lcfs_tmaker']['x'][i] = self._times[i]
        self._results['flux_lcfs_tmaker']['y'][i] = self._state['psi_prof'][i]['y'][-1]

        if calc_vloop:
            self._state['vloop'][i] = self._gs.calc_loopvoltage()
        
        # Update Results
        coils, _ = self._gs.get_coil_currents()
        if i == 0:
            self._results['COIL'] = {coil: {} for coil in coils}
        for coil, current in coils.items():
            self._results['COIL'][coil][self._times[i]] = current * 1.0 # TODO: handle nturns > 1

    def _get_torax_config(self, step):
        r'''! Generate config object for Torax simulation. Modifies BASE_CONFIG based on current simulation state.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @return Torax config object.
        '''
        myconfig = copy.deepcopy(BASE_CONFIG)

        myconfig['numerics'] = {
            't_initial': 0.0,
            't_final': 150.0,  # length of simulation time in seconds
            'fixed_dt': 1.0, # fixed timestep
            'evolve_ion_heat': True, # solve ion heat equation
            'evolve_electron_heat': True, # solve electron heat equation
            'evolve_current': True, # solve current equation
            'evolve_density': True, # solve density equation
        }

        myconfig['geometry'] = {
            'geometry_type': 'eqdsk',
            'geometry_directory': '/Users/johnl/Desktop/discharge-model', 
            'last_surface_factor': 0.90,  # TODO: tweak
            'Ip_from_parameters': True,
            'geometry_configs': {
                t: {'geometry_file': 'tmp/{:03}.{:03}.eqdsk'.format(step, i)} for i, t in enumerate(self._times)
            }
        }
        myconfig['profile_conditions']['Ip'] = {
            t: abs(self._state['Ip'][i]) for i, t in enumerate(self._times)
        }
        torax_config = torax.ToraxConfig.from_dict({**myconfig, **self._config_overrides})
        return torax_config

    def _run_transport(self, step, graph=False):
        r'''! Run the Torax simulation.
        @param step Iteration number of the Torax-Tokamaker simulation loop.
        @param graph Whether to display profiles at each iteration (for testing).
        @return Consumed flux.
        '''
        myconfig = self._get_torax_config(step)
        data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)
        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        v_loops = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._transport_update(i, data_tree)
            v_loops[i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest') # / self._state['a'][i]
        
        if graph:
            for var in ['ffp_prof', 'pp_prof', 'eta_prof']:
                fig, ax = plt.subplots(1, len(self._times))
                fig.suptitle(var)
                for i, _ in enumerate(self._times):
                    ax[i].plot(self._state[var][i]['x'], self._state[var][i]['y'])
                plt.show()

        consumed_flux = np.trapz(v_loops, self._times)
        return consumed_flux
    
    def _transport_update(self, i, data_tree, smooth=False):
        r'''! Update the simulation state and simulation results based on results of the Torax simulation.
        @param i Timestep of the solve.
        @param data_tree Result object from Torax.
        @smooth Whether to smooth profiles generated by Torax.
        '''
        t = self._times[i]
        
        self._state['Ip'][i ] = data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['beta_pol'][i] = data_tree.scalars.beta_pol.sel(time=t, method='nearest')
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')

        ffprime = data_tree.profiles.FFprime.sel(time=t, method='nearest')
        pprime = data_tree.profiles.pprime.sel(time=t, method='nearest')

        self._state['ffp_prof'][i] = {
            'x': np.pow(ffprime.coords['rho_face_norm'].values, 2),
            'y': ffprime.to_numpy(),
            'type': 'linterp',
        }

        psi_sample = np.linspace(0.0, 1.0, N_PSI)
        ffp_sample = np.interp(psi_sample, self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'])
        self._state['ffp_prof'][i]['x'] = psi_sample
        self._state['ffp_prof'][i]['y'] = ffp_sample

        self._state['pp_prof'][i] = {
            'x': np.pow(pprime.coords['rho_face_norm'].values, 2),
            'y': pprime.to_numpy(),
            'type': 'linterp',
        }

        pp_sample = np.interp(psi_sample, self._state['pp_prof'][i]['x'], self._state['pp_prof'][i]['y'])
        self._state['pp_prof'][i]['x'] = psi_sample
        self._state['pp_prof'][i]['y'] = pp_sample

        # Normalize profiles
        self._state['ffp_prof'][i]['y'] -= self._state['ffp_prof'][i]['y'][-1]
        self._state['pp_prof'][i]['y'] -= self._state['pp_prof'][i]['y'][-1]
        self._state['ffp_prof'][i]['y'] /= self._state['ffp_prof'][i]['y'][0]
        self._state['pp_prof'][i]['y'] /= self._state['pp_prof'][i]['y'][0]

        # Smooth Profiles
        def make_smooth(x, y):
            spline = make_smoothing_spline(x, y, lam=0.1)
            smoothed = spline(x)
            return smoothed

        if smooth:
            self._state['ffp_prof'][i]['y'] = make_smooth(self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'])
            self._state['pp_prof'][i]['y'] = make_smooth(self._state['pp_prof'][i]['x'], self._state['pp_prof'][i]['y'])

        t_i = data_tree.profiles.T_i.sel(time=t, method='nearest')
        t_e = data_tree.profiles.T_e.sel(time=t, method='nearest')
        n_i = data_tree.profiles.n_i.sel(time=t, method='nearest')
        n_e = data_tree.profiles.n_e.sel(time=t, method='nearest')
        
        self._state['T_i'][i] = {
            'x': np.pow(t_i.coords['rho_norm'].values, 2),
            'y': t_i.to_numpy(),
        }
        self._state['T_e'][i] = {
            'x': np.pow(t_e.coords['rho_norm'].values, 2),
            'y': t_e.to_numpy(),
        }
        self._state['n_i'][i] = {
            'x': np.pow(n_i.coords['rho_norm'].values, 2),
            'y': n_i.to_numpy(),
        }
        self._state['n_e'][i] = {
            'x': np.pow(n_e.coords['rho_norm'].values, 2),
            'y': n_e.to_numpy(),
        }

        ptot = data_tree.profiles.pressure_thermal_total.sel(time=t, method='nearest')
        self._state['Ptot'][i] = {
            'x': np.pow(ptot.coords['rho_norm'].values, 2),
            'y': ptot.to_numpy(),
        }

        conductivity = data_tree.profiles.sigma_parallel.sel(time=t, method='nearest')
        self._state['eta_prof'][i] = {
            'x': np.pow(conductivity.coords['rho_norm'].values, 2),
            'y': 1.0 / conductivity.to_numpy(),
            'type': 'linterp',
        }
        psi_sample = np.linspace(0.0, 1.0, N_PSI)
        eta_sample = np.interp(psi_sample, self._state['eta_prof'][i]['x'], self._state['eta_prof'][i]['y'])
        self._state['eta_prof'][i]['x'] = psi_sample
        self._state['eta_prof'][i]['y'] = eta_sample

        if i > 0:
            psi = data_tree.profiles.psi.sel(time=t, method='nearest')
            self._state['psi_prof'][i] = {
                'x': np.pow(psi.coords['rho_norm'].values, 2),
                'y': psi.to_numpy(),
            }
            psi_sample = np.linspace(0.0, 1.0, N_PSI)
            my_psi_sample = np.interp(psi_sample, self._state['psi_prof'][i]['x'], self._state['psi_prof'][i]['y'])
            self._state['psi_prof'][i]['x'] = psi_sample
            self._state['psi_prof'][i]['y'] = my_psi_sample

        # Update sim results
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

        flux_lcfs = data_tree.profiles.psi.sel(rho_norm = 1.0)
        self._results['flux_lcfs_torax'] = {
            'x': list(flux_lcfs.coords['time'].values),
            'y': flux_lcfs.to_numpy(),
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
        with open('tmp/res.json', 'w') as f:
            json.dump(self._results, f, cls=MyEncoder)

    def fly(self, convergence_threshold=1.0E-5, save_states=False, graph=False, max_step=100):
        r'''! Run Tokamaker-Torax simulation loop until convergence or max_step reached. Saves results to JSON object.
        @pararm convergence_threshold Maximum percent difference allowed for convergence.
        @param save_states Save intermediate simulation states.
        @param graph Whether to display psi and profile graphs at each iteration (for testing).
        @param max_step Maximum number of simulation iterations allowed.
        '''
        err = convergence_threshold + 1.0
        step = 0

        if graph:
            for var in ['ffp_prof', 'pp_prof']:
                fig, ax = plt.subplots(1, len(self._times))
                fig.suptitle(var)
                for i, _ in enumerate(self._times):
                    ax[i].plot(self._state[var][i]['x'], self._state[var][i]['y'])
                plt.show()
                        
        del_tmp = input('Delete temporary storage? [y/n] ')
        if del_tmp != 'y':
            quit()
        with open('convergence_history.txt', 'w'):
            pass
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')

        cflux_prev = 0.0
        while err > convergence_threshold and step < max_step:
            cflux_gs = self._run_gs(step, graph=graph)
            if save_states:
                self.save_state('tmp/gs_state{}.json'.format(step))

            cflux = self._run_transport(step, graph=graph)
            if save_states:
                self.save_state('tmp/ts_state{}.json'.format(step))
            self.save_res()

            with open('convergence_history.txt', 'a') as f:
                print("GS CF = {}".format(cflux_gs), file=f)
                print("TS CF = {}".format(cflux), file=f)
            step += 1

            err = np.abs(cflux - cflux_prev) / cflux_prev
            cflux_prev = cflux