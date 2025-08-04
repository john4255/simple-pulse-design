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

class CGTS:
    def __init__(self, times, g_eqdsk_arr, p_eqdsk=None):
        self._oftenv = OFT_env(nthreads=2)
        self._gs = TokaMaker(self._oftenv)
        self._state = {}
        self._times = times
        self._boundary = {}

        self._state['R'] = np.zeros(len(times))
        self._state['Z'] = np.zeros(len(times))
        self._state['a'] = np.zeros(len(times))
        self._state['kappa'] = np.zeros(len(times))
        self._state['delta'] = np.zeros(len(times))    
        self._state['deltaU'] = np.zeros(len(times))    
        self._state['deltaL'] = np.zeros(len(times))    
        self._state['B0'] = np.zeros(len(times))
        self._state['V0'] = np.zeros(len(times))
        # self._state['zmin'] = np.zeros(len(times))
        # self._state['zmax'] = np.zeros(len(times))
        # self._state['rmin'] = np.zeros(len(times))
        # self._state['rmax'] = np.zeros(len(times))
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
        # self._state['f_pol'] = {}

        for i, _ in enumerate(times):
            # Calculate geometry
            g = read_eqdsk(g_eqdsk_arr[i])

            self._boundary[i] = g['rzout']
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
            self._state['Ip'][i] = abs(g['ip'])
            self._state['pax'][i] = g['pres'][0]
            self._state['q95'][i] = np.percentile(g['qpsi'], 95)

            # Default Profiles
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
            }
            self._state['psi_prof'][i] = {
                'x': np.linspace(0.0, 1.0, N_PSI),
                'y': np.zeros(N_PSI),
            }
            # self._state['f_pol'][i] = g_eqdsk['fpol']

        if p_eqdsk is not None:
            for i, _ in enumerate(times):
                self._state['T_e'][i] = p_eqdsk['te(KeV)']
                self._state['T_i'][i] = p_eqdsk['ti(KeV)']
                self._state['n_e'][i] = {key: 1.0E20 * val for key, val in p_eqdsk['ne(10^20/m^3)'].items()}
                self._state['n_i'][i] = p_eqdsk['ni(10^20/m^3)']

    def initialize_gs(self, weight_mult=1.0):
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh('ITER_mesh.h5')
        self._gs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._gs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._gs.setup(order = 2, F0 = self._state['R'][0]*self._state['B0'][0])

        self._gs.set_coil_vsc({'VS': 1.0})

        coil_bounds = {key: [-50.E6, 50.E6] for key in self._gs.coil_sets}
        self._gs.set_coil_bounds(coil_bounds)

        # Set regularization weights
        # TODO: generalize to allow for non-ITER cases
        regularization_terms = []
        for name, coil in self._gs.coil_sets.items():
            # Set zero target current and different small weights to help conditioning of fit
            if name.startswith('CS'):
                if name.startswith('CS1'):
                    regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=2.E-2 * weight_mult))
                else:
                    regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2 * weight_mult))
            elif name.startswith('PF'):
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2 * weight_mult))
            elif name.startswith('VS'):
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2 * weight_mult))
        # Disable VSC virtual coil
        regularization_terms.append(self._gs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E2))
        
        # Pass regularization terms to TokaMaker
        self._gs.set_coil_reg(reg_terms=regularization_terms)

        self._gs.settings.maxits = 500
        # self._gs.settings.nl_tol = 1.E-4
        self._gs.update_settings()

    def _get_boundary(self, i, npts=20): # TODO: use create_isoflux
        thp = np.linspace(0, 2*np.pi, npts+1)
        thp = thp[:-1]

        r0 = self._state['R'][i]
        z0 = self._state['Z'][i]
        a0 = self._state['a'][i]
        kappa = self._state['kappa'][i]
        delta = self._state['delta'][i]
        squar = 0.0 # sim_vars['squar'][i]

        ra = r0 + a0*np.cos(thp + delta*np.sin(thp) - squar*np.sin(2*thp))
        za = z0 + kappa*a0*np.sin(thp + squar*np.sin(2*thp))
        return np.vstack([ra, za]).transpose()

    def _run_gs(self, step, graph=False):
        dt = 0
        v_loop = 0.0
        v_loops = np.array([])

        for i, _ in enumerate(self._times):
            if i > 0:
                dt = self._times[i] - self._times[i-1]
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)


            Ip_target = self._state['Ip'][i]
            # P0_target = self._state['pax'][i]
            # V0_target = self._state['V0'][i]
            self._gs.set_targets(Ip=Ip_target, Ip_ratio=2.0)

            ffp_prof = self._state['ffp_prof'][i]
            pp_prof = self._state['pp_prof'][i]

            self._gs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

            if step:
                self._gs.set_resistivity(eta_prof=self._state['eta_prof'][i])

            # lcfs = self._boundary[::10] # TODO: make time-dependent
            isoflux_pts = np.array([
                [ 8.20,  0.41],
                [ 8.06,  1.46],
                [ 7.51,  2.62],
                [ 6.14,  3.78],
                [ 4.51,  3.02],
                [ 4.26,  1.33],
                [ 4.28,  0.08],
                [ 4.49, -1.34],
                [ 7.28, -1.89],
                [ 8.00, -0.68]
            ])
            x_point = np.array([[5.125, -3.4],])
            # self._gs.set_isoflux(np.vstack((isoflux_pts,x_point)))
            # self._gs.set_saddles(x_point)

            lcfs = np.vstack((isoflux_pts,x_point))
            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            if i == 0:
                self._gs.set_isoflux(lcfs, isoflux_weights)
            else: # TODO: fix
                lcfs_psi_target -= dt * v_loop / (2.0 * np.pi)
                self._gs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)
                self._gs.set_psi_dt(psi0,dt)

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
                ax.plot(self._boundary[:, 0], self._boundary[:, 1], color='r')
                plt.show()

            self._gs.update_settings()
            err_flag = self._gs.solve()

            self._gs.print_info()

            if step:
                v_loop = self._gs.calc_loopvoltage()
                v_loops = np.append(v_loops, v_loop)

            self._gs_update(i, calc_vloop = step)
            self._gs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, i),lcfs_pad=0.001,run_info='TokaMaker EQDSK', cocos=2)

            lcfs_psi_target = self._gs.psi_bounds[0]
            psi0 = self._gs.get_psi(False)

        consumed_flux = 0.0
        if step:
            consumed_flux = np.trapz(v_loops, self._times)
        return consumed_flux
        
    def _gs_update(self, i, calc_vloop=False):
        eq_stats = self._gs.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']

        psi = self._gs.get_psi(False)
        default_space = np.linspace(0.0, 1.0, len(psi))
        psi_space = np.linspace(0.0, 1.0, N_PSI)
        psi_sample = np.interp(psi_space, default_space, psi)
        self._state['psi_prof'][i] = {'x': psi_space, 'y': psi_sample}

        if calc_vloop:
            self._state['vloop'][i] = self._gs.calc_loopvoltage()

    def _get_torax_config(self, step):
        myconfig = copy.deepcopy(BASE_CONFIG)

        myconfig['numerics'] = {
            't_initial': 0.0,
            't_final': 150.0,  # length of simulation time in seconds
            # 'fixed_dt': (self._times[1] - self._times[0]) / 10.0, # fixed timestep
            'fixed_dt': 1, # fixed timestep
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
        # myconfig = set_LH_transition_time(myconfig, LH_transition_time = 80)
        torax_config = torax.ToraxConfig.from_dict(myconfig)
        return torax_config

    def _run_transport(self, step, graph=False):
        myconfig = self._get_torax_config(step)
        data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)
        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        v_loops = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._transport_update(i, data_tree)
            v_loops[i] = data_tree.scalars.v_loop_lcfs.sel(time=t, method='nearest')
            self._state['vloop'][i] = v_loops[i] # TODO: move
        
        if graph:
            for var in ['ffp_prof', 'pp_prof', 'eta_prof']:
                fig, ax = plt.subplots(1, len(self._times))
                fig.suptitle(var)
                for i, _ in enumerate(self._times):
                    ax[i].plot(self._state[var][i]['x'], self._state[var][i]['y'])
                plt.show()

        consumed_flux = np.trapz(v_loops, self._times)
        # consumed_flux = 0.0
        return consumed_flux

    def _transport_update(self, i, data_tree, smooth=False):
        t = self._times[i]

        # self._state['R'][i] = np.abs(data_tree.scalars.R_major.sel(time=t, method='nearest'))
        # self._state['a'][i] = np.abs(data_tree.scalars.a_minor.sel(time=t, method='nearest'))
        # self._state['kappa'][i] = data_tree.profiles.elongation.sel(time=t, rho_norm=1.0, method='nearest')

        # self._state['deltaU'][i] = data_tree.profiles.delta_upper.sel(time=t, rho_face_norm=1.0, method='nearest').to_numpy()
        # self._state['deltaL'][i] = data_tree.profiles.delta_lower.sel(time=t, rho_face_norm=1.0, method='nearest').to_numpy()
        # self._state['delta'][i] = (self._state['deltaU'][i] + self._state['deltaL'][i]) / 2.0

        self._state['Ip'][i] = data_tree.scalars.Ip.sel(time=t, method='nearest')
        self._state['beta_pol'][i] = data_tree.scalars.beta_pol.sel(time=t, method='nearest')
        self._state['q95'][i] = data_tree.scalars.q95.sel(time=t, method='nearest')

        eta = 1.0 / data_tree.profiles.sigma_parallel.sel(time=t, method='nearest')
        self._state['eta_prof'][i] = {
            'x': eta.coords['rho_norm'].values,
            'y': eta.to_numpy(),
            'type': 'linterp',
        }

        ffprime = data_tree.profiles.FFprime.sel(time=t, method='nearest')
        pprime = data_tree.profiles.pprime.sel(time=t, method='nearest')

        self._state['ffp_prof'][i] = {
            'x': ffprime.coords['rho_face_norm'].values,
            'y': ffprime.to_numpy(),
            'type': 'linterp',
        }

        psi_sample = np.linspace(0.0, 1.0, N_PSI)
        ffp_sample = np.interp(psi_sample, self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'])
        self._state['ffp_prof'][i]['x'] = psi_sample
        self._state['ffp_prof'][i]['y'] = ffp_sample

        self._state['pp_prof'][i] = {
            'x': pprime.coords['rho_face_norm'].values,
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
            spline = make_smoothing_spline(x, y, lam=0.01)
            smoothed = spline(x)
            return smoothed

        if smooth:
            self._state['ffp_prof'][i]['y'] = make_smooth(self._state['ffp_prof'][i]['x'], self._state['ffp_prof'][i]['y'])
            self._state['pp_prof'][i]['y'] = make_smooth(self._state['pp_prof'][i]['x'], self._state['pp_prof'][i]['y'])

        # self._state['pp_prof'][i]['y'] = 1.0 - self._state['pp_prof'][i]['x']

        t_i = data_tree.profiles.T_i.sel(time=t, method='nearest')
        t_e = data_tree.profiles.T_e.sel(time=t, method='nearest')
        n_i = data_tree.profiles.n_i.sel(time=t, method='nearest')
        n_e = data_tree.profiles.n_e.sel(time=t, method='nearest')
        
        self._state['T_i'][i] = {
            'x': t_i.coords['rho_norm'].values,
            'y': t_i.to_numpy(),
        }
        self._state['T_e'][i] = {
            'x': t_e.coords['rho_norm'].values,
            'y': t_e.to_numpy(),
        }
        self._state['n_i'][i] = {
            'x': n_i.coords['rho_norm'].values,
            'y': n_i.to_numpy(),
        }
        self._state['n_e'][i] = {
            'x': n_e.coords['rho_norm'].values,
            'y': n_e.to_numpy(),
        }

        ptot = data_tree.profiles.pressure_thermal_total.sel(time=t, method='nearest')
        self._state['Ptot'][i] = {
            'x': ptot.coords['rho_norm'].values,
            'y': ptot.to_numpy(),
        }

        psi = data_tree.profiles.psi.sel(time=t, method='nearest')
        self._state['psi_prof'][i] = {
            'x': psi.coords['rho_norm'].values,
            'y': psi.to_numpy(),
        }

    def save_state(self, fname):
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open(fname, 'w') as f:
            json.dump(self._state, f, cls=MyEncoder)
        
    def fly(self, convergence_threshold=1.0E-3, save_states=False, graph=False, max_step=50):
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

        while err > convergence_threshold and step < max_step:
            cflux_gs = self._run_gs(step, graph=graph)
            if save_states:
                self.save_state('tmp/gs_state{}.json'.format(step))

            cflux_transport = self._run_transport(step, graph=graph)
            if save_states:
                self.save_state('tmp/ts_state{}.json'.format(step))

            if step > 0:
                err = ((cflux_gs - cflux_transport) / cflux_transport) ** 2
            with open('convergence_history.txt', 'a') as f:
                # print("Err = {}".format(err), file=f)
                print("GS CF = {}".format(cflux_gs), file=f)
                print("TS CF = {}".format(cflux_transport), file=f)
            step += 1