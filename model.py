import numpy as np
import matplotlib.pyplot as plt
import torax

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, create_power_flux_fun

from baseconfig import BASE_CONFIG

LCFS_WEIGHT = 100.0

class CGTS:
    def __init__(self, g_eqdsk, times, p_eqdsk=None):
        self._oftenv = OFT_env(nthreads=2)
        self._gs = TokaMaker(self._oftenv)
        self._state = {}
        self._times = times

        self._state['R'] = np.zeros(len(times))
        self._state['Z'] = np.zeros(len(times))
        self._state['a'] = np.zeros(len(times))
        self._state['kappa'] = np.zeros(len(times))
        self._state['delta'] = np.zeros(len(times))    
        self._state['deltaU'] = np.zeros(len(times))    
        self._state['deltaL'] = np.zeros(len(times))    
        self._state['B0'] = np.zeros(len(times))
        self._state['V0'] = np.zeros(len(times))
        # self._state['zbot'] = np.zeros(len(times))
        # self._state['ztop'] = np.zeros(len(times))
        # self._state['rbot'] = np.zeros(len(times))
        # self._state['rtop'] = np.zeros(len(times))
        self._state['Ip'] = np.zeros(len(times))
        self._state['pax'] = np.zeros(len(times))

        self._state['ffp_prof'] = {}
        self._state['pp_prof'] = {}
        self._state['eta'] = {}
        self._state['psi'] = {}
        self._state['T_e'] = {}
        self._state['T_i'] = {}
        self._state['n_e'] = {}
        # self._state['n_i'] = {}
        # self._state['f_pol'] = {}
        # self._state['pressure_thermal_total'] = {}

        # Calculate geometry
        g = read_eqdsk(g_eqdsk)
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

        # Calculate profiles
        ffp_prof = create_power_flux_fun(40,1.5,2.0)
        pp_prof = create_power_flux_fun(40,4.0,1.0)

        for i, _ in enumerate(times):
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

            # Default Profiles
            # sim_vars['ffp_prof'][i] = {psi_sample[i]: ffp_prof[i] for i in range(len(psi_sample))}
            # sim_vars['pp_prof'][i] = {psi_sample[i]: pp_prof[i] for i in range(len(psi_sample))}
            self._state['ffp_prof'][i] = ffp_prof
            self._state['pp_prof'][i] = pp_prof

            self._state['eta'][i]= {}
            self._state['psi'][i] = {}
            # self._state['f_pol'][i] = g_eqdsk['fpol']

        if p_eqdsk is not None:
            for i, _ in enumerate(times):
                self._state['T_e'][i] = p_eqdsk['te(KeV)']
                self._state['T_i'][i] = p_eqdsk['ti(KeV)']
                self._state['n_e'][i] = {key: 1.0E20 * val for key, val in p_eqdsk['ne(10^20/m^3)'].items()}
                self._state['n_i'][i] = p_eqdsk['ni(10^20/m^3)']

    def initialize_gs(self):
        mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh('ITER_mesh.h5')
        self._gs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        self._gs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
        self._gs.setup(order = 2, F0 = 5.3*6.2)

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
                    regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=2.E-2))
                else:
                    regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
            elif name.startswith('PF'):
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
            elif name.startswith('VS'):
                regularization_terms.append(self._gs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
        # Disable VSC virtual coil
        regularization_terms.append(self._gs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E2))
        
        # Pass regularization terms to TokaMaker
        self._gs.set_coil_reg(reg_terms=regularization_terms)

        self._gs.maxits = 800
        self._gs.update_settings()

    def _get_boundary(self, i, npts=20):
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

    def _run_gs(self, step, calc_vloop=True, graph=False):
        dt = self._times[1] - self._times[0]
        v_loop = np.zeros(len(self._times))

        for i, _ in enumerate(self._times):
            self._gs.set_isoflux(None)
            self._gs.set_flux(None,None)

            Ip_target = self._state['Ip'][i]
            P0_target = self._state['pax'][i]
            V0_target = self._state['V0'][i]
            self._gs.set_targets(Ip=Ip_target, pax=P0_target, V0=V0_target)

            ffp_prof = self._state['ffp_prof'][i]
            pp_prof = self._state['pp_prof'][i]
            self._gs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

            if calc_vloop:
                self._gs.set_resistivity(eta_prof=self._state['eta_prof'][i])

            err_flag = self._gs.init_psi(self._state['R'][0],
                                        self._state['Z'][0],
                                        self._state['a'][0],
                                        self._state['kappa'][0], 
                                        self._state['delta'][0])
            if err_flag:
                print("Error initializing psi.")

            lcfs = self._get_boundary(i)
            isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
            if i == 0:
                self._gs.set_isoflux(lcfs, isoflux_weights)
            else:
                vloop = v_loop[i]
                lcfs_psi_target -= dt * vloop / 2 / np.pi
                self._gs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)
                self._gs.set_psi_dt(psi0,dt)

            self._gs.update_settings()
            err_flag = self._gs.solve()

            if graph:
                fig, ax = plt.subplots(1,1)
                self._gs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
                self._gs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
                # ax.plot(g_eqdsk['rzout'][:, 0], g_eqdsk['rzout'][:, 1], color='r')
                plt.show()

            self._gs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, i),lcfs_pad=0.001,run_info='TokaMaker EQDSK', cocos=2)
            self._gs_update(i)
            if calc_vloop:
                v_loop[i] = self._gs.calc_loopvoltage()

            lcfs_psi_target = self._gs.psi_bounds[0]
            psi0 = self._gs.get_psi(False)
        consumed_flux = np.trapz(self._times, v_loop)
        return consumed_flux
        
    def _gs_update(self, i):
        eq_stats = self._gs.get_stats()
        self._state['Ip'][i] = eq_stats['Ip']

        psi, _, _, _, _ = self._gs.get_profiles(npsi=100)
        rho_vals = np.linspace(0.0, 1.0, len(psi))
        self._state['psi'][i] = {rho: -psi[j] for j, rho in enumerate(rho_vals)}

    def _get_torax_config(self, step):
        myconfig = BASE_CONFIG.copy()

        myconfig['numerics'] = {
            't_initial': self._times[0],
            't_final': self._times[-1],  # length of simulation time in seconds
            'fixed_dt': (self._times[1] - self._times[0]) / 10.0, # fixed timestep
            'evolve_ion_heat': True, # solve ion heat equation
            'evolve_electron_heat': True, # solve electron heat equation
            'evolve_current': True, # solve current equation
            'evolve_density': True, # solve density equation
        }

        myconfig['geometry'] = {
            'geometry_type': 'eqdsk',
            'geometry_directory': '/Users/johnl/Desktop/discharge-model', 
            # 'geometry_file': self._, TODO: fill-in
            'last_surface_factor': 0.95,
            'Ip_from_parameters': True,
            'geometry_configs': {
                t: {'geometry_file': 'tmp/{:03}.{:03}.eqdsk'.format(step, i)} for i, t in enumerate(self._times)
            }
        }

        # myconfig['profile_conditions'] = {
        #     'Ip': {
        #         t: self._state['Ip'][i] for i, t in enumerate(self._times)
        #     },
        #     'psi': {
        #         t: self._state['psi'][i] for i, t in enumerate(self._times)
        #     },
        # }

        torax_config = torax.ToraxConfig.from_dict(myconfig)
        return torax_config

    def _run_transport(self, step, graph=False):
        myconfig = self._get_torax_config(step)
        data_tree, hist = torax.run_simulation(myconfig, log_timestep_info=False)
        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        v_loop = np.zeros(len(self._times))
        for i, t in enumerate(self._times):
            self._transport_update(i, data_tree)
            v_loop[i] = data_tree.profiles.v_loop.sel(time=t, rho_norm=1.0, method='nearest')
        
        if graph:
            for var in ['ffp_prof', 'pp_prof', 'eta']:
                fig, ax = plt.subplots(1, len(self._times))
                fig.suptitle(var)
                for i, _ in enumerate(self._times):
                    ax[i].plot(self._state[var][i]['x'], self._state[var][i]['y'])
                plt.show()

        consumed_flux = np.trapz(self._times, v_loop)
        return consumed_flux

    def _transport_update(self, i, data_tree):
        t = self._times[i]

        self._state['R'][i] = np.abs(data_tree.scalars.R_major.sel(time=t, method='nearest'))
        self._state['a'][i] = np.abs(data_tree.scalars.a_minor.sel(time=t, method='nearest'))
        self._state['kappa'][i] = data_tree.profiles.elongation.sel(time=t, method='nearest')[-1] # TODO: inspect

        # print("TORAX DELTA")
        self._state['deltaU'][i] = data_tree.profiles.delta_upper.sel(time=t, rho_face_norm=1.0, method='nearest').to_numpy()
        self._state['deltaL'][i] = data_tree.profiles.delta_lower.sel(time=t, rho_face_norm=1.0, method='nearest').to_numpy()
        self._state['delta'][i] = (self._state['deltaU'][i] + self._state['deltaL'][i]) / 2.0

        self._state['Ip'][i] = data_tree.scalars.Ip.sel(time=t, method='nearest')

        eta = 1.0 / data_tree.profiles.sigma_parallel.sel(time=t, method='nearest')
        self._state['eta'][i] = {
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
        self._state['pp_prof'][i] = {
            'x': pprime.coords['rho_face_norm'].values,
            'y': pprime.to_numpy(),
            'type': 'linterp',
        }

    def fly(self, convergence_threshold=1.0E-6):
        err = convergence_threshold + 1.0
        step = 0

        while err > convergence_threshold:
            cflux_gs = self._run_gs(step, calc_vloop=False, graph=True)
            cflux_transport = self._run_transport(step, graph=True)

            err = (cflux_gs - cflux_transport) ** 2
            step += 1