import torax
from matplotlib import pyplot as plt
import numpy as np
from typing import Any
import json
from defaultconfig import default_tconfig
from decimal import Decimal
from visualization import graph_var

N_PSI = 100
N_RHO = 25
LCFS_WEIGHT = 100.0

def update_config(step, sim_vars, times, calc_vloop=True):
    myconfig = default_tconfig.copy()
    
    myconfig['geometry'] = {
        'geometry_type': 'eqdsk',
        # 'geometry_file': geo_file,
        'geometry_directory': '/Users/johnl/Desktop/discharge-model',
        'last_surface_factor': 0.9,
        'n_surfaces': 100,
        # 'nrho': N_RHO,
        # 'Ip_from_parameters': False,
        'geometry_configs': {
            t: {'geometry_file': 'tmp/{:03}.{:03}.eqdsk'.format(step, i)} for i, t in enumerate(times)
        }
    }
    myconfig['numerics'] = {
        't_initial': times[0],
        't_final': times[-1],
        'fixed_dt': (times[1] - times[0]) / 100.0,
        'evolve_ion_heat': True, # solve ion heat equation
        'evolve_electron_heat': True, # solve electron heat equation
        'evolve_current': False, # solve current equation
        'evolve_density': False, # solve density equation
    }
    
    # rho = np.linspace(0.0, 1.0, N_RHO)

    myconfig['profile_conditions'] = {
        'Ip': {
            # 0.0: sim_vars['Ip'][0]
            t: sim_vars['Ip'][i] for i, t in enumerate(times)
        },
        'psi': {
            t: sim_vars['psi'][i] for i, t in enumerate(times)
        },
        'T_e': {
            0.0: sim_vars['T_e'][0]
            # t: sim_vars['T_e'][i] for i, t in enumerate(times)
        },
        'T_i': {
            0.0: sim_vars['T_i'][0]
            # t: sim_vars['T_i'][i] for i, t in enumerate(times)
        },
        'n_e': {
            # 0.0: sim_vars['n_e'][0]
            t: sim_vars['n_e'][i] for i, t in enumerate(times)
        },
        # 'nbar': 0.85,
    }
    if calc_vloop:
        myconfig['profile_conditions']['v_loop_lcfs'] = {0.0: sim_vars['v_loop'][0]}
    torax_config = torax.ToraxConfig.from_dict(myconfig)
    return torax_config

def init_vars(times, g_eqdsk, a_eqdsk, p_eqdsk):
    sim_vars = {}

    # Initialize Scalars
    sim_vars['R'] = np.zeros(len(times))
    sim_vars['Z'] = np.zeros(len(times))
    sim_vars['a'] = np.zeros(len(times))
    sim_vars['kappa'] = np.zeros(len(times))
    sim_vars['delta'] = np.zeros(len(times))    
    sim_vars['deltaU'] = np.zeros(len(times))    
    sim_vars['deltaL'] = np.zeros(len(times))    
    sim_vars['B0'] = np.zeros(len(times))
    sim_vars['V0'] = np.zeros(len(times))
    sim_vars['zbot'] = np.zeros(len(times))
    sim_vars['ztop'] = np.zeros(len(times))
    sim_vars['rbot'] = np.zeros(len(times))
    sim_vars['rtop'] = np.zeros(len(times))
    sim_vars['Ip'] = np.zeros(len(times))
    sim_vars['pax'] = np.zeros(len(times))
    sim_vars['v_loop'] = 1.5 * np.ones(len(times))

    # Initialize Profiles
    sim_vars['ffp_prof'] = {}
    sim_vars['pp_prof'] = {}
    sim_vars['eta'] = {}
    sim_vars['psi'] = {}
    sim_vars['T_e'] = {}
    sim_vars['T_i'] = {}
    sim_vars['n_e'] = {}
    sim_vars['n_i'] = {}
    sim_vars['f_pol'] = {}
    sim_vars['vol'] = {}
    sim_vars['area'] = {}
    sim_vars['Bp'] = {}

    # Setup Profiles
    ffprim = g_eqdsk['ffprim']
    pprime = g_eqdsk['pprime']

    # plt.plot(ffprim)
    # Normalize ffprim
    # ffprim /= ffprim[0]
    # pprime /= pprime[0]
    psi_eqdsk = np.linspace(0.0,1.0,len(ffprim))
    psi_sample = np.linspace(0.0,1.0,N_PSI)
    ffp_prof = np.interp(psi_sample,psi_eqdsk,ffprim)
    pp_prof = np.interp(psi_sample,psi_eqdsk,pprime)

    # ffp_prof -= np.min(ffp_prof)
    # pp_prof -= np.min(pp_prof)
    # ffp_prof /= np.max(ffp_prof)
    # pp_prof /= np.max(pp_prof)

    # Shaping parameters
    zmax = np.max(g_eqdsk['rzout'][:,1])
    zmin = np.min(g_eqdsk['rzout'][:,1])
    rmax = np.max(g_eqdsk['rzout'][:,0])
    rmin = np.min(g_eqdsk['rzout'][:,0])
    minor_radius = (rmax - rmin) / 2.0
    rgeo = (rmax + rmin) / 2.0
    highest_pt_idx = np.argmax(g_eqdsk['rzout'][:,1])
    lowest_pt_idx = np.argmin(g_eqdsk['rzout'][:,1])
    rupper = g_eqdsk['rzout'][highest_pt_idx][0]
    rlower = g_eqdsk['rzout'][lowest_pt_idx][0]
    delta_upper = (rgeo - rupper) / minor_radius
    delta_lower = (rgeo - rlower) / minor_radius

    for i,_ in enumerate(times):
        # Default Scalars
        sim_vars['R'][i] = g_eqdsk['rcentr']
        sim_vars['Z'][i] = g_eqdsk['zmid']
        sim_vars['a'][i] = minor_radius
        sim_vars['kappa'][i] = (zmax - zmin) / (2.0 * minor_radius)
        sim_vars['delta'][i] = (delta_upper + delta_lower) / 2.0
        sim_vars['deltaU'][i] = delta_upper
        sim_vars['deltaL'][i] = delta_lower
        sim_vars['B0'][i] = g_eqdsk['bcentr']
        sim_vars['V0'][i] = g_eqdsk['zaxis']
        sim_vars['zbot'][i] = zmin
        sim_vars['ztop'][i] = zmax
        sim_vars['rbot'][i] = rmin
        sim_vars['rtop'][i] = rmax
        sim_vars['Ip'][i] = abs(g_eqdsk['ip'])
        sim_vars['pax'][i] = g_eqdsk['pres'][0]
        sim_vars['v_loop'][i] = 0.0

        # Default Profiles
        sim_vars['ffp_prof'][i] = {psi_sample[i]: ffp_prof[i] for i in range(len(psi_sample))}
        sim_vars['pp_prof'][i] = {psi_sample[i]: pp_prof[i] for i in range(len(psi_sample))}

        sim_vars['eta'][i]= {}
        sim_vars['psi'][i] = {}

        # print(p_eqdsk)
        sim_vars['T_e'][i] = p_eqdsk['te(KeV)']
        sim_vars['T_i'][i] = p_eqdsk['ti(KeV)']
        sim_vars['n_e'][i] = {key: 1.0E20 * val for key, val in p_eqdsk['ne(10^20/m^3)'].items()}
        sim_vars['n_i'][i] = p_eqdsk['ni(10^20/m^3)']
        sim_vars['f_pol'][i] = g_eqdsk['fpol']
        
        # sim_vars['vol'][i] = aeqdsk['vout']
        # sim_vars['area'] = aeqdsk['psurfa']
        # sim_vars['Bp'] = aeqdsk['betap']

    return sim_vars

def transport_update(sim_vars, i, times, data_tree):
    # psi_interp1 = np.linspace(0.0, 1.0, len(data_tree.profiles.FFprime[0]))
    # psi_interp2 = np.linspace(0.0, 1.0, len(data_tree.profiles.pprime[0]))
    t = times[i]

    sim_vars['R'][i] = np.abs(data_tree.scalars.R_major.sel(time=t, method='nearest'))
    sim_vars['a'][i] = np.abs(data_tree.scalars.a_minor.sel(time=t, method='nearest'))
    sim_vars['kappa'][i] = data_tree.profiles.elongation.sel(time=t, method='nearest')[-1] # TODO: inspect

    # rmax = data_tree.profiles.R_in.sel(time=t, method='nearest')[-1]
    # rmin = data_tree.profiles.R_in.sel(time=t, method='nearest')[-1]
    # rgeo = (rmax + rmin) / 2.0
    # highest_pt_idx = np.argmax
    # deltaU = (rgeo - rupper) / sim_vars['a']
    # sim_vars['delta'][i] = 0.0

    sim_vars['Ip'][i] = data_tree.scalars.Ip.sel(time=t, method='nearest')
    # sim_vars['pax'][i] = 0.0
    # if calc_vloop:
    #     sim_vars['v_loop'][i] = data_tree.profiles.v_loop[t][-1]
    eta_prof = 1.0 / data_tree.profiles.sigma_parallel.sel(time=t, method='nearest').to_numpy()
    sim_vars['eta'][i] = dict(zip(
        data_tree.profiles.sigma_parallel.sel(time=t, method='nearest').coords['rho_norm'].values,
        eta_prof
    ))

    ffp_prof = data_tree.profiles.FFprime.sel(time=t, method='nearest').to_numpy()
    pp_prof = data_tree.profiles.pprime.sel(time=t, method='nearest').to_numpy()

    plt.plot(ffp_prof)

    # ffp_prof /= ffp_prof[0]
    # pp_prof /= pp_prof[0]
    ffp_prof /= np.max(abs(ffp_prof))
    pp_prof /= np.max(abs(pp_prof))
    ffp_prof -= np.min(ffp_prof)
    pp_prof -= np.min(pp_prof)
    # ffp_prof[-1] = 0
    # pp_prof[-1] = 0

    sim_vars['ffp_prof'][i] = dict(zip(
        data_tree.profiles.FFprime.sel(time=t, method='nearest').coords['rho_face_norm'].values,
        ffp_prof
    ))
    sim_vars['pp_prof'][i] = dict(zip(
        data_tree.profiles.pprime.sel(time=t, method='nearest').coords['rho_face_norm'].values,
        pp_prof
    ))
    # sim_vars['pp_prof'][i] = {0.0: 1.0, 1.0: 0.0}

    sim_vars['T_e'][i] = dict(zip(
        data_tree.profiles.T_e.sel(time=t, method='nearest').coords['rho_norm'].values,
        data_tree.profiles.T_e.sel(time=t, method='nearest').to_numpy()
    ))
    sim_vars['T_i'][i] = dict(zip(
        data_tree.profiles.T_i.sel(time=t, method='nearest').coords['rho_norm'].values,
        data_tree.profiles.T_i.sel(time=t, method='nearest').to_numpy()
    ))
    sim_vars['n_e'][i] = dict(zip(
        data_tree.profiles.n_e.sel(time=t, method='nearest').coords['rho_norm'].values,
        data_tree.profiles.n_e.sel(time=t, method='nearest').to_numpy()
    ))

    return sim_vars

def gs_update(sim_vars, i, mygs, calc_vloop=True):
    eq_stats = mygs.get_stats()
    psi,f,fp, _, pp = mygs.get_profiles(npsi=N_PSI)

    # Update scalars
    sim_vars['R'][i] = np.abs(mygs.o_point[0])
    sim_vars['Z'][i] = np.abs(mygs.o_point[1])
    sim_vars['kappa'][i] = np.abs(eq_stats['kappa'])
    sim_vars['delta'][i] = np.abs(eq_stats['delta'])
    sim_vars['deltaU'][i] = np.abs(eq_stats['deltaU'])
    sim_vars['deltaL'][i] = np.abs(eq_stats['deltaL'])
    sim_vars['zbot'][i] = mygs.x_points[0,1]
    sim_vars['ztop'][i] = mygs.x_points[1,1]
    sim_vars['rbot'][i] = mygs.x_points[0,0]
    sim_vars['rtop'][i] = mygs.x_points[1,0]
    sim_vars['a'][i] = np.abs((sim_vars['rtop'][i] - sim_vars['rbot'][i]) / 2.0)
    sim_vars['Ip'][i] = eq_stats['Ip']

    # Update profiles
    mu0 = np.pi*4.E-7

    sim_vars['f_pol'][i] = -f

    rho_vals = np.linspace(0.0, 1.0, len(psi))
    sim_vars['psi'][i] = {rho: -psi[i] for i, rho in enumerate(rho_vals)} # TODO: get psi from GS or transport?
    # sim_vars['ffp_prof'][i] = fp
    # plt.plot(fp)
    sim_vars['pp_prof'][i] = pp * mu0

    if calc_vloop:
        sim_vars['v_loop'][i] = mygs.calc_loopvoltage()
    return sim_vars

def set_coil_reg(mygs, machine_dict, e_coil_dict, f_coil_dict):
    # Set coil regularization to weakly track measured coil currents
    regularization_terms = []
    for key in e_coil_dict:
        if e_coil_dict[key][1] == 0:
            print('{} not selected.'.format(key))
            continue
        regularization_terms.append(mygs.coil_reg_term({key: 1.0},
                                                    target=e_coil_dict[key][0],
                                                    weight=1.0E1))
    for key in f_coil_dict:
        if f_coil_dict[key][1] == 0:
            print('{} not selected.'.format(key))
            continue
        regularization_terms.append(mygs.coil_reg_term({key: 1.0},
                                                    target=f_coil_dict[key][0] / machine_dict['FCOIL'][key][4],
                                                    weight=1.0E2))

    # Set zero target current and small weight on virtual VSC to allow up-down adjustment
    regularization_terms.append(mygs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E-2))

    # Pass regularization terms to TokaMaker
    mygs.set_coil_reg(reg_terms=regularization_terms)

def get_boundary(sim_vars, i, npts=20):
    thp = np.linspace(0, 2*np.pi, npts+1)
    thp = thp[:-1]

    r0 = sim_vars['R'][i]
    z0 = sim_vars['Z'][i]
    a0 = sim_vars['a'][i]
    kappa = sim_vars['kappa'][i]
    delta = sim_vars['delta'][i]
    squar = 0.0 # sim_vars['squar'][i]

    ra = r0 + a0*np.cos(thp + delta*np.sin(thp) - squar*np.sin(2*thp))
    za = z0 + kappa*a0*np.sin(thp + squar*np.sin(2*thp))
    return np.vstack([ra, za]).transpose()

def run_eqs(mygs, sim_vars, times, machine_dict, e_coil_dict, f_coil_dict, g_eqdsk, step, calc_vloop=True, graph=False, verbose=False):
    if verbose:
        print("\n\n\n")
        print("=== SIMVARS ===")
        i = 1
        for key, val in sim_vars.items():
            if type(val) is list or type(val) is np.ndarray:
                print(key)
                print(val[i])
            elif len(val) == 0:
                continue
            elif key in 'ffp_prof pp_prof':
                vals = [v for _, v in val.items()]
                print("min {} = {}".format(key, np.min(vals)))
            elif type(val) is dict:
                continue
            else:
                print(key)
                print("({}, {})".format(np.max(val[i]), np.min(val[i])))
        print("\n\n\n")

    if graph:
        graph_var(sim_vars, 'ffp_prof', step)
        graph_var(sim_vars, 'pp_prof', step)
        if step > 0:
            graph_var(sim_vars, 'eta', step)
        
    save_state(sim_vars, step)

    # sim_vars['v_loop'] = np.zeros(len(times))

    vsc_signs = {key: 0 for key in mygs.coil_sets}
    vsc_signs['F9A'] = 1.0
    vsc_signs['F9B'] = -1.0
    mygs.set_coil_vsc(vsc_signs)

    mygs.set_targets(Ip=sim_vars['Ip'][0], pax=sim_vars['pax'][0], V0=sim_vars['V0'][0])

    psi_sample = np.linspace(0.0, 1.0, N_PSI)
    ffp_rho = list(sim_vars['ffp_prof'][0].keys())
    pp_rho = list(sim_vars['pp_prof'][0].keys())
    ffp_vals = [sim_vars['ffp_prof'][0][rho] for rho in ffp_rho]
    pp_vals = [sim_vars['pp_prof'][0][rho] for rho in pp_rho]
    ffp_interp = np.interp(psi_sample, ffp_rho, ffp_vals)
    pp_interp = np.interp(psi_sample, pp_rho, pp_vals)
    mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': ffp_interp, 'x': psi_sample},
                      pp_prof={'type': 'linterp', 'y': pp_interp, 'x': psi_sample},
                      foffset=sim_vars['R'][0]*sim_vars['B0'][0])

    set_coil_reg(mygs, machine_dict, e_coil_dict, f_coil_dict)
    mygs.set_flux(None,None)

    # mygs.set_resistivity(eta_prof={'type': 'linterp', 'x': np.linspace(0.0, 1.0, N_PSI), 'y': 1.0E-7 * np.ones(N_PSI)})
    
    if calc_vloop:
        psi_sample = np.linspace(0.0, 1.0, N_PSI)
        eta_rho_vals = list(sim_vars['eta'][0].keys())
        eta_prof = [sim_vars['eta'][0][rho] for rho in eta_rho_vals]
        eta_interp = np.interp(psi_sample, eta_rho_vals, eta_prof)
        mygs.set_resistivity(eta_prof={'type': 'linterp', 'x': psi_sample, 'y': eta_interp})

    err_flag = mygs.init_psi(sim_vars['R'][0],
                             sim_vars['Z'][0],
                             sim_vars['a'][0],
                             sim_vars['kappa'][0], 
                             sim_vars['delta'][0])

    if err_flag:
        print("Error initializing Psi.")

    # lcfs = g_eqdsk['rzout']
    lcfs = get_boundary(sim_vars, 0)
    isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))
    mygs.set_isoflux(lcfs, isoflux_weights)

    mygs.update_settings()
    try:
        err_flag = mygs.solve()
        if err_flag:
            print("Error during initial solve.")
    except:
        print("Solve failed.")
        save_state(sim_vars, step)
        graph_var(sim_vars, 'ffp_prof', step)
        graph_var(sim_vars, 'pp_prof', step)
        if step > 0:
            graph_var(sim_vars, 'eta', step)
        quit()

    sim_vars = gs_update(sim_vars, 0, mygs, calc_vloop=calc_vloop)

    # print(lcfs_psi_target)
    # print(mygs.psi_bounds[0])

    if graph:
        fig, ax = plt.subplots(1,1)
        mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
        mygs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
        ax.plot(g_eqdsk['rzout'][:, 0], g_eqdsk['rzout'][:, 1], color='r')
        plt.show()

    mygs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, 0),lcfs_pad=0.001,run_info='TokaMaker EQDSK', cocos=2)
    lcfs_psi_target = mygs.psi_bounds[0]
    psi0 = mygs.get_psi(False)

    mygs.set_isoflux(None)

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]

        mygs.set_targets(Ip=sim_vars['Ip'][i], Ip_ratio=1.0E-2, pax=sim_vars['pax'][i], V0=sim_vars['V0'][i])

        psi_sample = np.linspace(0.0,1.0,N_PSI)
        ffp_rho = list(sim_vars['ffp_prof'][i].keys())
        pp_rho = list(sim_vars['pp_prof'][i].keys())
        ffp_vals = [sim_vars['ffp_prof'][i][rho] for rho in ffp_rho]
        pp_vals = [sim_vars['pp_prof'][i][rho] for rho in pp_rho]
        ffp_interp = np.interp(psi_sample, ffp_rho, ffp_vals)
        pp_interp = np.interp(psi_sample, pp_rho, pp_vals)
        mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': ffp_interp, 'x': psi_sample},
                        pp_prof={'type': 'linterp', 'y': pp_interp, 'x': psi_sample},
                        foffset=sim_vars['R'][i]*sim_vars['B0'][i])
        
        mygs.set_psi_dt(psi0,dt)

        if calc_vloop:
            psi_sample = np.linspace(0.0, 1.0, N_PSI)
            eta_rho_vals = list(sim_vars['eta'][i].keys())
            eta_prof = [sim_vars['eta'][i][rho] for rho in eta_rho_vals]
            eta_interp = np.interp(psi_sample, eta_rho_vals, eta_prof)
            mygs.set_resistivity(eta_prof={'type': 'linterp', 'x': psi_sample, 'y': eta_interp})

        set_coil_reg(mygs, machine_dict, e_coil_dict, f_coil_dict)

        lcfs = get_boundary(sim_vars, i, len(lcfs))
        isoflux_weights = LCFS_WEIGHT * np.ones(len(lcfs))

        vloop = sim_vars['v_loop'][i]
        # vloop = 0.1
        lcfs_psi_target -= dt * vloop / 2 / np.pi
        mygs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)

        mygs.set_psi_dt(psi0,dt)
        try:
            err_flag = mygs.solve()
            if err_flag:
                print("Error solving at t={}.".format(times[i]))
        except:
            print("Solve failed.")
            print(sim_vars)
            return sim_vars, 0.0
        
        mygs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, i), lcfs_pad=0.001, run_info='TokaMaker  EQDSK', cocos=2)
        sim_vars = gs_update(sim_vars, i, mygs, calc_vloop=calc_vloop)

        if graph:
            fig, ax = plt.subplots(1,1)
            mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
            mygs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
            plt.show()
        print(lcfs_psi_target)
        print(mygs.psi_bounds[0])

        lcfs_psi_target = mygs.psi_bounds[0]
        psi0 = mygs.get_psi(False)


    consumed_flux = 0.0
    if calc_vloop:
        consumed_flux = np.trapz(times, sim_vars['v_loop'])
    return sim_vars, np.abs(consumed_flux)

def run_sims(sim_vars, times, step):
    t_config = update_config(step, sim_vars, times, calc_vloop=True)
    data_tree, hist = torax.run_simulation(t_config, log_timestep_info=False)

    if hist.sim_error != torax.SimError.NO_ERROR:
        print(hist.sim_error)
        raise ValueError(f'TORAX failed to run the simulation.')
    
    for i in range(len(times)):
        sim_vars = transport_update(sim_vars, i, times, data_tree)

    for i, t in enumerate(times):
        sim_vars['v_loop'][i] = data_tree.profiles.v_loop.sel(time=t, method='nearest').to_numpy()[-1]
    consumed_flux = np.trapz(times, sim_vars['v_loop'])

    return sim_vars, np.abs(consumed_flux)

def save_state(sim_vars, step):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open('state_{}.json'.format(step), 'w') as f:
        json.dump(sim_vars, f, cls=MyEncoder)