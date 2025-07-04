import copy
import jax
import torax
# from torax._src.geometry import standard_geometry
from torax._src.geometry import geometry
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from typing import Any
from defaultconfig import default_tconfig
# from freeqdsk import geqdsk
from decimal import Decimal
from visualization import graph_var

N_PSI = 50
N_RHO = 50

def update_config(geo_file, sim_vars, times, i, calc_vloop=True):
    myconfig = default_tconfig.copy()
    myconfig['geometry'] = {
        'geometry_type': 'eqdsk',
        'geometry_file': geo_file,
        'geometry_directory': '/Users/johnl/Desktop/discharge-model',
        'last_surface_factor': 0.95,
        # 'n_surfaces': 50,
        # 'nrho': N_RHO,
    }
    myconfig['numerics'] = {
        't_initial': times[i],
        't_final': times[i + 1],
        'fixed_dt': (times[i + 1] - times[i]) / 10,
    }
    if calc_vloop:
        myconfig['profile_conditions']['v_loop_lcfs'] = {times[i]: sim_vars['v_loop'][i][-1] for i in range(len(times))}
    torax_config = torax.ToraxConfig.from_dict(myconfig)
    return torax_config

def init_vars(times, geqdsk, aeqdsk, pdict):
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
    ffprim = geqdsk['ffprim']
    pprime = geqdsk['pprime']
    # Normalize ffprim
    ffprim /= ffprim[0]
    pprime /= pprime[0]
    psi_eqdsk = np.linspace(0.0,1.0,len(ffprim))
    psi_sample = np.linspace(0.0,1.0,N_PSI)
    ffp_prof = np.interp(psi_sample,psi_eqdsk,ffprim)
    pp_prof = np.interp(psi_sample,psi_eqdsk,pprime)

    ffp_prof /= np.max(ffp_prof)
    pp_prof /= np.max(pp_prof)
    ffp_prof -= np.min(ffp_prof)
    pp_prof -= np.min(pp_prof)

    # Shaping parameters
    zmax = np.max(geqdsk['rzout'][:,1])
    zmin = np.min(geqdsk['rzout'][:,1])
    rmax = np.max(geqdsk['rzout'][:,0])
    rmin = np.min(geqdsk['rzout'][:,0])
    minor_radius = (rmax - rmin) / 2.0
    rgeo = (rmax + rmin) / 2.0
    highest_pt_idx = np.argmax(geqdsk['rzout'][:,1])
    lowest_pt_idx = np.argmin(geqdsk['rzout'][:,1])
    rupper = geqdsk['rzout'][highest_pt_idx][0]
    rlower = geqdsk['rzout'][lowest_pt_idx][0]
    delta_upper = (rgeo - rupper) / minor_radius
    delta_lower = (rgeo - rlower) / minor_radius

    for i,_ in enumerate(times):
        # Default Scalars
        sim_vars['R'][i] = geqdsk['rcentr']
        sim_vars['Z'][i] = geqdsk['zmid']
        sim_vars['a'][i] = minor_radius
        sim_vars['kappa'][i] = (zmax - zmin) / (2.0 * minor_radius)
        sim_vars['delta'][i] = (delta_upper + delta_lower) / 2.0
        sim_vars['deltaU'][i] = delta_upper
        sim_vars['deltaL'][i] = delta_lower
        sim_vars['B0'][i] = geqdsk['bcentr']
        sim_vars['zbot'][i] = zmin
        sim_vars['ztop'][i] = zmax
        sim_vars['rbot'][i] = rmin
        sim_vars['rtop'][i] = rmax
        sim_vars['Ip'][i] = abs(geqdsk['ip'])
        sim_vars['pax'][i] = geqdsk['pres'][0]
        sim_vars['v_loop'][i] = 0.0

        # Default Profiles
        sim_vars['ffp_prof'][i] = ffp_prof.copy()
        sim_vars['pp_prof'][i] = pp_prof.copy()

        sim_vars['eta'][i]= np.zeros(N_RHO)
        sim_vars['psi'][i] = np.zeros(N_RHO)

        # sim_vars['T_e'][i] = pdict['te(KeV)']
        # sim_vars['T_i'][i] = pdict['ti(KeV)']
        # sim_vars['n_e'][i] = pdict['ne(10^20/m^3)']
        # sim_vars['n_i'][i] = pdict['ni(10^20/m^3)']
        sim_vars['f_pol'][i] = geqdsk['fpol']
        
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
    sim_vars['eta'][i] = 1.0 / data_tree.profiles.sigma_parallel.sel(time=t, method='nearest').to_numpy()

    ffp_prof = data_tree.profiles.FFprime.sel(time=t, method='nearest').to_numpy()
    pp_prof = data_tree.profiles.pprime.sel(time=t, method='nearest').to_numpy()
    ffp_prof /= ffp_prof[0]
    pp_prof /= pp_prof[0]
    ffp_prof -= np.min(ffp_prof)
    pp_prof -= np.min(pp_prof)
    ffp_prof /= np.max(abs(ffp_prof))
    pp_prof /= np.max(abs(pp_prof))
    ffp_prof[-1] = 0
    pp_prof[-1] = 0

    sim_vars['ffp_prof'][i] = ffp_prof
    sim_vars['pp_prof'][i] = pp_prof
    
    return sim_vars

def gs_update(sim_vars, i, mygs, calc_vloop=True):
    eq_stats = mygs.get_stats()
    psi,f,fp, _, pp = mygs.get_profiles(npsi=N_PSI)

    # Update scalars
    sim_vars['R'][i] = np.abs(mygs.o_point[0])
    sim_vars['a'][i] = np.abs(mygs.o_point[1])
    sim_vars['kappa'][i] = np.abs(eq_stats['kappa'])
    sim_vars['delta'][i] = np.abs(eq_stats['delta'])
    sim_vars['deltaU'][i] = np.abs(eq_stats['deltaU'])
    sim_vars['deltaL'][i] = np.abs(eq_stats['deltaL'])
    sim_vars['zbot'][i] = mygs.x_points[0,1]
    sim_vars['ztop'][i] = mygs.x_points[1,1]
    sim_vars['rbot'][i] = mygs.x_points[0,0]
    sim_vars['rtop'][i] = mygs.x_points[1,0]
    sim_vars['Ip'][i] = eq_stats['Ip']

    # Update profiles
    mu0 = np.pi*4.E-7

    sim_vars['f_pol'][i] = -f
    sim_vars['psi'][i] = -psi
    sim_vars['ffp_prof'][i] = fp
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

def run_eqs(mygs, sim_vars, times, machine_dict, e_coil_dict, f_coil_dict, geqdsk, step, calc_vloop=True):
    print("\n\n\n")
    print("=== SIMVARS ===")
    i = 1
    for key, val in sim_vars.items():
        if type(val) is list or type(val) is np.ndarray:
            print(key)
            print(val[i])
        elif len(val) == 0:
            continue
        else:
            print(key)
            print("({}, {})".format(np.max(val[i]), np.min(val[i])))
    print("\n\n\n")
    graph_var(sim_vars, 'ffp_prof', step)

    vsc_signs = {key: 0 for key in mygs.coil_sets}
    vsc_signs['F9A'] = 1.0
    vsc_signs['F9B'] = -1.0
    mygs.set_coil_vsc(vsc_signs)

    mygs.set_targets(Ip=sim_vars['Ip'][0], Ip_ratio=1.0E-2, pax=sim_vars['pax'][0])

    psi_sample = np.linspace(0.0,1.0,N_PSI)
    psi_ffp_values = np.linspace(0.0, 1.0, len(sim_vars['ffp_prof'][0]))
    psi_pp_values = np.linspace(0.0, 1.0, len(sim_vars['pp_prof'][0]))
    ffp_interp = np.interp(psi_sample, psi_ffp_values, sim_vars['ffp_prof'][0])
    pp_interp = np.interp(psi_sample, psi_pp_values, sim_vars['pp_prof'][0])
    mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': ffp_interp, 'x': psi_sample},
                      pp_prof={'type': 'linterp', 'y': pp_interp, 'x': psi_sample},
                      foffset=sim_vars['R'][0]*sim_vars['B0'][0])

    set_coil_reg(mygs, machine_dict, e_coil_dict, f_coil_dict)
    mygs.set_flux(None,None)    

    err_flag = mygs.init_psi(sim_vars['R'][0],
                             sim_vars['Z'][0],
                             sim_vars['a'][0],
                             sim_vars['kappa'][0], 
                             sim_vars['delta'][0])

    if err_flag:
        print("Error initializing Psi.")

    # lcfs = get_boundary(sim_vars, 0)
    lcfs = geqdsk['rzout']
    isoflux_weights = np.ones(len(lcfs))
    mygs.set_isoflux(lcfs, isoflux_weights)

    mygs.update_settings()
    err_flag = mygs.solve()

    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
    mygs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
    plt.show()

    if err_flag:
        print("Error during initial solve.")

    mygs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, 0),lcfs_pad=0.001,run_info='TokaMaker EQDSK', cocos=2)
    sim_vars = gs_update(sim_vars, 0, mygs, calc_vloop=False)
    lcfs_psi_target = mygs.psi_bounds[0]
    psi0 = mygs.get_psi(False)

    mygs.set_isoflux(None)

    for i in range(1, len(times)):
        dt = times[i] - times[i-1]

        mygs.set_targets(Ip=sim_vars['Ip'][i], Ip_ratio=1.0E-2, pax=sim_vars['pax'][i])

        psi_sample = np.linspace(0.0,1.0,N_PSI)
        psi_ffp_values = np.linspace(0.0, 1.0, len(sim_vars['ffp_prof'][i]))
        psi_ffp_values = np.linspace(0.0, 1.0, len(sim_vars['ffp_prof'][i]))
        psi_pp_values = np.linspace(0.0, 1.0, len(sim_vars['pp_prof'][i]))
        ffp_interp = np.interp(psi_sample, psi_ffp_values, sim_vars['ffp_prof'][i])
        pp_interp = np.interp(psi_sample, psi_pp_values, sim_vars['pp_prof'][i])
        mygs.set_profiles(ffp_prof={'type': 'linterp', 'y': ffp_interp, 'x': psi_sample},
                        pp_prof={'type': 'linterp', 'y': pp_interp, 'x': psi_sample},
                        foffset=sim_vars['R'][i]*sim_vars['B0'][i])
        
        mygs.set_psi_dt(psi0,dt)

        if calc_vloop:
            psi_sample = np.linspace(0.0, 1.0, N_PSI)
            psi_eta_values = np.linspace(0.0, 1.0, len(sim_vars['eta'][i]))
            eta_interp = np.interp(psi_sample, psi_eta_values, sim_vars['eta'][i])
            mygs.set_resistivity(eta_prof={'type': 'linterp', 'x': psi_sample, 'y': eta_interp})

        set_coil_reg(mygs, machine_dict, e_coil_dict, f_coil_dict)

        lcfs = get_boundary(sim_vars, i)
        isoflux_weights = np.ones(len(lcfs))

        vloop = sim_vars['v_loop'][i]
        lcfs_psi_target -= dt * vloop / 2 / np.pi
        mygs.set_flux(lcfs, targets=lcfs_psi_target*np.ones_like(isoflux_weights), weights=isoflux_weights)
        mygs.set_isoflux(lcfs, isoflux_weights)

        mygs.set_psi_dt(psi0,dt)
        err_flag = mygs.solve()
        if err_flag:
            print("Error solving at t={}.".format(times[i]))
        mygs.save_eqdsk('tmp/{:03}.{:03}.eqdsk'.format(step, i),lcfs_pad=0.001,run_info='TokaMaker  EQDSK', cocos=2)

        fig, ax = plt.subplots(1,1)
        mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_symmap=True,coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]')
        mygs.plot_psi(fig,ax,xpoint_color='r',vacuum_nlevels=4)
        plt.show()

        psi0 = mygs.get_psi(False)

        sim_vars = gs_update(sim_vars, i, mygs, calc_vloop=calc_vloop)
        if calc_vloop:
            sim_vars['v_loop'][i] = mygs.calc_loopvoltage()

    consumed_flux = 0.0
    if calc_vloop:
        consumed_flux = np.trapz(sim_vars['v_loop'][:,1], sim_vars['v_loop'][:,0])
    return sim_vars, consumed_flux

def run_sims(sim_vars, times, step):
    for i, _ in enumerate(times[:-1]):
        eqdsk_file = 'tmp/{:03}.{:03}.eqdsk'.format(step, i)

        t_config = update_config(eqdsk_file, sim_vars, times, i, calc_vloop=step)
        data_tree, hist = torax.run_simulation(t_config, log_timestep_info=False)

        if hist.sim_error != torax.SimError.NO_ERROR:
            print(hist.sim_error)
            raise ValueError(f'TORAX failed to run the simulation.')
        
        sim_vars = transport_update(sim_vars, i + 1, times, data_tree)
        # graph_sim(sim_vars, i)

    return sim_vars, 0.0