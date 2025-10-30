import os
import json

import numpy as np
from model import CGTS

eqdsk_names = sorted(os.listdir('163303/eqs_safe'))
eqdsk = []
times = []
for fname in eqdsk_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    tag, _ = fname.split('.')
    _, t = tag.split('-')
    t = float(t) / 1e3
    times.append(t)
    eqdsk.append(f'163303/eqs_safe/{fname}')

# eqdsk1 = ['eqdsk/iter_i=0.eqdsk'] * len(eqdsk)

prof_names = sorted(os.listdir('163303/profs'))
t_res = []
for fname in prof_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    _, t = fname.split('.')
    t = float(t) / 1e3
    t_res.append(t)

mysim = CGTS(0.6, 5.0, times, eqdsk, dt=1.0E-2, t_res=t_res)
mysim.initialize_gs('163303/DIIID_mesh.h5')

target_currents = {
    'ECOILA': 0.0,
    'ECOILB': 0.0,
    'F1A': 0.0,
    'F2A': 0.0,
    'F3A': 0.0,
    'F4A': 0.0,
    'F5A': 0.0,
    'F6A': 0.0,
    'F7A': 0.0,
    'F8A': 0.0,
    'F9A': 0.0,
    'F1B': 0.0,
    'F2B': 0.0,
    'F3B': 0.0,
    'F4B': 0.0,
    'F5B': 0.0,
    'F6B': 0.0,
    'F7B': 0.0,
    'F8B': 0.0,
    'F9B': 0.0,
}
mysim.set_coil_reg(targets=target_currents)

ip_f = open('163303/ip.json')
ip = json.load(ip_f)
ip = {t: ip['data'][i] for i, t in enumerate(ip['time'])}
mysim.set_Ip(ip)

ech_f = open('163303/pech.json')
ech = json.load(ech_f)
ech = {t: ech['data'][i] for i, t in enumerate(ech['time'])}
inj_f = open('163303/pinj.json')
inj = json.load(inj_f)
inj = {t: inj['data'][i] for i, t in enumerate(inj['time'])}
mysim.set_heating(eccd=ech, eccd_loc=0.1, nbi=inj, nbi_loc=0.1)

def read_pfile(path):
    data = {}
    key = ''
    with open(path) as f:
        for line in f:
            if '3 N Z A' in line:
                break
            if line.startswith('201'):
                key = line.split()[2]
                data[key] = {}
            else:
                psi, dat, _ = line.split()
                psi = float(psi)
                dat = float(dat)
                data[key][psi] = dat
    return data

nbar = {}

T_e_ped = {}
T_i_ped = {}
n_e_ped = {}

T_e_right_bc = {}
T_i_right_bc = {}
n_e_right_bc = {}

ne = {}
Te = {}
Ti = {}

prof_names = sorted(os.listdir('163303/profs'))
prof_times = []
for fname in prof_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    _, t = fname.split('.')
    t = float(t) / 1e3
    prof_times.append(t)

    path = f'163303/profs/{fname}'
    data = read_pfile(path)

    psi_space = sorted(data['te(KeV)'].keys())
    rho_space = [np.sqrt(psi) for psi in psi_space]
    te_space = [data['te(KeV)'][psi] for psi in psi_space]
    Te[t] = {rho: te_space[i] for i, rho in enumerate(rho_space)}
    te_ped = np.interp(0.95, rho_space, te_space)

    psi_space = sorted(data['ti(KeV)'].keys())
    rho_space = [np.sqrt(psi) for psi in psi_space]
    ti_space = [data['ti(KeV)'][psi] for psi in psi_space]
    Ti[t] = {rho: ti_space[i] for i, rho in enumerate(rho_space)}
    ti_ped = np.interp(0.95, rho_space, ti_space)

    psi_space = sorted(data['ne(10^20/m^3)'].keys())
    rho_space = [np.sqrt(psi) for psi in psi_space]
    ne_space = [data['ne(10^20/m^3)'][psi] * 1e20 for psi in psi_space]
    ne[t] = {rho: ne_space[i] for i, rho in enumerate(rho_space)} # Use rho coords
    ne_ped = np.interp(0.95, rho_space, ne_space)
    nbar[t] = np.mean(ne_space)

    T_e_ped[t] = te_ped
    T_i_ped[t] = ti_ped
    n_e_ped[t] = ne_ped

    T_e_right_bc[t] = te_space[-1]
    T_i_right_bc[t] = ti_space[-1]
    n_e_right_bc[t] = ne_space[-1]

mysim.set_nbar(nbar)
mysim.set_pedestal(T_e_ped=T_e_ped, T_i_ped=T_i_ped, n_e_ped=n_e_ped)
mysim.set_right_bc(Te_right_bc=T_e_right_bc, Ti_right_bc=T_i_right_bc, ne_right_bc=n_e_right_bc)

Te_init = {1.02: Te[1.02]}
Ti_init = {1.02: Ti[1.02]}
ne_init = {1.02: ne[1.02]}
# mysim.set_density(ne)
mysim.set_Te(Te_init)
mysim.set_Ti(Ti_init)
mysim.set_density(ne_init)
# mysim.set_evolve(density=False)

mysim.fly(save_states=True, graph=False)