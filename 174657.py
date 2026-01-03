import os
import json

import numpy as np
from model import DISMAL

eqdsk_names = sorted(os.listdir('174657/eqs_safe'))
eqdsk = []
eqtimes = []
for fname in eqdsk_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    tag, _ = fname.split('.')
    _, t = tag.split('-')
    t = float(t) / 1e3
    eqtimes.append(t)
    eqdsk.append(f'174657/eqs_safe/{fname}')

prof_names = sorted(os.listdir('174657/profs'))
prof_t = []
for fname in prof_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    t = fname.split('.')[1]
    t = float(t) / 1e3
    if t >= 0.5:
        prof_t.append(t)

prof_t = sorted(prof_t)

print(f'profs t = {prof_t[0]}')

# alltimes = np.append(prof_t[::10], eqtimes)

mysim = DISMAL(0.0, 5.5, eqtimes, eqdsk, dt=1.0E-2, times=eqtimes)
mysim.initialize_gs('174657/DIIID_mesh.h5')

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

ip_f = open('174657/ip.json')
ip = json.load(ip_f)
ip = {t / 1e3: abs(ip['data'][i]) for i, t in enumerate(ip['time'])}
mysim.set_Ip(ip)

ech_f = open('174657/pech.json')
ech = json.load(ech_f)
ech = {t / 1e3: ech['data'][i] for i, t in enumerate(ech['time'])}
# ech = {t: 1e5 for i, t in enumerate(ech['time'])}
inj_f = open('174657/pinj.json')
inj = json.load(inj_f)
inj = {t / 1e3: 1e3 * inj['data'][i] for i, t in enumerate(inj['time'])}
mysim.set_heating(eccd=ech, eccd_loc=0.0, nbi=inj, nbi_loc=0.0) # TODO: fix

zeff_f = open('174657/zeff.json')
zeff = json.load(zeff_f)
zeff = {float(k): v for k, v in zeff.items()}
mysim.set_Zeff(zeff)

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

prof_names = sorted(os.listdir('174657/profs'))
prof_times = []
for fname in prof_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    _, t, _ = fname.split('.')
    t = float(t) / 1e3
    prof_times.append(t)

    path = f'174657/profs/{fname}'
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

# Smooth Pedestals
Te_ped_list = [T_e_ped[t] for t in prof_times]
Ti_ped_list = [T_i_ped[t] for t in prof_times]
ne_ped_list = [n_e_ped[t] for t in prof_times]

Te_bc_list = [T_e_right_bc[t] for t in prof_times]
Ti_bc_list = [T_i_right_bc[t] for t in prof_times]
ne_bc_list = [n_e_right_bc[t] for t in prof_times]

nbar_list = [nbar[t] for t in prof_times]

t_inc = np.linspace(0.9, 1.3, 5000)
for t in t_inc:
    T_e_ped[t] = np.interp(t, prof_times, Te_ped_list)
    T_i_ped[t] = np.interp(t, prof_times, Ti_ped_list)
    n_e_ped[t] = np.interp(t, prof_times, ne_ped_list)

    T_e_right_bc[t] = np.interp(t, prof_times, Te_bc_list)
    T_i_right_bc[t] = np.interp(t, prof_times, Ti_bc_list)
    n_e_right_bc[t] = np.interp(t, prof_times, ne_bc_list)

    nbar[t] = np.interp(t, prof_times, nbar_list)

mysim.set_nbar({0.56: nbar[0.56]})
mysim.set_pedestal(T_e_ped=T_e_ped, T_i_ped=T_i_ped, n_e_ped=n_e_ped, ped_top=0.95)
# mysim.set_right_bc(Te_right_bc={0.56: T_e_right_bc[0.56]},
#                    Ti_right_bc={0.56: T_i_right_bc[0.56]},
#                    ne_right_bc={0.56: n_e_right_bc[0.56]})
# mysim.set_right_bc(Te_right_bc=T_e_right_bc[0.56],
#                    Ti_right_bc=T_i_right_bc[0.56],
#                    ne_right_bc=n_e_right_bc[0.56])

Te_init = {0.5: Te[0.5]}
Ti_init = {0.5: Ti[0.5]}
ne_init = {0.5: ne[0.5]}
# mysim.set_density(ne)
mysim.set_Te(Te_init)
mysim.set_Ti(Ti_init)
mysim.set_density(ne_init)
# mysim.set_evolve(density=False)

# gaspuff_s = {0.0: 0.0, 2.0: 5.0e21}
gaspuff_s = {0.5: 0.0, 1.0: 1.0e21, 0.8: 1.0e22, 1.5: 1.0e22}
mysim.set_gaspuff(s=gaspuff_s, decay_length=0.2) # 1.0e22

mysim.fly(graph=False)