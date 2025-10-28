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
times = times[1:]
eqdsk = eqdsk[1:]

mysim = CGTS(1.0, 5.0, times, eqdsk, dt=1.0E-2)
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
mysim.set_heating(eccd=ech, eccd_loc=0.3, nbi=inj, nbi_loc=0.3)

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

for path in eqdsk:
    name = path.split('/')[-1]
    name = name.split('.')[0]
    t_tag = int(name.split('-')[1][1:])
    time = float(t_tag) / 1e3
    t_tag = f'{t_tag:05d}'
    path = f'163303/profs/p163303.{t_tag}'
    data = read_pfile(path)

    psi_space = sorted(data['te(KeV)'].keys())
    te_space = [data['te(KeV)'][psi] for psi in psi_space]
    te_ped = np.interp(0.95, psi_space, te_space)

    psi_space = sorted(data['ti(KeV)'].keys())
    ti_space = [data['ti(KeV)'][psi] for psi in psi_space]
    ti_ped = np.interp(0.95, psi_space, ti_space)

    psi_space = sorted(data['ne(10^20/m^3)'].keys())
    ne_space = [data['ne(10^20/m^3)'][psi] * 1e20 for psi in psi_space]
    ne_ped = np.interp(0.95, psi_space, ne_space)

    nbar[time] = np.mean(ne_space)

    T_e_ped[time] = te_ped
    T_i_ped[time] = ti_ped
    n_e_ped[time] = ne_ped

    T_e_right_bc[time] = te_space[-1]
    T_i_right_bc[time] = ti_space[-1]
    n_e_right_bc[time] = ne_space[-1]

mysim.set_nbar(nbar)
mysim.set_pedestal(T_e_ped=T_e_ped, T_i_ped=T_i_ped, n_e_ped=n_e_ped)
mysim.set_right_bc(Te_right_bc=T_e_right_bc, Ti_right_bc=T_i_right_bc, ne_right_bc=n_e_right_bc)

mysim.fly(graph=False)