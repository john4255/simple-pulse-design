import numpy as np
import matplotlib.pyplot as plt

import os
import json

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

f = 'tmp/ts_state2.json'
state = {}
with open(f, 'r') as f:
    state = json.loads(f.read())

Te_sim = {t: state['T_e'][str(i)] for i, t in enumerate(times)}

times = times[::6]
eqdsk = eqdsk[::6]

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

Te = {}
for path in eqdsk:
    name = path.split('/')[-1]
    name = name.split('.')[0]
    t_tag = int(name.split('-')[1][1:])
    time = float(t_tag) / 1e3
    t_tag = f'{t_tag:05d}'
    path = f'163303/profs/p163303.{t_tag}'
    data = read_pfile(path)

    psi_space = sorted(data['ne(10^20/m^3)'].keys())
    Te_space = [data['te(KeV)'][psi] for psi in psi_space]
    Te[time] = {psi: Te_space[i] for i, psi in enumerate(psi_space)}

fig, ax = plt.subplots(2,3)
for i in range(2):
    for j in range(3):
        t = times[3*i + j]
        ax[i][j].set_title(f't={t}')
        Te_prof = Te[t]
        x0 = sorted(Te_prof.keys())
        y0 = [Te_prof[k] for k in x0]
        ax[i][j].plot(x0, y0, label='Experiment')
        Te_sim_prof = Te_sim[t]
        x1 = Te_sim_prof['x']
        y1 = Te_sim_prof['y']
        ax[i][j].plot(x1, y1, label='Torax')
plt.legend()
plt.show()