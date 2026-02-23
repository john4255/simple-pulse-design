from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import json

shot_tag = sys.argv[1]

eq_names = sorted(os.listdir(f'{shot_tag}/eqs_safe'))
if shot_tag == '163303':
    eq_names = sorted(os.listdir(f'{shot_tag}/eqs_cocos2'))
t_eq = []
q_eq = {}
for fname in eq_names:
    if 'OMFIT' in fname or 'DS_Store' in fname:
        continue
    t = fname.split('.')[0].split('-')[1]
    t = float(t) / 1e3
    t_eq.append(t)
    if shot_tag == '163303':
        g = read_eqdsk(f'{shot_tag}/eqs_cocos2/{fname}')
    else:
        g = read_eqdsk(f'{shot_tag}/eqs_safe/{fname}')
    qpsi = g['qpsi']
    if shot_tag == '163303':
        qpsi = -qpsi
    q_eq[t] = qpsi

if shot_tag == '163303':
    t_eq = t_eq[::8]

# def interp_prof(time, profs, times):
#     if time <= times[0]:
#         return profs[times[0]]
#     for i in range(1, len(times)):
#         if time > times[i-1] and time < times[i]:
#             dt = times[i] - times[i-1]
#             alpha = (time - times[i-1]) / dt
#             return (1.0 - alpha) * profs[times[i-1]] + alpha * profs[times[i]]
#     return profs[times[-1]]

f = 'res.json'
res = {}
with open(f, 'r') as f:
    res = json.loads(f.read())

# times = t_eq[::2]
rampup_t = 0.8
if shot_tag == '163303':
    rampup_t = 1.0
times = sorted([t for t in t_eq if t > rampup_t])

q = res['q']

fig, ax = plt.subplots(2,10, figsize=(12, 8))
for i in range(2):
    for j in range(10):
        if 10*i + j >= len(times):
            break
        t = times[10*i + j]
        q_prof = q[str(t)]
        ax[i][j].plot(q_prof['x'], q_prof['y'], c='lime', label='Simulation')

        x0 = np.linspace(0.0, 1.0, len(q_eq[t]))
        y0 = q_eq[t]
        ax[i][j].plot(x0, y0, c='magenta', label='Experiment')

        ax[i][j].set_title(f't={t}')
        ax[i][j].set_xlabel(r'$\psi$', labelpad=0)
        if j == 0:
            ax[i][j].set_ylabel(r'$q$')


plt.suptitle(f'q ({shot_tag})')
plt.legend()
plt.savefig(f'{shot_tag}-q.png')
plt.show()