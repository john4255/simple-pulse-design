import numpy as np
import matplotlib.pyplot as plt

import json

def get_data(fname, mult):
    f = open(fname)
    raw = f.readlines()
    x_res = []
    y_res = []
    for line in raw:
        x, y = line.split(',')
        x = float(x.strip())
        y = float(y.strip()) * mult
        x_res.append(x)
        y_res.append(y)
    return x_res, y_res

te_rampup_casp = get_data('T_e_80s.txt', 1.0)
te_flattop_casp = get_data('T_e_300s.txt', 1.0)

te_rampup_ped = np.interp(0.95, te_rampup_casp[0], te_rampup_casp[1])
print(te_rampup_ped)
te_flattop_ped = np.interp(0.95, te_flattop_casp[0], te_flattop_casp[1])
print(te_flattop_ped)


f = 'tmp/ts_state2.json'
state = {}
with open(f, 'r') as f:
    state = json.loads(f.read())

te_rampup_sim_1 = state['T_e']['0']
te_rampup_sim_2 = state['T_e']['4']
te_flattop_sim = state['T_e']['10']

fig, ax = plt.subplots(1,3, figsize=(10,5))

fig.suptitle('T_e')

ax[0].set_title('Rampup t=5s')
ax[0].plot(te_rampup_casp[0], te_rampup_casp[1], color='magenta', label='Casper')
ax[0].plot(np.sqrt(te_rampup_sim_1['x']), te_rampup_sim_1['y'], color='lime', label='Tokamaker-Torax')
ax[0].scatter(0.95, te_rampup_ped, c='blue', label='Pedestal')

ax[1].set_title('Rampup t=38s')
ax[1].plot(te_rampup_casp[0], te_rampup_casp[1], color='magenta', label='Casper')
ax[1].plot(np.sqrt(te_rampup_sim_2['x']), te_rampup_sim_2['y'], color='lime', label='Tokamaker-Torax')
ax[1].scatter(0.95, te_rampup_ped, c='blue', label='Pedestal')


ax[2].set_title('Flattop t=100s')
ax[2].plot(te_flattop_casp[0], te_flattop_casp[1], color='magenta', label='Casper')
ax[2].plot(np.sqrt(te_flattop_sim['x']), te_flattop_sim['y'], color='lime', label='Tokamaker-Torax')
ax[2].scatter(0.95, te_flattop_ped, c='blue', label='Pedestal')

plt.legend()
plt.show()