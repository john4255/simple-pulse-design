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

ne_rampup_casp = get_data('n_e_80s.txt', 1.0E20)
ne_flattop_casp = get_data('n_e_300s.txt', 1.0E20)

ne_rampup_ped = np.interp(0.95, ne_rampup_casp[0], ne_rampup_casp[1])
print(ne_rampup_ped)
ne_flattop_ped = np.interp(0.95, ne_flattop_casp[0], ne_flattop_casp[1])
print(ne_flattop_ped)

f = 'tmp/ts_state2.json'
state = {}
with open(f, 'r') as f:
    state = json.loads(f.read())

ne_rampup_sim = state['n_e']['0']
ne_flattop_sim = state['n_e']['10']

fig, ax = plt.subplots(1,2)

ax[0].set_title('Rampup')
ax[0].plot(ne_rampup_casp[0], ne_rampup_casp[1], color='r', label='Casper')
ax[0].plot(np.sqrt(ne_rampup_sim['x']), ne_rampup_sim['y'], color='g', label='Tokamaker-Torax')
ax[0].scatter(0.95, ne_rampup_ped, c='blue', label='Pedestal')

ax[1].set_title('Flattop')
ax[1].plot(ne_flattop_casp[0], ne_flattop_casp[1], color='r', label='Casper')
ax[1].plot(np.sqrt(ne_flattop_sim['x']), ne_flattop_sim['y'], color='g', label='Tokamaker-Torax')
ax[1].scatter(0.95, ne_flattop_ped, c='blue', label='Pedestal')

plt.legend()
plt.show()