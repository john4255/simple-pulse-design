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

ti_rampup_casp = get_data('T_i_80s.txt', 1.0)
ti_flattop_casp = get_data('T_i_300s.txt', 1.0)

f = 'tmp/ts_state2.json'
state = {}
with open(f, 'r') as f:
    state = json.loads(f.read())

ti_rampup_sim = state['T_i']['0']
ti_flattop_sim = state['T_i']['10']

fig, ax = plt.subplots(1,2)

fig.suptitle('T_i')

ax[0].set_title('Rampup')
ax[0].plot(ti_rampup_casp[0], ti_rampup_casp[1], color='r', label='Casper')
ax[0].plot(np.sqrt(ti_rampup_sim['x']), ti_rampup_sim['y'], color='g', label='Tokamaker-Torax')

ax[1].set_title('Flattop')
ax[1].plot(ti_flattop_casp[0], ti_flattop_casp[1], color='r', label='Casper')
ax[1].plot(np.sqrt(ti_flattop_sim['x']), ti_flattop_sim['y'], color='g', label='Tokamaker-Torax')

plt.legend()
plt.show()