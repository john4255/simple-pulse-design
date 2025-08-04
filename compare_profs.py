import json
import matplotlib.pyplot as plt
import numpy as np

import sys

sim_states = []

n_steps = int(sys.argv[1])

for i in range(n_steps):
    f = 'tmp/ts_state{}.json'.format(i)

    with open(f, 'r') as f:
        sim_states = np.append(sim_states, json.loads(f.read()))


var = sys.argv[2]

fig, ax = plt.subplots(1, 6, figsize=(12,6))

for j, st in enumerate(sim_states):
    keys = sorted(st[var].keys())
    for i, key in enumerate(keys):
        ax[i].set_title("i={}".format(i), weight='bold')
        x = st[var][key]['x']
        y = st[var][key]['y']
        ax[i].plot(x, y, label=j)
        ax[i].legend()

plt.tight_layout()
plt.show()