import json
import matplotlib.pyplot as plt
import numpy as np

import sys

sim_states = []

n_steps = int(sys.argv[1])

for i in range(n_steps):
    f = 'tmp/gs_state{}.json'.format(i)

    with open(f, 'r') as f:
        sim_states = np.append(sim_states, json.loads(f.read()))


var = sys.argv[2]

fig, ax = plt.subplots(3, 10, figsize=(12,6))

for st_idx, st in enumerate(sim_states):
    for idx in range(30):
        i = idx // 10
        j = idx % 10
        data = st[var]
        x = data[str(idx)]['x']
        y = data[str(idx)]['y']
        ax[i][j].set_title(f'i={idx}')
        ax[i][j].plot(x, y, label=st_idx)

plt.legend()
plt.tight_layout()
plt.show()