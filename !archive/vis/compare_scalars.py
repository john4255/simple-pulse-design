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

cmap = plt.get_cmap('plasma')
var = sys.argv[2]

fig, ax = plt.subplots(2, 10, figsize=(12,6))

meas = {}
for i in range(len(sim_states[0][var])):
    meas[i] = [st[var][i] for st in sim_states]
    ax[i // 10, i % 10].set_title("i={}".format(i), weight='bold')
    steps = [j for j in range(len(meas[i]))]
    ax[i // 10, i % 10].bar(steps, meas[i])

plt.tight_layout()
plt.show()