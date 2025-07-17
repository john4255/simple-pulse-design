import json
import matplotlib.pyplot as plt

fname = 'tmp/ts_state0.json'

state = {}
with open(fname, 'r') as f:
    state = json.loads(f.read())

print(state.keys())

vars = ['ffp_prof', 'pp_prof', 'eta_prof', 'psi_prof', 'T_e', 'T_i', 'n_e', 'n_i']

fig, ax = plt.subplots(3, 3)
for i, var in enumerate(vars):
    print(var)
    for t_idx, prof in state[var].items():
        ax[i // 3][i % 3].set_title(var, weight='bold')
        
        if len(prof) == 0:
            continue
        x = prof['x']
        y = prof['y']
        ax[i // 3][i % 3].plot(x, y, label=t_idx)
        ax[i // 3][i % 3].legend()

plt.tight_layout()
plt.show()