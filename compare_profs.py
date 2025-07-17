import json
import matplotlib.pyplot as plt

f1 = 'tmp/ts_state0.json'
f2 = 'tmp/notebook.json'

sim_state = {}
nb_state = {}
with open(f1, 'r') as f:
    sim_state = json.loads(f.read())
with open(f2, 'r') as f:
    nb_state = json.loads(f.read())

var = 'ffp_prof'

print(sim_state['ffp_prof'].keys())

fig, ax = plt.subplots(1, 6, figsize=(12,6))
for t_key in sim_state[var].keys():
    i = int(t_key)

    ax[i].set_title("i={}".format(i), weight='bold')
    x1 = sim_state[var][t_key]['x']
    x2 = nb_state[var][t_key]['x']
    y1 = sim_state[var][t_key]['y']
    y2 = nb_state[var][t_key]['y']
    ax[i].plot(x1, y1, label="Sim")
    ax[i].plot(x2, y2, label="NB")
    ax[i].legend()

plt.tight_layout()
plt.show()