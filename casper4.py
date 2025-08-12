import matplotlib.pyplot as plt
import numpy as np

import json

fname = 'tmp/res.json'

res = None
with open(fname, 'r') as f:
    res = json.loads(f.read())

fig, ax = plt.subplots(3, 2)

ax[0,0].set_title('Power (MW) and Q (Dimensionless)')
ax[0,1].set_title('Fast Control Voltage (V)')
ax[0,0].plot(res['Q']['x'], 1.0E7 * np.array(res['Q']['y']), label='Q')
ax[0,0].plot(res['P_alpha_total']['x'], res['P_alpha_total']['y'], label='Alpha')
ax[0,0].plot(res['P_aux_total']['x'], res['P_aux_total']['y'], label='Aux')
ax[0,0].legend()

ax[1,0].set_title('CS Coil Currents (MA-turns)')
ax[1,1].set_title('CS Coil Voltage (V)')
for coil in res['COIL']:
    if coil.startswith('PF') or coil.startswith('VS'):
        continue
    times = res['COIL'][coil].keys()
    currents = res['COIL'][coil].values()
    ax[1,0].plot(times, currents, label=coil)
ax[1,0].legend()

ax[2,0].set_title('PF Coil Currents (MA-turns)')
ax[2,1].set_title('PF Coil Voltage (V)')
for coil in res['COIL']:
    if coil.startswith('CS') or coil.startswith('VS'):
        continue
    times = res['COIL'][coil].keys()
    currents = res['COIL'][coil].values()
    ax[2,0].plot(times, currents, label=coil)
ax[2,0].legend()

plt.tight_layout()
plt.show()