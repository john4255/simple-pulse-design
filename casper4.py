import matplotlib.pyplot as plt
import numpy as np

import json

fname = 'tmp/res.json'

res = None
with open(fname, 'r') as f:
    res = json.loads(f.read())

fig, ax = plt.subplots(2, 2)

ax[0,0].set_title('Power (W)')
ax[0,0].plot(res['P_alpha_total']['x'], res['P_alpha_total']['y'], label='Alpha')
ax[0,0].plot(res['P_aux_total']['x'], res['P_aux_total']['y'], label='Aux')
ax[0,0].plot(res['P_ohmic_e']['x'], res['P_ohmic_e']['y'], label='P_ohmic_e')
ax[0,0].plot(res['P_radiation_e']['x'], res['P_radiation_e']['y'], label='P_radiation_e')
ax[0,0].legend()

ax[1,0].set_title('Q (Dimensionless)')
ax[1,0].plot(res['Q']['x'], res['Q']['y'], label='Q')

ax[1,1].set_title('CS Coil Currents (MA-turns)')
for coil in res['COIL']:
    if coil.startswith('PF') or coil.startswith('VS'):
        continue
    times = res['COIL'][coil].keys()
    currents = res['COIL'][coil].values()
    ax[1,1].plot(times, currents, label=coil)
ax[1,1].legend()

ax[0,1].set_title('PF Coil Currents (MA-turns)')
for coil in res['COIL']:
    if coil.startswith('CS') or coil.startswith('VS'):
        continue
    times = res['COIL'][coil].keys()
    currents = res['COIL'][coil].values()
    ax[0,1].plot(times, currents, label=coil)
ax[0,1].legend()

plt.tight_layout()
plt.show()