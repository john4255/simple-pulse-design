import matplotlib.pyplot as plt
import numpy as np

import json

fname = 'tmp/res.json'

res = None
with open(fname, 'r') as f:
    res = json.loads(f.read())

fig, ax = plt.subplots(3, 2) #, figsize=(20,10))

ax[0,0].set_title('Heating (MW)')
ax[0,0].plot(res['P_alpha_total']['x'], np.array(res['P_alpha_total']['y']) / 1.0E6, label='Alpha', c='darkorange')
ax[0,0].plot(res['P_aux_total']['x'], np.array(res['P_aux_total']['y']) / 1.0E6, label='Aux', c='mediumorchid')
ax[0,0].plot(res['P_ohmic_e']['x'], np.array(res['P_ohmic_e']['y']) / 1.0E6, label='Ohmic', c='limegreen')
# ax[0,0].plot(res['P_radiation_e']['x'], np.array(res['P_radiation_e']['y']) / 1.0E6, label='P_radiation_e', c='')
ax[0,0].grid(True)
ax[0,0].legend(fontsize=5, loc='upper right')

ax[1,0].set_title('Q (Dimensionless)')
ax[1,0].plot(res['Q']['x'][5:], res['Q']['y'][5:], label='Q')
ax[1,0].set_yticks([0, 10, 20, 30, 40, 50])
ax[1,0].grid(True)

ax[2,0].set_title('Fusion Power (MW)')
ax[2,0].plot(res['P_alpha_total']['x'], 5.0 * np.array(res['P_alpha_total']['y']) / 1.0E6)
ax[2,0].grid(True)

ax[0,1].set_title('Plasma Current (MA)')
ax[0,1].plot(res['Ip']['x'], np.array(res['Ip']['y']) / 1.0E6)
ax[0,1].grid(True)

# ax[1,1].set_title('CS Coil Currents (A-turns)')
# for coil in res['COIL']:
#     if coil.startswith('PF') or coil.startswith('VS'):
#         continue
#     times = res['COIL'][coil].keys()
#     times = [float(t) for t in times]
#     currents = res['COIL'][coil].values()
#     ax[1,1].plot(times, currents, label=coil)
# ax[1,1].legend()

# ax[0,1].set_title('PF Coil Currents (A-turns)')
# for coil in res['COIL']:
#     if coil.startswith('CS') or coil.startswith('VS'):
#         continue
#     times = res['COIL'][coil].keys()
#     times = [float(t) for t in times]
#     currents = res['COIL'][coil].values()
#     ax[0,1].plot(times, currents, label=coil)
# ax[0,1].legend()

ax[2,1].set_title('Core Temperature (keV)')
ax[2,1].plot(res['T_e_core']['x'], res['T_e_core']['y'], label='T_e')
ax[2,1].plot(res['T_i_core']['x'], res['T_i_core']['y'], label='T_i')
ax[2,1].grid(True)
ax[2,1].legend()

ax[1,1].set_title('Line Average Temperature (keV)')
ax[1,1].plot(res['T_e_line_avg']['x'], res['T_e_line_avg']['y'], label='T_e')
ax[1,1].plot(res['T_i_line_avg']['x'], res['T_i_line_avg']['y'], label='T_i')
ax[1,1].grid(True)
ax[1,1].legend()

ax[0,0].set_xticks(np.linspace(0, 600, 7))
ax[0,1].set_xticks(np.linspace(0, 600, 7))
ax[1,0].set_xticks(np.linspace(0, 600, 7))
ax[1,1].set_xticks(np.linspace(0, 600, 7))
ax[2,0].set_xticks(np.linspace(0, 600, 7))
ax[2,1].set_xticks(np.linspace(0, 600, 7))

ax[0,0].set_xlabel('time (s)')
ax[1,0].set_xlabel('time (s)')
ax[2,0].set_xlabel('time (s)')
ax[0,1].set_xlabel('time (s)')
ax[1,1].set_xlabel('time (s)')
ax[2,1].set_xlabel('time (s)')

plt.tight_layout(pad=0.5)

plt.savefig('casp4.png')
plt.show()