import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.io as sio

from model import CGTS

# Reproduciton of Next Step Fusion Sim

# Load times and setup CGTS objct
f = open('nextstep/pulse_json/psi_dist_out.json')
psi_dist = json.load(f)
f.close()

eq_times = np.array([float(eq['ttt']) for eq in psi_dist['a_equ']]) / 1e3
# print(len(times))
# tstep = times[::25]
# eqdsk = [f'nsf_eq_2/nsf-eq/NSF_i={i}.eqdsk' for i in range(20)]

tstep = np.arange(0.16, 5.06, 0.1)
tstep = [np.round(t, 2) for t in tstep]
eqdsk = [f'pulse_abs/TokaMaker_{t}.eqdsk' for t in tstep]

mysim = CGTS(5.0, tstep, eqdsk, dt=1.0E-2)
mysim.initialize_gs('nextstep/nextstep_mesh.h5')

# Load Scalars
f = open('nextstep/pulse_json/zerodout.json')
zerod = json.load(f)
f.close()

zerod_sim = np.array(zerod['simul_m'])
zerod_times = np.array([float(t) for t in zerod['time']]) / 1e3

Ip_values = zerod_sim[:, 0] * 1e3
Ip = {zerod_times[i]: Ip for i, Ip in enumerate(Ip_values)}
mysim.set_Ip(Ip)

zeff_values = zerod_sim[:, 9]
zeff = {zerod_times[i]: {0: zeff, 1: zeff} for i, zeff in enumerate(zeff_values)}
mysim.set_Zeff(zeff)

nbar_values = zerod_sim[:, 8] * 1e20
nbar = {zerod_times[i]: nb for i, nb in enumerate(nbar_values)}
mysim.set_nbar(nbar)

P_ecrh_values = zerod_sim[:, 21] * 1e6
eccd = {zerod_times[i]: heating for i, heating in enumerate(P_ecrh_values)}
ohmic_values = zerod_sim[:, 22]
# rhon = np.linspace(0.0, 1.0, 100)
# ohmic = np.zeros([len(zerod_times), len(rhon)])
# for i in range(len(zerod_times)):
#     for j in range(len(rhon)):
#         ohmic[i][j] = ohmic_values[i]
# ohmic = {t: {0.0: ohmic_values[i], 1.0: ohmic_values[i]} for i, t in enumerate(zerod_times)}
mysim.set_heating(eccd=eccd, eccd_loc=0.3)

Bp_values = zerod_sim[:, 1]
Bp = {zerod_times[i]: bp for i, bp in enumerate(Bp_values)}
Bp_arr = [np.interp(t, zerod_times, Bp_values) for t in tstep]
mysim.set_Bp(Bp_arr)

def boxcar_smooth(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

vloop_values = boxcar_smooth(zerod_sim[:,12], 100)
vloop = {zerod_times[i]: v for i, v in enumerate(vloop_values)}
vloop_arr = [np.interp(t, zerod_times, vloop_values) for t in tstep]
mysim.set_Vloop(vloop_arr)

# Load profiles
f = open('nextstep/pulse_json/profiles_out.json')
profiles = json.load(f)
f.close()

n_e = {}
T_e = {}
T_i = {}
prof_t = {}
for n, prof in enumerate(profiles['ne']):
    n_e[n] = prof
for n, prof in enumerate(profiles['te']):
    T_e[n] = prof
for n, prof in enumerate(profiles['ti']):
    T_i[n] = prof
for n, t in enumerate(profiles['t']):
    prof_t[n] = t / 1e3

psi = np.linspace(0.0, 1.0, len(n_e[0]))
rho = [np.sqrt(p) for p in psi]
n_e = {prof_t[i]: {rho[j]: n_e[i][j] * 1e20 for j in range(len(rho))} for i in n_e.keys()}

mysim.set_density(n_e)

T_e = {prof_t[i]: {rho[j]: T_e[i][j] / 1e3 for j in range(len(rho))} for i in T_e.keys()}
T_i = {prof_t[i]: {rho[j]: T_i[i][j] / 1e3 for j in range(len(rho))} for i in T_i.keys()}
mysim.set_Te(T_e)
mysim.set_Ti(T_i)

# Set pedestal
T_i_ped = {time: np.interp(0.95, rho, [T_i[time][x] for x in rho]) for time in T_i.keys()}
T_e_ped = {time: np.interp(0.95, rho, [T_e[time][x] for x in rho]) for time in T_e.keys()}
n_e_ped = {time: np.interp(0.95, rho, [n_e[time][x] for x in rho]) for time in n_e.keys()}

mysim.set_pedestal(T_i_ped=T_i_ped, T_e_ped=T_e_ped, n_e_ped=n_e_ped)

# Set edge conditions
ne_right_bc = {t: n_e[t][rho[-1]] for t in n_e.keys()}
Te_right_bc = {t: T_e[t][rho[-1]] for t in T_e.keys()}
Ti_right_bc = {t: T_i[t][rho[-1]] for t in T_i.keys()}
mysim.set_right_bc(ne_right_bc=ne_right_bc, Te_right_bc=Te_right_bc, Ti_right_bc=Ti_right_bc)

# mysim.set_evolve(density=False)

# Function to load and reorganize .mat files into Python dictionaries
def load_mat_as_dict(filepath):
    #print(f"\nLoading file: {os.path.basename(filepath)}")
    raw_data = sio.loadmat(filepath)

    # Filter out MATLAB's internal metadata keys
    keys = [k for k in raw_data.keys() if not k.startswith("__")]

    data_dict = {}
    for k in keys:
        value = raw_data[k]
        # Convert numpy arrays with a single element to scalars
        if isinstance(value, np.ndarray):
            if value.shape == (1, 1):
                data_dict[k] = value.item()
            elif value.shape[0] == 1 and value.ndim == 2:
                data_dict[k] = value.flatten()
            else:
                data_dict[k] = value
        else:
            data_dict[k] = value

        #print(f"  - {k}: type={type(data_dict[k])}, shape={getattr(data_dict[k], 'shape', 'N/A')}")

    return data_dict

data_currents = load_mat_as_dict("currents_coils_cam.mat")
coil_targets = {}
coil_map = {
    'CS_1': 3,
    'CS_2': 2,
    'CS_3': 1,
    'PF_1U': 4,
    'PF_1L': 11,
    'PF_2U': 5,
    'PF_2L': 10,
    'PF_3U': 6,
    'PF_3L': 9,
    'PF_4U': 7,
    'PF_4L': 8,
    'DIV_1U': 12,
    'DIV_1L': 12,
    'DIV_2U': 13,
    'DIV_2L': 13,
}
coil_names = coil_map.keys()
coil_targets['time'] = data_currents['t_fc'] / 1e3
for coil_name in coil_names:
    coil_targets[coil_name] = data_currents['fc'][:,coil_map[coil_name]-1] * 1e3
    # coil_targets[coil_name] = 0.0
mysim.set_coil_reg(coil_targets, t=tstep[0], strict_limit=1.0E8)

# Run the simulation
mysim.fly(graph=False, save_states=True)