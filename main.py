import numpy as np

from model import CGTS

rampup_times = np.linspace(20.0, 80.0, 5)
flattop_times = np.linspace(120.0, 130.0, 5)
times = np.r_[rampup_times, flattop_times]

g_arr1 = [f'ramp3/iter_i={i}.eqdsk' for i in range(5)]
g_arr2 = [f'Lmode_ex.eqdsk'] * 5
g_arr = np.r_[g_arr1, g_arr2]

mysim = CGTS(times, g_arr)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=False)