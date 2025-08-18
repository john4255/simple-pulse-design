import numpy as np

from model import CGTS

rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 140.0, 10)
times = np.r_[rampup_times, flattop_times]

g_arr1 = [f'ramp4/iter_i={i}.eqdsk' for i in range(10)]
# g_arr2 = ['Lmode_ex.eqdsk'] * 3 + ['ramp3/iter_i=2'] * 2
g_arr2 = ['Lmode_ex.eqdsk'] * 10
g_arr = np.r_[g_arr1, g_arr2]

mysim = CGTS(times, g_arr)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=False)