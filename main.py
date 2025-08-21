import numpy as np

from model import CGTS

rampup_times = np.linspace(10.0, 90.0, 10)
flattop_times = np.linspace(110.0, 140.0, 10)
times = np.r_[rampup_times, flattop_times]

g_arr_rampup = [f'ramp4/iter_i={i}.eqdsk' for i in range(10)]
g_arr_flattop = ['Lmode_ex.eqdsk'] * 10
g_arr = np.r_[g_arr_rampup, g_arr_flattop]

mysim = CGTS(times, g_arr)

mysim.initialize_gs('ITER_mesh.h5', vsc='VS')

mysim.fly(save_states=True, graph=False)