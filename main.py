import numpy as np

from model import CGTS

rampup_times = np.linspace(20.0, 80.0, 5)
flattop_times = np.linspace(110.0, 150.0, 5)
times = np.r_[rampup_times, flattop_times]

g_arr1 = ['./Lmode_ex.eqdsk'] * len(rampup_times)
g_arr2 = ['./EQDSK_ITERhybrid_COCOS02.eqdsk'] * len(flattop_times)
g_arr = np.r_[g_arr1, g_arr2]

mysim = CGTS(times, g_arr)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=False)