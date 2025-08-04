import numpy as np

from model import CGTS

rampup_times = np.linspace(20.0, 80.0, 3)
flattop_times = np.linspace(110.0, 150.0, 5)
times = np.r_[rampup_times, flattop_times]

# times = np.linspace(20.0, 150.0, 14)

mysim = CGTS('./EQDSK_ITERhybrid_COCOS02.eqdsk', times)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=False)