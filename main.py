import numpy as np

from model import CGTS

times = np.linspace(0.0, 5.0, 6)
mysim = CGTS('./eqdsk_cocos07.eqdsk', times)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=True)