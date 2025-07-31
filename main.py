import numpy as np

from model import CGTS

times = np.linspace(145.0, 150.0, 5)
mysim = CGTS('./eqdsk_cocos02.eqdsk', times)
mysim.initialize_gs()
mysim.fly(save_states=True, graph=False)