import numpy as np

from model import CGTS

times = np.linspace(145.0, 150.0, 6)
mysim = CGTS('./ITER_Lmode.eqdsk', times)
mysim.initialize_gs()
mysim.fly()