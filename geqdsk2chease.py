import matplotlib.pyplot as plt

from my_omfit_chease import OMFITchease
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

my_geqdsk = OMFITgeqdsk('tmp/000.000.eqdsk')
my_geqdsk.plot()
plt.show()

my_chease = OMFITchease(filename=None)
my_chease = my_chease.from_gEQDSK(my_geqdsk, rhotype=1)

my_chease.plot()
plt.show()

my_chease.save()