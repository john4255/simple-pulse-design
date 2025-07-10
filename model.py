import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline
import torax
from decimal import Decimal
from freeqdsk import aeqdsk

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk, read_mhdin, read_kfile, read_pfile

from helper import *

CONV_THRESHOLD = 0.05

os.makedirs('tmp', exist_ok=True)

myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)

# Load Mesh
mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh("DIIID/DIIID_mesh_197555.h5")

mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
mygs.lim_zmax = 1.15
mygs.settings.nl_tol=1.0E-5
mygs.settings.maxits = 800

# Load Files
geqdsk = read_eqdsk('DIIID/g163520.02200.1.eqdsk')

a_eqdsk = {}
# with open('DIIID/a163520.02200', 'r') as f:
#     a_eqdsk = aeqdsk.read(f)

e_coil_names = ['ECOILA','ECOILB','E567UP','E567DN','E89DN','E89UP']
f_coil_names = ['F1A', 'F2A', 'F3A', 'F4A', 'F5A', 'F6A', 'F7A', 'F8A', 'F9A', 'F1B', 'F2B', 'F3B', 'F4B', 'F5B', 'F6B', 'F7B', 'F8B', 'F9B']
machine_dict, _ = read_mhdin('DIIID/mhdin_156001.dat', e_coil_names, f_coil_names)
_, _, e_coil_dict, f_coil_dict, _ = read_kfile('DIIID/k163520.02200', machine_dict, e_coil_names, f_coil_names)
pdict = read_pfile('DIIID/p163520.02200')

mygs.setup(order=2, F0=geqdsk['rcentr']*geqdsk['bcentr'])

times = np.linspace(0.0, 8.0, 5)

step = 0
err = 0
sim_vars = init_vars(times, geqdsk, a_eqdsk, pdict)
# graph_sim(sim_vars, 0)

sim_vars, cflux_gs = run_eqs(mygs, sim_vars, times, machine_dict, e_coil_dict, f_coil_dict, geqdsk, step, calc_vloop=False)

while err > CONV_THRESHOLD:
    sim_vars, cflux_transport = run_sims(sim_vars, times, step)
    step += 1

    sim_vars, cflux_gs = run_eqs(mygs, sim_vars, times, machine_dict, e_coil_dict, f_coil_dict, geqdsk, step, calc_vloop=True)
    err = (cflux_gs - cflux_transport) ** 2
    
    
print("Discharge model complete.")
print("Error = {}.".format(err))