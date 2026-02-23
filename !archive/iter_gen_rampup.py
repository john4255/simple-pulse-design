from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux

import numpy as np
import matplotlib.pyplot as plt

myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)
n_eqdsk = 10

mesh_pts,mesh_lc,mesh_reg,coil_dict,cond_dict = load_gs_mesh('ITER_mesh.h5')
mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict,coil_dict=coil_dict)
mygs.setup(order = 2, F0 = 5.3*6.2)

mygs.set_coil_vsc({'VS': 1.0})

coil_bounds = {key: [-50.E6, 50.E6] for key in mygs.coil_sets}
mygs.set_coil_bounds(coil_bounds)

# Set regularization weights
regularization_terms = []
for name, coil in mygs.coil_sets.items():
    # Set zero target current and different small weights to help conditioning of fit
    if name.startswith('CS'):
        if name.startswith('CS1'):
            regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=2.E-2))
        else:
            regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
    elif name.startswith('PF'):
        regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
    elif name.startswith('VS'):
        regularization_terms.append(mygs.coil_reg_term({name: 1.0},target=0.0,weight=1.E-2))
# Disable VSC virtual coil
regularization_terms.append(mygs.coil_reg_term({'#VSC': 1.0},target=0.0,weight=1.E2))

# Pass regularization terms to TokaMaker
mygs.set_coil_reg(reg_terms=regularization_terms)

# Set profiles
ffp_prof = create_power_flux_fun(40,1.5,2.0)
pp_prof = create_power_flux_fun(40,4.0,1.0)

mygs.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)

Ip_target = np.linspace(3.0E6, 15.0E6, n_eqdsk)
P0_target = np.linspace(3.76E4, 6.4E5, n_eqdsk)

R0 = np.linspace(6.599, 6.245, n_eqdsk)
Z0 = 0.5
a = np.linspace(1.762, 2.020, n_eqdsk)
kappa = np.linspace(1.393, 1.520, n_eqdsk)
delta = np.linspace(-0.0146, -0.097, n_eqdsk)

for eq_idx in range(n_eqdsk):
    # Ip_ratio = 2.0
    mygs.set_targets(Ip=Ip_target[eq_idx], pax=P0_target[eq_idx])

    isoflux_pts = create_isoflux(20, R0[eq_idx], Z0, a[eq_idx], kappa[eq_idx], delta[eq_idx])

    mygs.set_isoflux(isoflux_pts)

    mygs.init_psi(R0[eq_idx], Z0, a[eq_idx], kappa[eq_idx], delta[eq_idx])
    mygs.solve()

    fig, ax = plt.subplots(1,1)
    mygs.plot_machine(fig,ax,coil_colormap='seismic',coil_scale=1.E-6,coil_clabel=r'$I_C$ [MA]',coil_symmap=True)
    mygs.plot_psi(fig,ax,plasma_nlevels=5,vacuum_nlevels=5)
    psi_target = mygs.get_psi()
    mygs.plot_psi(fig,ax,psi_target,plasma_levels=[1.0,],plasma_color='red',vacuum_nlevels=0,plasma_linestyles='dashed')

    plt.show()
    mygs.save_eqdsk(f'rampup/iter_i={eq_idx}.eqdsk', cocos=2)