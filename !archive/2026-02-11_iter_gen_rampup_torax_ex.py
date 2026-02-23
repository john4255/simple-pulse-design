"""
TokaMaker equilibrium generation for TORAX tutorial notebook (extended)
Matches: torax_tutorial_exercises_with_solutions.ipynb

TORAX tutorial uses:
- Rampup: 0-100s, Ip: 3 MA → 12.5 MA
- Extended flattop: 100-400s at 12.5 MA
- CHEASE geometry: R0=6.2m, a=2.0m, B0=5.3T
- ITER-like shaping: kappa~1.7, delta~0.3-0.4
- Density: 0.85 Greenwald fraction
- Temperature: 6 keV initial → evolves
- No heating during rampup, 53 MW total at flattop

This script generates equilibria for:
- Rampup phase: 0-100s (10 files)
- Flattop phase: 100-400s (6 files) at 12.5 MA
"""

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun, create_isoflux

import numpy as np
import matplotlib.pyplot as plt

myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)

# Generate 16 equilibria total: 10 during rampup + 6 during flattop
n_rampup = 10
n_flattop = 6
n_eqdsk = n_rampup + n_flattop

mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh('ITER_mesh.h5')
mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
mygs.settings.maxits = 500
mygs.setup(order=2, F0=5.3*6.2)  # B0=5.3T, R0=6.2m

mygs.set_coil_vsc({'VS': 1.0})

coil_bounds = {key: [-50.E6, 50.E6] for key in mygs.coil_sets}
mygs.set_coil_bounds(coil_bounds)

# Set regularization weights
regularization_terms = []
for name, coil in mygs.coil_sets.items():
    if name.startswith('CS'):
        if name.startswith('CS1'):
            regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=2.E-2))
        else:
            regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.E-2))
    elif name.startswith('PF'):
        regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.E-2))
    elif name.startswith('VS'):
        regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.E-2))
# Disable VSC virtual coil
regularization_terms.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0, weight=1.E2))

mygs.set_coil_reg(reg_terms=regularization_terms)

# Set profiles - use similar power law profiles to original script
ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
pp_prof = create_power_flux_fun(40, 4.0, 1.0)
mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

# --- FIXED GEOMETRY matching TORAX iterhybrid_rampup ---
R0 = 6.2   # major radius [m]
a = 2.0    # minor radius [m]
Z0 = 0.5   # vertical position [m]
kappa = 1.7  # elongation (typical ITER H-mode)
delta = 0.35  # triangularity (typical ITER H-mode)

# Rampup: Ip from 3 MA to 12.5 MA over 100s (matches tutorial)
Ip_rampup = np.linspace(3.0E6, 12.5E6, n_rampup)
# Flattop: Ip stays at 12.5 MA
Ip_flattop = np.array([12.5E6] * n_flattop)
Ip_target = np.concatenate([Ip_rampup, Ip_flattop])

# Pressure evolution - estimates for tutorial scenario
# Start low (cold initial, no heating), ramp up when NBI turns on at t=100s
# At 12.5 MA with 53 MW heating and 0.85 fGW: eventually reaches ~300-500 kPa
P0_rampup = np.linspace(5E3, 1.8E5, n_rampup)  # 5 kPa → 180 kPa (mild heating from ohmic)
P0_flattop = np.linspace(3.0E5, 5.0E5, n_flattop)  # 300 kPa → 500 kPa (NBI heats plasma over time)
P0_target = np.concatenate([P0_rampup, P0_flattop])

# Time stamps for equilibria
t_rampup = np.linspace(0, 100, n_rampup)
t_flattop = np.array([150, 200, 250, 300, 350, 400])  # Spread across extended flattop
t_stamps = np.concatenate([t_rampup, t_flattop])

print(f"Generating {n_eqdsk} equilibria for TORAX tutorial (extended):")
print(f"  Rampup (0-100s): {n_rampup} files")
print(f"  Flattop (100-400s): {n_flattop} files")
print(f"  Fixed geometry: R0={R0}m, a={a}m, kappa={kappa}, delta={delta}")
print(f"  Ip range: {Ip_target[0]/1e6:.1f} - {Ip_target[-1]/1e6:.1f} MA")
print(f"  P0 range: {P0_target[0]/1e3:.1f} - {P0_target[-1]/1e3:.1f} kPa")
print()

for eq_idx in range(n_eqdsk):
    print(f"Solving equilibrium {eq_idx}: t={t_stamps[eq_idx]:.1f}s, Ip={Ip_target[eq_idx]/1e6:.2f} MA, P0={P0_target[eq_idx]/1e3:.1f} kPa")
    
    mygs.set_targets(Ip=Ip_target[eq_idx], pax=P0_target[eq_idx])
    
    # Create isoflux points with fixed geometry
    isoflux_pts = create_isoflux(20, R0, Z0, a, kappa, delta)
    mygs.set_isoflux(isoflux_pts)
    
    # Initialize psi with fixed geometry
    mygs.init_psi(R0, Z0, a, kappa, delta)
    
    # Solve equilibrium
    mygs.solve()
    
    # Plot result
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    mygs.plot_machine(fig, ax, coil_colormap='seismic', coil_scale=1.E-6, 
                     coil_clabel=r'$I_C$ [MA]', coil_symmap=True)
    mygs.plot_psi(fig, ax, plasma_nlevels=5, vacuum_nlevels=5)
    psi_target = mygs.get_psi()
    mygs.plot_psi(fig, ax, psi_target, plasma_levels=[1.0,], 
                 plasma_color='red', vacuum_nlevels=0, plasma_linestyles='dashed')
    
    # Add title with info
    ax.set_title(f'ITER Rampup Eq {eq_idx}: t={t_stamps[eq_idx]:.1f}s, Ip={Ip_target[eq_idx]/1e6:.2f} MA')
    
    plt.tight_layout()
    plt.savefig(f'out/iter_rampup_torax_ex_i={eq_idx}.png', dpi=150)
    plt.close()
    
    # Save EQDSK
    mygs.save_eqdsk(f'out/iter_rampup_torax_ex_i={eq_idx}.eqdsk', cocos=2)
    
    # Print equilibrium info
    mygs.print_info()
    print()

print("=" * 60)
print("Equilibrium generation complete!")
print(f"Files saved to: out/iter_rampup_torax_ex_i={{0..{n_eqdsk-1}}}.eqdsk")
print("=" * 60)
