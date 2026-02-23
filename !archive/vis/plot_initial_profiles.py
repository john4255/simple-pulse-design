"""
Diagnostic: compare the initial kinetic profiles given to TORAX vs
the values implied by the seed EQDSKs.

Shows why p_axis from TORAX is so much lower than the seed EQDSK pressure —
the Lmode/Hmode profile shapes do not match the Budny 2008 equilibrium data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from omfit_classes.utils_fusion import Hmode_profiles, Lmode_profiles
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

# ─────────────────────────────────────────────────────────────────────────────
# Reproduce exactly the time / EQDSK arrays from toktox_budny2008_iter.py
# ─────────────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rampup_times  = np.linspace(0.0,   80.0, 10)
flattop_times = np.linspace(100.0, 450.0, 10)
rampdown_times= np.linspace(500.0, 600.0, 10)
times = np.r_[rampup_times, flattop_times, rampdown_times]   # 30 eqtimes
n_rampup_eqdsk = 9

g_arr_rampup   = [os.path.join(REPO, f'budny2008_iter/budny2008_iter_i={i}.eqdsk')
                  for i in range(0, n_rampup_eqdsk + 1)]
g_arr_flattop  = [os.path.join(REPO, 'budny2008_iter/budny2008_iter_flattop.eqdsk')] * 10
g_arr_rampdown = g_arr_rampup[::-1]
g_arr = np.r_[g_arr_rampup, g_arr_flattop, g_arr_rampdown]  # 30 EQDSKs

# simulation sample times (what TORAX is asked to solve)
t_res = np.arange(0.0, 600.0, 20.0)   # 30 points

# ─────────────────────────────────────────────────────────────────────────────
# Load scalars from each seed EQDSK (one per eqtime)
# ─────────────────────────────────────────────────────────────────────────────
eq_pax   = []   # Pa
eq_Ip    = []   # MA
eq_psi_swing = []  # Wb/rad  (|psibry - psimag|)
eq_B0    = []   # T
eq_R     = []   # m
eq_a     = []   # m
eq_kappa = []
eq_delta = []

psi_norm = np.linspace(0.0, 1.0, 50)

eq_pres_prof = []   # pressure profile on psi_norm

for i, g_path in enumerate(g_arr):
    g = read_eqdsk(g_path)
    eq_pax.append(g['pres'][0] / 1e3)          # kPa
    eq_Ip.append(abs(g['ip']) / 1e6)           # MA
    eq_psi_swing.append(abs(g['psibry'] - g['psimag']))
    eq_B0.append(abs(g['bcentr']))

    rzout = g['rzout']
    rmax, rmin = np.max(rzout[:,0]), np.min(rzout[:,0])
    zmax, zmin = np.max(rzout[:,1]), np.min(rzout[:,1])
    a  = (rmax - rmin) / 2.0
    rg = (rmax + rmin) / 2.0
    eq_R.append(g['rcentr'])
    eq_a.append(a)
    eq_kappa.append((zmax - zmin) / (2.0 * a))
    rupper = rzout[np.argmax(rzout[:,1]), 0]
    rlower = rzout[np.argmin(rzout[:,1]), 0]
    d_u = (rg - rupper) / a
    d_l = (rg - rlower) / a
    eq_delta.append((d_u + d_l) / 2.0)

    psi_g = np.linspace(0.0, 1.0, g['nr'])
    eq_pres_prof.append(np.interp(psi_norm, psi_g, g['pres']) / 1e3)  # kPa

eq_pax      = np.array(eq_pax)
eq_Ip       = np.array(eq_Ip)
eq_psi_swing= np.array(eq_psi_swing)
eq_B0       = np.array(eq_B0)
eq_R        = np.array(eq_R)
eq_a        = np.array(eq_a)
eq_kappa    = np.array(eq_kappa)
eq_delta    = np.array(eq_delta)

# ─────────────────────────────────────────────────────────────────────────────
# Reproduce the initial kinetic profiles (copy of toktox_budny2008_iter.py)
# ─────────────────────────────────────────────────────────────────────────────
n_sample = 100
psi_sample = np.linspace(0.0, 1.0, n_sample)

# --- rampup ---
ne_axis = np.linspace(1.e19, 1.e20, n_rampup_eqdsk)
Te_axis = np.linspace(1.0,   12.5,  n_rampup_eqdsk)          # keV
ne_edge = np.linspace(.1e20, .4e20, n_rampup_eqdsk)
Te_edge = np.linspace(.050,  .100,  n_rampup_eqdsk)           # keV

ne_prof_list = []
Te_prof_list = []
for idx in range(n_rampup_eqdsk):
    ne_prof_list.append(Lmode_profiles(edge=ne_edge[idx], core=ne_axis[idx], rgrid=n_sample))
    Te_prof_list.append(Lmode_profiles(edge=Te_edge[idx], core=Te_axis[idx], rgrid=n_sample))

# --- flattop ---
xphalf     = 0.965
widthp_Te  = 0.1
widthp_ne  = 0.35
ne_ped     = 0.6e20
Te_ped     = 5.0   # keV
ne_flattop = Hmode_profiles(edge=0.35, ped=ne_ped/1e20, core=1.1,  rgrid=n_sample,
                             expin=1.6, expout=1.6, widthp=widthp_ne, xphalf=xphalf) * 1e20
Te_flattop = Hmode_profiles(edge=0.1,  ped=Te_ped,      core=21.0, rgrid=n_sample,
                             expin=1.3, expout=1.7, widthp=widthp_Te, xphalf=xphalf)  # keV

# --- rampdown mirrors rampup ---
ne_rampdown = ne_prof_list[::-1]
Te_rampdown = Te_prof_list[::-1]

# ─────────────────────────────────────────────────────────────────────────────
# Build time-resolved scalars from the kinetic profiles
#   Interpolate between the profile "knots" to every eqtime/t_res point
# ─────────────────────────────────────────────────────────────────────────────
# Map: simulation time → initial profile (same logic as the TORAX ne/Te dicts)
profile_knot_times  = (list(range(0, 90, 10))         # 0..80  (9 rampup + transition)
                     + [100]                           # flattop start
                     + list(range(500, 610, 10)))      # rampdown 500..600

def build_ne_at_time(t):
    """Return (ne_axis [m^-3], ne_edge [m^-3]) by interpolating profile knots."""
    if t <= 80:
        # rampup: knots at t=0,10,...,80 → ne_prof_list[0..8]
        frac = t / 80.0
        idx  = frac * (n_rampup_eqdsk - 1)
        lo   = int(idx)
        hi   = min(lo + 1, n_rampup_eqdsk - 1)
        alpha = idx - lo
        ne_c = (1 - alpha) * ne_axis[lo] + alpha * ne_axis[hi]
        ne_e = (1 - alpha) * ne_edge[lo] + alpha * ne_edge[hi]
        return ne_c, ne_e
    elif t <= 500:
        return ne_flattop[0], ne_flattop[-1]
    else:
        frac = (t - 500) / 100.0
        idx  = frac * (n_rampup_eqdsk - 1)
        lo   = int(idx)
        hi   = min(lo + 1, n_rampup_eqdsk - 1)
        alpha = idx - lo
        # rampdown reverses rampup
        ne_c = (1 - alpha) * ne_axis[n_rampup_eqdsk-1-lo] + alpha * ne_axis[n_rampup_eqdsk-1-hi]
        ne_e = (1 - alpha) * ne_edge[n_rampup_eqdsk-1-lo] + alpha * ne_edge[n_rampup_eqdsk-1-hi]
        return ne_c, ne_e

def build_Te_at_time(t):
    """Return (Te_axis [keV], Te_edge [keV])."""
    if t <= 80:
        frac = t / 80.0
        idx  = frac * (n_rampup_eqdsk - 1)
        lo   = int(idx)
        hi   = min(lo + 1, n_rampup_eqdsk - 1)
        alpha= idx - lo
        Te_c = (1 - alpha) * Te_axis[lo] + alpha * Te_axis[hi]
        Te_e = (1 - alpha) * Te_edge[lo] + alpha * Te_edge[hi]
        return Te_c, Te_e
    elif t <= 500:
        return Te_flattop[0], Te_flattop[-1]
    else:
        frac = (t - 500) / 100.0
        idx  = frac * (n_rampup_eqdsk - 1)
        lo   = int(idx)
        hi   = min(lo + 1, n_rampup_eqdsk - 1)
        alpha= idx - lo
        Te_c = (1 - alpha) * Te_axis[n_rampup_eqdsk-1-lo] + alpha * Te_axis[n_rampup_eqdsk-1-hi]
        Te_e = (1 - alpha) * Te_edge[n_rampup_eqdsk-1-lo] + alpha * Te_edge[n_rampup_eqdsk-1-hi]
        return Te_c, Te_e

# Evaluate for every eqtime
init_ne_axis = np.array([build_ne_at_time(t)[0] for t in times])
init_ne_edge = np.array([build_ne_at_time(t)[1] for t in times])
init_Te_axis = np.array([build_Te_at_time(t)[0] for t in times])
init_Te_edge = np.array([build_Te_at_time(t)[1] for t in times])

# Pressure from initial profiles (2 species, quasineutral: p = 2 n_e k T_e)
# 1.602e-16 [J/keV] converts keV→J, giving Pa
init_pax = 2.0 * init_ne_axis * init_Te_axis * 1.602e-16 / 1e3  # kPa

# Also evaluate at t_res (the TORAX solve times)
tres_ne_ax = np.array([build_ne_at_time(t)[0] for t in t_res])
tres_Te_ax = np.array([build_Te_at_time(t)[0] for t in t_res])
tres_pax   = 2.0 * tres_ne_ax * tres_Te_ax * 1.602e-16 / 1e3   # kPa

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle('Initial profiles vs seed EQDSK values\n'
             '(Dashed=seed EQDSK, Solid=initial kinetic profile given to TORAX)',
             fontsize=13)
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax_pax  = fig.add_subplot(gs[0, 0])
ax_ne   = fig.add_subplot(gs[0, 1])
ax_Te   = fig.add_subplot(gs[0, 2])
ax_Ip   = fig.add_subplot(gs[1, 0])
ax_psi  = fig.add_subplot(gs[1, 1])
ax_geo  = fig.add_subplot(gs[1, 2])
ax_pprof_early   = fig.add_subplot(gs[2, 0])
ax_pprof_flattop = fig.add_subplot(gs[2, 1])
ax_ratio = fig.add_subplot(gs[2, 2])

def vline(ax, t, color='grey', lw=0.8, ls='--'):
    ax.axvline(t, color=color, lw=lw, ls=ls, zorder=0)

for ax in [ax_pax, ax_ne, ax_Te, ax_Ip, ax_psi, ax_geo, ax_ratio]:
    vline(ax, 100)
    vline(ax, 500)
    ax.set_xlabel('Time [s]')

# ── p_axis ────────────────────────────────────────────────────────────────────
ax_pax.semilogy(times, eq_pax,    'k--', lw=1.5, label='EQDSK p_axis')
ax_pax.semilogy(times, init_pax,  'C0-', lw=1.5, label='Profile p_axis (2neTe)')
ax_pax.semilogy(t_res, tres_pax,  'C0o', ms=4,   label='TORAX solve times')
ax_pax.set_ylabel('p_axis [kPa]')
ax_pax.set_title('Axis pressure')
ax_pax.legend(fontsize=7)

# ── n_e axis ──────────────────────────────────────────────────────────────────
ax_ne.semilogy(times, init_ne_axis / 1e20, 'C1-',  lw=1.5, label='profile n_e(0)')
ax_ne.semilogy(times, init_ne_edge / 1e20, 'C1--', lw=1.0, label='profile n_e(edge)')
ax_ne.set_ylabel(r'$n_e$ [10²⁰ m⁻³]')
ax_ne.set_title('Electron density (initial profile)')
ax_ne.legend(fontsize=7)

# ── T_e axis ──────────────────────────────────────────────────────────────────
ax_Te.semilogy(times, init_Te_axis, 'C2-',  lw=1.5, label='profile T_e(0)  [keV]')
ax_Te.semilogy(times, init_Te_edge, 'C2--', lw=1.0, label='profile T_e(edge) [keV]')
ax_Te.set_ylabel('T_e [keV]')
ax_Te.set_title('Electron temperature (initial profile)')
ax_Te.legend(fontsize=7)

# ── Ip ───────────────────────────────────────────────────────────────────────
ax_Ip.plot(times, eq_Ip, 'k--', lw=1.5, label='EQDSK Ip')
ax_Ip.set_ylabel('Ip [MA]')
ax_Ip.set_title('Plasma current')
ax_Ip.legend(fontsize=7)

# ── psi_swing ────────────────────────────────────────────────────────────────
ax_psi.plot(times, eq_psi_swing, 'k--', lw=1.5, label='|ψ_bry − ψ_mag| EQDSK')
ax_psi.set_ylabel('Enclosed flux [Wb/rad]')
ax_psi.set_title('Poloidal flux swing (EQDSK)')
ax_psi.legend(fontsize=7)

# ── geometry ─────────────────────────────────────────────────────────────────
ax_geo.plot(times, eq_kappa, 'C4-', lw=1.5, label='κ')
ax_geo.plot(times, eq_delta, 'C5--', lw=1.5, label='δ')
ax_geo.set_ylabel('Shape')
ax_geo.set_title('Equilibrium shape (seed EQDSK)')
ax_geo.legend(fontsize=7)

# ── Pressure profile snapshots: early rampup ─────────────────────────────────
early_idxs = [0, 2, 5, 9]   # t ≈ 0, 17.8, 44.4, 80 s
cmap = plt.cm.plasma(np.linspace(0.2, 0.85, len(early_idxs)))
for k, idx in enumerate(early_idxs):
    t_label = f't={times[idx]:.0f}s'
    ax_pprof_early.plot(psi_norm, eq_pres_prof[idx],  '--', color=cmap[k], lw=1.2)
    # compute initial profile pressure on psi_norm
    if times[idx] <= 80:
        frac_  = times[idx] / 80.0
        idx_   = frac_ * (n_rampup_eqdsk - 1)
        lo_    = int(idx_)
        hi_    = min(lo_ + 1, n_rampup_eqdsk - 1)
        alpha_ = idx_ - lo_
        ne_p   = (1 - alpha_) * ne_prof_list[lo_] + alpha_ * ne_prof_list[hi_]
        Te_p   = (1 - alpha_) * Te_prof_list[lo_] + alpha_ * Te_prof_list[hi_]
    else:
        ne_p = ne_flattop
        Te_p = Te_flattop
    p_prof_kpa = 2.0 * ne_p * Te_p * 1.602e-16 / 1e3
    ne_pn = np.interp(psi_norm, psi_sample, ne_p)
    Te_pn = np.interp(psi_norm, psi_sample, Te_p)
    p_pn  = 2.0 * ne_pn * Te_pn * 1.602e-16 / 1e3
    ax_pprof_early.plot(psi_norm, p_pn, '-', color=cmap[k], lw=1.5, label=t_label)

ax_pprof_early.set_xlabel('ψ_N')
ax_pprof_early.set_ylabel('p [kPa]')
ax_pprof_early.set_title('Rampup pressure profiles\n(solid=TORAX init, dashed=EQDSK)')
ax_pprof_early.legend(fontsize=7)

# ── Pressure profile snapshot: flattop ───────────────────────────────────────
flat_idx = 10  # first flattop time (t=100s)
ne_pn = np.interp(psi_norm, psi_sample, ne_flattop)
Te_pn = np.interp(psi_norm, psi_sample, Te_flattop)
p_flat_kpa = 2.0 * ne_pn * Te_pn * 1.602e-16 / 1e3
ax_pprof_flattop.plot(psi_norm, eq_pres_prof[flat_idx], 'k--', lw=1.5, label='EQDSK t=100s')
ax_pprof_flattop.plot(psi_norm, p_flat_kpa,              'C0-', lw=1.5, label='TORAX init t=100s')
ax_pprof_flattop.set_xlabel('ψ_N')
ax_pprof_flattop.set_ylabel('p [kPa]')
ax_pprof_flattop.set_title('Flattop pressure profile')
ax_pprof_flattop.legend(fontsize=7)

# ── Ratio p_axis(EQDSK) / p_axis(init profile) ───────────────────────────────
ratio = eq_pax / np.where(init_pax > 0, init_pax, np.nan)
ax_ratio.semilogy(times, ratio, 'C3-', lw=1.5)
ax_ratio.axhline(1.0, color='k', lw=0.8, ls='--')
ax_ratio.set_ylabel('p_ax(EQDSK) / p_ax(profile)')
ax_ratio.set_xlabel('Time [s]')
ax_ratio.set_title('Pressure mismatch ratio\n(1 = matched)')

plt.tight_layout()
plt.savefig(os.path.join(REPO, 'vis', 'initial_profiles_vs_eqdsk.png'), dpi=150, bbox_inches='tight')
print('Saved: vis/initial_profiles_vs_eqdsk.png')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
print()
print(f"{'t [s]':>8}  {'EQDSK Ip [MA]':>14}  {'EQDSK p_ax [kPa]':>18}  "
      f"{'Init p_ax [kPa]':>16}  {'ratio':>7}  {'Init ne [1e20]':>14}  {'Init Te [keV]':>14}")
print('-' * 105)
for k, t in enumerate(times):
    r = eq_pax[k] / init_pax[k] if init_pax[k] > 0 else float('nan')
    print(f"{t:8.1f}  {eq_Ip[k]:14.4f}  {eq_pax[k]:18.2f}  "
          f"{init_pax[k]:16.2f}  {r:7.1f}  "
          f"{init_ne_axis[k]/1e20:14.4f}  {init_Te_axis[k]:14.3f}")
