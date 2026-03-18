"""Generate pulse movie frames from TokTox simulation data.

Each frame is a 3-column, 16:9 composite:
  Column 0  – run info (top 1/3) + equilibrium plot (bottom 2/3)
  Column 1  – scalar time-series with moving vertical time indicator
  Column 2  – radial profiles at the current time slice
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

# ── Colour / style variables (edit once, changes propagate everywhere) ──
COLOR_TM  = 'steelblue'   # TokaMaker traces
COLOR_TX  = 'crimson'      # TORAX traces
LS_PRI    = '-'            # primary (left) y-axis
LS_SEC    = '--'           # secondary (right) y-axis
MK_TM     = '.'            # marker for TokaMaker
MK_SZ     = 3              # marker size
LW        = 1.6            # default line width
# Colours for multi-TORAX outputs (neither steelblue nor crimson)
COLORS_MULTI = [
    'darkorange', 'forestgreen', 'mediumpurple',
    'goldenrod', 'deeppink', 'teal', 'sienna',
]
VLINE_COLOR = 'black'
VLINE_LS    = ':'
VLINE_LW    = 1.0
GRID_ALPHA  = 0.2
LEGEND_FS   = 9            # legend font size
TITLE_FS    = 13           # subplot title font size
LABEL_FS    = 11           # axis-label font size
TICK_FS     = 10           # tick-label font size
INFO_FS     = 13           # info panel font size
DIAG_FS     = 13           # diagnostic text under equil

FIG_W, FIG_H = 19.2, 10.8  # inches  (16 : 9)
DPI = 200                   # → 3840 × 2160 px


# ═══════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════

def generate_pulse_movie(tt, step, run_name='', save_frames=True, speed_factor=1.0):
    """Create one PNG frame per time-slice, then encode MP4 + GIF.

    Parameters
    ----------
    tt : TokTox
        Fully-populated TokTox object (after _run_transport + _run_gs).
    step : int
        Current coupling-iteration index.
    run_name : str
        Human-readable run label shown in the info panel.
    save_frames : bool
        If True, keep individual frame PNG files. If False, generate them
        for MP4 encoding but delete them afterward.
    speed_factor : float
        Playback speed relative to real time. 1.0 = real-time, 2.0 = 2x speed, etc.
        Default is 1.0 (real-time pulse duration).
    """
    vid_dir = os.path.join(tt._out_dir, 'vid')
    os.makedirs(vid_dir, exist_ok=True)

    times = tt._times
    n     = len(times)

    # Pre-compute flux consumed (Wb)
    psi_lcfs_tm = np.array(tt._state['psi_lcfs_tm'])
    psi_lcfs_tx = np.array(tt._state['psi_lcfs_tx'])
    flux_con_tm = (psi_lcfs_tm - psi_lcfs_tm[0]) * 2.0 * np.pi
    flux_con_tx = (psi_lcfs_tx - psi_lcfs_tx[0]) * 2.0 * np.pi

    for idx in range(n):
        fpath = os.path.join(vid_dir, f'frame_{idx:04d}.png')
        _render_frame(tt, step, idx, times[idx], times,
                      flux_con_tm, flux_con_tx, fpath, run_name)
        plt.close('all')

    # Calculate fps based on real-time speed factor
    total_time = times[-1] - times[0] if len(times) > 1 else 1.0
    fps = speed_factor * n / total_time
    _encode_video(vid_dir, step, fps=fps)
    
    # If not saving frames, delete them after encoding
    if not save_frames:
        for idx in range(n):
            fpath = os.path.join(vid_dir, f'frame_{idx:04d}.png')
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Frame renderer
# ═══════════════════════════════════════════════════════════════════════

def _render_frame(tt, step, idx, t_now, times,
                  flux_con_tm, flux_con_tx, out_path, run_name):

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    gs  = GridSpec(6, 3, figure=fig,
                   width_ratios=[1.2, 1.0, 1.0],
                   wspace=0.30, hspace=0.18, # wspace = whitespace between columns, .42 is too much, .15 is too little
                   left=0.045, right=0.965, top=0.96, bottom=0.05)

    # ── Column 0: info + equil ──────────────────────────────────────
    ax_text  = fig.add_subplot(gs[0:1, 0])
    ax_equil = fig.add_subplot(gs[1:5, 0])
    _draw_info(ax_text, tt, step, run_name)
    _draw_equil(ax_equil, tt, step, idx, t_now)

    # ── Column 1: scalar time-series ────────────────────────────────
    sax = [fig.add_subplot(gs[j, 1]) for j in range(6)]
    _draw_scalars(sax, tt, times, t_now, flux_con_tm, flux_con_tx)
    for ax in sax[:-1]:
        ax.tick_params(labelbottom=False)
    sax[-1].set_xlabel('Time [s]', fontsize=LABEL_FS)

    # ── Column 2: profile snapshots ─────────────────────────────────
    pax = [fig.add_subplot(gs[j, 2]) for j in range(6)]
    _draw_profiles(pax, tt, idx)
    for ax in pax[:-1]:
        ax.tick_params(labelbottom=False)
    pax[-1].set_xlabel(r'$\psi_N$', fontsize=LABEL_FS)

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Column 0 helpers
# ═══════════════════════════════════════════════════════════════════════

def _draw_info(ax, tt, step, run_name):
    ax.axis('off')
    lines = [
        f'Run:   {run_name}            Step:  {step}',
        f'n_rho: {tt._n_rho}          dt:    {tt._dt} s',
        f'time range:     [{tt._t_init}, {tt._t_final}] s          times: {len(tt._times)}',
        f'LSF:   {tt._last_surface_factor}',
    ]
    ax.text(0.05, 0.95, '\n'.join(lines),
            transform=ax.transAxes, fontsize=INFO_FS, va='top',
            fontfamily='monospace')


def _draw_equil(ax, tt, step, idx, t_now):
    s = tt._state
    eq_img = os.path.join(tt._out_dir, 'equil',
                          f'equil_{step:03}.{idx:03}.png')
    tm_ok = os.path.exists(eq_img)

    if tm_ok:
        img = imread(eq_img)
        ax.imshow(img, aspect='equal')
        ax.axis('off')
        ax.set_title(f't = {t_now:.2f} s', fontsize=TITLE_FS + 2)

        # Build diagnostic text for below equil image
        div_flags = getattr(tt, '_diverted_flags', {})
        div_flag = div_flags.get(idx)
        config_str = 'Diverted' if div_flag else ('Limited' if div_flag is not None else '?')

        diag = (
            f"i = {idx}    "
            f"Ip = {abs(s['Ip_tm'][idx])/1e6:.3f} MA    "
            f"pax = {s['pax_tm'][idx]/1e3:.1f} kPa\n"
            f"R = {s['R0_mag'][idx]:.3f} m    "
            f"a = {s['a'][idx]:.3f} m    "
            f"B0 = {s['B0'][idx]:.3f} T\n"
            f"κ = {s['kappa'][idx]:.3f}    "
            f"δ = {s['delta'][idx]:.3f}    "
            f"Configuration = {config_str}"
            f"ψ_lcfs = {s['psi_lcfs_tm'][idx]:.4f} Wb/rad\n"
        )
        ax.text(0.5, -0.02, diag, transform=ax.transAxes, fontsize=DIAG_FS,
                ha='center', va='top', fontfamily='monospace')
    else:
        ax.axis('off')
        ax.text(0.5, 0.55, 'TokaMaker failed\nto converge',
                transform=ax.transAxes, fontsize=16,
                ha='center', va='center', color='darkred', fontweight='bold')
        diag = (
            f"Ip target = {abs(s['Ip'][idx])/1e6:.3f} MA\n"
            f"pax target = {s['pax'][idx]/1e3:.1f} kPa\n"
            f"R = {s['R0_mag'][idx]:.3f} m    a = {s['a'][idx]:.3f} m\n"
            f"B0 = {s['B0'][idx]:.3f} T    κ = {s['kappa'][idx]:.3f}    δ = {s['delta'][idx]:.3f}"
            f"ψ_lcfs TX = {s['psi_lcfs_tx'][idx]:.4f} Wb/rad\n"
        )
        ax.text(0.5, 0.25, diag, transform=ax.transAxes, fontsize=DIAG_FS,
                ha='center', va='center', fontfamily='monospace')
        ax.set_title(f't = {t_now:.2f} s  (FAILED)', fontsize=TITLE_FS + 2,
                     color='darkred')


# ═══════════════════════════════════════════════════════════════════════
#  Column 1 – scalar time-series
# ═══════════════════════════════════════════════════════════════════════

def _style(ax):
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=TICK_FS)


def _vline(axes, t_now):
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.axvline(t_now, color=VLINE_COLOR, ls=VLINE_LS, lw=VLINE_LW,
                   zorder=10)


def _draw_scalars(axes, tt, times, t_now, flux_con_tm, flux_con_tx):
    s = tt._state
    r = tt._results

    # 1 ── Ip and Ip_NI (left) / l_i (right) ────────────────────────
    ax = axes[0]
    ax.plot(times, np.array(s['Ip_tm']) / 1e6,
            color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='Ip TM')
    ax.plot(times, np.array(s['Ip_tx']) / 1e6,
            color=COLOR_TX, ls=LS_PRI, lw=LW, label='Ip TX')
    ax.plot(times, np.array(s['Ip_NI_tx']) / 1e6,
            color=COLOR_TX, ls=LS_SEC, lw=LW, label='Ip_NI TX')
    ax.set_ylabel('Ip [MA]', fontsize=LABEL_FS)
    ax.set_title('Plasma Current', fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper left')
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(times, s['l_i_tm'],
             color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label='l_i TM')
    ax2.set_ylabel('l_i', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 2 ── V_loop (left) / flux consumed (right) ─────────────────────
    ax = axes[1]
    ax.plot(times, s['vloop_tm'],
            color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='Vloop TM')
    ax.plot(times, s['vloop_tx'],
            color=COLOR_TX, ls=LS_PRI, lw=LW, label='Vloop TX')
    ax.set_ylabel('V_loop [V]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper left')
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(times, flux_con_tm,
             color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label='Φ TM')
    ax2.plot(times, flux_con_tx,
             color=COLOR_TX, ls=LS_SEC, lw=LW, label='Φ TX')
    ax2.set_ylabel('Flux Consumed [Wb]', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 3 ── pax ────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(times, np.array(s['pax_tm']) / 1e3,
            color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='pax TM')
    ax.plot(times, np.array(s['pax']) / 1e3,
            color=COLOR_TX, ls=LS_PRI, lw=LW, label='pax TX')
    ax.set_ylabel('pax [kPa]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper left')
    _style(ax)

    # 4 ── Power channels (TORAX only → multi-colours) ───────────────
    ax = axes[3]
    pkeys = [
        ('P_ohmic_e',     'Ohmic',     COLORS_MULTI[0]),
        ('P_aux_total',   'Aux',       COLORS_MULTI[1]),
        ('P_alpha_total', 'Fusion',    COLORS_MULTI[2]),
        ('P_radiation_e', 'Radiation', COLORS_MULTI[3]),
        ('P_SOL_total',   'SOL',       COLORS_MULTI[4]),
    ]
    for key, label, clr in pkeys:
        if key in r:
            ax.plot(r[key]['x'], np.array(r[key]['y']) / 1e6,
                    color=clr, ls=LS_PRI, lw=LW, label=label)
    ax.set_ylabel('Power [MW]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper left', ncol=2)
    _style(ax)
    # Secondary axis: Q
    ax2 = ax.twinx()
    if 'Q' in r:
        ax2.plot(r['Q']['x'], r['Q']['y'],
                 color='indigo', ls=LS_SEC, lw=LW, label='Q')
    # if 'P_alpha_total' in r:
    #     ax2.plot(r['P_alpha_total']['x'],
    #              np.array(r['P_alpha_total']['y']) / 1e6,
    #              color='darkgreen', ls=LS_SEC, lw=LW, label='P_fus [MW]')
    ax2.set_ylabel('Q', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 5 ── ψ_axis (left) / ψ_lcfs (right) ────────────────────────────
    ax = axes[4]
    ax.plot(times, s['psi_axis_tm'],
            color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='ψ_ax TM')
    ax.plot(times, s['psi_axis_tx'],
            color=COLOR_TX, ls=LS_PRI, lw=LW, label='ψ_ax TX')
    ax.set_ylabel('ψ_axis [Wb/rad]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper left')
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(times, s['psi_lcfs_tm'],
             color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label='ψ_lcfs TM')
    ax2.plot(times, s['psi_lcfs_tx'],
             color=COLOR_TX, ls=LS_SEC, lw=LW, label='ψ_lcfs TX')
    ax2.set_ylabel('ψ_lcfs [Wb/rad]', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='lower right')

    # 6 ── Coil currents ─────────────────────────────────────────────
    ax = axes[5]
    coil_data = tt._results.get('COIL', {})
    coil_bounds = getattr(tt, '_coil_bounds', {})
    if coil_data:
        coil_colors = plt.cm.tab10(np.linspace(0, 1, max(len(coil_data), 1)))
        for ci, (cname, cvals) in enumerate(sorted(coil_data.items())):
            ct = sorted(cvals.keys())
            ci_vals = [cvals[t] * 1e-3 for t in ct]  # kA
            ax.plot(ct, ci_vals, ls=LS_PRI, lw=LW * 0.7, color=coil_colors[ci],
                    label=cname)
        # Draw limit lines from first coil bound (all same by default)
        # if coil_bounds:
        #     first_bounds = next(iter(coil_bounds.values()))
        #     ax.axhline(first_bounds[0] * 1e-3, color='r', ls='--', lw=0.6, alpha=0.6)
        #     ax.axhline(first_bounds[1] * 1e-3, color='r', ls='--', lw=0.6, alpha=0.6)
    ax.set_ylabel('I_coil [kA]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS - 2, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
    _style(ax)

    # Vertical time indicator on every scalar panel
    _vline(axes, t_now)


# ═══════════════════════════════════════════════════════════════════════
#  Column 2 – radial profiles
# ═══════════════════════════════════════════════════════════════════════

def _prof(state_dict, idx):
    """Return (x, y) arrays from a profile state dict, or (None, None)."""
    d = state_dict.get(idx)
    if d is None:
        return None, None
    return d['x'], d['y']


def _draw_profiles(axes, tt, idx):
    s = tt._state

    # 1 ── q profile ──────────────────────────────────────────────────
    ax = axes[0]
    x, y = _prof(s['q_prof_tm'], idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='q TM')
    x, y = _prof(s['q_prof_tx'], idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TX, ls=LS_PRI, lw=LW, label='q TX')
    ax.set_ylabel('q', fontsize=LABEL_FS)
    ax.set_title('Profiles', fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS, loc='best')
    _style(ax)

    # 2 ── n_e (left) / T_e (right) ──────────────────────────────────
    ax = axes[1]
    x, y = _prof(s.get('n_e', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TX, ls=LS_PRI, lw=LW, label='ne TX')
    ax.set_ylabel('ne [m⁻³]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='lower left')
    _style(ax)
    x, y = _prof(s.get('T_e', {}), idx)
    if x is not None:
        ax2 = ax.twinx()
        ax2.plot(x, y, color=COLOR_TX, ls=LS_SEC, lw=LW, label='Te TX')
        x_ti, y_ti = _prof(s.get('T_i', {}), idx)
        if x_ti is not None:
            ax2.plot(x_ti, y_ti, color='forestgreen', ls='--', lw=LW, label='Ti TX')
        ax2.set_ylabel('T [keV]', fontsize=LABEL_FS)
        ax2.tick_params(labelsize=TICK_FS)
        ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 3 ── j components (left, multi-colour) / FF' (right) ───────────
    ax = axes[2]
    j_keys = [
        ('j_tot',             'j_tot',  COLORS_MULTI[0]),
        ('j_ohmic',           'j_ohm',  COLORS_MULTI[1]),
        ('j_ni',              'j_NI',   COLORS_MULTI[2]),
        ('j_bootstrap',       'j_BS',   COLORS_MULTI[3]),
        ('j_ecrh',            'j_EC',   COLORS_MULTI[4]),
        ('j_generic_current', 'j_gen',  COLORS_MULTI[5]),
    ]
    for skey, label, clr in j_keys:
        x, y = _prof(s.get(skey, {}), idx)
        if x is not None:
            ax.plot(x, y / 1e6, color=clr, ls=LS_PRI, lw=LW, label=label)
    ax.set_ylabel('j [MA/m²]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS - 1, loc='upper left', ncol=2)
    _style(ax)
    ax2 = ax.twinx()
    x, y = _prof(s.get('ffp_prof_tm', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label="FF' TM")
    x, y = _prof(s.get('ffp_prof_tx', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TX, ls=LS_SEC, lw=LW, label="FF' TX")
    ax2.set_ylabel("FF'", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 4 ── p' (left) / p (right) ──────────────────────────────────────
    ax = axes[3]
    x, y = _prof(s.get('pp_prof_tm', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label="p' TM")
    x, y = _prof(s.get('pp_prof_tx', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TX, ls=LS_PRI, lw=LW, label="p' TX")
    ax.set_ylabel("p'", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='center')
    _style(ax)
    ax2 = ax.twinx()
    x, y = _prof(s.get('p_prof_tm', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label='p TM')
    x, y = _prof(s.get('p_prof_tx', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TX, ls=LS_SEC, lw=LW, label='p TX')
    ax2.set_ylabel('p [Pa]', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='upper right')

    # 5 ── <1/R> (left) / ψ profile (right) ──────────────────────────
    ax = axes[4]
    x, y = _prof(s.get('R_inv_avg_tm', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TM, ls=LS_PRI, lw=LW, marker=MK_TM, ms=MK_SZ, label='<1/R> TM')
    x, y = _prof(s.get('R_inv_avg_tx', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TX, ls=LS_PRI, lw=LW, label='<1/R> TX')
    ax.set_ylabel('<1/R> [1/m]', fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc='center left')
    _style(ax)
    ax2 = ax.twinx()
    x, y = _prof(s.get('psi_tm', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TM, ls=LS_SEC, lw=LW, marker=MK_TM, ms=MK_SZ, label='ψ TM')
    x, y = _prof(s.get('psi_tx', {}), idx)
    if x is not None:
        ax2.plot(x, y, color=COLOR_TX, ls=LS_SEC, lw=LW, label='ψ TX')
    ax2.set_ylabel('ψ [Wb/rad]', fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.legend(fontsize=LEGEND_FS, loc='center right')

    # 6 ── η (resistivity, TORAX only) ───────────────────────────────
    ax = axes[5]
    x, y = _prof(s.get('eta_prof', {}), idx)
    if x is not None:
        ax.plot(x, y, color=COLOR_TX, ls=LS_PRI, lw=LW, label='η TX')
    ax.set_ylabel('η [Ω·m]', fontsize=LABEL_FS)
    ax.set_yscale('log')
    ax.legend(fontsize=LEGEND_FS)
    _style(ax)


# ═══════════════════════════════════════════════════════════════════════
#  Video encoding  (frames → MP4 + GIF via ffmpeg, best-effort)
# ═══════════════════════════════════════════════════════════════════════

def _encode_video(vid_dir, step, fps=2):
    pattern  = os.path.join(vid_dir, 'frame_%04d.png')
    mp4_path = os.path.join(vid_dir, f'!pulse_step{step:03}.mp4')
    gif_path = os.path.join(vid_dir, f'!pulse_step{step:03}.gif')

    # MP4
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps),
             '-i', pattern,
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
             mp4_path],
            check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f'Warning: ffmpeg failed to encode MP4 for step {step}. Check if ffmpeg is installed and in PATH.')
        pass  # ffmpeg unavailable or error – frames are still saved

    # GIF
    # try:
    #     subprocess.run(
    #         ['ffmpeg', '-y', '-framerate', str(fps),
    #          '-i', pattern,
    #          '-vf', 'fps=2,scale=1920:-1:flags=lanczos',
    #          gif_path],
    #         check=True, capture_output=True)
    # except (subprocess.CalledProcessError, FileNotFoundError):
    #     print(f'Warning: ffmpeg failed to encode GIF for step {step}. Check if ffmpeg is installed and in PATH.')
    #     pass
