# script to run basic torax sim and plot results

import torax
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

def run_torax_sim(input_config, plot_time=0):

    config = torax.ToraxConfig.from_dict(input_config)
    data_tree, _ = torax.run_simulation(config, log_timestep_info=False)

    detailed_plot_single_sim(data_tree, time=plot_time)

    return(data_tree)




def detailed_plot_single_sim(dt: xr.DataTree, time: float | None = None):
  """Plot simulation results with graceful handling of missing variables."""
  spr = dt.profiles.spr.sel(rho_norm=dt.rho_cell_norm)
  drho_norm = dt.scalars.drho_norm
  j_ohmic = dt.profiles.j_ohmic
  I_ohm = np.sum(spr*j_ohmic*drho_norm, axis=1)

  jnbi = dt.profiles.j_generic_current
  I_nbi = np.sum(spr*jnbi*drho_norm, axis=1)

  te_line_avg = np.mean(dt.profiles.T_e, axis=1)
  ti_line_avg = np.mean(dt.profiles.T_i, axis=1)

  te_core = dt.profiles.T_e[:, 0]
  te_edge = dt.profiles.T_e[:, -1]
  ti_core = dt.profiles.T_i[:, 0]
  ti_edge = dt.profiles.T_i[:, -1]

  ne_core = dt.profiles.n_e[:, 0]
  ne_edge = dt.profiles.n_e[:, -1]

  pax_new = dt.profiles.pressure_thermal_total.sel(rho_norm=0.0, method='nearest').values

  # Choose a time index for plotting
  if time is None:
    time_index = -1
  else:
    time_index = np.argmin(np.abs(dt.time.values - time))

  _, axes = plt.subplots(5, 5, figsize=(24, 15))

  # Set fontsize
  fsize = 13
  fontreduction = 1
  plt.rcParams.update({'font.size': fsize})

  # Helper function to safely get profile or scalar
  def safe_get(obj, attr_path, default=None):
    """Safely access nested attributes, return default if missing."""
    try:
      attrs = attr_path.split('.')
      val = obj
      for attr in attrs:
        val = getattr(val, attr)
      return val
    except (AttributeError, KeyError):
      return default

  # First Row: Currents and Q
  axes[0, 0].plot(dt.time, dt.profiles.Ip_profile[:, -1]/1e6, 'b-', label=r'$I_p$')
  axes[0, 0].set_xlabel(r"Time [s]")
  axes[0, 0].set_ylabel(r"Plasma current [MA]")
  axes[0, 0].legend(fontsize=fsize-fontreduction)

  # Plot current components
  axes[0, 1].plot(dt.time, dt.scalars.I_bootstrap/1e6, 'b-', label=r'$I_{bootstrap}$')
  if hasattr(dt.scalars, 'I_ecrh'):
    axes[0, 1].plot(dt.time, dt.scalars.I_ecrh/1e6, 'r-', label=r'$I_{ecrh}$')
  axes[0, 1].plot(dt.time, I_ohm/1e6, 'm-', label=r'$I_{ohmic}$')
  axes[0, 1].plot(dt.time, I_nbi/1e6, 'k-', label=r'$I_{nbi}$')
  axes[0, 1].set_xlabel(r"Time [s]")
  axes[0, 1].set_ylabel(r"Current [MA]")
  axes[0, 1].legend(fontsize=fsize-fontreduction)
  axes[0, 1].set_title(r"Total currents", fontsize=fsize-fontreduction)

  # Plot Q over time
  if hasattr(dt.scalars, 'Q_fusion'):
    axes[0, 2].plot(dt.time[10:], dt.scalars.Q_fusion[10:], 'r-')
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel(r"Q")
    axes[0, 2].set_title(r"Fusion Q", fontsize=fsize-fontreduction)

  # Plot H20 over time
  if hasattr(dt.scalars, 'H20'):
    axes[0, 3].plot(dt.time[10:], dt.scalars.H20[10:], 'r-')
    axes[0, 3].set_xlabel("Time [s]")
    axes[0, 3].set_ylabel(r"H20")
    axes[0, 3].set_title(r"H20 confinement factor", fontsize=fsize-fontreduction)

  axes[0, 4].plot(dt.time, pax_new, label='pax')
  axes[0, 4].set_xlabel("Time [s]")
  axes[0, 4].set_ylabel(r"Pressure on axis (Pa)")
  axes[0, 4].set_title(r"Pressure on axis", fontsize=fsize-fontreduction)
  axes[0, 4].legend(fontsize=fsize-fontreduction)
  
  # Second Row: Vloop and heating powers
  axes[1, 0].plot(dt.time, dt.scalars.v_loop_lcfs, 'r-')
  axes[1, 0].set_xlabel("Time [s]")
  axes[1, 0].set_ylabel(r"$V_{loop}(LCFS)$")
  axes[1, 0].set_title(r"Loop voltage (at LCFS)", fontsize=fsize-fontreduction)

  # Plot heating powers
  if hasattr(dt.scalars, 'P_ecrh_e'):
    axes[1, 1].plot(dt.time, dt.scalars.P_ecrh_e/1e6, 'b-', label=r'$P_{ECRH}$')
  if hasattr(dt.scalars, 'P_aux_generic_total'):
    axes[1, 1].plot(dt.time, dt.scalars.P_aux_generic_total/1e6, 'r-', label=r'$P_{NBI}$')
  if hasattr(dt.scalars, 'P_ohmic_e'):
    axes[1, 1].plot(dt.time, dt.scalars.P_ohmic_e/1e6, 'm-', label=r'$P_{ohmic}$')
  if hasattr(dt.scalars, 'P_alpha_total'):
    axes[1, 1].plot(dt.time, dt.scalars.P_alpha_total/1e6/5, 'k-', label=r'$P_{fusion}/5$')
  axes[1, 1].set_xlabel("Time [s]")
  axes[1, 1].set_ylabel(r"Heating powers $[MW]$")
  axes[1, 1].legend(fontsize=fsize-fontreduction)
  axes[1, 1].set_title(r"Total heating powers", fontsize=fsize-fontreduction)

  # Plot sinks
  if hasattr(dt.scalars, 'P_cyclotron_e'):
    axes[1, 2].plot(dt.time, dt.scalars.P_cyclotron_e/1e6, 'b-', label=r'$P_{cyclotron}$')
  if hasattr(dt.scalars, 'P_radiation_e'):
    axes[1, 2].plot(dt.time, dt.scalars.P_radiation_e/1e6, 'r-', label=r'$P_{rad}$')
  axes[1, 2].set_xlabel("Time [s]")
  axes[1, 2].set_ylabel(r"Heat sinks $[MW]$")
  axes[1, 2].legend(fontsize=fsize-fontreduction)
  axes[1, 2].set_title(r"Total sinks", fontsize=fsize-fontreduction)

  # Plot q-profile
  axes[1, 3].plot(dt.time, dt.scalars.q_min, 'b-', label=r'$q_{min}$')
  axes[1, 3].plot(dt.time, dt.profiles.q[:, 0], 'r-', label=r'$q_{0}$')
  if hasattr(dt.scalars, 'q95'):
    axes[1, 3].plot(dt.time, dt.scalars.q95, 'm-', label=r'$q_{95}$')
  axes[1, 3].set_xlabel("Time [s]")
  axes[1, 3].set_ylabel(r"$q$")
  axes[1, 3].legend(fontsize=fsize-fontreduction)
  axes[1, 3].set_title(r"Safety factor (q)", fontsize=fsize-fontreduction)

  # Plot inductance
  axes[1, 4].plot(dt.time, dt.scalars.li3, 'r-')
  axes[1, 4].set_xlabel("Time [s]")
  axes[1, 4].set_ylabel(r"li(3)")
  axes[1, 4].set_title(r"Normalized internal inductance", fontsize=fsize-fontreduction)

  # Third Row: Transport and profiles
  axes[2, 0].plot(dt.rho_face_norm, dt.profiles.chi_turb_e[time_index, :], 'b-', label=r'$\chi_e$')
  axes[2, 0].plot(dt.rho_face_norm, dt.profiles.chi_turb_i[time_index, :], 'r-', label=r'$\chi_i$')
  axes[2, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[2, 0].set_ylabel(r"Heat conductivity $[m^2/s]")
  axes[2, 0].legend(fontsize=fsize-fontreduction)
  axes[2, 0].set_title(r"Heat transport coefficients", fontsize=fsize-fontreduction)

  axes[2, 1].plot(dt.rho_norm, dt.profiles.T_e[time_index, :], 'b-', label=r'$T_e$')
  axes[2, 1].plot(dt.rho_norm, dt.profiles.T_i[time_index, :], 'r-', label=r'$T_i$')
  axes[2, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[2, 1].set_ylabel(r"Temperature [keV]")
  axes[2, 1].legend(fontsize=fsize-fontreduction)
  axes[2, 1].set_title(r"Temperature profiles", fontsize=fsize-fontreduction)

  axes[2, 2].plot(dt.rho_norm, dt.profiles.n_e[time_index, :], 'b-', label=r'$n_e$')
  axes[2, 2].plot(dt.rho_norm, dt.profiles.n_i[time_index, :], 'b--', label=r'$n_i$')
  axes[2, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[2, 2].set_ylabel(r"Density [$10^{20}$ m$^{-3}$]")
  axes[2, 2].legend(fontsize=fsize-fontreduction)
  axes[2, 2].set_title(r"$n_e$, $n_i$", fontsize=fsize-fontreduction)

  axes[2, 3].plot(dt.rho_face_norm, dt.profiles.q[time_index, :], 'b-', label='q')
  axes[2, 3].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[2, 3].set_ylabel(r"q")
  axes[2, 3].legend(fontsize=fsize-fontreduction)
  axes[2, 3].set_title(r"Safety Factor ($q$)", fontsize=fsize-fontreduction)

  axes[2, 4].plot(dt.rho_face_norm, dt.profiles.magnetic_shear[time_index, :], 'b-', label=r'$\hat{s}$')
  axes[2, 4].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[2, 4].set_ylabel(r"$\hat{s}$")
  axes[2, 4].legend(fontsize=fsize-fontreduction)
  axes[2, 4].set_title(r"Magnetic shear ($\hat{s}$)", fontsize=fsize-fontreduction)

  # Fourth Row: Current sources and sinks
  psidot = dt.profiles.v_loop[time_index, :]
  ymin = min(min(psidot), 0) * 1.2
  ymax = max(max(psidot), 0) * 1.2
  axes[3, 0].plot(dt.rho_norm, psidot, 'b-')
  axes[3, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[3, 0].set_ylabel(r"Vloop [V]")
  axes[3, 0].set_ylim([ymin, ymax])
  axes[3, 0].set_title(r"Loop voltage profile", fontsize=fsize-fontreduction)

  axes[3, 1].plot(dt.rho_norm, dt.profiles.j_total[time_index, :]/1e6, 'b-', label=r'$j_{total}$')
  axes[3, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[3, 1].set_ylabel(r"Currents $[MA/m^2]$")
  axes[3, 1].legend(fontsize=fsize-fontreduction)
  axes[3, 1].set_title(r"Total current density", fontsize=fsize-fontreduction)

  # Current components
  if hasattr(dt.profiles, 'j_ecrh'):
    axes[3, 2].plot(dt.rho_cell_norm, dt.profiles.j_ecrh[time_index, :]/1e6, 'b-', label=r'$j_{ecrh}$')
  axes[3, 2].plot(dt.rho_cell_norm, dt.profiles.j_generic_current[time_index, :]/1e6, 'r-', label=r'$j_{nbi}$')
  axes[3, 2].plot(dt.rho_cell_norm, dt.profiles.j_ohmic[time_index, :]/1e6, 'm-', label=r'$j_{ohmic}$')
  axes[3, 2].plot(dt.rho_norm, dt.profiles.j_bootstrap[time_index, :]/1e6, 'g-', label=r'$j_{bootstrap}$')
  axes[3, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
  axes[3, 2].set_ylabel(r"Currents $[MA/m^2]$")
  axes[3, 2].legend(fontsize=fsize-fontreduction)
  axes[3, 2].set_title(r"Current source densities", fontsize=fsize-fontreduction)

  # Radiation
  if hasattr(dt.profiles, 'p_impurity_radiation_e'):
    axes[3, 3].plot(dt.rho_cell_norm, dt.profiles.p_impurity_radiation_e[time_index, :]/1e6, 'b-', label=r'$P_{rad}$')
    axes[3, 3].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 3].set_ylabel(r"Heat sink density $[MW/m^3]$")
    axes[3, 3].legend(fontsize=fsize-fontreduction)
    axes[3, 3].set_title(r"Radiation heat sink", fontsize=fsize-fontreduction)

  # Heat sources (safely plot available variables)
  heat_sources = []
  if hasattr(dt.profiles, 'p_ecrh_e'):
    axes[3, 4].plot(dt.rho_cell_norm, dt.profiles.p_ecrh_e[time_index, :]/1e6, 'b-', label=r'$Q_{ecrh}$')
    heat_sources.append(True)
  if hasattr(dt.profiles, 'p_generic_heat_i'):
    axes[3, 4].plot(dt.rho_cell_norm, dt.profiles.p_generic_heat_i[time_index, :]/1e6, 'r-', label=r'$Q_{nbi_i}$')
    heat_sources.append(True)
  if hasattr(dt.profiles, 'p_generic_heat_e'):
    axes[3, 4].plot(dt.rho_cell_norm, dt.profiles.p_generic_heat_e[time_index, :]/1e6, 'm-', label=r'$Q_{nbi_e}$')
    heat_sources.append(True)
  if hasattr(dt.profiles, 'p_alpha_i'):
    axes[3, 4].plot(dt.rho_cell_norm, dt.profiles.p_alpha_i[time_index, :]/1e6, 'g-', label=r'$Q_{fus_i}$')
    heat_sources.append(True)
  if hasattr(dt.profiles, 'p_alpha_e'):
    axes[3, 4].plot(dt.rho_cell_norm, dt.profiles.p_alpha_e[time_index, :]/1e6, 'k-', label=r'$Q_{fus_e}$')
    heat_sources.append(True)
  if heat_sources:
    axes[3, 4].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 4].set_ylabel(r"Heat source densities $[MW/m^3]$")
    axes[3, 4].legend(fontsize=fsize-fontreduction)
    axes[3, 4].set_title(r"Heat sources", fontsize=fsize-fontreduction)

  # Fifth Row: Core time series
  axes[4, 0].plot(dt.time, te_core, 'b-', label=r'$T_{e,core}$')
  axes[4, 0].plot(dt.time, ti_core, 'r-', label=r'$T_{i,core}$')
  axes[4, 0].set_xlabel("Time [s]")
  axes[4, 0].set_ylabel(r"Temperature [keV]")
  axes[4, 0].set_title(r"Core temperatures", fontsize=fsize-fontreduction)
  axes[4, 0].legend(fontsize=fsize-fontreduction)

  axes[4, 1].plot(dt.time, ne_core/1e20, 'b-', label=r'$n_{e,core}$')
  axes[4, 1].set_xlabel("Time [s]")
  axes[4, 1].set_ylabel(r"Density [$10^{20}$ m$^{-3}$]")
  axes[4, 1].set_title(r"Core electron density", fontsize=fsize-fontreduction)
  axes[4, 1].legend(fontsize=fsize-fontreduction)

  # Hide unused subplots
  for col in range(2, 5):
    axes[4, col].set_visible(False)

  plt.tight_layout()
  plt.show()
