"""
read_eqdsk_extended.py

Extended gEQDSK reader that extracts the same fields as  TokaMaker. read_eqdsk,
plus derived flux-surface geometry:

    psi_n         - normalised poloidal flux grid (0 = axis, 1 = boundary)
    volume        - volume enclosed by each flux surface  V(psi_n)  [m^3]
    vpr           - dV/dpsi  (not normalised psi)  [m^3 / (Wb/rad)]
    R_avg         - flux-surface average <R>  [m]
    R_inv_avg     - flux-surface average <1/R>  [1/m]

Uses OMFIT's contour tracing algorithm.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import integrate
import contourpy
from scipy import ndimage
from matplotlib import path


# ============================================================================
# OMFIT-derived helper functions for robust contour tracing
# Based on omfit_classes.utils_math.contourPaths and supporting functions
# ============================================================================

def remove_adjacent_duplicates(x):
    """Remove adjacent duplicate points from array."""
    if len(x) == 0:
        return x
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x) - 1):
        if np.allclose(x[i], x[i + 1]):
            keep[i + 1] = False
    return x[keep]


def get_contour_generator(X, Y, Z, corner_mask=True):
    """
    Create a contour generator using contourpy with matplotlib-compatible algorithm.
    Wrapper to maintain compatibility with different matplotlib/contourpy versions.
    """
    # Use contourpy's mpl2014 algorithm (default matplotlib algorithm)
    return contourpy.contour_generator(
        X, Y, Z,
        name='mpl2014',
        line_type=contourpy.LineType.SeparateCode,
        corner_mask=corner_mask
    )


def contourPaths(x, y, Z, levels, remove_boundary_points=False, smooth_factor=1):
    """
    Trace contour paths using contourpy, returning matplotlib Path objects.
    
    Direct copy of OMFIT's contourPaths implementation.
    
    Parameters:
    -----------
    x : 1D x coordinate
    y : 1D y coordinate  
    Z : 2D data
    levels : levels to trace
    remove_boundary_points : remove traces at the boundary
    smooth_factor : smooth contours by cranking up grid resolution
        
    Returns:
    --------
    list of segments (list of matplotlib.path.Path objects per level)
    """
    
    sf = int(round(smooth_factor))
    if sf > 1:
        x = ndimage.zoom(x, sf)
        y = ndimage.zoom(y, sf)
        Z = ndimage.zoom(Z, sf)
    
    [X, Y] = np.meshgrid(x, y)
    contour_generator = get_contour_generator(X, Y, Z, True)
    
    mx = min(x)
    Mx = max(x)
    my = min(y)
    My = max(y)
    
    allsegs = []
    for level in levels:
        verts = contour_generator.create_contour(level)
        
        if isinstance(verts, tuple):
            # comment from OMFIT: Matplotlib>3.4.3 and ContourPy return vertices and codes as a tuple. Prior it was just vertices.
            segs = verts[0]
        else:
            segs = verts
        
        for i, seg in enumerate(segs):
            segs[i] = remove_adjacent_duplicates(seg)
        
        if not remove_boundary_points:
            segs_ = segs
        else:
            segs_ = []
            for segarray in segs:
                segarray = np.array(segarray)
                x_ = segarray[:, 0]
                y_ = segarray[:, 1]
                valid = []
                for i in range(len(x_) - 1):
                    if np.isclose(x_[i], x_[i + 1]) and (np.isclose(x_[i], Mx) or np.isclose(x_[i], mx)):
                        continue
                    if np.isclose(y_[i], y_[i + 1]) and (np.isclose(y_[i], My) or np.isclose(y_[i], my)):
                        continue
                    valid.append((x_[i], y_[i]))
                    if i == len(x_):
                        valid.append((x_[i + 1], y_[i + 1]))
                if len(valid):
                    segs_.append(np.array(valid))
        
        segs = list(map(path.Path, segs_))
        allsegs.append(segs)
    return allsegs


# ---------------------------------------------------------------------------
# helper functions copied from TokaMaker.read_eqdsk
# ---------------------------------------------------------------------------

def _read_1d(fid, n):
    """Read n floats from a 5-per-line, 16-chars-per-value Fortran record."""
    j = 0
    output = np.zeros(n)
    for i in range(n):
        if j == 0:
            line = fid.readline()
        output[i] = line[j:j + 16]
        j += 16
        if j == 16 * 5:
            j = 0
    return output


def _read_2d(fid, n, m):
    """Read n*m floats into shape (n, m) from a 5-per-line Fortran record."""
    j = 0
    output = np.zeros((n, m))
    for k in range(n):
        for i in range(m):
            if j == 0:
                line = fid.readline()
            output[k, i] = line[j:j + 16]
            j += 16
            if j == 16 * 5:
                j = 0
    return output


# ---------------------------------------------------------------------------
# copy of TokaMaker.read_eqdsk
# ---------------------------------------------------------------------------

def read_eqdsk(filename):
    """Read gEQDSK file.

    Returns a dictionary with exactly the same keys as the original
    tokamaker read_eqdsk:

        case, nr, nz,
        rdim, zdim, rcentr, rleft, zmid,
        raxis, zaxis, psimag, psibry, bcentr, ip,
        fpol, pres, ffprim, pprime,
        psirz,
        qpsi,
        nbbs, nlim,
        rzout, rzlim
    """
    obj = {}
    with open(filename, 'r') as fid:
        # --- header line ---
        line = fid.readline()
        obj['case'] = line[:48]
        split = line[48:].split()
        obj['nr'] = int(split[-2])
        obj['nz'] = int(split[-1])

        # --- scalar block (4 lines × 5 values) ---
        keys = [['rdim',  'zdim',   'rcentr', 'rleft',  'zmid'],
                ['raxis', 'zaxis',  'psimag', 'psibry', 'bcentr'],
                ['ip',    'skip',   'skip',   'skip',   'skip'],
                ['skip',  'skip',   'skip',   'skip',   'skip']]
        for row in keys:
            line = fid.readline()
            for j, key in enumerate(row):
                if key != 'skip':
                    obj[key] = float(line[j * 16:(j + 1) * 16])

        # --- 1D flux profiles ---
        for key in ['fpol', 'pres', 'ffprim', 'pprime']:
            obj[key] = _read_1d(fid, obj['nr'])

        # --- 2D psi map ---
        obj['psirz'] = _read_2d(fid, obj['nz'], obj['nr'])

        # --- q profile ---
        obj['qpsi'] = _read_1d(fid, obj['nr'])

        # --- boundary / limiter counts ---
        line = fid.readline()
        obj['nbbs'] = int(line.split()[0])
        obj['nlim'] = int(line.split()[1])

        # --- LCFS and wall coordinates ---
        obj['rzout'] = _read_2d(fid, obj['nbbs'], 2)
        obj['rzlim'] = _read_2d(fid, obj['nlim'], 2)

    return obj


# ---------------------------------------------------------------------------
# Flux-surface geometry using OMFIT's flux surface calculation
# ---------------------------------------------------------------------------

def add_flux_surface_geometry(eqdsk, filename=None):
    """
    Compute flux-surface geometry and add it to the eqdsk dictionary in-place.

    Uses OMFIT's contour tracing algorithm for accurate calculations.

    Adds the following keys:
        psi_n       - normalised psi grid  shape (nr,)
        psi         - psi values [Wb/rad]  shape (nr,)
        volume      - V(psi_n)  [m^3]      shape (nr,)
        vpr         - dV/dpsi  [m^3/(Wb/rad)]  shape (nr,)
        R_avg       - <R>(psi_n)  [m]       shape (nr,)
        R_inv_avg   - <1/R>(psi_n)  [1/m]   shape (nr,)

    Parameters
    ----------
    eqdsk : dict returned by read_eqdsk
    filename : str, optional (unused, kept for API compatibility)
    """
    nr     = eqdsk['nr']
    nz     = eqdsk['nz']
    psimag = eqdsk['psimag']
    psibry = eqdsk['psibry']
    raxis  = eqdsk['raxis']
    zaxis  = eqdsk['zaxis']
    
    # Grid setup
    R = np.linspace(eqdsk['rleft'], eqdsk['rleft'] + eqdsk['rdim'], nr)
    Z = np.linspace(eqdsk['zmid'] - eqdsk['zdim']/2, 
                    eqdsk['zmid'] + eqdsk['zdim']/2, nz)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    
    # Psi grid
    psi_n_vals = np.linspace(0.0, 1.0, nr)
    psi_vals   = psimag + psi_n_vals * (psibry - psimag)
    
    # Get psi on grid - shape (nz, nr) to match contourPaths expectation
    psi_grid = eqdsk['psirz']
    
    # Compute poloidal magnetic field components using OMFIT's formula
    # For COCOS=1: sigma_RpZ=1, sigma_Bp=1, exp_Bp=0
    # Br = sigma_RpZ * sigma_Bp * (∂ψ/∂Z) / (R * (2π)^exp_Bp)
    # Bz = -sigma_RpZ * sigma_Bp * (∂ψ/∂R) / (R * (2π)^exp_Bp)
    # Note: np.gradient returns [dZ, dR] when given (dZ, dR) spacing
    [dpsi_dZ, dpsi_dR] = np.gradient(psi_grid, dZ, dR)
   
    # Create 2D R grid for division
    R_2d, Z_2d = np.meshgrid(R, Z)
    
    # OMFIT's formula for COCOS=1
    Br = dpsi_dZ / R_2d  
    Bz = -dpsi_dR / R_2d
    
    # Create interpolators for Br and Bz separately (OMFIT's approach)
    # Then compute |Bp| at contour points, not on the grid
    psi_interp = RectBivariateSpline(Z, R, psi_grid)
    Br_interp = RectBivariateSpline(Z, R, Br)
    Bz_interp = RectBivariateSpline(Z, R, Bz)
    
    # Initialize output arrays
    vpr = np.zeros(nr)
    volume = np.zeros(nr)
    R_avg = np.zeros(nr)
    R_inv_avg = np.zeros(nr)
    
    # Trace contours
    paths_list = contourPaths(R, Z, psi_grid, psi_vals, 
                              remove_boundary_points=False, smooth_factor=1)
    
    # Process each flux surface
    for i, (psi, psi_n, level_paths) in enumerate(zip(psi_vals, psi_n_vals, paths_list)):
        if psi_n == 0.0:
            # Skip magnetic axis - too close to Bp=0 singularity
            vpr[i] = 0.0
            volume[i] = 0.0
            R_avg[i] = raxis
            R_inv_avg[i] = 1.0 / raxis
            continue
        
        if len(level_paths) == 0:
            # No contour found - use previous values or axis values
            if i > 0:
                vpr[i] = vpr[i-1]
                volume[i] = volume[i-1]
                R_avg[i] = R_avg[i-1]
                R_inv_avg[i] = R_inv_avg[i-1]
            else:
                vpr[i] = 0.0
                volume[i] = 0.0
                R_avg[i] = raxis
                R_inv_avg[i] = 1.0 / raxis
            continue
        
        # Select the contour that encloses the axis
        # Special case: use boundary points from file for LCFS (psi_n = 1.0)
        if psi_n == 1.0 and 'rzout' in eqdsk and len(eqdsk['rzout']) > 0:
            # Use boundary points from eqdsk file for LCFS like OMFIT does
            contour_R = eqdsk['rzout'][:, 0]
            contour_Z = eqdsk['rzout'][:, 1]
        else:
            # For other surfaces, trace contours
            best_path = None
            for path in level_paths:
                if path.contains_point((raxis, zaxis)):
                    best_path = path
                    break
            
            if best_path is None:
                # No contour encloses axis - use largest contour
                best_path = max(level_paths, key=lambda p: len(p.vertices))
            
            # Get contour points
            contour_R = best_path.vertices[:, 0]
            contour_Z = best_path.vertices[:, 1]
        
        if len(contour_R) < 3:
            # Too few points
            if i > 0:
                vpr[i] = vpr[i-1]
                volume[i] = volume[i-1]
                R_avg[i] = R_avg[i-1]
                R_inv_avg[i] = R_inv_avg[i-1]
            else:
                vpr[i] = 0.0
                volume[i] = 0.0
                R_avg[i] = raxis
                R_inv_avg[i] = 1.0 / raxis
            continue
        
        # Compute |Bp| at contour points
        # Interpolate Br and Bz separately, then compute magnitude (same as OMFIT)
        Br_contour = Br_interp.ev(contour_Z, contour_R)
        Bz_contour = Bz_interp.ev(contour_Z, contour_R)
        Bp_contour = np.sqrt(Br_contour**2 + Bz_contour**2)
        
        # Avoid div by zero near axis by setting a minimum Bp value
        Bp_contour = np.maximum(Bp_contour, 1e-10)
        
        # Compute flux expansion: dV/dψ = 2π ∮ dl/|Bp|
        # Since |Bp| = |∇ψ|/R, we have: R*dl/|∇ψ| = dl/|Bp|
        # Contour integral using trapezoidal rule
        # dl = √((dR)² + (dZ)²)
        dR_contour = np.diff(contour_R, append=contour_R[0])
        dZ_contour = np.diff(contour_Z, append=contour_Z[0])
        dl = np.sqrt(dR_contour**2 + dZ_contour**2)
        
        # Integrand: dl / |Bp| 
        integrand = dl / Bp_contour
        int_flux_expansion = np.sum(integrand)
        
        # Sign convention for vpr following OMFIT's approach
        # For COCOS=1: vp = sign(Bp) * 2π * ∑(dl / |Bp|)
        vpr[i] = 2.0 * np.pi * int_flux_expansion
        
        # Flux-surface averages
        # OMFIT uses fluxexpansion_dl = dl/|Bp| as the weight
        # <R> = ∫ R * (dl/|Bp|) / ∫ (dl/|Bp|)
        # <1/R> = ∫ (1/R) * (dl/|Bp|) / ∫ (dl/|Bp|)
        integrand_weight = dl / Bp_contour
        norm = np.sum(integrand_weight)
        if norm > 0:
            R_avg[i] = np.sum(contour_R * integrand_weight) / norm
            R_inv_avg[i] = np.sum(integrand_weight / contour_R) / norm
        else:
            R_avg[i] = np.mean(contour_R)
            R_inv_avg[i] = 1.0 / R_avg[i]
    
    # Integrate vpr to get volume
    # V(ψ) = ∫₀^ψ (dV/dψ') dψ'
    volume = integrate.cumulative_trapezoid(vpr, psi_vals, initial=0.0)
    
    # Store results
    eqdsk['psi_n'] = psi_n_vals
    eqdsk['psi'] = psi_vals
    eqdsk['vol'] = volume
    eqdsk['vol_lcfs'] = volume[-1]
    eqdsk['vpr'] = vpr
    eqdsk['R_avg'] = R_avg
    eqdsk['R_inv_avg'] = R_inv_avg
    
    return eqdsk


# ---------------------------------------------------------------------------
# Expanded read_eqdsk function
# ---------------------------------------------------------------------------

def read_eqdsk_extended(filename):
    """
    Read a gEQDSK file and compute derived flux-surface geometry.

    Returns a dictionary containing all fields from read_eqdsk plus:
        psi_n       - normalised psi grid       shape (nr,)
        psi         - psi values [Wb/rad]       shape (nr,)
        volume      - V(psi_n) [m^3]            shape (nr,)
        vpr         - dV/dpsi [m^3/(Wb/rad)]    shape (nr,)
        R_avg       - <R>(psi_n) [m]            shape (nr,)
        R_inv_avg   - <1/R>(psi_n) [1/m]        shape (nr,)

    All derived quantities are on the same psi_n grid as the other
    eqdsk profiles (fpol, pres, ffprim, pprime, qpsi).

    Flux surface geometry is computed using OMFIT's robust fluxSurfaces
    calculation when available. Requires omfit_classes to be installed.

    Parameters
    ----------
    filename : str     Path to gEQDSK file
    """
    eqdsk = read_eqdsk(filename)
    add_flux_surface_geometry(eqdsk, filename=filename)
    return eqdsk


# ---------------------------------------------------------------------------
# Tester function
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python read_eqdsk_extended.py <geqdsk_file>')
        sys.exit(1)

    fname = sys.argv[1]

    print(f'Reading {fname} ...')
    eq = read_eqdsk_extended(fname)

    print(f"  Grid:          {eq['nr']} x {eq['nz']}")
    print(f"  psi_mag:       {eq['psimag']:.4f}  psi_bry: {eq['psibry']:.4f}  [Wb/rad]")
    print(f"  Axis (R,Z):    ({eq['raxis']:.4f}, {eq['zaxis']:.4f})  [m]")
    print(f"  Ip:            {eq['ip'] / 1e6:.3f}  MA")
    print(f"  LCFS volume:   {eq['vol_lcfs']:.4f}  m^3")
    print(f"  <R> at LCFS:   {eq['R_avg'][-1]:.4f}  m")
    print(f"  <1/R> at LCFS: {eq['R_inv_avg'][-1]:.4f}  1/m")
    print(f"  Max dV/dpsi:   {np.max(np.abs(eq['vpr'])):.4e}  m^3/(Wb/rad)")
