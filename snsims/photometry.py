"""
Photometry utilities for TNG simulations.

Functions for aperture photometry on synthetic images, supporting both
regular TNG broadband images and KIDS survey data.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
try:
    from photutils.aperture import CircularAperture, aperture_photometry
    HAS_PHOTUTILS = True
except Exception:
    HAS_PHOTUTILS = False

DEFAULT_BANDS = ['g', 'r', 'i', 'z']
DEFAULT_ZP = {b: 25.0 for b in DEFAULT_BANDS}
DEFAULT_INDEX_MAP = {'g': 0, 'r': 1, 'i': 2, 'z': 3}

def do_aperture_photometry(image_path, *, bands=None, positions=None,
                           zero_points=None, band_index_map=None, aperture_radius_pix=None):
    """
    SKIRT-style multi-band aperture photometry.
    Handles data[band, y, x] or 2D single-plane images.
    Returns dict {band: pd.Series of magnitudes}.
    """
    if positions is None:
        raise ValueError("positions is required (Nx2 array of pixel coords)")
    bands = list(bands or DEFAULT_BANDS)
    zero_points = (zero_points or DEFAULT_ZP).copy()
    band_index_map = band_index_map or DEFAULT_INDEX_MAP

    with fits.open(image_path) as hdul:
        data = np.asarray(hdul[0].data)

    # Detect geometry and a plane getter
    if data.ndim == 3:
        n_bands, ny, nx = data.shape
        def get_plane(b):
            idx = band_index_map.get(b)
            if idx is None or idx >= n_bands:
                return None
            return data[idx]
    else:
        ny, nx = data.shape[-2], data.shape[-1]
        plane2d = data
        def get_plane(b):
            # Single-plane: prefer 'i'
            if b == ('i' if 'i' in bands else bands[0]):
                return plane2d
            return None

    r_pix = float(aperture_radius_pix or 3.0)
    out = {b: pd.Series(np.full(len(positions), np.nan)) for b in bands}

    if HAS_PHOTUTILS:
        for b in bands:
            plane = get_plane(b)
            if plane is None:
                continue
            apertures = CircularAperture(positions, r=r_pix)
            phot_table = aperture_photometry(plane.astype(np.float64), apertures)
            flux = np.array(phot_table['aperture_sum'])
            good = flux > 0
            mags = np.full_like(flux, np.nan, dtype=float)
            mags[good] = -2.5 * np.log10(flux[good]) + zero_points.get(b, 25.0)
            out[b] = pd.Series(mags)
    else:
        # Manual fallback
        for b in bands:
            plane = get_plane(b)
            if plane is None:
                continue
            mags = np.full(len(positions), np.nan)
            for i, (x, y) in enumerate(positions):
                xi, yi = int(round(x)), int(round(y))
                x0, x1 = max(0, xi - int(r_pix)), min(plane.shape[1], xi + int(r_pix) + 1)
                y0, y1 = max(0, yi - int(r_pix)), min(plane.shape[0], yi + int(r_pix) + 1)
                sub = plane[y0:y1, x0:x1]
                if sub.size == 0:
                    continue
                yyw, xxw = np.ogrid[y0:y1, x0:x1]
                mask = (xxw - xi)**2 + (yyw - yi)**2 <= r_pix**2
                flux = float(np.nansum(sub[mask]))
                if flux > 0:
                    mags[i] = -2.5 * np.log10(flux) + zero_points.get(b, 25.0)
            out[b] = pd.Series(mags)

    return out

def add_local_colors(sn_data, band_mags):
    sn = sn_data.copy()
    def s(b): return band_mags.get(b, pd.Series(np.full(len(sn), np.nan))).values
    if 'g' in band_mags and 'z' in band_mags:
        sn['localrestframe_gz'] = s('g') - s('z')
    if 'g' in band_mags and 'r' in band_mags:
        sn['localrestframe_gr'] = s('g') - s('r')
    if 'r' in band_mags and 'i' in band_mags:
        sn['localrestframe_ri'] = s('r') - s('i')
    if 'i' in band_mags and 'z' in band_mags:
        sn['localrestframe_iz'] = s('i') - s('z')
    return sn

def determine_photometry_strategy(subhalo_ids, galmeta, *, mass_threshold=10**9.5, kids_results_dir=None):
    """
    Decide 'kids' vs 'regular' per subhalo. KIDS for low-mass hosts, regular otherwise.
    """
    mass_col = None
    for c in ('mass_stars_true', 'stellar_mass', 'mass_stars'):
        if c in galmeta.columns:
            mass_col = c
            break
    if mass_col is None:
        masses = pd.Series(np.inf, index=subhalo_ids)
    else:
        masses = galmeta.reindex(subhalo_ids)[mass_col]

    types = {}
    for sid, m in zip(subhalo_ids, masses.fillna(np.inf)):
        types[int(sid)] = 'kids' if np.isfinite(m) and float(m) < float(mass_threshold) else 'regular'

    kids_results = load_kids_photometry_results(kids_results_dir) if kids_results_dir else None
    return types, kids_results

def load_kids_photometry_results(kids_results_dir):
    """
    Load precomputed KIDS photometry results if present.
    """
    if not kids_results_dir:
        return None
    for fname in ('kids_photometry.parquet', 'kids_photometry.csv'):
        path = os.path.join(kids_results_dir, fname)
        if os.path.isfile(path):
            try:
                return pd.read_parquet(path) if fname.endswith('.parquet') else pd.read_csv(path)
            except Exception:
                pass
    return None

def get_photometry_for_subhalo(subhalo_id, sn_data, photometry_type, simulation,
                               *, bands, image_path=None, kids_results=None,
                               kids_aperture_mode='direct', kids_image_type='noisy',
                               zero_points=None, **kwargs):
    """
    Attach local_* magnitudes and return (band_mags, sn_data).
    """
    if photometry_type == 'regular':
        if image_path is None or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Regular image not found: {image_path}")
        positions = sn_data[['x_pix', 'y_pix']].to_numpy()
        band_mags = do_aperture_photometry(
            image_path,
            bands=bands,
            positions=positions,
            zero_points=zero_points
        )
        for b in bands:
            sn_data[f'local_{b}'] = band_mags.get(b, pd.Series(np.full(len(sn_data), np.nan))).values
        return band_mags, sn_data

    if photometry_type == 'kids':
        # Placeholder: return NaNs unless kids_results provides values
        out = {b: pd.Series(np.full(len(sn_data), np.nan)) for b in bands}
        if isinstance(kids_results, pd.DataFrame) and 'subhalo_id' in kids_results.columns:
            row = kids_results[kids_results['subhalo_id'] == int(subhalo_id)]
            if not row.empty:
                for b in bands:
                    col = f'local_{b}'
                    if col in row.columns:
                        sn_data[col] = np.full(len(sn_data), float(row.iloc[0][col]))
                        out[b] = pd.Series(sn_data[col].values)
        for b in bands:
            if f'local_{b}' not in sn_data.columns:
                sn_data[f'local_{b}'] = np.nan
        return out, sn_data

    raise ValueError(f"Unknown photometry_type: {photometry_type}")