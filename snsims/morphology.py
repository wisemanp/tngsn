"""
Galaxy morphology and DLR calculation functions.

Loads morphology tables (HDF5) and computes DLR given per-subhalo params.
"""
import os
import math
import logging
import h5py
import numpy as np
import pandas as pd


def _normalize_morph_path(path_or_dir: str, band: str = "i") -> str:
    """
    Normalize to morphs_{band}.hdf5 if a directory is given; otherwise return as-is.
    """
    p = str(path_or_dir or "")
    fname = f"morphs_{band}.hdf5"
    # Trim accidental duplication .../morphs_i.hdf5/morphs_i.hdf5
    dup = f"/{fname}/{fname}"
    if p.endswith(dup):
        p = p[: -len(f"/{fname}")]
    if os.path.isdir(p):
        return os.path.join(p, fname)
    return p


def load_regular_morphology(path_or_dir: str, band: str = "i"):
    """
    Load morphology like the legacy script:
      with h5py.File('.../morphs_i.hdf5') as f:
          subfind_id = f['subfind_id'][:]
          elongation_asymmetry = f['elongation_asymmetry'][:]
          ...
    Returns {'source_path': <file>, 'table': DataFrame} where table has the legacy columns
    and is indexed by subhalo_id (alias of subfind_id).
    """
    morph_path = _normalize_morph_path(path_or_dir, band=band)
    if not os.path.isfile(morph_path):
        raise FileNotFoundError(f"Morphology file not found: {morph_path}")

    with h5py.File(morph_path, "r") as f:
        # Prefer legacy dataset names; tolerate simple aliases
        def get(name, *aliases):
            for key in (name, *aliases):
                if key in f:
                    return np.array(f[key])
            raise KeyError(f"Missing dataset '{name}' in {morph_path}")

        subfind_ids = get('subfind_id', 'SubhaloID', 'subhalo_id')
        elongation_asymmetry = get('elongation_asymmetry')
        ellipticity_asymmetry = get('ellipticity_asymmetry')
        orientation_asymmetry = get('orientation_asymmetry')
        rhalf_ellip = get('rhalf_ellip')
        sersic_ellip = get('sersic_ellip')
        sersic_n = get('sersic_n')
        sersic_rhalf = get('sersic_rhalf')
        sersic_theta = get('sersic_theta')

    df = pd.DataFrame(
        np.array([
            subfind_ids,
            elongation_asymmetry,
            ellipticity_asymmetry,
            orientation_asymmetry,
            rhalf_ellip,
            sersic_ellip,
            sersic_n,
            sersic_rhalf,
            sersic_theta,
        ]).T,
        columns=[
            'subfind_id',
            'elongation_asymmetry',
            'ellipticity_asymmetry',
            'orientation_asymmetry',
            'rhalf_ellip',
            'sersic_ellip',
            'sersic_n',
            'sersic_rhalf',
            'sersic_theta',
        ],
    )

    # Provide a consistent integer index by subhalo id, preserving your legacy column names
    if 'subfind_id' in df.columns:
        df['subhalo_id'] = df['subfind_id'].astype(int)
        df = df.set_index('subhalo_id')
    elif 'subhalo_id' in df.columns:
        df = df.set_index('subhalo_id')
    else:
        # Last resort: create an integer index
        df.index = np.arange(len(df), dtype=int)

    return {"source_path": morph_path, "table": df}


def _coerce_regular_df(regular_morphs):
    """Coerce to a DataFrame indexed by subhalo id."""
    if isinstance(regular_morphs, pd.DataFrame):
        return regular_morphs
    if isinstance(regular_morphs, dict) and isinstance(regular_morphs.get("table"), pd.DataFrame):
        return regular_morphs["table"]
    if isinstance(regular_morphs, str):
        return load_regular_morphology(regular_morphs)["table"]
    return None


def calculate_ellipse_params(rhalf_ellip, elongation_asymmetry, rhalf_correction):
    if not np.isfinite(rhalf_ellip):
        return np.nan, np.nan
    A = float(rhalf_ellip) * float(rhalf_correction)
    q = float(elongation_asymmetry) if np.isfinite(elongation_asymmetry) and elongation_asymmetry > 0 else 1.0
    B = A / q
    return A, B


def _rotate(x, y, theta):
    ct, st = math.cos(theta), math.sin(theta)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    return xr, yr


def safe_DLR_calculation(x, y, A, B, theta, default=99.99):
    if not np.isfinite(A) or not np.isfinite(B) or A <= 0 or B <= 0:
        return np.full_like(x, default, dtype=float)
    xr, yr = _rotate(x, y, float(theta or 0.0))
    dlr = np.sqrt((xr / A)**2 + (yr / B)**2)
    bad = ~np.isfinite(dlr)
    dlr[bad] = default
    return dlr


def get_morphology_for_subhalo(subhalo_id, morph_df: pd.DataFrame):
    """
    Extract rhalf_ellip, elongation_asymmetry, orientation_asymmetry for a subhalo_id
    from the legacy-structured DataFrame.
    """
    sid = int(subhalo_id)
    if morph_df is None or len(morph_df) == 0 or sid not in morph_df.index:
        return None
    row = morph_df.loc[sid]
    return {
        "rhalf_ellip": float(row.get("rhalf_ellip", np.nan)),
        "elongation_asymmetry": float(row.get("elongation_asymmetry", np.nan)),
        "orientation_asymmetry": float(row.get("orientation_asymmetry", 0.0)),
    }


def calculate_morphology_for_subhalo(subhalo_id, sn_data, morph_df: pd.DataFrame,
                                     rhalf_correction=1/0.68, dlr_default=99.99):
    """
    Compute DLR using legacy morphology parameters from morph_df for subhalo_id.
    """
    sn = sn_data.copy()
    params = get_morphology_for_subhalo(subhalo_id, morph_df)
    if params is None:
        sn["d_DLR"] = dlr_default
        return sn

    A, B = calculate_ellipse_params(params["rhalf_ellip"], params["elongation_asymmetry"], rhalf_correction)
    theta = params.get("orientation_asymmetry", 0.0)
    x = np.asarray(sn.get("x_pos", np.zeros(len(sn))), dtype=float)
    y = np.asarray(sn.get("y_pos", np.zeros(len(sn))), dtype=float)
    sn["d_DLR"] = safe_DLR_calculation(x, y, A, B, theta, default=dlr_default)
    return sn