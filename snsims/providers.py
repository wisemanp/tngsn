"""
Data provider abstraction for TNG-SN, enabling local-file and TNG API backends.

Initial scope:
- TNGAPIProvider can fetch subhalo info and download cutouts (and attempt images)
- Provides a galmeta builder when no local galmeta is present

Notes:
- Uses TNG_API_KEY from the environment for authenticated API access.
- Caches downloads under a configurable cache_dir to avoid reliance on data/ layout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import requests
import tempfile

DEFAULT_BASE_URL = 'http://www.tng-project.org/api/'

def _get_api_key() -> str:
    key = os.environ.get('TNG_API_KEY')
    if not key:
        raise RuntimeError("TNG_API_KEY environment variable is not set; required for API backend.")
    return key

def get_tng_data(path: str, params: Optional[dict] = None, savepath: Optional[str] = None):
    headers = {"api-key": _get_api_key()}
    r = requests.get(path, params=params, headers=headers)
    r.raise_for_status()
    ctype = r.headers.get('content-type', '')
    if 'application/json' in ctype:
        return r.json()
    if 'content-disposition' in r.headers:
        if savepath is None:
            savepath = ''
        filename = savepath + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename
    # Fallback: return raw response
    return r


class TNGAPIProvider:
    """
    Provider that pulls required artifacts directly from the TNG public API.

    It downloads artifacts to a cache directory but does not require the
    strict data/ layout used by the local backend.
    """

    def __init__(self, base_url: Optional[str] = None, cache_dir: Optional[str] = None):
        self.base_url = base_url or DEFAULT_BASE_URL
        self.cache_dir = Path(cache_dir or '.cache/tngsn').resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- URL helpers ----------
    def _snap_base(self, simulation: str, snapshot: int) -> str:
        return f"{self.base_url}{simulation}/snapshots/{snapshot}/"

    # ---------- Subhalo meta ----------
    def get_subhalo_info(self, simulation: str, snapshot: int, subhalo_id: int) -> Dict:
        base = self._snap_base(simulation, snapshot)
        info = get_tng_data(base + f'subhalos/{int(subhalo_id)}/')
        return info

    def build_galmeta_for_ids(self, simulation: str, snapshot: int, subhalo_ids: List[int]) -> pd.DataFrame:
        rows = {}
        for sid in subhalo_ids:
            try:
                info = self.get_subhalo_info(simulation, snapshot, sid)
                mass_stars = info.get('mass_stars', 0.0)
                sfr = info.get('sfr', 0.0)
                ssfr = (sfr / mass_stars) if mass_stars else 0.0
                rows[int(sid)] = [mass_stars, sfr, ssfr]
            except Exception:
                # Skip failures silently for now; caller can filter later
                continue
        df = pd.DataFrame(rows).T
        if not df.empty:
            df.rename(columns={0: 'mass_stars', 1: 'sfr', 2: 'ssfr'}, inplace=True)
            df.index.name = 'subhalo_id'
            try:
                df.index = df.index.astype(int)
            except Exception:
                pass
            # mass_stars_true in solar masses (API mass in 1e10 Msun / h)
            try:
                df['mass_stars_true'] = df['mass_stars'].astype(float) * 1e10 / 0.6774
            except Exception:
                pass
        return df

    # ---------- Cutouts ----------
    def get_cutout_path(self, simulation: str, snapshot: int, subhalo_id: int) -> Optional[Path]:
        """
        Download subhalo cutout to cache if needed and return path.
        """
        cache_dir = self.cache_dir / simulation / f'snap{snapshot}' / str(int(subhalo_id))
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Use a canonical filename
        target = cache_dir / f'cutout_{int(subhalo_id)}.hdf5'
        if target.exists() and target.stat().st_size > 0:
            return target

        info = self.get_subhalo_info(simulation, snapshot, subhalo_id)
        if 'cutouts' not in info or 'subhalo' not in info['cutouts']:
            return None
        cutout_url = info['cutouts']['subhalo']
        # Default fields used elsewhere in the pipeline
        star_fields = 'Masses,Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime,GFM_StellarPhotometrics'
        gas_fields = 'Coordinates,StarFormationRate,GFM_Metallicity'
        query = f"gas={gas_fields}&stars={star_fields}"
        # get_tng_data expects a directory ending with '/'
        dl_path = get_tng_data(f"{cutout_url}?{query}", savepath=str(cache_dir) + "/")
        # Move/rename to canonical if API names differently
        try:
            dl_file = Path(dl_path)
            if dl_file != target:
                if target.exists():
                    target.unlink()
                dl_file.rename(target)
        except Exception:
            # If rename fails, still return the downloaded file
            return Path(dl_path)
        return target

    # ---------- Images (best-effort) ----------
    def get_image_path(self, simulation: str, snapshot: int, subhalo_id: int) -> Optional[Path]:
        """
        Attempt to download a broadband image via supplementary data if available.
        Returns a cached local path or None if unavailable.
        """
        try:
            info = self.get_subhalo_info(simulation, snapshot, subhalo_id)
        except Exception:
            return None
        supp = info.get('supplementary_data', {})
        skirt = supp.get('skirt_images', {}) if isinstance(supp, dict) else {}
        # Try a couple of known keys used by TNG examples
        for key in ('fits_pogs', 'fits_pogs_noiseless'):
            url = skirt.get(key)
            if not url:
                continue
            cache_dir = self.cache_dir / simulation / f'snap{snapshot}' / str(int(subhalo_id))
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Name by key to avoid clobbering
            target = cache_dir / f"{key}_{int(subhalo_id)}.fits"
            if target.exists() and target.stat().st_size > 0:
                return target
            try:
                dl_path = get_tng_data(url, savepath=str(cache_dir) + "/")
                dl_file = Path(dl_path)
                if dl_file != target:
                    if target.exists():
                        target.unlink()
                    dl_file.rename(target)
                return target
            except Exception:
                continue
        return None


class TNGCloudProvider:
    """
    Provider for IllustrisTNG Lab environment with local, direct access to full snapshots.

    This class is a skeleton meant to be wired to Lab-provided helper functions that can:
      - extract per-subhalo particle cutouts from full snapshot files into a temporary HDF5
      - return subhalo centers/metadata efficiently without API

    Until those helpers are provided, methods will raise informative errors to guide integration.
    """

    def __init__(self, snapshot_root: Optional[str] = None, temp_base: Optional[str] = None):
        self.snapshot_root = Path(snapshot_root).resolve() if snapshot_root else None
        self.temp_base = Path(temp_base).resolve() if temp_base else Path(tempfile.gettempdir())
        self._temps: List[Path] = []

    def _mktemp_dir(self, prefix: str = 'tngsn_cloud_') -> Path:
        td = Path(tempfile.mkdtemp(prefix=prefix, dir=str(self.temp_base)))
        self._temps.append(td)
        return td

    def _basepath(self, simulation: str) -> Path:
        if self.snapshot_root is None:
            raise RuntimeError("TNGCloudProvider requires 'snapshot_root' to be set (e.g., via SNAPSHOT_ROOT).")
        return (self.snapshot_root / simulation).resolve()

    def cleanup(self):
        for p in list(self._temps):
            try:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
            finally:
                self._temps.remove(p)

    # ---- Stubs to be implemented with Lab helpers ----
    def get_subhalo_center(self, simulation: str, snapshot: int, subhalo_id: int) -> dict:
        """Return a dict with pos_x, pos_y, pos_z for the subhalo center using groupcat.loadSingle."""
        try:
            from illustris_python import groupcat
        except Exception as e:
            raise RuntimeError("illustris_python package is required for cloud backend.") from e
        basePath = str(self._basepath(simulation))
        data = groupcat.loadSingle(basePath, int(snapshot), subhaloID=int(subhalo_id))
        # Prefer SubhaloPos if present
        if isinstance(data, dict):
            if 'SubhaloPos' in data:
                pos = data['SubhaloPos']
                return {'pos_x': float(pos[0]), 'pos_y': float(pos[1]), 'pos_z': float(pos[2])}
            # Fallbacks
            for key in ('GroupPos', 'HaloPos'):
                if key in data:
                    pos = data[key]
                    return {'pos_x': float(pos[0]), 'pos_y': float(pos[1]), 'pos_z': float(pos[2])}
        raise RuntimeError(f"Unable to get center for subhalo {subhalo_id} at snapshot {snapshot}.")

    def extract_cutout(self, simulation: str, snapshot: int, subhalo_id: int,
                       star_fields: Optional[List[str]] = None, gas_fields: Optional[List[str]] = None) -> Path:
        """Create a temporary HDF5 cutout file containing required star particle fields using snapshot.loadSubhalo."""
        try:
            from illustris_python import snapshot
        except Exception as e:
            raise RuntimeError("illustris_python package is required for cloud backend.") from e

        basePath = str(self._basepath(simulation))
        sid = int(subhalo_id)
        snap = int(snapshot)

        # default fields needed by pipeline
        if star_fields is None:
            star_fields = ['Masses', 'Coordinates', 'GFM_InitialMass', 'GFM_Metallicity', 'GFM_StellarFormationTime', 'GFM_StellarPhotometrics']

        # load stars for this subhalo
        stars = snapshot.loadSubhalo(basePath, snap, sid, 'stars', fields=star_fields)
        if isinstance(stars, dict) and (stars.get('count', 0) == 0 or (len(stars) == 1 and 'count' in stars)):
            raise RuntimeError(f"No star particles returned for subhalo {sid} at snapshot {snap}.")

        # ensure all requested fields exist; if some are missing, fill placeholders
        count = stars['count'] if isinstance(stars, dict) and 'count' in stars else None
        for fld in star_fields:
            if fld not in stars:
                # allocate zeros placeholder with appropriate shape
                import numpy as np
                if fld == 'Coordinates':
                    stars[fld] = np.zeros((count, 3), dtype=np.float32)
                elif fld == 'GFM_StellarPhotometrics':
                    stars[fld] = np.zeros((count, 9), dtype=np.float32)
                else:
                    stars[fld] = np.zeros((count,), dtype=np.float32)

        # write minimal HDF5 to a temp file
        td = self._mktemp_dir()
        out = td / f"cutout_{sid}.hdf5"
        import h5py
        with h5py.File(out, 'w') as f:
            g = f.create_group('PartType4')
            for fld in star_fields:
                g.create_dataset(fld, data=stars[fld])
        return out

