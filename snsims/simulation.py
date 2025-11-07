"""
Main TNG-SN Simulation Class
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from astropy.cosmology import Planck15 as cosmo

from . import morphology, photometry
from . import sn_luminosity
from . import lc_params
from . import host_properties
from . import subhalo_selection

def _merge_overrides(cfg, overrides):
    for k, v in overrides.items():
        if "__" in k:
            sect, key = k.split("__", 1)
            cfg.setdefault(sect, {})
            cfg[sect][key] = v
        else:
            cfg[k] = v
    return cfg

class TNGSNSimulation:
    def __init__(self, config=None, **overrides):
        self.config = {
            "paths": {
                "root_path": str(Path(__file__).resolve().parents[2]),  # repo root fallback
            },
            "simulation": {
                "name": "TNG50-1",
                "snapshot": 99,
                "kids_snapshot": 96,
                "min_stellar_mass": 0.0,
            },
            "photometry": {
                "mode": "auto",
                "bands": ["g", "r", "i", "z"],
            },
            "morphology": {
                "band": "i",
                "rhalf_correction": 1/0.68,
                "dlr_default": 99.99,
            },
            "subhalo_selection": {
                "method": "observed",
                "mass_distribution": "uniform",
            },
            "light_curve": {
                "x1_params": {"mu_age": 0.0, "sigma_age": 1.0, "mu_metal": 0.0, "sigma_metal": 1.0},
                "age_step": {"Ka": 0.0, "Kb": 0.0, "threshold": 0.0},
            },
            "dtd": {"A": 1e-3, "beta": -1.1, "t0": 40.0},
            "cosmology": {"alpha": 0.14, "beta": 3.1},
        }
        if config:
            self.config.update(config)
        _merge_overrides(self.config, overrides)
        self._setup_paths()
        self.galmeta = None
        self.galmeta_filtered = None
        self.photometry_types = {}
        self.kids_results = None
        self.morphology_data = None

    def _setup_paths(self):
        self.root_path = Path(self.config["paths"]["root_path"]).resolve()
        self.data_root = self.root_path / "data"

    # Path templates
    def get_path(self, key: str, **fmt) -> str:
        sim = fmt.get("simulation") or self.config["simulation"]["name"]
        snap = self.config["simulation"].get("snapshot", 99)
        kids_snap = self.config["simulation"].get("kids_snapshot", 96)
        sub = fmt.get("subhalo_id")
        base = self.data_root / sim
        snapdir = base / f"snap{snap:02d}"
        if key == "galmeta_file":
            return str(base / "galmeta.parquet")
        if key == "morph_file":
            return str(snapdir / "morphs_i.hdf5")
        if key == "kids_results_dir":
            return str(base / f"KIDS/snapnum_{kids_snap:03d}/zx/data")
        if key == "cutout_pattern":
            return str(snapdir / f"{sub}/cutout_{sub}.hdf5")
        if key == "image_pattern":
            return str(snapdir / f"{sub}/broadband_{sub}.fits")
        if key == "ages_pattern":
            return str(snapdir / f"{sub}/{sub}_ages.dat")
        if key == "kids_image_pattern":
            return str(base / f"KIDS/snapnum_{kids_snap:03d}/zx/data/broadband_{sub}.fits")
        # default: return as-is to avoid crashes
        return ""

    def resolve_path(self, key: str, **fmt) -> str:
        planned = self.get_path(key, **fmt)
        if planned and os.path.exists(planned):
            return planned
        # simple legacy fallback: strip "/snapNN/"
        if "/snap" in planned:
            legacy = planned.replace(f"/snap{self.config['simulation'].get('snapshot',99):02d}/", "/")
            if os.path.exists(legacy):
                return legacy
        return planned

    # Metadata
    def load_metadata(self):
        sim = self.config["simulation"]["name"]
        # try common galmeta locations
        candidates = [
            self.resolve_path("galmeta_file", simulation=sim),
            str(self.data_root / sim / "galmeta.csv"),
            str(self.data_root / sim / "galmeta.h5"),
        ]
        path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f"galmeta not found under {self.data_root/sim}")
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            # best-effort: load first key
            try:
                df = pd.read_hdf(path)
            except Exception:
                df = pd.read_hdf(path, key="table")
        else:
            df = pd.read_parquet(path)
        # choose index
        for idx in ("subhalo_id", "SubhaloID", "subfind_id", "id"):
            if idx in df.columns:
                df[idx] = df[idx].astype(int, errors="ignore")
                df = df.set_index(idx)
                break
        # ensure mass column alias
        if "mass_stars_true" not in df.columns:
            for c in ("stellar_mass", "mass_stars", "Mstar_true"):
                if c in df.columns:
                    df["mass_stars_true"] = df[c]
                    break
        self.galmeta = df
        mcut = float(self.config["simulation"].get("min_stellar_mass", 0.0))
        self.galmeta_filtered = df[df.get("mass_stars_true", pd.Series(np.inf, index=df.index)) > mcut]

    # Strategy setup
    def setup_photometry_and_morphology(self, subhalo_ids):
        sim = self.config["simulation"]["name"]
        phot_cfg = self.config.get("photometry", {})
        morph_cfg = self.config.get("morphology", {})
        # photometry
        if phot_cfg.get("mode") == "auto":
            kids_dir = self.resolve_path("kids_results_dir", simulation=sim)
            self.photometry_types, self.kids_results = photometry.determine_photometry_strategy(
                subhalo_ids, self.galmeta, mass_threshold=phot_cfg.get("mass_threshold", 10**9.5),
                kids_results_dir=kids_dir
            )
        elif phot_cfg.get("mode") == "kids":
            self.photometry_types = {int(s): "kids" for s in subhalo_ids}
        else:
            self.photometry_types = {int(s): "regular" for s in subhalo_ids}
        # morphology (load once)
        morph_path = self.resolve_path("morph_file", simulation=sim)
        if os.path.isfile(morph_path) or os.path.isdir(morph_path):
            loaded = morphology.load_regular_morphology(morph_path, band=morph_cfg.get("band", "i"))
            self.morphology_data = loaded["table"]  # DataFrame
            logging.info(f"Loaded regular morphology data from {loaded.get('source_path', morph_path)}")
        else:
            logging.warning(f"Regular morphology path not found: {morph_path}")
            self.morphology_data = None

    # Pixel positions: simple passthrough (pixels == provided positions)
    def _calculate_pixel_positions(self, subhalo_id, sn_df: pd.DataFrame) -> pd.DataFrame:
        sn = sn_df.copy()
        if "x_pix" not in sn.columns:
            sn["x_pix"] = sn["x_pos"]
        if "y_pix" not in sn.columns:
            sn["y_pix"] = sn["y_pos"]
        return sn

    # Photometry
    def _calculate_photometry(self, subhalo_id, sn_df: pd.DataFrame) -> pd.DataFrame:
        sim = self.config["simulation"]["name"]
        bands = self.config["photometry"].get("bands", ["g","r","i","z"])
        ptype = self.photometry_types.get(int(subhalo_id), "regular")
        image_path = None
        if ptype == "regular":
            image_path = self.resolve_path("image_pattern", simulation=sim, subhalo_id=subhalo_id)
        try:
            band_mags, sn = photometry.get_photometry_for_subhalo(
                subhalo_id, sn_df, ptype, sim,
                bands=bands, image_path=image_path, kids_results=self.kids_results
            )
            sn = photometry.add_local_colors(sn, band_mags)
            return sn
        except Exception as e:
            print(f"Error in photometry for {subhalo_id}: {e}")
            sn = sn_df.copy()
            for b in bands:
                sn[f"local_{b}"] = np.nan
            return sn

    # Morphology / DLR
    def _calculate_morphology(self, subhalo_id, sn_df: pd.DataFrame) -> pd.DataFrame:
        morph_cfg = self.config.get("morphology", {})
        try:
            return morphology.calculate_morphology_for_subhalo(
                subhalo_id,
                sn_df,
                self.morphology_data,
                rhalf_correction=morph_cfg.get("rhalf_correction", 1/0.68),
                dlr_default=morph_cfg.get("dlr_default", 99.99),
            )
        except Exception as e:
            print(f"Error calculating morphology for {subhalo_id}: {e}")
            sn = sn_df.copy()
            sn["d_DLR"] = np.nan
            return sn