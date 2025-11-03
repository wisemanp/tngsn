"""
CLI tool to build a simple galaxy metadata table (galmeta.csv) for a set of
downloaded subhalos using the TNG public API. It classifies galaxies into
star_formers, star_bursts, and passives based on specific SFR thresholds.

This script expects subhalo directories to already exist under:
  {root_path}/data/{simulation}/snap{snapshot}/*

For each subhaloId directory found, we query the TNG API for that subhalo's
mass_stars and sfr, compute ssfr = sfr / mass_stars, and write a CSV:
  {root_path}/data/{simulation}/snap{snapshot}/galmeta.csv

Usage example:
  python -m io.get_meta --simulation TNG50-1 --snapshot 96 --root-path .

Note: API key is read from the TNG_API_KEY environment variable or prompted.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Reuse TNG API helpers from get_cutouts
from .get_cutouts import (
    BASE_URL,
    get_tng_data,
)


def setup_logging(level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("get_meta")


logger = setup_logging()


def build_base_url(simulation: str, snapshot: int) -> str:
    """Return the TNG API base url for a given simulation and snapshot."""
    return f"{BASE_URL}{simulation}/snapshots/{snapshot}/"


def fetch_subhalo_meta(simulation: str, snapshot: int, subhalo_id: int) -> Dict:
    """Fetch subhalo JSON from TNG API."""
    base = build_base_url(simulation, snapshot)
    url = f"{base}subhalos/{subhalo_id}/"
    return get_tng_data(url)


def classify_by_ssfr(mass_stars: float, sfr: float,
                     burst_threshold: float = 10.0,
                     passive_threshold: float = 0.1) -> str:
    """
    Classify a galaxy based on specific SFR: ssfr = sfr / mass_stars.

    Returns one of: 'star_bursts', 'passives', 'star_formers'.
    """
    if mass_stars <= 0:
        return 'passives'  # degenerate; treat as passive
    ssfr = sfr / mass_stars
    if ssfr > burst_threshold:
        return 'star_bursts'
    if ssfr < passive_threshold:
        return 'passives'
    return 'star_formers'


def collect_subhalo_ids(root_path: Path, simulation: str, snapshot: int) -> List[int]:
    snap_dir = root_path / 'data' / simulation / f'snap{snapshot}'
    if not snap_dir.exists():
        logger.error(f"Snapshot directory not found: {snap_dir}")
        return []
    subdirs = [p for p in snap_dir.iterdir() if p.is_dir()]
    ids: List[int] = []
    for d in subdirs:
        try:
            ids.append(int(d.name))
        except ValueError:
            logger.debug(f"Skipping non-integer directory name: {d}")
    ids.sort()
    return ids


def build_galmeta(simulation: str, snapshot: int, subhalo_ids: List[int]) -> pd.DataFrame:
    star_formers: Dict[str, List[float]] = {}
    star_bursts: Dict[str, List[float]] = {}
    passives: Dict[str, List[float]] = {}

    for subhalo_id in tqdm(subhalo_ids, desc="Fetching subhalo meta"):
        try:
            info = fetch_subhalo_meta(simulation, snapshot, subhalo_id)
            mass_stars = info.get('mass_stars', 0.0)
            sfr = info.get('sfr', 0.0)
            cls = classify_by_ssfr(mass_stars, sfr)
            rec = [mass_stars, sfr, (sfr / mass_stars) if mass_stars else 0.0]
            key = str(subhalo_id)
            if cls == 'star_bursts':
                star_bursts[key] = rec
            elif cls == 'passives':
                passives[key] = rec
            else:
                star_formers[key] = rec
        except Exception as e:
            logger.warning(f"Failed to fetch meta for subhalo {subhalo_id}: {e}")

    def to_df(d: Dict[str, List[float]]) -> pd.DataFrame:
        if not d:
            return pd.DataFrame(columns=['mass_stars', 'sfr', 'ssfr'])
        df = pd.DataFrame(d).T
        df.rename(columns={0: 'mass_stars', 1: 'sfr', 2: 'ssfr'}, inplace=True)
        return df

    SFmeta = to_df(star_formers)
    SBmeta = to_df(star_bursts)
    Pmeta = to_df(passives)
    galmeta = pd.concat([SFmeta, SBmeta, Pmeta])
    galmeta.index.name = 'subhalo_id'
    # Add true stellar mass in solar masses assuming API mass is in 1e10 Msun/h
    try:
        galmeta['mass_stars_true'] = galmeta['mass_stars'].astype(float) * 1e10 / 0.6774
    except Exception:
        # Leave column absent if computation fails; logging handled by caller if needed
        pass
    return galmeta


def main():
    parser = argparse.ArgumentParser(description="Build galmeta.csv from TNG API for existing subhalo cutouts")
    parser.add_argument('--simulation', '-s', default='TNG50-1',
                        choices=['TNG50-1', 'TNG100-1', 'TNG300-1'],
                        help='TNG simulation to use')
    parser.add_argument('--snapshot', '-n', type=int, default=96,
                        help='Snapshot number (e.g., 96, 99)')
    parser.add_argument('--root-path', '-r', default='.',
                        help='Root path containing data/{simulation}/snap{snapshot}/...')
    parser.add_argument('--output', '-o', default=None,
                        help='Optional output CSV path (defaults to data/{sim}/snap{snap}/galmeta.csv)')
    parser.add_argument('--subhalo-ids', '-i', nargs='*', type=int, default=None,
                        help='Optional explicit subhalo IDs; if omitted, discovered from directories')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    root_path = Path(args.root_path).resolve()
    if args.subhalo_ids:
        subhalo_ids = sorted(set(args.subhalo_ids))
    else:
        subhalo_ids = collect_subhalo_ids(root_path, args.simulation, args.snapshot)

    if not subhalo_ids:
        logger.error("No subhalo IDs found. Ensure cutout directories exist or pass --subhalo-ids explicitly.")
        return 1

    logger.info(f"Building galmeta for {len(subhalo_ids)} subhalos: sim={args.simulation}, snap={args.snapshot}")
    galmeta = build_galmeta(args.simulation, args.snapshot, subhalo_ids)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = root_path / 'data' / args.simulation / f'snap{args.snapshot}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'galmeta.csv'

    galmeta.to_csv(out_path, index=True)
    logger.info(f"Wrote galmeta CSV: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())