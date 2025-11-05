"""
Main TNG-SN Simulation Class

A class-based simulation tool for generating supernova populations
from TNG simulation data with configurable parameters.
"""

import os
import yaml
import h5py
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import warnings
from pandas.errors import SettingWithCopyWarning
from pathlib import Path
import logging
import shutil

# Import local modules
from . import sn_luminosity
from . import lc_params
from . import host_properties
from . import morphology
from . import photometry
from . import subhalo_selection

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class TNGSNSimulation:
    """
    Main class for TNG supernova simulations.
    
    This class handles the full pipeline from TNG data to simulated
    supernova observations including host selection, parameter generation,
    and photometric measurements.
    """
    
    def __init__(self, config_file=None, **config_overrides):
        """
        Initialize simulation with configuration.
        
        Parameters:
        -----------
        config_file : str, optional
            Path to YAML configuration file
        **config_overrides : dict
            Override configuration parameters
        """
        # Load configuration
        self.config = self._load_config(config_file, **config_overrides)
        
        # Set up root path and resolve all paths
        self.setup_paths()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data containers
        self.galmeta = None
        self.galaxy_morphs = None
        self.simulation_results = []
        
        # Photometry and morphology strategy
        self.photometry_types = None
        self.morphology_data = None
        self.kids_results = None
        
        # Track ephemeral artifacts for cleanup
        self._temp_files = []
        self._temp_dirs = []
        
    def setup_paths(self):
        """Setup root path and resolve all relative paths."""
        # Get root path from config or use current working directory
        root_path_config = self.config.get('paths', {}).get('root_path', os.getcwd())
        self.root_path = Path(root_path_config)
        
        # Make sure root path is absolute
        if not self.root_path.is_absolute():
            self.root_path = Path(os.getcwd()) / self.root_path
            
        self.root_path = self.root_path.resolve()
        
        # Log the root path
        logging.info(f"Using root path: {self.root_path}")
        
        # Resolve all paths relative to root
        self.paths = {}
        path_config = self.config.get('paths', {})
        
        for key, path_template in path_config.items():
            if key == 'root_path':
                continue
                
            # Convert template to absolute path
            if isinstance(path_template, str):
                if os.path.isabs(path_template):
                    # If already absolute, use as-is
                    resolved_path = path_template
                else:
                    # If relative, combine with root path
                    resolved_path = str(self.root_path / path_template)
                self.paths[key] = resolved_path
            else:
                self.paths[key] = path_template
                
        logging.info(f"Resolved {len(self.paths)} path templates")
        
        # Debug: Print some resolved paths
        for key in ['galmeta_file', 'morph_file']:
            if key in self.paths:
                logging.info(f"  {key}: {self.paths[key]}")
    
    def get_path(self, path_key: str, **format_kwargs) -> str:
        """
        Get resolved absolute path with formatting.
        
        Parameters:
        -----------
        path_key : str
            Key in paths configuration
        **format_kwargs : dict
            Format arguments for path template
            
        Returns:
        --------
        str : Absolute path
        """
        if path_key not in self.paths:
            raise ValueError(f"Path key '{path_key}' not found in configuration")
            
        path_template = self.paths[path_key]
        
        # Add defaults to format kwargs
        sim_cfg = self.config['simulation']
        if 'simulation' not in format_kwargs:
            format_kwargs['simulation'] = sim_cfg['name']
        if 'snapshot' not in format_kwargs:
            format_kwargs['snapshot'] = sim_cfg.get('snapshot')
        # kids_snapshot may be different; fall back to snapshot if missing
        if 'kids_snapshot' not in format_kwargs:
            format_kwargs['kids_snapshot'] = sim_cfg.get('kids_snapshot', sim_cfg.get('snapshot'))
            
        try:
            formatted_path = path_template.format(**format_kwargs)
            return formatted_path
        except KeyError as e:
            raise ValueError(f"Missing format parameter {e} for path '{path_key}'")

    def resolve_path(self, path_key: str, **format_kwargs) -> str:
        """Resolve a path and apply fallbacks for legacy layouts.

        Strategy:
        - Try the configured pattern (with snapshot/kids_snapshot)
        - If missing and not a KIDS path, try without the '/snap{snapshot}/' segment
        - If a KIDS path and missing, try legacy locations (e.g., without snapnum segment)
        """
        primary = self.get_path(path_key, **format_kwargs)
        if os.path.exists(primary):
            return primary

        # For catalog-like singletons, prefer snapshot-specific location if the directory exists
        if path_key in ('galmeta_file', 'morph_file'):
            primary_parent = os.path.dirname(primary)
            if os.path.isdir(primary_parent):
                # Directory exists; trust the primary path (let downstream code error clearly if file missing)
                return primary

        # Attempt fallback without '/snap{snapshot}/'
        try:
            sim_cfg = self.config['simulation']
            snapshot = sim_cfg.get('snapshot')
            if snapshot is not None:
                without_snap = primary.replace(f"/snap{snapshot}/", "/")
                if os.path.exists(without_snap):
                    return without_snap
        except Exception:
            pass

        # KIDS-specific fallbacks
        if path_key.startswith('kids_'):
            # Try legacy results dir without snapnum
            if 'results' in path_key:
                sim = format_kwargs.get('simulation', self.config['simulation']['name'])
                legacy = os.path.join(self.root_path, f"data/{sim}/KIDS/results")
                if os.path.exists(legacy):
                    return legacy
            # Try default snapnum_096 if different kids_snapshot was set
            try:
                sim = format_kwargs.get('simulation', self.config['simulation']['name'])
                legacy_img = os.path.join(self.root_path, f"data/{sim}/KIDS/snapnum_096/zx/data")
                if os.path.isdir(legacy_img):
                    # if it's an image pattern, reconstruct full path including filename if provided
                    if 'subhalo_id' in format_kwargs:
                        subhalo_id = format_kwargs['subhalo_id']
                        candidate = os.path.join(legacy_img, f"broadband_{subhalo_id}.fits")
                        if os.path.exists(candidate):
                            return candidate
                    return legacy_img
            except Exception:
                pass

        # If this is a regular image or cutout pattern, try rebuilding without the snap segment
        # Note: galmeta_file and morph_file are handled above to avoid incorrect fallback when their
        # snapshot-specific directories exist.
        if path_key in ('image_pattern', 'cutout_pattern', 'ages_pattern'):
            try:
                # Remove '/snap{snapshot}' segment if present
                parts = primary.split('/snap')
                if len(parts) > 1:
                    # e.g., data/TNG50-1/snap99/... -> data/TNG50-1/...
                    no_snap = parts[0] + '/' + parts[1].split('/', 1)[1]
                    no_snap = no_snap if no_snap.startswith('/') else os.path.join(self.root_path, no_snap)
                    # Only fallback if the no-snap sibling exists
                    if os.path.exists(no_snap):
                        return no_snap
            except Exception:
                pass

        return primary
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self, config_file=None, **overrides):
        """Load configuration from file and apply overrides."""
        
        # Start with default config
        config = self.get_default_config()
        
        # Load from file if provided
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                # Look for config file relative to current directory first,
                # then relative to this script's directory
                if not config_path.exists():
                    script_dir = Path(__file__).parent.parent
                    config_path = script_dir / config_file
                    
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                config = self.merge_configs(config, file_config)
                logging.info(f"Loaded configuration from {config_path}")
            else:
                logging.warning(f"Config file not found: {config_file}")
        
        # Apply overrides
        config = self.apply_config_overrides(config, overrides)
        
        return config
    
    def get_default_config(self):
        """Get default configuration."""
        return {
            'paths': {
                'root_path': os.getcwd(),
                # Snapshot-aware canonical layout
                'galmeta_file': "data/{simulation}/snap{snapshot}/galmeta.csv",
                'morph_file': "data/{simulation}/snap{snapshot}/morphs_i.hdf5", 
                'cutout_pattern': "data/{simulation}/snap{snapshot}/{subhalo_id}/cutout_{subhalo_id}.hdf5",
                'ages_pattern': "data/{simulation}/snap{snapshot}/{subhalo_id}/{subhalo_id}_ages.dat",
                'image_pattern': "data/{simulation}/snap{snapshot}/{subhalo_id}/broadband_{subhalo_id}.fits",
                # KIDS: results live outside snapnum; images use per-snapnum folders
                'kids_results_dir': "data/{simulation}/KIDS/results",
                'kids_image_pattern': "data/{simulation}/KIDS/snapnum_{kids_snapshot:03d}/zx/data/broadband_{subhalo_id}.fits"
            },
            'backend': {
                'mode': 'local',  # 'local' | 'api' | 'cloud'
                'api_cache_dir': '.cache/tngsn',
                'api_base_url': None,
                'api_seed_ids': [],
                # Cloud/temp behavior
                'cleanup_temp': False,    # if True, remove temp artifacts post run
                'persist_ages': True      # if False, do not write ages files to disk
            },
            'simulation': {
                'name': 'TNG50-1',
                'snapshot': 99,
                'kids_snapshot': 96,
                'n_cores': 4,
                'output_dir': 'simout',
                'min_stellar_mass': 1e8,
                'max_stellar_mass': 1e13
            },
            'subhalo_selection': {
                'method': 'uniform',
                'mass_distribution': 'uniform',
                'mass_bins': None,
                'selection_seed': None
            },
            'photometry': {
                'mode': 'auto',
                'mass_threshold': 3.16e9,
                'force_kids': False,
                'force_regular': False,
                'bands': ['g', 'r', 'i', 'z'],
                'zero_points': {'g': 25.0, 'r': 25.0, 'i': 25.0, 'z': 25.0},
                'error_floor': 0.1,
                'kids_aperture_mode': 'direct',
                'kids_image_type': 'noisy'
            },
            'morphology': {
                'mode': 'auto',
                'band': 'i',
                'rhalf_correction': 1.470588,
                'dlr_default': 99.99
            },
            'dtd': {
                'function': 'powerlaw',
                'A': 2.11e-13,
                'beta': -1.13,
                't0': 40
            },
            'light_curve': {
                'x1_params': [0.25, 0.55, -1.33, 0.63, -0.3, 0.18, 0.98, -0.128, 5],
                'age_step': {
                    'agesplit': 3.0,
                    'Ka': -0.128,
                    'step_high': 0.1,
                    'step_low': -0.1,
                    'mabs_base': -19.3
                },
                'color': {
                    'mean': 0.0,
                    'sigma': 0.1
                }
            },
            'cosmology': {
                'alpha': -0.14,
                'beta': 3.15,
                'mabs': -19.3
            }
        }
    
    def merge_configs(self, base_config, new_config):
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                base_config[key] = self.merge_configs(base_config[key], value)
            else:
                base_config[key] = value
        return base_config
    
    def apply_config_overrides(self, config, overrides):
        """Apply configuration overrides using double underscore notation."""
        for key, value in overrides.items():
            if '__' in key:
                # Handle nested configuration like 'simulation__n_cores'
                parts = key.split('__')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value
        return config
    
    def _set_nested_config(self, key, value):
        """Set nested configuration parameter using dot notation."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config[k]
        config[keys[-1]] = value
    
    def load_metadata(self):
        """Load galaxy metadata and determine processing strategy."""
        sim = self.config['simulation']['name']
        snapshot = int(self.config['simulation'].get('snapshot', 99))
        backend_mode = self.config.get('backend', {}).get('mode', 'local')
        
        # Load galaxy metadata using resolved path
        galmeta_path = self.resolve_path('galmeta_file', simulation=sim)
        
        if not os.path.exists(galmeta_path):
            if backend_mode == 'api':
                # Build minimal galmeta via API for IDs inferred from available cutouts or a small query.
                from .providers import TNGAPIProvider
                provider = TNGAPIProvider(
                    base_url=self.config.get('backend', {}).get('api_base_url'),
                    cache_dir=self.config.get('backend', {}).get('api_cache_dir')
                )
                # Heuristic: if cutouts exist locally, use their IDs; else, pick top few from API
                subdirs = []
                snap_dir = Path(self.root_path) / f"data/{sim}/snap{snapshot}"
                if snap_dir.exists():
                    subdirs = [p for p in snap_dir.iterdir() if p.is_dir()]
                ids = [int(p.name) for p in subdirs if p.name.isdigit()]
                if not ids:
                    ids = list(self.config.get('backend', {}).get('api_seed_ids') or [])
                if not ids:
                    # Query API for a small seed sample (e.g., first 100 massive)
                    try:
                        base = provider._snap_base(sim, snapshot)
                        res = provider.get_subhalo_info(sim, snapshot, 0)  # will fail; using next block
                    except Exception:
                        pass
                    # Fall back to a predictable range; caller can refine later
                    ids = []
                galmeta_df = provider.build_galmeta_for_ids(sim, snapshot, ids)
                if galmeta_df is None or galmeta_df.empty:
                    raise FileNotFoundError(f"Galaxy metadata file not found and API fallback yielded no entries: {galmeta_path}")
                self.galmeta = galmeta_df
            else:
                raise FileNotFoundError(f"Galaxy metadata file not found: {galmeta_path}")
        else:
            logging.info(f"Loading galaxy metadata from {galmeta_path}")
            self.galmeta = pd.read_csv(galmeta_path, index_col=0)
        
        # Calculate true stellar masses
        self.galmeta['mass_stars_true'] = (
            self.galmeta['mass_stars'] * 1e10 / cosmo.h
        )
        
        # Apply mass cuts
        min_mass = self.config['simulation']['min_stellar_mass']
        max_mass = self.config['simulation'].get('max_stellar_mass')
        
        mass_mask = self.galmeta['mass_stars_true'] >= min_mass
        if max_mass is not None:
            mass_mask &= self.galmeta['mass_stars_true'] <= max_mass
            
        self.galmeta_filtered = self.galmeta[mass_mask]
        
        print(f"Loaded {len(self.galmeta)} galaxies, {len(self.galmeta_filtered)} after mass cuts")
    
    def select_subhalos(self, n_subhalos=None, subhalo_ids=None):
        """
        Select subhalos for processing based on configuration.
        
        Parameters:
        -----------
        n_subhalos : int, optional
            Number of subhalos to select
        subhalo_ids : list, optional
            Specific subhalo IDs to use
            
        Returns:
        --------
        array : Selected subhalo IDs
        """
        if subhalo_ids is not None:
            return np.array(subhalo_ids)
        
        # Use subhalo selection function
        selection_config = self.config.get('subhalo_selection', {})
        
        selected_galmeta, selected_ids = subhalo_selection.select_subhalos_by_mass(
            self.galmeta_filtered,
            n_subhalos=n_subhalos,
            mass_distribution=selection_config.get('method', 'uniform'),
            min_mass=self.config['simulation']['min_stellar_mass'],
            max_mass=self.config['simulation'].get('max_stellar_mass'),
            mass_bins=selection_config.get('mass_bins'),
            seed=selection_config.get('selection_seed', 42)
        )
        
        return selected_ids
    
    def setup_photometry_and_morphology(self, subhalo_ids):
        """
        Set up photometry and morphology processing strategy.
        
        Parameters:
        -----------
        subhalo_ids : array
            Subhalo IDs to process
        """
        sim = self.config['simulation']['name']
        phot_config = self.config.get('photometry', {})
        morph_config = self.config.get('morphology', {})
        
        # Determine photometry strategy
        if phot_config.get('mode') == 'auto':
            self.photometry_types, self.kids_results = photometry.determine_photometry_strategy(
                subhalo_ids, self.galmeta,
                mass_threshold=phot_config.get('mass_threshold', 10**9.5),
                kids_results_dir=self.resolve_path('kids_results_dir')
            )
        elif phot_config.get('mode') == 'kids':
            self.photometry_types = {sid: 'kids' for sid in subhalo_ids}
            kids_results_dir = self.resolve_path('kids_results_dir')
            self.kids_results = photometry.load_kids_photometry_results(self.config['simulation']['name'], results_dir=kids_results_dir)
        else:
            self.photometry_types = {sid: 'regular' for sid in subhalo_ids}
            self.kids_results = None
        
        # Load morphology data
        morph_file = self.resolve_path('morph_file')
        if os.path.exists(morph_file):
            self.morphology_data = morphology.load_regular_morphology(morph_file, band=morph_config.get('band', 'i'))
            logging.info(f"Loaded regular morphology data from {morph_file}")
        else:
            logging.warning(f"Regular morphology file not found: {morph_file}")
            self.morphology_data = None
    
    def process_subhalo(self, subhalo_id):
        """
        Process a single subhalo to generate SN population.
        
        Parameters:
        -----------
        subhalo_id : int
            Subhalo ID to process
            
        Returns:
        --------
        pandas.DataFrame or None : Simulation results for this subhalo
        """
        subhalo_id = int(subhalo_id)
        sim = self.config['simulation']['name']
        
        # Check for required files using resolved paths
        cutout_path = self.resolve_path('cutout_pattern', simulation=sim, subhalo_id=subhalo_id)
        if not os.path.isfile(cutout_path):
            backend_mode = self.config.get('backend', {}).get('mode', 'local')
            # Try API backend
            if backend_mode == 'api':
                try:
                    from .providers import TNGAPIProvider
                    provider = TNGAPIProvider(
                        base_url=self.config.get('backend', {}).get('api_base_url'),
                        cache_dir=self.config.get('backend', {}).get('api_cache_dir')
                    )
                    api_cutout = provider.get_cutout_path(sim, int(self.config['simulation']['snapshot']), subhalo_id)
                    if api_cutout and api_cutout.exists():
                        cutout_path = str(api_cutout)
                    else:
                        print(f'{subhalo_id} has no snapshot (local) and API fetch failed, skipping')
                        return None
                except Exception as e:
                    print(f"API backend error for subhalo {subhalo_id}: {e}")
                    return None
            elif backend_mode == 'cloud':
                # Use cloud provider to extract a temp cutout directly from local snapshot files
                try:
                    from .providers import TNGCloudProvider
                    provider = TNGCloudProvider(
                        snapshot_root=os.environ.get('SNAPSHOT_ROOT'),
                        temp_base=self.config.get('backend', {}).get('api_cache_dir') or None
                    )
                    tmp = provider.extract_cutout(sim, int(self.config['simulation']['snapshot']), subhalo_id)
                    if tmp and os.path.isfile(tmp):
                        cutout_path = str(tmp)
                        # track temp for cleanup
                        self._temp_files.append(cutout_path)
                        try:
                            self._temp_dirs.append(str(Path(cutout_path).parent))
                        except Exception:
                            pass
                    else:
                        print(f'{subhalo_id}: cloud extract returned no cutout, skipping')
                        return None
                except Exception as e:
                    print(f"Cloud backend error for subhalo {subhalo_id}: {e}")
                    return None
            else:
                print(f'{subhalo_id} has no snapshot, skipping')
                return None
        
        try:
            # Load stellar particle data
            galaxy_catalog = self._load_stellar_data(subhalo_id, cutout_path)
            
            if galaxy_catalog is None or len(galaxy_catalog) == 0:
                return None
            
            # Generate SN population
            sn_data = self._generate_sn_population(subhalo_id, galaxy_catalog)
            
            # Add global galaxy properties
            sn_data = self._add_global_properties(subhalo_id, sn_data)
            
            # Calculate pixel positions for photometry/morphology
            sn_data = self._calculate_pixel_positions(subhalo_id, sn_data)
            
            # Calculate photometric properties
            sn_data = self._calculate_photometry(subhalo_id, sn_data)
            
            # Calculate morphological properties
            sn_data = self._calculate_morphology(subhalo_id, sn_data)
            
            # Save results using resolved output path
            output_dir = self.config['simulation']['output_dir']
            if not os.path.isabs(output_dir):
                output_dir = self.root_path / output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = output_dir / f'{subhalo_id}_skysurvey_out.h5'
            sn_data.to_hdf(output_path, key='main', mode='w')
            
            return sn_data
            
        except Exception as e:
            print(f"Error processing subhalo {subhalo_id}: {e}")
            return None
    
    def _load_stellar_data(self, subhalo_id, cutout_path):
        """Load and process stellar particle data."""
        sim = self.config['simulation']['name']
        
        with h5py.File(cutout_path, 'r') as f:
            stars = f['PartType4']
            star_inds = np.array(stars['GFM_StellarFormationTime']) > 0
            
            if np.sum(star_inds) == 0:
                return None
            
            # Extract stellar properties
            metallicity = np.log10(
                stars['GFM_Metallicity'][star_inds] / 0.0127
            )
            initial_mass = stars['GFM_InitialMass'][star_inds] * 1e10 / cosmo.h
            stellar_mass = stars['Masses'][star_inds] * 1e10 / cosmo.h
            
            # Photometric properties
            V = stars['GFM_StellarPhotometrics'][star_inds, 2]
            Ur = (stars['GFM_StellarPhotometrics'][star_inds, 0] - 
                  (stars['GFM_StellarPhotometrics'][star_inds, 5] - 0.16))
            gz = (stars['GFM_StellarPhotometrics'][star_inds, 4] - 
                  stars['GFM_StellarPhotometrics'][star_inds, 7])
            
            # Positions (need subhalo center)
            subinfo = self._get_subhalo_info(subhalo_id)
            x = stars['Coordinates'][star_inds, 0] - subinfo['pos_x']
            y = stars['Coordinates'][star_inds, 1] - subinfo['pos_y']
            
            # Calculate ages
            stfs = np.array(stars['GFM_StellarFormationTime'][star_inds])
        
        # Load or calculate ages using resolved path
        ages_path = self.resolve_path('ages_pattern', simulation=sim, subhalo_id=subhalo_id)
        
        if os.path.isfile(ages_path):
            ages = np.loadtxt(ages_path, dtype=float)
        else:
            print(f'Calculating ages for {subhalo_id}')
            zs = (1 / stfs) - 1
            t = cosmo.lookback_time(zs)
            ages = np.clip(t.value, a_min=0.05, a_max=None)
            
            # Save ages for future use only if persistence enabled
            persist_ages = bool(self.config.get('backend', {}).get('persist_ages', True))
            if persist_ages:
                os.makedirs(os.path.dirname(ages_path), exist_ok=True)
                np.savetxt(ages_path, ages)
            else:
                # treat as ephemeral; nothing written to disk
                pass
        
        # Create galaxy catalog
        galaxy_catalog = host_properties.create_galaxy_catalog(
            ages, metallicity, initial_mass, stellar_mass,
            V, Ur, gz, x, y
        )
        
        return galaxy_catalog
    
    def _generate_sn_population(self, subhalo_id, galaxy_catalog):
        """Generate supernova population for this subhalo."""
        import skysurvey
        
        ages = galaxy_catalog['age'].values
        masses = galaxy_catalog['initial_mass'].values
        
        # Calculate sample size based on stellar mass
        minlen = len(galaxy_catalog)  # This should be set more intelligently
        n_samp = max(1, int(len(ages) / minlen))
        
        # Compute DTD weights
        dtd_config = self.config['dtd']
        dtd_weights = host_properties.compute_dtd_weights(
            ages, masses, 
            A=dtd_config['A'],
            beta=dtd_config['beta'],
            t0=dtd_config['t0']
        )
        
        # Define sampling model
        full_model = {
            "hostId": {
                "func": np.random.choice,
                "kwargs": {
                    "a": np.arange(len(ages)),
                    "p": dtd_weights
                },
                "as": "hostId"
            },
            'hostmet': {
                "func": host_properties.get_metallicity,
                "kwargs": {
                    "hostId": "@hostId",
                    "galaxy_catalog": galaxy_catalog
                },
                'as': "hostmet"
            },
            'progage': {
                "func": host_properties.get_progenitor_age,
                "kwargs": {
                    "hostId": "@hostId",
                    "galaxy_catalog": galaxy_catalog
                },
                'as': "progage"
            },
            "x1": {
                "func": lc_params.x1_from_host_properties,
                "kwargs": {
                    "age": "@progage",
                    "metallicity": "@hostmet",
                    "params": self.config['light_curve']['x1_params']
                }
            },
            "mabs": {
                "func": sn_luminosity.age_step,
                "kwargs": {
                    "age": "@progage",
                    **self.config['light_curve']['age_step']
                }
            }
        }
        
        # Generate SN sample
        snia = skysurvey.SNeIa.from_draw(
            size=n_samp, 
            model=full_model, 
            magabs={"mabs": "@mabs"}
        )
        
        # Join with host properties
        drawn_gals = galaxy_catalog.iloc[snia.data['hostId']]
        sim_data = snia.data.join(drawn_gals.reset_index(drop=True))
        
        return sim_data
    
    def _add_global_properties(self, subhalo_id, sn_data):
        """Add global galaxy properties to SN data."""
        # Add Hubble residuals
        cosmo_config = self.config['cosmology']
        sn_data['MURES'] = sn_luminosity.hubble_residual(
            sn_luminosity.distmod(
                sn_data['magobs'], 
                sn_data['x1'], 
                sn_data['c'],
                mabs=cosmo_config['mabs'],
                alpha=cosmo_config['alpha'],
                beta=cosmo_config['beta']
            ),
            sn_data['z']
        )
        
        # Add global galaxy properties from metadata
        if self.galmeta is not None and subhalo_id in self.galmeta.index:
            galaxy_info = self.galmeta.loc[subhalo_id]
            sn_data['globalmass'] = galaxy_info['mass_stars']
            sn_data['globalsfr'] = galaxy_info.get('sfr', np.nan)
            sn_data['globalssfr'] = galaxy_info.get('ssfr', np.nan)
        
        # Add magnitude error (placeholder)
        sn_data['MUERR'] = self.config['photometry']['error_floor']
        
        return sn_data
    
    def _calculate_pixel_positions(self, subhalo_id, sn_data):
        """Calculate pixel positions for photometry and morphology."""
        sim = self.config['simulation']['name']
        
        try:
            # Get image header for pixel scale conversion using resolved path
            image_path = self.resolve_path('image_pattern', simulation=sim, subhalo_id=subhalo_id)
            if not os.path.isfile(image_path) and self.config.get('backend', {}).get('mode') == 'api':
                try:
                    from .providers import TNGAPIProvider
                    provider = TNGAPIProvider(
                        base_url=self.config.get('backend', {}).get('api_base_url'),
                        cache_dir=self.config.get('backend', {}).get('api_cache_dir')
                    )
                    api_img = provider.get_image_path(sim, int(self.config['simulation']['snapshot']), subhalo_id)
                    if api_img and api_img.exists():
                        image_path = str(api_img)
                except Exception:
                    pass
            
            if os.path.isfile(image_path):
                with fits.open(image_path) as hdul:
                    h = hdul[0].header
                
                # Convert positions to pixels
                sn_data['x_pix'] = (
                    (sn_data['x_pos'] * 1000 / h['CDELT1'] / cosmo.h) + 
                    (h['NAXIS1'] / 2)
                )
                sn_data['y_pix'] = (
                    (sn_data['y_pos'] * 1000 / h['CDELT2'] / cosmo.h) + 
                    (h['NAXIS2'] / 2)
                )
                sn_data['r_pix'] = np.sqrt(
                    (sn_data['x_pos'] * 1000 / h['CDELT1'] / cosmo.h) ** 2 + 
                    (sn_data['y_pos'] * 1000 / h['CDELT2'] / cosmo.h) ** 2
                )
            else:
                # Default pixel scale if image not available
                sn_data['x_pix'] = sn_data['x_pos'] * 100  # Rough conversion
                sn_data['y_pix'] = sn_data['y_pos'] * 100
                sn_data['r_pix'] = np.sqrt(sn_data['x_pos']**2 + sn_data['y_pos']**2) * 100
                
        except Exception as e:
            print(f"Error calculating pixel positions for {subhalo_id}: {e}")
            sn_data['x_pix'] = 0
            sn_data['y_pix'] = 0
            sn_data['r_pix'] = 0
        
        return sn_data
    
    def _calculate_photometry(self, subhalo_id, sn_data):
        """Calculate photometric properties using appropriate method."""
        sim = self.config['simulation']['name']
        bands = self.config['photometry']['bands']
        phot_config = self.config['photometry']
        
        photometry_type = self.photometry_types.get(subhalo_id, 'regular')
        
        try:
            # Resolve image path with API fallback if needed
            image_path = self.resolve_path('image_pattern', simulation=sim, subhalo_id=subhalo_id)
            if not os.path.isfile(image_path) and self.config.get('backend', {}).get('mode') == 'api':
                try:
                    from .providers import TNGAPIProvider
                    provider = TNGAPIProvider(
                        base_url=self.config.get('backend', {}).get('api_base_url'),
                        cache_dir=self.config.get('backend', {}).get('api_cache_dir')
                    )
                    api_img = provider.get_image_path(sim, int(self.config['simulation']['snapshot']), subhalo_id)
                    if api_img and api_img.exists():
                        image_path = str(api_img)
                except Exception:
                    pass

            # Get photometry using appropriate method
            band_mags, sn_data = photometry.get_photometry_for_subhalo(
                subhalo_id, sn_data, photometry_type, sim,
                bands=bands, 
                kids_results=self.kids_results,
                kids_aperture_mode=phot_config.get('kids_aperture_mode', 'direct'),
                kids_image_type=phot_config.get('kids_image_type', 'noisy'),
                root_path=self.root_path,
                kids_image_pattern=self.get_path('kids_image_pattern', subhalo_id=subhalo_id),
                kids_snapshot=self.config['simulation'].get('kids_snapshot'),
                kids_results_dir=self.resolve_path('kids_results_dir'),
                image_path=image_path
            )
            
            # Add local colors
            sn_data = photometry.add_local_colors(sn_data, band_mags)
            
        except Exception as e:
            print(f"Error in photometry for {subhalo_id}: {e}")
            # Fill with NaNs on error
            for band in bands:
                sn_data[f'local_{band}'] = np.nan
            sn_data['localrestframe_gz'] = np.nan
        
        return sn_data
    
    def _calculate_morphology(self, subhalo_id, sn_data):
        """Calculate morphological properties including DLR."""
        morph_config = self.config.get('morphology', {})
        photometry_type = self.photometry_types.get(subhalo_id, 'regular')
        
        # Use same type for morphology as photometry by default
        morphology_type = morph_config.get('mode', 'auto')
        if morphology_type == 'auto':
            morphology_type = photometry_type
        
        try:
            sn_data = morphology.calculate_morphology_for_subhalo(
                subhalo_id, sn_data, morphology_type,
                self.config['simulation']['name'],
                morphology_data=self.morphology_data,
                rhalf_correction=morph_config.get('rhalf_correction', 1/0.68),
                dlr_default=morph_config.get('dlr_default', 99.99),
                root_path=self.root_path
            )
        except Exception as e:
            print(f"Error calculating morphology for {subhalo_id}: {e}")
            sn_data['d_DLR'] = np.nan
        
        return sn_data
    
    def _get_subhalo_info(self, subhalo_id):
        """Get subhalo information from TNG API."""
        try:
            mode = self.config.get('backend', {}).get('mode')
            sim = self.config['simulation']['name']
            snap = int(self.config['simulation'].get('snapshot', 99))
            if mode == 'api':
                from .providers import TNGAPIProvider
                provider = TNGAPIProvider(
                    base_url=self.config.get('backend', {}).get('api_base_url'),
                    cache_dir=self.config.get('backend', {}).get('api_cache_dir')
                )
                info = provider.get_subhalo_info(sim, snap, int(subhalo_id))
                if isinstance(info, dict):
                    if all(k in info for k in ('pos_x', 'pos_y', 'pos_z')):
                        return {'pos_x': info['pos_x'], 'pos_y': info['pos_y'], 'pos_z': info['pos_z']}
                    if 'pos' in info and isinstance(info['pos'], (list, tuple)) and len(info['pos']) >= 3:
                        return {'pos_x': info['pos'][0], 'pos_y': info['pos'][1], 'pos_z': info['pos'][2]}
            elif mode == 'cloud':
                from .providers import TNGCloudProvider
                provider = TNGCloudProvider(snapshot_root=os.environ.get('SNAPSHOT_ROOT'))
                center = provider.get_subhalo_center(sim, snap, int(subhalo_id))
                if isinstance(center, dict) and all(k in center for k in ('pos_x','pos_y','pos_z')):
                    return center
        except Exception:
            pass
        # Fallback if API not used/available
        return {'pos_x': 0, 'pos_y': 0, 'pos_z': 0}
    
    def run_simulation(self, n_subhalos=None, subhalo_ids=None):
        """
        Run the full simulation pipeline.
        
        Parameters:
        -----------
        n_subhalos : int, optional
            Number of subhalos to process (from filtered catalog)
        subhalo_ids : list, optional
            Specific subhalo IDs to process
            
        Returns:
        --------
        pandas.DataFrame : Combined simulation results
        """
        # Load metadata if not already loaded
        if self.galmeta is None:
            self.load_metadata()
        
        # Determine which subhalos to process
        process_ids = self.select_subhalos(n_subhalos, subhalo_ids)
        
        print(f"Processing {len(process_ids)} subhalos")
        
        # Setup photometry and morphology strategy
        self.setup_photometry_and_morphology(process_ids)
        
        # Setup multiprocessing
        n_cores = min(
            self.config['simulation']['n_cores'], 
            mp.cpu_count(), 
            len(process_ids)
        )
        
        # Process subhalos
        if n_cores > 1:
            with mp.Pool(n_cores) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self.process_subhalo, process_ids),
                    total=len(process_ids),
                    desc="Processing subhalos"
                ))
        else:
            results = []
            for subhalo_id in tqdm(process_ids, desc="Processing subhalos"):
                results.append(self.process_subhalo(subhalo_id))
        
        # Combine results
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            combined_results = pd.concat(valid_results, ignore_index=True)
            
            # Save combined results using resolved path
            sim = self.config['simulation']['name']
            output_dir = self.config['simulation']['output_dir']
            if not os.path.isabs(output_dir):
                output_dir = self.root_path / output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = output_dir / f'{sim}_combined_skysurvey_out.h5'
            combined_results.to_hdf(output_path, key='main', mode='w')
            
            print(f"Simulation complete. Results saved to {output_path}")
            print(f"Generated {len(combined_results)} SNe from {len(valid_results)} galaxies")
            
            # Cleanup temp artifacts if requested
            if bool(self.config.get('backend', {}).get('cleanup_temp', False)):
                self._post_run_cleanup()
            
            return combined_results
        else:
            print("No valid results generated")
            # Even on empty results, still perform cleanup if requested
            if bool(self.config.get('backend', {}).get('cleanup_temp', False)):
                self._post_run_cleanup()
            return pd.DataFrame()

    def _post_run_cleanup(self):
        """Remove temporary files and directories produced during run when cleanup enabled."""
        # Remove tracked temp files
        for f in list(self._temp_files):
            try:
                if isinstance(f, (str, Path)) and os.path.isfile(f):
                    os.remove(f)
            except Exception:
                pass
            finally:
                self._temp_files.remove(f)
        # Remove tracked temp directories
        for d in list(self._temp_dirs):
            try:
                if isinstance(d, (str, Path)) and os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
            finally:
                self._temp_dirs.remove(d)
        # Optional: clear API cache in cloud mode (defensive; usually unused)
        if self.config.get('backend', {}).get('mode') == 'cloud':
            cache_dir = self.config.get('backend', {}).get('api_cache_dir')
            if cache_dir:
                try:
                    cache_dir = Path(cache_dir)
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass

    def run_snapshots(self, snapshots, n_subhalos=None, subhalo_ids=None):
        """
        Convenience helper: run the pipeline across multiple snapshots.

        Parameters:
        - snapshots: iterable of int snapshot numbers
        - n_subhalos, subhalo_ids: forwarded to run_simulation

        Returns: dict mapping snapshot -> DataFrame
        """
        results_by_snap = {}
        original_snapshot = int(self.config['simulation'].get('snapshot', 99))
        try:
            for snap in snapshots:
                # Set snapshot and clear state to force reload
                self._set_nested_config('simulation.snapshot', int(snap))
                self.galmeta = None
                self.morphology_data = None
                # Run and store
                df = self.run_simulation(n_subhalos=n_subhalos, subhalo_ids=subhalo_ids)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = df.copy()
                    df['snapshot'] = int(snap)
                results_by_snap[int(snap)] = df
        finally:
            # Restore original snapshot
            self._set_nested_config('simulation.snapshot', original_snapshot)
        return results_by_snap


def main():
    """Command line interface for running simulations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TNG-SN simulation')
    parser.add_argument('--config', help='Configuration YAML file')
    parser.add_argument('--simulation', default='TNG50-1', help='Simulation name')
    parser.add_argument('--n-subhalos', type=int, help='Number of subhalos to process')
    parser.add_argument('--output-dir', default='simout', help='Output directory')
    parser.add_argument('--n-cores', type=int, help='Number of CPU cores to use')
    parser.add_argument('--root-path', type=str, help='Root path for data files')
    
    args = parser.parse_args()
    
    # Create configuration overrides
    config_overrides = {}
    if args.simulation:
        config_overrides['simulation__name'] = args.simulation
    if args.output_dir:
        config_overrides['simulation__output_dir'] = args.output_dir
    if args.n_cores:
        config_overrides['simulation__n_cores'] = args.n_cores
    if args.root_path:
        config_overrides['paths__root_path'] = args.root_path
    
    # Initialize and run simulation
    sim = TNGSNSimulation(args.config, **config_overrides)
    results = sim.run_simulation(n_subhalos=args.n_subhalos)
    
    return results


if __name__ == "__main__":
    main()