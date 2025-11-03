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
        self.config = self._load_config(config_file)
        
        # Apply any overrides
        for key, value in config_overrides.items():
            self._set_nested_config(key, value)
        
        # Initialize data containers
        self.galmeta = None
        self.galaxy_morphs = None
        self.simulation_results = []
        
        # Photometry and morphology strategy
        self.photometry_types = None
        self.morphology_data = None
        self.kids_results = None
        
    def _load_config(self, config_file):
        """Load configuration from YAML file."""
        if config_file is None:
            # Use default config
            config_dir = os.path.dirname(__file__)
            config_file = os.path.join(config_dir, 'config', 'default_config.yaml')
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
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
        
        # Load galaxy metadata
        galmeta_path = self.config['paths']['galmeta_file'].format(simulation=sim)
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
                force_kids=phot_config.get('force_kids', False),
                force_regular=phot_config.get('force_regular', False),
                sim=sim
            )
        elif phot_config.get('mode') == 'kids':
            self.photometry_types = {sid: 'kids' for sid in subhalo_ids}
            self.kids_results = photometry.load_kids_photometry_results(sim)
        else:
            self.photometry_types = {sid: 'regular' for sid in subhalo_ids}
            self.kids_results = None
        
        # Load morphology data
        self.morphology_data = morphology.preload_morphology_data(
            subhalo_ids, self.photometry_types, sim
        )
    
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
        
        # Check for required files
        cutout_path = self.config['paths']['cutout_pattern'].format(
            simulation=sim, subhalo_id=subhalo_id
        )
        
        if not os.path.isfile(cutout_path):
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
            
            # Save results
            output_path = os.path.join(
                self.config['simulation']['output_dir'],
                f'{subhalo_id}_skysurvey_out.h5'
            )
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
        
        # Load or calculate ages
        ages_path = self.config['paths']['ages_pattern'].format(
            simulation=sim, subhalo_id=subhalo_id
        )
        
        if os.path.isfile(ages_path):
            ages = np.loadtxt(ages_path, dtype=float)
        else:
            print(f'Calculating ages for {subhalo_id}')
            zs = (1 / stfs) - 1
            t = cosmo.lookback_time(zs)
            ages = np.clip(t.value, a_min=0.05, a_max=None)
            
            # Save ages for future use
            os.makedirs(os.path.dirname(ages_path), exist_ok=True)
            np.savetxt(ages_path, ages)
        
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
            # Get image header for pixel scale conversion
            image_path = self.config['paths']['image_pattern'].format(
                simulation=sim, subhalo_id=subhalo_id
            )
            
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
            # Get photometry using appropriate method
            band_mags, sn_data = photometry.get_photometry_for_subhalo(
                subhalo_id, sn_data, photometry_type, sim,
                bands=bands, 
                kids_results=self.kids_results,
                kids_aperture_mode=phot_config.get('kids_aperture_mode', 'direct')
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
                dlr_default=morph_config.get('dlr_default', 99.99)
            )
        except Exception as e:
            print(f"Error calculating morphology for {subhalo_id}: {e}")
            sn_data['d_DLR'] = np.nan
        
        return sn_data
    
    def _get_subhalo_info(self, subhalo_id):
        """Get subhalo information from TNG API."""
        # This would use the TNG API to get subhalo center
        # For now, return dummy data
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
            
            # Save combined results
            sim = self.config['simulation']['name']
            output_path = os.path.join(
                self.config['simulation']['output_dir'],
                f'{sim}_combined_skysurvey_out.h5'
            )
            combined_results.to_hdf(output_path, key='main', mode='w')
            
            print(f"Simulation complete. Results saved to {output_path}")
            print(f"Generated {len(combined_results)} SNe from {len(valid_results)} galaxies")
            
            return combined_results
        else:
            print("No valid results generated")
            return pd.DataFrame()


def main():
    """Command line interface for running simulations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TNG-SN simulation')
    parser.add_argument('--config', help='Configuration YAML file')
    parser.add_argument('--simulation', default='TNG50-1', help='Simulation name')
    parser.add_argument('--n-subhalos', type=int, help='Number of subhalos to process')
    parser.add_argument('--output-dir', default='simout', help='Output directory')
    parser.add_argument('--n-cores', type=int, help='Number of CPU cores to use')
    
    args = parser.parse_args()
    
    # Create configuration overrides
    config_overrides = {}
    if args.simulation:
        config_overrides['simulation.name'] = args.simulation
    if args.output_dir:
        config_overrides['simulation.output_dir'] = args.output_dir
    if args.n_cores:
        config_overrides['simulation.n_cores'] = args.n_cores
    
    # Initialize and run simulation
    sim = TNGSNSimulation(args.config, **config_overrides)
    results = sim.run_simulation(n_subhalos=args.n_subhalos)
    
    return results


if __name__ == "__main__":
    main()