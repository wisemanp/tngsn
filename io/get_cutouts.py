import requests
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import glob
import re
from pathlib import Path
import logging

# TNG API configuration
BASE_URL = 'http://www.tng-project.org/api/'

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('get_cutouts.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_api_key():
    """Get API key from environment variable or prompt user."""
    api_key = os.environ.get('TNG_API_KEY')
    if not api_key:
        api_key = input("Enter your TNG API key: ")
        if not api_key:
            raise ValueError("TNG API key is required. Set TNG_API_KEY environment variable or provide when prompted.")
    logger.info(f"API key loaded: {'***' + api_key[-4:] if len(api_key) > 4 else '***'}")
    return api_key

def get_headers():
    """Get headers with API key."""
    api_key = get_api_key()
    headers = {"api-key": api_key}
    logger.debug(f"Headers created with API key ending in: {'***' + api_key[-4:] if len(api_key) > 4 else '***'}")
    return headers

def get_tng_data(path, params=None, savepath=None):
    """
    Make HTTP GET request to TNG API.
    
    Parameters:
    -----------
    path : str
        API endpoint path
    params : dict, optional
        Query parameters
    savepath : str, optional
        Directory to save downloaded files
        
    Returns:
    --------
    dict or str : JSON response or filename if file downloaded
    """
    headers = get_headers()
    
    # Log the exact request details
    logger.info(f"Making API request:")
    logger.info(f"  URL: {path}")
    logger.info(f"  Params: {params}")
    logger.info(f"  Headers: {dict(headers)}")
    
    try:
        r = requests.get(path, params=params, headers=headers)
        logger.info(f"Response status: {r.status_code}")
        logger.info(f"Response headers: {dict(r.headers)}")
        
        r.raise_for_status()

        if r.headers['content-type'] == 'application/json':
            response_data = r.json()
            logger.info(f"JSON response received with {len(response_data)} keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'List response'}")
            return response_data
        
        if 'content-disposition' in r.headers:
            if savepath is None:
                savepath = ''
            
            filename = savepath + r.headers['content-disposition'].split("filename=")[1]
            logger.info(f'Downloading file: {filename}')
            logger.info(f'File size: {len(r.content)} bytes')
            
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            logger.info(f'File saved successfully: {filename}')
            return filename
        
        logger.warning(f"Unexpected content type: {r.headers.get('content-type', 'Unknown')}")
        return r
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error {r.status_code}: {e}")
        logger.error(f"Response content: {r.text[:500]}...")
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

def get_simulation_base_url(sim):
    """Get base URL for simulation."""
    bases = {
        'TNG50-1': BASE_URL + 'TNG50-1/snapshots/96/',
        'TNG100-1': BASE_URL + 'TNG100-1/snapshots/96/',
        'TNG300-1': BASE_URL + 'TNG300-1/snapshots/96/'
    }
    
    if sim not in bases:
        logger.error(f"Simulation {sim} not supported. Available: {list(bases.keys())}")
        raise ValueError(f"Simulation {sim} not supported. Available: {list(bases.keys())}")
    
    base_url = bases[sim]
    logger.info(f"Base URL for {sim}: {base_url}")
    return base_url

def get_subhalo_cutout(subhalo_id, sim, output_dir, star_fields=None, gas_fields=None):
    """
    Download cutout data for a specific subhalo.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID to download
    sim : str
        Simulation name (e.g., 'TNG50-1')
    output_dir : str
        Output directory for cutouts
    star_fields : list, optional
        Star particle fields to download
    gas_fields : list, optional
        Gas particle fields to download
        
    Returns:
    --------
    str : Path to downloaded file, or None if failed
    """
    
    logger.info(f"Starting cutout download for subhalo {subhalo_id}")
    
    if star_fields is None:
        star_fields = ['Masses', 'Coordinates', 'GFM_InitialMass', 
                      'GFM_Metallicity', 'GFM_StellarFormationTime']
    
    if gas_fields is None:
        gas_fields = ['Coordinates', 'StarFormationRate', 'GFM_Metallicity']
    
    logger.info(f"Star fields: {star_fields}")
    logger.info(f"Gas fields: {gas_fields}")
    
    base_url = get_simulation_base_url(sim)
    
    try:
        # Get subhalo info
        subhalo_url = base_url + f'subhalos/{subhalo_id}/'
        logger.info(f"Fetching subhalo info from: {subhalo_url}")
        
        subinfo = get_tng_data(subhalo_url)
        
        logger.info(f"Subhalo info received. Available fields: {list(subinfo.keys())}")
        logger.info(f"Cutout URL: {subinfo.get('cutouts', {}).get('subhalo', 'Not found')}")
        
        # Create output directory
        savepath = os.path.join(output_dir, sim, '96', str(subhalo_id))
        os.makedirs(savepath, exist_ok=True)
        logger.info(f"Output directory created: {savepath}")
        savepath += '/'  # Add trailing slash for API
        
        # Build query string
        star_query = ','.join(star_fields)
        gas_query = ','.join(gas_fields)
        cutout_query = f'gas={gas_query}&stars={star_query}'
        
        logger.info(f"Cutout query string: {cutout_query}")
        
        # Build full cutout URL
        cutout_base_url = subinfo['cutouts']['subhalo']
        full_cutout_url = f"{cutout_base_url}?{cutout_query}"
        logger.info(f"Full cutout URL: {full_cutout_url}")
        
        print(f'Downloading cutout for subhalo {subhalo_id}')
        
        # Download cutout
        snap_fn = get_tng_data(full_cutout_url, savepath=savepath)
        
        logger.info(f"Cutout download completed: {snap_fn}")
        return snap_fn
        
    except Exception as e:
        logger.error(f'Error downloading subhalo {subhalo_id}: {e}')
        print(f'Error downloading subhalo {subhalo_id}: {e}')
        return None

def get_random_subhalos(sim, n_halos=1000, min_stellar_mass=0.01, output_dir='data'):
    """
    Download cutouts for random subhalos above stellar mass threshold.
    
    Parameters:
    -----------
    sim : str
        Simulation name
    n_halos : int
        Target number of halos to download
    min_stellar_mass : float
        Minimum stellar mass (10^10 Msun units)
    output_dir : str
        Output directory
        
    Returns:
    --------
    list : List of successfully downloaded subhalo IDs
    """
    
    logger.info(f"Starting random subhalo download")
    logger.info(f"Target: {n_halos} halos from {sim}")
    logger.info(f"Minimum stellar mass: {min_stellar_mass} x 10^10 Msun")
    logger.info(f"Output directory: {output_dir}")
    
    base_url = get_simulation_base_url(sim)
    downloaded_ids = []
    
    # Calculate how many API calls we need (100 halos per call)
    n_calls = (n_halos + 199) // 200  # 2 random halos per 100-halo call
    logger.info(f"Estimated API calls needed: {n_calls}")
    
    print(f"Downloading {n_halos} random subhalos from {sim}")
    print(f"Minimum stellar mass: {min_stellar_mass} x 10^10 Msun")
    
    for i in tqdm(range(n_calls), desc="API calls"):
        try:
            # Build subhalo list URL
            subhalo_list_url = base_url + f'subhalos/?mass_stars__gte={min_stellar_mass}&limit=100&offset={100*i}&order_by=-mass_stars'
            logger.info(f"API call {i+1}/{n_calls}: {subhalo_list_url}")
            
            # Get 100 massive subhalos
            subs = get_tng_data(subhalo_list_url)
            
            if not subs['results']:
                logger.warning(f"No more subhalos found at offset {100*i}")
                print(f"No more subhalos found at offset {100*i}")
                break
            
            logger.info(f"Retrieved {len(subs['results'])} subhalos from API call {i+1}")
            
            # Select 2 random subhalos from this batch
            n_available = len(subs['results'])
            n_select = min(2, n_available, n_halos - len(downloaded_ids))
            
            if n_select <= 0:
                logger.info("Target number of halos reached")
                break
                
            rand_indices = np.random.choice(n_available, size=n_select, replace=False)
            logger.info(f"Selected random indices: {rand_indices}")
            
            for rand_idx in rand_indices:
                subhalo_id = subs['results'][rand_idx]['id']
                logger.info(f"Processing subhalo {subhalo_id} (index {rand_idx})")
                
                # Download cutout
                result = get_subhalo_cutout(subhalo_id, sim, output_dir)
                
                if result:
                    downloaded_ids.append(subhalo_id)
                    logger.info(f"Successfully downloaded subhalo {subhalo_id}")
                else:
                    logger.error(f"Failed to download subhalo {subhalo_id}")
                
                if len(downloaded_ids) >= n_halos:
                    logger.info(f"Reached target of {n_halos} downloads")
                    break
            
            if len(downloaded_ids) >= n_halos:
                break
                
        except Exception as e:
            logger.error(f"Error in API call {i}: {e}")
            print(f"Error in API call {i}: {e}")
            continue
    
    logger.info(f"Download completed: {len(downloaded_ids)} successful downloads")
    print(f"Successfully downloaded {len(downloaded_ids)} subhalos")
    return downloaded_ids

def find_kids_subhalo_ids(kids_data_dir):
    """
    Scrape KIDS directory to find all available subhalo IDs.
    
    Parameters:
    -----------
    kids_data_dir : str
        Path to KIDS data directory (e.g., data/TNG50-1/KIDS/snapnum_096/zx/data)
        
    Returns:
    --------
    list : List of subhalo IDs found in KIDS directory
    """
    
    logger.info(f"Scanning KIDS directory: {kids_data_dir}")
    print(f"Scanning KIDS directory: {kids_data_dir}")
    
    if not os.path.exists(kids_data_dir):
        logger.error(f"KIDS directory not found: {kids_data_dir}")
        raise FileNotFoundError(f"KIDS directory not found: {kids_data_dir}")
    
    # Look for files matching pattern: broadband_*.fits
    pattern = os.path.join(kids_data_dir, 'broadband_*.fits')
    fits_files = glob.glob(pattern)
    
    logger.info(f"Found {len(fits_files)} FITS files")
    print(f"Found {len(fits_files)} FITS files")
    
    # Extract subhalo IDs from filenames
    subhalo_ids = []
    for fits_file in fits_files:
        filename = os.path.basename(fits_file)
        # Extract ID from filename like "broadband_12345.fits"
        match = re.search(r'broadband_(\d+)\.fits', filename)
        if match:
            subhalo_id = int(match.group(1))
            subhalo_ids.append(subhalo_id)
    
    subhalo_ids = sorted(list(set(subhalo_ids)))  # Remove duplicates and sort
    
    logger.info(f"Found {len(subhalo_ids)} unique subhalo IDs")
    logger.info(f"ID range: {min(subhalo_ids) if subhalo_ids else 'None'} to {max(subhalo_ids) if subhalo_ids else 'None'}")
    print(f"Found {len(subhalo_ids)} unique subhalo IDs")
    if subhalo_ids:
        print(f"ID range: {min(subhalo_ids)} to {max(subhalo_ids)}")
    
    return subhalo_ids

def download_cutouts_for_kids(kids_data_dir, sim, output_dir='data', max_download=None):
    """
    Find subhalo IDs from KIDS directory and download corresponding cutouts.
    
    Parameters:
    -----------
    kids_data_dir : str
        Path to KIDS data directory
    sim : str
        Simulation name
    output_dir : str
        Output directory for cutouts
    max_download : int, optional
        Maximum number of cutouts to download (for testing)
        
    Returns:
    --------
    dict : Download results with success/failure counts
    """
    
    logger.info(f"Starting KIDS-based cutout download")
    logger.info(f"KIDS directory: {kids_data_dir}")
    logger.info(f"Simulation: {sim}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max download limit: {max_download}")
    
    # Find subhalo IDs from KIDS
    subhalo_ids = find_kids_subhalo_ids(kids_data_dir)
    
    if max_download:
        subhalo_ids = subhalo_ids[:max_download]
        logger.info(f"Limited to first {max_download} subhalos for testing")
        print(f"Limited to first {max_download} subhalos for testing")
    
    # Check which cutouts already exist
    existing_cutouts = []
    missing_cutouts = []
    
    for subhalo_id in subhalo_ids:
        cutout_dir = os.path.join(output_dir, sim, 'snap96', str(subhalo_id))
        if os.path.exists(cutout_dir) and os.listdir(cutout_dir):
            existing_cutouts.append(subhalo_id)
        else:
            missing_cutouts.append(subhalo_id)
    
    logger.info(f"Cutout status:")
    logger.info(f"  Already downloaded: {len(existing_cutouts)}")
    logger.info(f"  Need to download: {len(missing_cutouts)}")
    
    print(f"Cutout status:")
    print(f"  Already downloaded: {len(existing_cutouts)}")
    print(f"  Need to download: {len(missing_cutouts)}")
    
    # Download missing cutouts
    downloaded = []
    failed = []
    
    if missing_cutouts:
        logger.info(f"Starting download of {len(missing_cutouts)} missing cutouts")
        print(f"Downloading {len(missing_cutouts)} missing cutouts...")
        
        for subhalo_id in tqdm(missing_cutouts, desc="Downloading cutouts"):
            logger.info(f"Downloading cutout for subhalo {subhalo_id}")
            result = get_subhalo_cutout(subhalo_id, sim, output_dir)
            
            if result:
                downloaded.append(subhalo_id)
                logger.info(f"Successfully downloaded subhalo {subhalo_id}")
            else:
                failed.append(subhalo_id)
                logger.error(f"Failed to download subhalo {subhalo_id}")
    
    results = {
        'kids_subhalos': len(subhalo_ids),
        'already_downloaded': len(existing_cutouts),
        'newly_downloaded': len(downloaded),
        'failed': len(failed),
        'failed_ids': failed
    }
    
    logger.info(f"Download summary: {results}")
    return results

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Download TNG subhalo cutouts')
    
    # Mode selection
    parser.add_argument('--mode', choices=['random', 'kids'], default='random',
                       help='Download mode: random subhalos or based on KIDS data')
    
    # Common parameters
    parser.add_argument('--sim', default='TNG50-1', choices=['TNG50-1', 'TNG100-1', 'TNG300-1'],
                       help='Simulation name')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for cutouts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Random mode parameters
    parser.add_argument('--n-halos', type=int, default=1000,
                       help='Number of random halos to download (random mode)')
    parser.add_argument('--min-mass', type=float, default=0.01,
                       help='Minimum stellar mass in 10^10 Msun (random mode)')
    
    # KIDS mode parameters
    parser.add_argument('--kids-dir', 
                       help='Path to KIDS data directory (required for kids mode)')
    parser.add_argument('--max-download', type=int,
                       help='Maximum number to download (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    logger.info(f"Starting get_cutouts.py with arguments: {vars(args)}")
    
    if args.mode == 'random':
        logger.info("=== RANDOM MODE ===")
        print("=== RANDOM MODE ===")
        downloaded_ids = get_random_subhalos(
            sim=args.sim,
            n_halos=args.n_halos,
            min_stellar_mass=args.min_mass,
            output_dir=args.output_dir
        )
        
        # Save list of downloaded IDs
        output_file = f'downloaded_subhalos_{args.sim}.txt'
        with open(output_file, 'w') as f:
            for subhalo_id in downloaded_ids:
                f.write(f"{subhalo_id}\n")
        
        logger.info(f"Downloaded subhalo IDs saved to {output_file}")
        print(f"Downloaded subhalo IDs saved to {output_file}")
    
    elif args.mode == 'kids':
        logger.info("=== KIDS MODE ===")
        print("=== KIDS MODE ===")
        
        if not args.kids_dir:
            # Try to infer KIDS directory
            kids_dir = os.path.join(args.output_dir, args.sim, 'KIDS', 'snapnum_096', 'zx', 'data')
            if os.path.exists(kids_dir):
                args.kids_dir = kids_dir
                logger.info(f"Using inferred KIDS directory: {kids_dir}")
                print(f"Using inferred KIDS directory: {kids_dir}")
            else:
                logger.error("--kids-dir is required for kids mode")
                raise ValueError("--kids-dir is required for kids mode")
        
        results = download_cutouts_for_kids(
            kids_data_dir=args.kids_dir,
            sim=args.sim,
            output_dir=args.output_dir,
            max_download=args.max_download
        )
        
        logger.info("=== DOWNLOAD SUMMARY ===")
        print("=== DOWNLOAD SUMMARY ===")
        print(f"Total KIDS subhalos: {results['kids_subhalos']}")
        print(f"Already downloaded: {results['already_downloaded']}")
        print(f"Newly downloaded: {results['newly_downloaded']}")
        print(f"Failed downloads: {results['failed']}")
        
        if results['failed_ids']:
            logger.warning(f"Failed IDs: {results['failed_ids']}")
            print(f"Failed IDs: {results['failed_ids']}")

if __name__ == "__main__":
    main()

