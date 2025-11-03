"""
Photometry utilities for TNG simulations.

Functions for aperture photometry on synthetic images, supporting both
regular TNG broadband images and KIDS survey data.
"""

import numpy as np
import pandas as pd
import os
from astropy.io import fits
from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry
from astropy.cosmology import Planck15 as cosmo


def convert_flux_to_magnitude(aperture_sum, zp):
    """Convert aperture flux sum to magnitude."""
    return -2.5 * np.log10(aperture_sum) + zp


def load_kids_photometry_results(sim, results_dir=None):
    """
    Load KIDS photometry results.
    
    Parameters:
    -----------
    sim : str
        Simulation name (e.g., 'TNG50-1')
    results_dir : str, optional
        Path to KIDS results directory
        
    Returns:
    --------
    pandas.DataFrame : KIDS photometry results with morphology
    """
    if results_dir is None:
        # Fallback legacy location without snapshot
        results_dir = f'data/{sim}/KIDS/results'
    
    results_file = os.path.join(results_dir, 'photometry_results.csv')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"KIDS results not found: {results_file}")
    
    kids_results = pd.read_csv(results_file)
    kids_results = kids_results.set_index('subhaloId')
    
    return kids_results


def do_aperture_photometry(img_path, positions, bands=[0, 1, 2, 3], 
                          zp=0, aperture_radius=None):
    """
    Perform aperture photometry on multiple positions.
    
    Parameters:
    -----------
    img_path : str
        Path to FITS image file
    positions : list
        List of [x, y] pixel positions
    bands : list
        List of band indices to process
    zp : float
        Zero point magnitude
    aperture_radius : float, optional
        Aperture radius. If None, calculated from pixel scale
        
    Returns:
    --------
    dict : Results with magnitudes and errors for each band
    """
    results = {}
    
    with fits.open(img_path) as hdul:
        img_data = hdul[0].data
        header = hdul[0].header
        
        if aperture_radius is None:
            # Calculate physical aperture radius
            pixscale_physical = 1000 / header['CDELT1'] / cosmo.h
            aperture_radius = pixscale_physical
        
        # Create aperture
        aperture = CircularAperture(positions, r=aperture_radius)
        
        for i, band in enumerate(bands):
            if len(img_data.shape) == 3:
                # Multi-band image
                band_data = img_data[band]
            else:
                # Single band
                band_data = img_data
            
            # Perform photometry
            phot_table = aperture_photometry(
                band_data, 
                aperture, 
                error=0.1 * band_data
            )
            
            # Convert to magnitudes
            magnitudes = convert_flux_to_magnitude(
                phot_table['aperture_sum'], zp
            )
            mag_errors = convert_flux_to_magnitude(
                phot_table['aperture_sum_err'], zp
            )
            
            results[f'band_{band}'] = {
                'magnitudes': magnitudes,
                'errors': mag_errors
            }
    
    return results


def multi_band_photometry(img_path, positions, bands=['g', 'r', 'i', 'z']):
    """
    Perform photometry in multiple bands with proper band indexing.
    
    Parameters:
    -----------
    img_path : str
        Path to FITS image file
    positions : list
        List of [x, y] pixel positions  
    bands : list
        List of band names
        
    Returns:
    --------
    dict : Band magnitudes for each position
    """
    band_indices = list(range(len(bands)))
    results = do_aperture_photometry(
        img_path, positions, bands=band_indices
    )
    
    # Reformat results by band name
    band_results = {}
    for i, band_name in enumerate(bands):
        if f'band_{i}' in results:
            band_results[band_name] = pd.Series(
                results[f'band_{i}']['magnitudes']
            )
    
    return band_results


def get_kids_image_path(subhalo_id, sim, image_type='noisy', kids_snapshot=None, root_path=None, pattern=None):
    """
    Get path to KIDS image file.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID
    sim : str
        Simulation name
    image_type : str
        'original' or 'noisy'
        
    Returns:
    --------
    str : Path to KIDS image file
    """
    # If a pattern is provided (already resolved with root and kids_snapshot), prefer it if it exists
    if pattern is not None:
        if os.path.isfile(pattern):
            return pattern
        # continue to try other constructions if missing

    # Build path with kids_snapshot if provided; fallback to legacy snapnum_096
    snapnum = f"{int(kids_snapshot):03d}" if kids_snapshot is not None else "096"
    # Preferred layout: with snapnum folder
    base = os.path.join(root_path or ".", f"data/{sim}/KIDS/snapnum_{snapnum}/zx")
    candidates = []
    if image_type == 'noisy':
        candidates.append(os.path.join(base, 'noisy', f'noisy_broadband_{subhalo_id}.fits'))
    else:
        candidates.append(os.path.join(base, 'data', f'broadband_{subhalo_id}.fits'))

    # Legacy: default to snapnum_096 if specific kids_snapshot missing
    if kids_snapshot is not None and int(kids_snapshot) != 96:
        legacy_base = os.path.join(root_path or ".", f"data/{sim}/KIDS/snapnum_096/zx")
        if image_type == 'noisy':
            candidates.append(os.path.join(legacy_base, 'noisy', f'noisy_broadband_{subhalo_id}.fits'))
        else:
            candidates.append(os.path.join(legacy_base, 'data', f'broadband_{subhalo_id}.fits'))

    # Extra legacy: no snapnum folder at all
    nosnap_base = os.path.join(root_path or ".", f"data/{sim}/KIDS/zx")
    if image_type == 'noisy':
        candidates.append(os.path.join(nosnap_base, 'noisy', f'noisy_broadband_{subhalo_id}.fits'))
    else:
        candidates.append(os.path.join(nosnap_base, 'data', f'broadband_{subhalo_id}.fits'))

    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fall back to the first constructed path
    return candidates[0]


def perform_kids_aperture_photometry(subhalo_id, sn_data, sim, bands=['g', 'r', 'i', 'z'],
                                    image_type='noisy', use_source_params=True,
                                    root_path=None, kids_image_pattern=None,
                                    kids_snapshot=None, kids_results_dir=None):
    """
    Perform aperture photometry on KIDS images at SN positions.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID
    sn_data : pandas.DataFrame
        SN position data with 'x_pix', 'y_pix' columns
    sim : str
        Simulation name
    bands : list
        Band names
    image_type : str
        'original' or 'noisy' KIDS images
    use_source_params : bool
        Use KIDS source parameters for aperture if available
        
    Returns:
    --------
    dict : Photometry results by band
    """
    # Get KIDS image path (pattern takes precedence)
    kids_image_path = get_kids_image_path(
        subhalo_id, sim, image_type,
        kids_snapshot=kids_snapshot, root_path=root_path,
        pattern=kids_image_pattern
    )
    
    if not os.path.isfile(kids_image_path):
        print(f"Warning: KIDS image not found: {kids_image_path}")
        return {band: pd.Series([np.nan] * len(sn_data)) for band in bands}
    
    try:
        # Load KIDS image
        with fits.open(kids_image_path) as hdul:
            img_data = hdul[0].data
            header = hdul[0].header
            
            # Check data structure - KIDS images are 3D (bands, height, width)
            if len(img_data.shape) == 3:
                n_bands, height, width = img_data.shape
                print(f"DEBUG: KIDS image shape: {img_data.shape}")
            else:
                raise ValueError(f"Expected 3D KIDS image, got shape: {img_data.shape}")
        
        # Get SN positions
        positions = sn_data[['x_pix', 'y_pix']].values.tolist()
        
        # Try to use KIDS source parameters for better apertures
        if use_source_params:
            try:
                kids_results = load_kids_photometry_results(sim, results_dir=kids_results_dir)
                if subhalo_id in kids_results.index:
                    kids_data = kids_results.loc[subhalo_id]
                    
                    # Use elliptical aperture based on KIDS source detection
                    if all(param in kids_data for param in ['x', 'y', 'a', 'b', 'theta']):
                        from photutils.aperture import EllipticalAperture
                        
                        # Create elliptical apertures for each SN position
                        # Scale aperture size based on distance from galaxy center
                        galaxy_center = [kids_data['x'], kids_data['y']]
                        apertures = []
                        
                        for pos in positions:
                            # Use KIDS detected parameters
                            aperture = EllipticalAperture(
                                pos, kids_data['a'], kids_data['b'], kids_data['theta']
                            )
                            apertures.append(aperture)
                    else:
                        use_source_params = False
                else:
                    use_source_params = False
            except:
                use_source_params = False
        
        if not use_source_params:
            # Use circular apertures
            aperture_radius = 3.0  # pixels, could be configurable
            apertures = [CircularAperture(positions, r=aperture_radius)]
        
        # Perform photometry for each band
        band_mags = {}
        zp_dict = {'g': 25.0, 'r': 25.0, 'i': 25.0, 'z': 25.0}  # Default zero points
        
        for i, band in enumerate(bands):
            if i >= n_bands:
                print(f"Warning: Band {band} (index {i}) not available in KIDS image")
                band_mags[band] = pd.Series([np.nan] * len(sn_data))
                continue
            
            # Extract band data
            band_data = img_data[i].astype(np.float64)
            
            # Background subtraction
            import sep
            bkg = sep.Background(band_data)
            band_data_sub = band_data - bkg
            
            # Perform aperture photometry
            if use_source_params:
                # Individual apertures for each position
                fluxes = []
                for aperture in apertures:
                    phot_table = aperture_photometry(band_data_sub, aperture)
                    fluxes.append(phot_table['aperture_sum'][0])
                total_flux = np.array(fluxes)
            else:
                # Single aperture for all positions
                phot_table = aperture_photometry(band_data_sub, apertures[0])
                total_flux = phot_table['aperture_sum'].data
            
            # Convert to magnitudes
            magnitudes = []
            for flux in total_flux:
                if flux > 0:
                    mag = -2.5 * np.log10(flux) + zp_dict[band]
                    magnitudes.append(mag)
                else:
                    magnitudes.append(np.nan)
            
            band_mags[band] = pd.Series(magnitudes)
            print(f"KIDS photometry band {band}: {len(magnitudes)} measurements")
        
        return band_mags
        
    except Exception as e:
        print(f"Error in KIDS aperture photometry for subhalo {subhalo_id}: {e}")
        return {band: pd.Series([np.nan] * len(sn_data)) for band in bands}


def get_photometry_for_subhalo(subhalo_id, sn_data, photometry_type, sim,
                              bands=['g', 'r', 'i', 'z'], kids_results=None,
                              kids_aperture_mode='direct', kids_image_type='noisy',
                              root_path=None, kids_image_pattern=None, kids_snapshot=None,
                              kids_results_dir=None, image_path=None):
    """
    Get photometry for a subhalo using either regular or KIDS data.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID
    sn_data : pandas.DataFrame
        SN position data with 'x_pix', 'y_pix' columns
    photometry_type : str
        'regular' or 'kids'
    sim : str
        Simulation name
    bands : list
        Band names
    kids_results : pandas.DataFrame, optional
        Pre-loaded KIDS results
    kids_aperture_mode : str
        'direct' for aperture photometry on KIDS images,
        'precomputed' for using saved KIDS results
        
    Returns:
    --------
    dict : Photometry results by band
    pandas.DataFrame : Updated SN data with photometry
    """
    sn_data = sn_data.copy()
    
    if photometry_type == 'kids':
        if kids_aperture_mode == 'direct':
            # Perform aperture photometry directly on KIDS images
            band_mags = perform_kids_aperture_photometry(
                subhalo_id, sn_data, sim, bands,
                image_type=kids_image_type,
                root_path=root_path,
                kids_image_pattern=kids_image_pattern,
                kids_snapshot=kids_snapshot,
                kids_results_dir=kids_results_dir
            )
        else:
            # Use pre-computed KIDS photometry results
            if kids_results is None:
                kids_results = load_kids_photometry_results(sim)
            
            if subhalo_id not in kids_results.index:
                print(f"Warning: SubhaloID {subhalo_id} not found in KIDS results")
                # Fill with NaNs
                band_mags = {band: pd.Series([np.nan] * len(sn_data)) for band in bands}
            else:
                # Get KIDS photometry for this subhalo
                kids_data = kids_results.loc[subhalo_id]
                
                # KIDS has the galaxy center photometry, apply to all SNe in this galaxy
                band_mags = {}
                for band in bands:
                    if band in kids_data:
                        # Apply same magnitude to all SNe in this galaxy
                        band_mags[band] = pd.Series([kids_data[band]] * len(sn_data))
                    else:
                        band_mags[band] = pd.Series([np.nan] * len(sn_data))
    
    elif photometry_type == 'regular':
        # Use regular TNG broadband images. Prefer provided resolved image_path.
        if image_path is None:
            # Legacy fallback
            image_path = os.path.join(root_path or ".", f'data/{sim}/{subhalo_id}/broadband_{subhalo_id}.fits')
        
        if not os.path.isfile(image_path):
            print(f"Warning: Regular image not found for subhalo {subhalo_id}")
            band_mags = {band: pd.Series([np.nan] * len(sn_data)) for band in bands}
        else:
            # Perform aperture photometry at SN positions
            positions = sn_data[['x_pix', 'y_pix']].values.tolist()
            
            try:
                band_mags = multi_band_photometry(image_path, positions, bands)
            except Exception as e:
                print(f"Error in photometry for subhalo {subhalo_id}: {e}")
                band_mags = {band: pd.Series([np.nan] * len(sn_data)) for band in bands}
    
    else:
        raise ValueError(f"Unknown photometry type: {photometry_type}")
    
    return band_mags, sn_data


def add_local_colors(sn_data, band_mags):
    """
    Add local color measurements to SN data.
    
    Parameters:
    -----------
    sn_data : pandas.DataFrame
        SN data to update
    band_mags : dict
        Magnitude measurements by band
        
    Returns:
    --------
    pandas.DataFrame : Updated SN data with local colors
    """
    sn_data = sn_data.copy()
    
    # Calculate common colors
    if 'g' in band_mags and 'z' in band_mags:
        sn_data['localrestframe_gz'] = (
            band_mags['g'].values - band_mags['z'].values
        )
    
    if 'g' in band_mags and 'r' in band_mags:
        sn_data['localrestframe_gr'] = (
            band_mags['g'].values - band_mags['r'].values
        )
    
    if 'r' in band_mags and 'i' in band_mags:
        sn_data['localrestframe_ri'] = (
            band_mags['r'].values - band_mags['i'].values
        )
    
    if 'i' in band_mags and 'z' in band_mags:
        sn_data['localrestframe_iz'] = (
            band_mags['i'].values - band_mags['z'].values
        )
    
    # Add individual band magnitudes if desired
    for band, mags in band_mags.items():
        sn_data[f'local_{band}'] = mags.values
    
    return sn_data


def determine_photometry_strategy(subhalo_ids, galmeta, 
                                 mass_threshold=10**9.5,
                                 force_kids=False, 
                                 force_regular=False,
                                 sim='TNG50-1',
                                 kids_results_dir=None):
    """
    Determine photometry strategy for each subhalo and preload KIDS data if needed.
    
    Parameters:
    -----------
    subhalo_ids : array-like
        Subhalo IDs to process
    galmeta : pandas.DataFrame
        Galaxy metadata
    mass_threshold : float
        Mass threshold for KIDS vs regular
    force_kids : bool
        Force all to use KIDS
    force_regular : bool
        Force all to use regular
    sim : str
        Simulation name
        
    Returns:
    --------
    dict : Photometry type for each subhalo
    pandas.DataFrame or None : Preloaded KIDS results if needed
    """
    from .subhalo_selection import get_photometry_type
    
    # Get photometry types
    photometry_types = get_photometry_type(
        subhalo_ids, galmeta, 
        mass_threshold=mass_threshold,
        force_kids=force_kids,
        force_regular=force_regular
    )
    
    # Check if we need KIDS data
    needs_kids = any(ptype == 'kids' for ptype in photometry_types.values())
    
    kids_results = None
    if needs_kids:
        try:
            kids_results = load_kids_photometry_results(sim, results_dir=kids_results_dir)
            print(f"Loaded KIDS results for {len(kids_results)} galaxies")
        except FileNotFoundError as e:
            print(f"Warning: Could not load KIDS results: {e}")
            # Convert all KIDS requests to regular
            for subhalo_id in photometry_types:
                if photometry_types[subhalo_id] == 'kids':
                    photometry_types[subhalo_id] = 'regular'
                    print(f"Switched subhalo {subhalo_id} to regular photometry")
    
    return photometry_types, kids_results


# Legacy functions for backward compatibility
    """
    Perform aperture photometry on an image.
    
    Parameters:
    -----------
    img_path : str
        Path to FITS image file
    snx : float
        x position for single aperture (default: 0)
    sny : float  
        y position for single aperture (default: 0)
    pix_list : list, optional
        List of [x, y] pixel positions for multiple apertures
    pos_list : list, optional
        List of [x, y] physical positions for multiple apertures
    zp : float
        Zero point magnitude (default: 0)
    ax : matplotlib axis, optional
        Axis for plotting
    band : int
        Band index for multi-band FITS file (default: 0)
    plot_colour : str
        Color for plotting apertures (default: 'w')
    plot : bool
        Whether to plot apertures (default: False)
        
    Returns:
    --------
    tuple : (aperture, magnitudes, magnitude_errors)
    """
    # Load image data
    img = fits.getdata(img_path)[band]
    h = fits.getheader(img_path)
    
    # Set up positions
    if not pix_list:
        pix_list = [snx, sny]
    if not pos_list:
        pos_list = [snx, sny]
    
    # Calculate aperture radius in pixels
    pixscale_physical = 1000 / h['CDELT1'] / cosmo.h
    
    # Create aperture
    ap = CircularAperture(pix_list, r=pixscale_physical)
    
    # Perform photometry with error estimation
    phot_df = aperture_photometry(img, ap, error=0.1 * img).to_pandas()
    
    # Convert to magnitudes
    res = convert_phot(phot_df['aperture_sum'], zp)
    res_err = convert_phot_error(phot_df['aperture_sum'], 
                                phot_df['aperture_sum_err'], zp)
    
    return ap, res, res_err


def convert_phot(aperture_sum, zp):
    """
    Convert aperture sum to magnitude.
    
    Parameters:
    -----------
    aperture_sum : array_like
        Aperture photometry sum
    zp : float
        Zero point magnitude
        
    Returns:
    --------
    array_like : Magnitudes
    """
    return -2.5 * np.log10(aperture_sum) + zp


def convert_phot_error(aperture_sum, aperture_sum_err, zp):
    """
    Convert aperture sum error to magnitude error.
    
    Parameters:
    -----------
    aperture_sum : array_like
        Aperture photometry sum
    aperture_sum_err : array_like
        Error in aperture sum
    zp : float
        Zero point magnitude
        
    Returns:
    --------
    array_like : Magnitude errors
    """
    # Magnitude error propagation: dm = 2.5/ln(10) * (df/f)
    return 2.5 / np.log(10) * (aperture_sum_err / aperture_sum)


def convert_phot_fnu(aperture_sum, zp):
    """
    Convert aperture sum to flux density.
    
    Parameters:
    -----------
    aperture_sum : array_like
        Aperture photometry sum
    zp : float
        Zero point magnitude
        
    Returns:
    --------
    array_like : Flux density values
    """
    return 10**(np.log10(aperture_sum) - 0.4 * (zp - 8.9))


def multi_band_photometry(img_path, positions, bands=['g', 'r', 'i', 'z']):
    """
    Perform aperture photometry in multiple bands.
    
    Parameters:
    -----------
    img_path : str
        Path to multi-band FITS file
    positions : list
        List of [x, y] positions for apertures
    bands : list
        List of band names (default: ['g', 'r', 'i', 'z'])
        
    Returns:
    --------
    dict : Dictionary with band names as keys and magnitude arrays as values
    """
    band_mags = {}
    
    for counter, band in enumerate(bands):
        ap, mags, mag_errs = do_aperture_photometry(
            img_path, 
            pix_list=positions,
            band=counter, 
            plot=False
        )
        band_mags[band] = mags
        band_mags[f'{band}_err'] = mag_errs
    
    return band_mags