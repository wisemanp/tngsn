"""
Galaxy morphology and DLR calculation functions.

Functions for calculating directional light radius (DLR) and handling
both regular TNG morphology data and KIDS-derived morphological parameters.
"""

import numpy as np
import pandas as pd
import h5py
import os


def load_regular_morphology(sim, band='i', morph_dir=None):
    """
    Load regular TNG morphology data from HDF5 files.
    
    Parameters:
    -----------
    sim : str
        Simulation name (e.g., 'TNG50-1')
    band : str
        Band for morphology ('i', 'g', 'r', 'z')
    morph_dir : str, optional
        Directory containing morphology files
        
    Returns:
    --------
    pandas.DataFrame : Morphology data with standardized column names
    """
    if morph_dir is None:
        morph_dir = f'data/{sim}'
    
    morph_file = os.path.join(morph_dir, f'morphs_{band}.hdf5')
    
    if not os.path.exists(morph_file):
        raise FileNotFoundError(f"Morphology file not found: {morph_file}")
    
    with h5py.File(morph_file, 'r') as f:
        morph_data = {
            'subfind_id': np.array(f['subfind_id']),
            'elongation_asymmetry': np.array(f['elongation_asymmetry']),
            'ellipticity_asymmetry': np.array(f['ellipticity_asymmetry']),
            'orientation_asymmetry': np.array(f['orientation_asymmetry']),
            'rhalf_ellip': np.array(f['rhalf_ellip']),
            'sersic_ellip': np.array(f['sersic_ellip']),
            'sersic_n': np.array(f['sersic_n']),
            'sersic_rhalf': np.array(f['sersic_rhalf']),
            'sersic_theta': np.array(f['sersic_theta'])
        }
    
    morphology_df = pd.DataFrame(morph_data)
    morphology_df = morphology_df.set_index('subfind_id')
    
    return morphology_df


def load_kids_morphology(sim, results_dir=None):
    """
    Load KIDS morphology data from photometry results.
    
    Parameters:
    -----------
    sim : str
        Simulation name
    results_dir : str, optional
        Directory containing KIDS results
        
    Returns:
    --------
    pandas.DataFrame : Morphology data with standardized column names
    """
    if results_dir is None:
        results_dir = f'data/{sim}/KIDS/results'
    
    results_file = os.path.join(results_dir, 'photometry_results.csv')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"KIDS results not found: {results_file}")
    
    kids_results = pd.read_csv(results_file)
    kids_results = kids_results.set_index('subhaloId')
    
    # Extract morphological parameters from KIDS results
    # Convert from KIDS format to standard format
    morphology_data = {}
    
    if 'a' in kids_results.columns and 'b' in kids_results.columns:
        # Calculate elongation from semi-major/minor axes
        morphology_data['elongation_asymmetry'] = kids_results['a'] / kids_results['b']
        
        # Use rhalf_ellip as the geometric mean of a and b
        morphology_data['rhalf_ellip'] = np.sqrt(kids_results['a'] * kids_results['b'])
        
        # Ellipticity = 1 - b/a
        morphology_data['ellipticity_asymmetry'] = 1 - (kids_results['b'] / kids_results['a'])
    else:
        # Fill with default values if not available
        morphology_data['elongation_asymmetry'] = 1.0
        morphology_data['rhalf_ellip'] = 1.0
        morphology_data['ellipticity_asymmetry'] = 0.0
    
    if 'theta' in kids_results.columns:
        morphology_data['orientation_asymmetry'] = kids_results['theta']
    else:
        morphology_data['orientation_asymmetry'] = 0.0
    
    # Set default values for parameters not measured by KIDS
    n_galaxies = len(kids_results)
    morphology_data.update({
        'sersic_ellip': np.full(n_galaxies, np.nan),
        'sersic_n': np.full(n_galaxies, np.nan),
        'sersic_rhalf': np.full(n_galaxies, np.nan),
        'sersic_theta': np.full(n_galaxies, np.nan)
    })
    
    morphology_df = pd.DataFrame(morphology_data, index=kids_results.index)
    
    return morphology_df


def get_morphology_for_subhalo(subhalo_id, morphology_type, sim, 
                              regular_morphs=None, kids_morphs=None):
    """
    Get morphological parameters for a subhalo.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID
    morphology_type : str
        'regular' or 'kids'
    sim : str
        Simulation name
    regular_morphs : pandas.DataFrame, optional
        Pre-loaded regular morphology data
    kids_morphs : pandas.DataFrame, optional
        Pre-loaded KIDS morphology data
        
    Returns:
    --------
    dict : Morphological parameters
    """
    if morphology_type == 'regular':
        if regular_morphs is None:
            regular_morphs = load_regular_morphology(sim)
        
        if subhalo_id not in regular_morphs.index:
            print(f"Warning: SubhaloID {subhalo_id} not found in regular morphology")
            return None
        
        return regular_morphs.loc[subhalo_id].to_dict()
    
    elif morphology_type == 'kids':
        if kids_morphs is None:
            kids_morphs = load_kids_morphology(sim)
        
        if subhalo_id not in kids_morphs.index:
            print(f"Warning: SubhaloID {subhalo_id} not found in KIDS morphology")
            return None
        
        return kids_morphs.loc[subhalo_id].to_dict()
    
    else:
        raise ValueError(f"Unknown morphology type: {morphology_type}")


def calculate_ellipse_params(rhalf_ellip, elongation_asymmetry, correction=1/0.68):
    """
    Calculate ellipse parameters A_IMAGE and B_IMAGE from morphological data.
    
    Parameters:
    -----------
    rhalf_ellip : float
        Half-light radius along ellipse
    elongation_asymmetry : float
        Elongation (semi-major / semi-minor axis ratio)
    correction : float
        Correction factor for radius (default: 1/0.68)
        
    Returns:
    --------
    tuple : (A_IMAGE, B_IMAGE) - semi-major and semi-minor axis lengths
    """
    A_IMAGE = rhalf_ellip * correction
    B_IMAGE = rhalf_ellip * correction / elongation_asymmetry
    
    return A_IMAGE, B_IMAGE


def get_DLR_ABT(x_pos, y_pos, A_IMAGE, B_IMAGE, orientation_ellip, angsep):
    """
    Calculate the Directional Light Radius (DLR) for galaxy-SN pairs.
    
    This function calculates DLR following the DES-SN analysis approach.
    
    Parameters:
    -----------
    x_pos : float or array
        X position of SN relative to galaxy center
    y_pos : float or array  
        Y position of SN relative to galaxy center
    A_IMAGE : float
        Semi-major axis length
    B_IMAGE : float
        Semi-minor axis length
    orientation_ellip : float
        Orientation angle of galaxy ellipse (radians)
    angsep : float or array
        Angular separation between SN and galaxy center
        
    Returns:
    --------
    float or array : DLR values
    """
    # Angle between RA-axis and SN-host vector
    GAMMA = np.arctan2(-y_pos, x_pos)  # Use arctan2 for proper quadrant
    
    # Angle between semi-major axis of host and SN-host vector
    PHI = orientation_ellip + GAMMA
    
    # Calculate directional radius
    rPHI = A_IMAGE * B_IMAGE / np.sqrt(
        (A_IMAGE * np.sin(PHI))**2 + (B_IMAGE * np.cos(PHI))**2
    )
    
    # Calculate DLR
    d_DLR = angsep / rPHI
    
    return d_DLR


def safe_DLR_calculation(x_pos, y_pos, A_IMAGE, B_IMAGE, 
                        orientation_ellip, angsep, default_value=99.99):
    """
    Safely calculate DLR with error handling for bad morphological data.
    
    Parameters:
    -----------
    x_pos : array
        X positions
    y_pos : array
        Y positions  
    A_IMAGE : float
        Semi-major axis
    B_IMAGE : float
        Semi-minor axis
    orientation_ellip : float
        Orientation angle
    angsep : array
        Angular separations
    default_value : float
        Default value for failed calculations
        
    Returns:
    --------
    array : DLR values with fallback for bad data
    """
    try:
        # Check for valid morphological parameters
        if (np.isnan(A_IMAGE) or np.isnan(B_IMAGE) or 
            np.isnan(orientation_ellip) or 
            A_IMAGE <= 0 or B_IMAGE <= 0):
            return np.full_like(angsep, default_value)
        
        d_DLR = get_DLR_ABT(x_pos, y_pos, A_IMAGE, B_IMAGE, 
                           orientation_ellip, angsep)
        
        # Replace any NaN or infinite values
        d_DLR = np.where(np.isfinite(d_DLR), d_DLR, default_value)
        
        return d_DLR
        
    except Exception as e:
        print(f"Error calculating DLR: {e}")
        return np.full_like(angsep, default_value)


def preload_morphology_data(subhalo_ids, photometry_types, sim):
    """
    Preload morphology data for efficient processing.
    
    Parameters:
    -----------
    subhalo_ids : array-like
        Subhalo IDs to process
    photometry_types : dict
        Mapping of subhalo_id -> photometry type
    sim : str
        Simulation name
        
    Returns:
    --------
    dict : Preloaded morphology data by type
    """
    morphology_data = {'regular': None, 'kids': None}
    
    # Check what types of morphology we need
    needs_regular = any(ptype == 'regular' for ptype in photometry_types.values())
    needs_kids = any(ptype == 'kids' for ptype in photometry_types.values())
    
    if needs_regular:
        try:
            morphology_data['regular'] = load_regular_morphology(sim)
            print(f"Loaded regular morphology for {len(morphology_data['regular'])} galaxies")
        except FileNotFoundError as e:
            print(f"Warning: Could not load regular morphology: {e}")
    
    if needs_kids:
        try:
            morphology_data['kids'] = load_kids_morphology(sim)
            print(f"Loaded KIDS morphology for {len(morphology_data['kids'])} galaxies")
        except FileNotFoundError as e:
            print(f"Warning: Could not load KIDS morphology: {e}")
    
    return morphology_data


def calculate_morphology_for_subhalo(subhalo_id, sn_data, morphology_type, 
                                   sim, morphology_data=None, 
                                   rhalf_correction=1/0.68, 
                                   dlr_default=99.99):
    """
    Calculate morphological properties for SNe in a subhalo.
    
    Parameters:
    -----------
    subhalo_id : int
        Subhalo ID
    sn_data : pandas.DataFrame
        SN data with position information
    morphology_type : str
        'regular' or 'kids'
    sim : str
        Simulation name
    morphology_data : dict, optional
        Pre-loaded morphology data
    rhalf_correction : float
        Correction factor for half-light radius
    dlr_default : float
        Default DLR value for failed calculations
        
    Returns:
    --------
    pandas.DataFrame : Updated SN data with DLR measurements
    """
    sn_data = sn_data.copy()
    
    # Get morphological parameters
    if morphology_data and morphology_type in morphology_data:
        morphs = morphology_data[morphology_type]
    else:
        morphs = None
    
    morph_params = get_morphology_for_subhalo(
        subhalo_id, morphology_type, sim, 
        regular_morphs=morphs if morphology_type == 'regular' else None,
        kids_morphs=morphs if morphology_type == 'kids' else None
    )
    
    if morph_params is None:
        sn_data['d_DLR'] = dlr_default
        return sn_data
    
    # Calculate ellipse parameters
    A_IMAGE, B_IMAGE = calculate_ellipse_params(
        morph_params['rhalf_ellip'],
        morph_params['elongation_asymmetry'],
        rhalf_correction
    )
    
    # Calculate DLR for all SNe in this subhalo
    sn_data['d_DLR'] = safe_DLR_calculation(
        sn_data['x_pos'].values,
        sn_data['y_pos'].values,
        A_IMAGE,
        B_IMAGE,
        morph_params['orientation_asymmetry'],
        sn_data['r_pix'].values,
        dlr_default
    )
    
    return sn_data


def calculate_ellipse_params(rhalf_ellip, elongation_asymmetry, 
                           correction_factor=1.0/0.68):
    """
    Calculate ellipse parameters from morphological measurements.
    
    Parameters:
    -----------
    rhalf_ellip : float
        Half-light radius of ellipse
    elongation_asymmetry : float
        Elongation (axis ratio) of ellipse
    correction_factor : float
        Correction factor for half-light radius (default: 1/0.68)
        
    Returns:
    --------
    tuple : (A_IMAGE, B_IMAGE) semi-major and semi-minor axes
    """
    A_IMAGE = rhalf_ellip * correction_factor
    B_IMAGE = A_IMAGE / elongation_asymmetry
    
    return A_IMAGE, B_IMAGE


def angles_debug(x_pos, y_pos, orientation_ellip):
    """
    Debug function to print angle calculations.
    
    Parameters:
    -----------
    x_pos : float
        x position of SN relative to host center
    y_pos : float
        y position of SN relative to host center  
    orientation_ellip : float
        Orientation angle of galaxy ellipse (radians)
    """
    # Angle between RA-axis and SN-host vector
    GAMMA = np.arctan(-y_pos / x_pos)
    
    # Angle between semi-major axis of host and SN-host vector
    PHI = orientation_ellip + GAMMA
    
    print(f"GAMMA (rad): {GAMMA}, PHI (rad): {PHI}")
    print(f"GAMMA (deg): {np.rad2deg(GAMMA)}, PHI (deg): {np.rad2deg(PHI)}")


def safe_DLR_calculation(x_pos, y_pos, A_IMAGE, B_IMAGE, 
                        orientation_ellip, angsep, default_value=99.99):
    """
    Calculate DLR with error handling for bad morphological measurements.
    
    Parameters:
    -----------
    x_pos : array_like
        x position of SN relative to host center
    y_pos : array_like
        y position of SN relative to host center
    A_IMAGE : float
        Semi-major axis length of host galaxy
    B_IMAGE : float
        Semi-minor axis length of host galaxy
    orientation_ellip : float
        Orientation angle of galaxy ellipse (radians)
    angsep : array_like
        Angular separation between SN and host center
    default_value : float
        Default value for bad measurements (default: 99.99)
        
    Returns:
    --------
    array_like : DLR values (with default_value for bad measurements)
    """
    try:
        # Check for valid inputs
        if (np.isnan(A_IMAGE) or np.isnan(B_IMAGE) or 
            A_IMAGE <= 0 or B_IMAGE <= 0):
            return np.full_like(angsep, default_value)
        
        return get_DLR_ABT(x_pos, y_pos, A_IMAGE, B_IMAGE, 
                          orientation_ellip, angsep)
    
    except (ZeroDivisionError, ValueError, RuntimeWarning):
        return np.full_like(angsep, default_value)