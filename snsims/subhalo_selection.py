"""
Subhalo selection utilities for TNG simulations.

Functions for selecting subhalos based on stellar mass distributions
and other galaxy properties.
"""

import numpy as np
import pandas as pd
from astropy.cosmology import Planck15 as cosmo


def select_subhalos_by_mass(galmeta, n_subhalos=None, mass_distribution='uniform',
                           min_mass=1e8, max_mass=None, mass_bins=None, 
                           seed=None, selection_weights=None):
    """
    Select subhalos based on stellar mass distribution.
    
    Parameters:
    -----------
    galmeta : pandas.DataFrame
        Galaxy metadata with 'mass_stars' column
    n_subhalos : int, optional
        Number of subhalos to select. If None, use all available
    mass_distribution : str or array-like
        'uniform': uniform sampling over mass range
        'log_uniform': uniform sampling in log-mass space
        'observed': try to match observed mass function
        array: custom weights for each galaxy
    min_mass : float
        Minimum stellar mass in solar masses (default: 1e8)
    max_mass : float, optional
        Maximum stellar mass in solar masses. If None, use max available
    mass_bins : int or array-like, optional
        For binned sampling. If int, number of mass bins.
        If array, mass bin edges
    seed : int, optional
        Random seed for reproducible selection
    selection_weights : array-like, optional
        Custom weights for each galaxy in galmeta
        
    Returns:
    --------
    pandas.DataFrame : Selected subhalo metadata
    array : Selected subhalo IDs
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate true stellar masses
    if 'mass_stars_true' not in galmeta.columns:
        galmeta = galmeta.copy()
        galmeta['mass_stars_true'] = galmeta['mass_stars'] * 1e10 / cosmo.h
    
    # Apply mass cuts
    if max_mass is None:
        max_mass = galmeta['mass_stars_true'].max()
    
    mass_mask = (galmeta['mass_stars_true'] >= min_mass) & (galmeta['mass_stars_true'] <= max_mass)
    galmeta_filtered = galmeta[mass_mask]
    
    if len(galmeta_filtered) == 0:
        raise ValueError(f"No galaxies found in mass range [{min_mass:.1e}, {max_mass:.1e}]")
    
    # Default to all available galaxies if n_subhalos not specified
    if n_subhalos is None:
        n_subhalos = len(galmeta_filtered)
    elif n_subhalos > len(galmeta_filtered):
        print(f"Warning: Requested {n_subhalos} subhalos, but only {len(galmeta_filtered)} available")
        n_subhalos = len(galmeta_filtered)
    
    # Calculate selection weights
    if isinstance(mass_distribution, str):
        if mass_distribution == 'uniform':
            # Uniform sampling - all galaxies equal weight
            weights = np.ones(len(galmeta_filtered))
            
        elif mass_distribution == 'log_uniform':
            # Uniform in log-mass space
            # Weight inversely proportional to local density in log-mass
            log_masses = np.log10(galmeta_filtered['mass_stars_true'])
            hist, bin_edges = np.histogram(log_masses, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find which bin each galaxy belongs to
            bin_indices = np.digitize(log_masses, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
            
            # Weight inversely proportional to bin density
            weights = 1.0 / (hist[bin_indices] + 1)  # +1 to avoid divide by zero
            
        elif mass_distribution == 'observed':
            # Try to match observed galaxy mass function
            # Use Schechter function approximation
            masses = galmeta_filtered['mass_stars_true']
            log_masses = np.log10(masses)
            
            # Approximate Schechter function parameters
            M_star = 1e11  # Characteristic mass
            alpha = -1.1   # Faint-end slope
            
            weights = (masses / M_star) ** alpha * np.exp(-masses / M_star)
            
        elif mass_distribution == 'mass_weighted':
            # Weight by stellar mass (favor more massive galaxies)
            weights = galmeta_filtered['mass_stars_true']
            
        else:
            raise ValueError(f"Unknown mass distribution: {mass_distribution}")
            
    elif hasattr(mass_distribution, '__len__'):
        # Custom weight array
        if len(mass_distribution) != len(galmeta_filtered):
            raise ValueError("Custom weights must match number of filtered galaxies")
        weights = np.array(mass_distribution)
    else:
        raise ValueError("mass_distribution must be string or array-like")
    
    # Apply additional selection weights if provided
    if selection_weights is not None:
        if len(selection_weights) != len(galmeta):
            raise ValueError("selection_weights must match original galmeta length")
        # Extract weights for filtered galaxies
        filtered_weights = selection_weights[mass_mask]
        weights = weights * filtered_weights
    
    # Handle binned sampling
    if mass_bins is not None:
        masses = galmeta_filtered['mass_stars_true']
        
        if isinstance(mass_bins, int):
            # Create mass bins
            bin_edges = np.logspace(np.log10(min_mass), np.log10(max_mass), mass_bins + 1)
        else:
            # Use provided bin edges
            bin_edges = np.array(mass_bins)
        
        # Sample from each bin
        selected_indices = []
        bin_indices = np.digitize(masses, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        
        # Calculate number per bin
        n_per_bin = max(1, n_subhalos // len(bin_edges) - 1)
        
        for i in range(len(bin_edges) - 1):
            bin_mask = bin_indices == i
            if not np.any(bin_mask):
                continue
                
            bin_weights = weights[bin_mask]
            bin_weights = bin_weights / bin_weights.sum()
            
            bin_indices_filtered = np.where(bin_mask)[0]
            n_bin_select = min(n_per_bin, len(bin_indices_filtered))
            
            selected_bin = np.random.choice(
                bin_indices_filtered,
                size=n_bin_select,
                replace=False,
                p=bin_weights
            )
            selected_indices.extend(selected_bin)
        
        # If we haven't selected enough, fill randomly from remaining
        if len(selected_indices) < n_subhalos:
            remaining = n_subhalos - len(selected_indices)
            all_indices = np.arange(len(galmeta_filtered))
            available_indices = np.setdiff1d(all_indices, selected_indices)
            
            if len(available_indices) > 0:
                remaining_weights = weights[available_indices]
                remaining_weights = remaining_weights / remaining_weights.sum()
                
                additional_selected = np.random.choice(
                    available_indices,
                    size=min(remaining, len(available_indices)),
                    replace=False,
                    p=remaining_weights
                )
                selected_indices.extend(additional_selected)
        
        selected_indices = np.array(selected_indices)
        
    else:
        # Standard weighted sampling
        weights = weights / weights.sum()  # Normalize
        
        selected_indices = np.random.choice(
            len(galmeta_filtered),
            size=n_subhalos,
            replace=False,
            p=weights
        )
    
    # Get selected galaxies
    selected_galmeta = galmeta_filtered.iloc[selected_indices]
    selected_ids = selected_galmeta.index.values
    
    return selected_galmeta, selected_ids


def get_photometry_type(subhalo_ids, galmeta, mass_threshold=10**9.5, 
                       force_kids=False, force_regular=False):
    """
    Determine whether to use KIDS or regular photometry for each subhalo.
    
    Parameters:
    -----------
    subhalo_ids : array-like
        Subhalo IDs to classify
    galmeta : pandas.DataFrame
        Galaxy metadata with mass information
    mass_threshold : float
        Mass threshold for switching between photometry types (default: 10^9.5)
    force_kids : bool
        Force all galaxies to use KIDS photometry
    force_regular : bool
        Force all galaxies to use regular photometry
        
    Returns:
    --------
    dict : Mapping of subhalo_id -> 'kids' or 'regular'
    """
    if force_kids and force_regular:
        raise ValueError("Cannot force both KIDS and regular photometry")
    
    photometry_type = {}
    
    # Calculate true stellar masses if needed
    if 'mass_stars_true' not in galmeta.columns:
        galmeta = galmeta.copy()
        galmeta['mass_stars_true'] = galmeta['mass_stars'] * 1e10 / cosmo.h
    
    for subhalo_id in subhalo_ids:
        if force_kids:
            photometry_type[subhalo_id] = 'kids'
        elif force_regular:
            photometry_type[subhalo_id] = 'regular'
        else:
            # Use mass threshold
            if subhalo_id in galmeta.index:
                mass = galmeta.loc[subhalo_id, 'mass_stars_true']
                if mass < mass_threshold:
                    photometry_type[subhalo_id] = 'kids'
                else:
                    photometry_type[subhalo_id] = 'regular'
            else:
                # Default to regular if mass unknown
                photometry_type[subhalo_id] = 'regular'
    
    return photometry_type


def validate_subhalo_selection(selected_galmeta, target_distribution=None):
    """
    Validate that selected subhalos match target mass distribution.
    
    Parameters:
    -----------
    selected_galmeta : pandas.DataFrame
        Selected galaxy metadata
    target_distribution : str, optional
        Target distribution to validate against
        
    Returns:
    --------
    dict : Validation metrics
    """
    masses = selected_galmeta['mass_stars_true']
    log_masses = np.log10(masses)
    
    metrics = {
        'n_selected': len(selected_galmeta),
        'mass_range': (masses.min(), masses.max()),
        'log_mass_range': (log_masses.min(), log_masses.max()),
        'mean_mass': masses.mean(),
        'median_mass': masses.median(),
        'mass_std': masses.std(),
        'log_mass_mean': log_masses.mean(),
        'log_mass_std': log_masses.std()
    }
    
    # Add distribution-specific metrics
    if target_distribution == 'uniform':
        # Check for uniformity in linear space
        from scipy.stats import kstest
        uniform_masses = np.random.uniform(masses.min(), masses.max(), len(masses))
        ks_stat, ks_p = kstest(masses, uniform_masses)
        metrics['uniformity_ks_stat'] = ks_stat
        metrics['uniformity_p_value'] = ks_p
        
    elif target_distribution == 'log_uniform':
        # Check for uniformity in log space
        from scipy.stats import kstest
        log_uniform = np.random.uniform(log_masses.min(), log_masses.max(), len(log_masses))
        ks_stat, ks_p = kstest(log_masses, log_uniform)
        metrics['log_uniformity_ks_stat'] = ks_stat
        metrics['log_uniformity_p_value'] = ks_p
    
    return metrics


def create_mass_stratified_sample(galmeta, n_total, n_bins=5, 
                                 min_mass=1e8, max_mass=None):
    """
    Create a stratified sample across mass bins.
    
    Parameters:
    -----------
    galmeta : pandas.DataFrame
        Galaxy metadata
    n_total : int
        Total number of galaxies to select
    n_bins : int
        Number of mass bins
    min_mass : float
        Minimum mass
    max_mass : float, optional
        Maximum mass
        
    Returns:
    --------
    pandas.DataFrame : Stratified sample
    array : Selected subhalo IDs
    """
    # Calculate true stellar masses
    if 'mass_stars_true' not in galmeta.columns:
        galmeta = galmeta.copy()
        galmeta['mass_stars_true'] = galmeta['mass_stars'] * 1e10 / cosmo.h
    
    if max_mass is None:
        max_mass = galmeta['mass_stars_true'].max()
    
    # Create log-spaced mass bins
    mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), n_bins + 1)
    
    selected_galmeta, selected_ids = select_subhalos_by_mass(
        galmeta, 
        n_subhalos=n_total,
        mass_distribution='uniform',
        min_mass=min_mass,
        max_mass=max_mass,
        mass_bins=mass_bins
    )
    
    return selected_galmeta, selected_ids