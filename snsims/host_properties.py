"""
Host Galaxy Properties and Delay Time Distributions

Contains functions for managing host galaxy properties,
delay time distributions, and host selection.
"""

import numpy as np
import pandas as pd


def dtd_powerlaw(age, A=2.11e-13, beta=-1.13, t0=40):
    """
    Power-law delay time distribution.
    
    Parameters:
    -----------
    age : array_like
        Stellar ages in Gyr
    A : float
        Normalization constant (default: 2.11e-13)
    beta : float
        Power-law index (default: -1.13)
    t0 : float
        Minimum delay time in Myr (default: 40)
        
    Returns:
    --------
    array_like : DTD weights
    """
    age_myr = age * 1000  # Convert to Myr
    weights = np.where(age_myr < t0, 0, A * (age ** beta))
    return weights


def get_host_id(ages, dtd_func=dtd_powerlaw, **dtd_kwargs):
    """
    Select host star particle based on delay time distribution.
    
    Parameters:
    -----------
    ages : array_like
        Stellar ages in Gyr
    dtd_func : callable
        Delay time distribution function
    **dtd_kwargs : dict
        Keyword arguments for DTD function
        
    Returns:
    --------
    int : Selected host particle index
    """
    rates = dtd_func(ages, **dtd_kwargs)
    if np.sum(rates) == 0:
        # If no valid hosts, return random index
        return np.random.choice(len(ages))
    
    probabilities = rates / np.sum(rates)
    return np.random.choice(np.arange(len(ages)), p=probabilities)


def get_metallicity(hostId, galaxy_catalog):
    """
    Get metallicity of host particle.
    
    Parameters:
    -----------
    hostId : int
        Host particle index
    galaxy_catalog : pandas.DataFrame
        Galaxy catalog with metallicity column
        
    Returns:
    --------
    float : Host metallicity
    """
    return galaxy_catalog['metallicity'].iloc[hostId]


def get_progenitor_age(hostId, galaxy_catalog):
    """
    Get age of progenitor (host particle).
    
    Parameters:
    -----------
    hostId : int
        Host particle index
    galaxy_catalog : pandas.DataFrame
        Galaxy catalog with age column
        
    Returns:
    --------
    float : Progenitor age in Gyr
    """
    return galaxy_catalog['age'].iloc[hostId]


def create_galaxy_catalog(ages, metallicity, initial_mass, stellar_mass, 
                         V, Ur, gz, x, y):
    """
    Create standardized galaxy catalog DataFrame.
    
    Parameters:
    -----------
    ages : array_like
        Stellar ages in Gyr
    metallicity : array_like
        Stellar metallicities
    initial_mass : array_like
        Initial stellar masses
    stellar_mass : array_like
        Current stellar masses
    V : array_like
        V-band magnitudes
    Ur : array_like
        U-r colors
    gz : array_like
        g-z colors
    x : array_like
        x positions
    y : array_like
        y positions
        
    Returns:
    --------
    pandas.DataFrame : Galaxy catalog
    """
    return pd.DataFrame({
        'age': ages,
        'metallicity': metallicity,
        'initial_mass': initial_mass,
        'mass': stellar_mass,
        'V': V,
        'U-r': Ur,
        'g-z': gz,
        'x_pos': x,
        'y_pos': y
    })


def compute_dtd_weights(ages, masses, A=2.11e-13, beta=-1.13, t0=40):
    """
    Compute DTD weights for mass-weighted sampling.
    
    Parameters:
    -----------
    ages : array_like
        Stellar ages in Gyr
    masses : array_like
        Stellar masses
    A : float
        DTD normalization
    beta : float
        DTD power-law index
    t0 : float
        Minimum delay time in Myr
        
    Returns:
    --------
    array_like : Normalized DTD weights
    """
    # Only compute for valid ages
    valid = (ages * 1000 > t0)
    weights = np.zeros_like(ages)
    weights[valid] = masses[valid] * A * (ages[valid] ** beta)
    
    # Normalize
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    
    return weights