"""
Light Curve Parameters (x1, c)

Contains functions for generating light curve parameters like 
stretch (x1) and color (c) based on host galaxy properties.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm


def x1_g25_age_metallicity(age, metallicity, params):
    """
    Generate x1 (stretch) parameter based on Ginolin+25 age-metallicity relation.
    
    Parameters:
    -----------
    age : array_like
        Stellar ages in Gyr
    metallicity : array_like
        Stellar metallicities (log(Z/Zsol))
    params : list
        Model parameters [mu1, sig1, mu2, sig2, Km, rred, rblue, Kc, agesplit]
        
    Returns:
    --------
    array_like : x1 stretch parameters
    """
    mu1, sig1, mu2, sig2, Km, rred, rblue, Kc, agesplit = params
    
    # Age-dependent population mixing ratio
    r = rred + ((rblue - rred) * expit((age - agesplit) / Kc))
    
    # Metallicity-dependent x1 distributions
    Zsol = 0  # Solar metallicity reference
    x1low = (Km * (metallicity - Zsol)) + mu2
    x1high = (Km * (metallicity - Zsol)) + mu1
    
    # Create normal distributions
    norm1 = norm(x1high, sig1)
    norm2 = norm(x1low, sig2)
    
    # Random choice between populations
    choice = (np.random.rand(len(age)) > r).astype(int)
    
    # Sample from appropriate distribution
    n1rvs = norm1.rvs()
    n2rvs = norm2.rvs()
    
    x1 = ((choice == 0) * n1rvs) + ((choice == 1) * n2rvs)
    
    return x1


def c_parameter_simple(size, mean=0.0, sigma=0.1):
    """
    Generate simple color parameter from normal distribution.
    
    Parameters:
    -----------
    size : int
        Number of values to generate
    mean : float
        Mean color value (default: 0.0)
    sigma : float
        Standard deviation (default: 0.1)
        
    Returns:
    --------
    array_like : Color parameters
    """
    return np.random.normal(mean, sigma, size)


def x1_simple(size, mean=0.0, sigma=1.0):
    """
    Generate simple stretch parameter from normal distribution.
    
    Parameters:
    -----------
    size : int
        Number of values to generate  
    mean : float
        Mean x1 value (default: 0.0)
    sigma : float
        Standard deviation (default: 1.0)
        
    Returns:
    --------
    array_like : Stretch parameters
    """
    return np.random.normal(mean, sigma, size)


# Default Galbany+24 parameters
G24_AGE_MET_PARAMS = [0.25, 0.55, -1.33, 0.63, -0.3, 0.18, 0.98, -0.128, 5]


def x1_from_host_properties(**kwargs):
    """
    Wrapper function for x1 generation from host properties.
    
    Parameters:
    -----------
    **kwargs : dict
        Must contain 'age', 'metallicity', and 'params' keys
        
    Returns:
    --------
    array_like : x1 stretch parameters
    """
    return x1_g25_age_metallicity(
        kwargs['age'], 
        kwargs['metallicity'], 
        kwargs['params']
    )