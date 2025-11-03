"""
SN Luminosity and Step Functions

Contains functions related to supernova absolute magnitudes, 
luminosity steps, and magnitude calculations.
"""

import numpy as np
from scipy.special import expit
from astropy.cosmology import Planck18 as default_cosmo


def age_step(age, agesplit=3.0, Ka=-0.128, step_high=0.1, step_low=-0.1, mabs_base=-19.3):
    """
    Calculate absolute magnitude based on age step function.
    
    Parameters:
    -----------
    age : array_like
        Stellar ages in Gyr
    agesplit : float
        Age threshold for step transition (default: 3.0 Gyr)
    Ka : float
        Step transition steepness parameter (default: -0.128)
    step_high : float
        Magnitude step for young populations (default: 0.1)
    step_low : float  
        Magnitude step for old populations (default: -0.1)
    mabs_base : float
        Base absolute magnitude (default: -19.3)
        
    Returns:
    --------
    array_like : Absolute magnitudes
    """
    r = expit((age - agesplit) / Ka)
    choice = (np.random.rand(len(age)) > r).astype(int)
    step = ((choice == 0) * step_high) + ((choice == 1) * step_low)
    magabs = mabs_base + step
    return magabs


def distmod(mobs, x1, c, mabs=-19.3, alpha=-0.14, beta=3.15):
    """
    Calculate distance modulus from observed parameters.
    
    Parameters:
    -----------
    mobs : array_like
        Observed peak magnitude
    x1 : array_like
        Light curve stretch parameter
    c : array_like
        Light curve color parameter
    mabs : float
        Absolute magnitude (default: -19.3)
    alpha : float
        Stretch correction coefficient (default: -0.14)
    beta : float
        Color correction coefficient (default: 3.15)
        
    Returns:
    --------
    array_like : Distance modulus values
    """
    return mobs - mabs - (alpha * x1 + beta * c)


def hubble_residual(distmod, z, cosmo=None):
    """
    Calculate Hubble residual.
    
    Parameters:
    -----------
    distmod : array_like
        Distance modulus values
    z : array_like
        Redshift values
    cosmo : astropy.cosmology object, optional
        Cosmology to use (default: Planck18)
        
    Returns:
    --------
    array_like : Hubble residual values
    """
    if cosmo is None:
        cosmo = default_cosmo
    return distmod - cosmo.distmod(z).value


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