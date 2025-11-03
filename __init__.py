"""
TNGSN: TNG Supernova Simulation Pipeline

A modular Python package for generating supernova populations
from TNG simulation data with support for both regular TNG 
images and KIDS survey data.
"""

__version__ = "1.0.0"
__author__ = "Phil Wiseman"

# Import main components for easy access
try:
    from snsims import TNGSNSimulation, subhalo_selection
    from snsims import sn_luminosity, lc_params, host_properties
    from snsims import morphology, photometry
    
    __all__ = [
        'TNGSNSimulation',
        'subhalo_selection',
        'sn_luminosity',
        'lc_params', 
        'host_properties',
        'morphology',
        'photometry'
    ]
except ImportError:
    # Handle case where dependencies aren't installed yet
    __all__ = []