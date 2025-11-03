"""
SNSims: TNG Supernova Simulation Package

A modular package for generating supernova populations from TNG simulation data.
"""

from .simulation import TNGSNSimulation
from . import sn_luminosity
from . import lc_params
from . import host_properties
from . import morphology
from . import photometry
from . import subhalo_selection

__version__ = "1.0.0"
__author__ = "TNG-SN Team"

# Main simulation class
__all__ = [
    'TNGSNSimulation',
    'sn_luminosity',
    'lc_params', 
    'host_properties',
    'morphology',
    'photometry',
    'subhalo_selection'
]

# Convenience functions
def load_default_config():
    """Load the default configuration for simulations."""
    import os
    import yaml
    
    config_path = os.path.join(
        os.path.dirname(__file__), 
        'config', 
        'default_config.yaml'
    )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_simulation(config_file=None, **kwargs):
    """
    Convenience function to create a TNGSNSimulation instance.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to configuration YAML file
    **kwargs : dict
        Configuration overrides
        
    Returns:
    --------
    TNGSNSimulation : Configured simulation instance
    """
    return TNGSNSimulation(config_file=config_file, **kwargs)
