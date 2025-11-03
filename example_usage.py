#!/usr/bin/env python3
"""
Example script showing how to use the new modular TNGSNSimulation.

This demonstrates the transition from the old monolithic script
to the new class-based modular approach with subhalo selection
and dual photometry systems.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from snsims import TNGSNSimulation, subhalo_selection

def main():
    """Run example simulation."""
    
    # Example 1: Default configuration with automatic photometry selection
    print("Example 1: Default configuration with auto photometry")
    sim = TNGSNSimulation()
    
    # Run on a small subset
    results = sim.run_simulation(n_subhalos=5)
    print(f"Generated {len(results)} SNe")
    
    # Example 2: Custom mass distribution and photometry settings
    print("\nExample 2: Custom mass distribution and KIDS photometry")
    sim2 = TNGSNSimulation(
        simulation__name='TNG50-1',
        simulation__min_stellar_mass=1e8,
        simulation__max_stellar_mass=1e10,
        subhalo_selection__method='log_uniform',
        subhalo_selection__selection_seed=123,
        photometry__mode='kids',
        morphology__mode='kids'
    )
    
    # Process specific subhalos
    specific_ids = [102683, 102685, 102691]
    results2 = sim2.run_simulation(subhalo_ids=specific_ids)
    print(f"Generated {len(results2)} SNe from specific subhalos")
    
    # Example 3: Mass-stratified sampling
    print("\nExample 3: Mass-stratified sampling")
    sim3 = TNGSNSimulation(
        subhalo_selection__method='uniform',
        subhalo_selection__mass_bins=5,
        photometry__mode='auto',
        photometry__mass_threshold=1e9.5
    )
    
    results3 = sim3.run_simulation(n_subhalos=20)
    print(f"Generated {len(results3)} SNe with stratified sampling")
    
    # Example 4: Using subhalo selection functions directly
    print("\nExample 4: Direct subhalo selection")
    
    # Load metadata
    import pandas as pd
    galmeta = pd.read_csv('data/TNG50-1/galmeta.csv', index_col=0)
    
    # Select subhalos with different distributions
    selected_uniform, ids_uniform = subhalo_selection.select_subhalos_by_mass(
        galmeta, n_subhalos=10, mass_distribution='uniform',
        min_mass=1e8, max_mass=1e10
    )
    
    selected_log, ids_log = subhalo_selection.select_subhalos_by_mass(
        galmeta, n_subhalos=10, mass_distribution='log_uniform',
        min_mass=1e8, max_mass=1e10
    )
    
    print(f"Uniform selection: {len(ids_uniform)} subhalos")
    print(f"Log-uniform selection: {len(ids_log)} subhalos")
    
    # Example 5: Using custom YAML configuration
    print("\nExample 5: Custom YAML config with dual photometry")
    
    # Create a custom config file
    custom_config = """
simulation:
  name: TNG50-1
  min_stellar_mass: 1e8
  max_stellar_mass: 1e10
  output_dir: example_output
  n_cores: 2

subhalo_selection:
  method: observed  # Try to match observed mass function
  selection_seed: 456

photometry:
  mode: auto  # Automatic selection based on mass
  mass_threshold: 1e9.5  # KIDS below this, regular above
  bands: [g, r, i, z]

morphology:
  mode: auto  # Use same as photometry
  rhalf_correction: 1.470588

dtd:
  A: 2.6e-13
  beta: -1.0
  t0: 0.04

light_curve:
  age_step:
    tp: 0.04
    alpha: -1.0
"""
    
    with open('example_config.yaml', 'w') as f:
        f.write(custom_config)
    
    sim5 = TNGSNSimulation(config_file='example_config.yaml')
    print("Loaded custom configuration successfully")
    
    # This would automatically use KIDS for low-mass galaxies
    # and regular photometry for high-mass galaxies
    print("Configuration uses automatic photometry selection:")
    print(f"  - KIDS for galaxies < {sim5.config['photometry']['mass_threshold']:.1e} M☉")
    print(f"  - Regular for galaxies ≥ {sim5.config['photometry']['mass_threshold']:.1e} M☉")
    
    # Clean up
    os.remove('example_config.yaml')
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()