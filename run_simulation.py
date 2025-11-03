#!/usr/bin/env python3
"""
Command line interface for TNGSNSimulation.

This provides a convenient command line tool for running simulations
with YAML configuration files or command line overrides.
"""

import argparse
import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snsims import TNGSNSimulation

def main():
    """Main command line interface."""
    
    parser = argparse.ArgumentParser(
        description='TNG Supernova Simulation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings on TNG50-1
  python run_simulation.py
  
  # Use custom configuration file
  python run_simulation.py --config my_config.yaml
  
  # Override specific parameters
  python run_simulation.py --simulation TNG100-1 --n-cores 4
  
  # Process specific subhalos
  python run_simulation.py --subhalo-ids 102683 102685 102691
  
  # Run on first 100 subhalos
  python run_simulation.py --n-subhalos 100
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', 
        help='Path to YAML configuration file'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--simulation', 
        default='TNG50-1',
        choices=['TNG50-1', 'TNG100-1', 'TNG300-1'],
        help='TNG simulation to use (default: TNG50-1)'
    )
    
    parser.add_argument(
        '--n-subhalos', 
        type=int,
        help='Number of subhalos to process (from start of filtered catalog)'
    )
    
    parser.add_argument(
        '--subhalo-ids', 
        type=int, 
        nargs='+',
        help='Specific subhalo IDs to process'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='simout',
        help='Output directory for simulation results (default: simout)'
    )
    
    parser.add_argument(
        '--n-cores', 
        type=int,
        help='Number of CPU cores to use for multiprocessing'
    )
    
    # DTD parameters
    parser.add_argument(
        '--dtd-beta', 
        type=float,
        help='DTD power law slope (default: -1.13)'
    )
    
    parser.add_argument(
        '--dtd-A', 
        type=float,
        help='DTD normalization (default: 2.6e-13)'
    )
    
    # Light curve parameters
    parser.add_argument(
        '--age-step-tp', 
        type=float,
        help='Age step transition point in Gyr (default: 0.04)'
    )
    
    parser.add_argument(
        '--age-step-alpha', 
        type=float,
        help='Age step slope parameter (default: -1.13)'
    )
    
    # Subhalo selection
    parser.add_argument(
        '--mass-distribution',
        choices=['uniform', 'log_uniform', 'observed', 'mass_weighted'],
        help='Mass distribution for subhalo selection (default: uniform)'
    )
    
    parser.add_argument(
        '--max-mass',
        type=float,
        help='Maximum stellar mass in solar masses'
    )
    
    parser.add_argument(
        '--selection-seed',
        type=int,
        help='Random seed for subhalo selection'
    )
    
    # Photometry options
    parser.add_argument(
        '--photometry-mode',
        choices=['auto', 'kids', 'regular'],
        help='Photometry mode (default: auto)'
    )
    
    parser.add_argument(
        '--force-kids',
        action='store_true',
        help='Force all galaxies to use KIDS photometry'
    )
    
    parser.add_argument(
        '--force-regular',
        action='store_true',
        help='Force all galaxies to use regular photometry'
    )
    
    parser.add_argument(
        '--mass-threshold',
        type=float,
        help='Mass threshold for KIDS vs regular photometry (default: 1e9.5)'
    )
    
    parser.add_argument(
        '--kids-aperture-mode',
        choices=['direct', 'precomputed'],
        help='KIDS photometry mode: direct aperture photometry or use precomputed results'
    )
    
    parser.add_argument(
        '--kids-image-type',
        choices=['original', 'noisy'],
        help='Type of KIDS images to use (default: noisy)'
    )
    
    # Mass cuts
    parser.add_argument(
        '--min-mass', 
        type=float,
        help='Minimum stellar mass in solar masses (default: 1e8)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_overrides = {}
    
    if args.simulation:
        config_overrides['simulation.name'] = args.simulation
    if args.output_dir:
        config_overrides['simulation.output_dir'] = args.output_dir
    if args.n_cores:
        config_overrides['simulation.n_cores'] = args.n_cores
    if args.min_mass:
        config_overrides['simulation.min_stellar_mass'] = args.min_mass
    if args.max_mass:
        config_overrides['simulation.max_stellar_mass'] = args.max_mass
        
    # Subhalo selection overrides
    if args.mass_distribution:
        config_overrides['subhalo_selection.method'] = args.mass_distribution
    if args.selection_seed:
        config_overrides['subhalo_selection.selection_seed'] = args.selection_seed
        
    # Photometry overrides
    if args.photometry_mode:
        config_overrides['photometry.mode'] = args.photometry_mode
    if args.force_kids:
        config_overrides['photometry.force_kids'] = True
    if args.force_regular:
        config_overrides['photometry.force_regular'] = True
    if args.mass_threshold:
        config_overrides['photometry.mass_threshold'] = args.mass_threshold
    if args.kids_aperture_mode:
        config_overrides['photometry.kids_aperture_mode'] = args.kids_aperture_mode
    if args.kids_image_type:
        config_overrides['photometry.kids_image_type'] = args.kids_image_type
        
    if args.dtd_beta:
        config_overrides['dtd.beta'] = args.dtd_beta
    if args.dtd_A:
        config_overrides['dtd.A'] = args.dtd_A
        
    if args.age_step_tp:
        config_overrides['light_curve.age_step.tp'] = args.age_step_tp
    if args.age_step_alpha:
        config_overrides['light_curve.age_step.alpha'] = args.age_step_alpha
    
    # Check for conflicts
    if args.n_subhalos and args.subhalo_ids:
        parser.error("Cannot specify both --n-subhalos and --subhalo-ids")
    
    if args.force_kids and args.force_regular:
        parser.error("Cannot specify both --force-kids and --force-regular")
    
    # Print configuration
    if args.verbose:
        print("Configuration:")
        print(f"  Simulation: {args.simulation}")
        print(f"  Output directory: {args.output_dir}")
        if args.config:
            print(f"  Config file: {args.config}")
        if config_overrides:
            print("  Overrides:")
            for key, value in config_overrides.items():
                print(f"    {key}: {value}")
        print()
    
    try:
        # Initialize simulation
        sim = TNGSNSimulation(
            config_file=args.config, 
            **config_overrides
        )
        
        # Run simulation
        if args.subhalo_ids:
            print(f"Processing {len(args.subhalo_ids)} specific subhalos...")
            results = sim.run_simulation(subhalo_ids=args.subhalo_ids)
        else:
            if args.n_subhalos:
                print(f"Processing first {args.n_subhalos} subhalos...")
            else:
                print("Processing all available subhalos...")
            results = sim.run_simulation(n_subhalos=args.n_subhalos)
        
        print(f"\nSimulation completed successfully!")
        print(f"Generated {len(results)} SNe")
        
        # Print summary statistics
        if len(results) > 0:
            print("\nSummary statistics:")
            print(f"  Mean x1: {results['x1'].mean():.3f} ± {results['x1'].std():.3f}")
            print(f"  Mean host mass: {results.get('globalmass', []).mean():.2e} M☉")
            if 'MURES' in results:
                print(f"  Mean Hubble residual: {results['MURES'].mean():.3f} ± {results['MURES'].std():.3f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())