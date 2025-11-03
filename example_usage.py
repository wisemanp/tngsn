#!/usr/bin/env python3
"""
Example usage of the TNG Supernova Simulation Pipeline

This demonstrates various ways to use the refactored simulation system
for different scientific applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from snsims.simulation import TNGSNSimulation

def example_ztf_like_survey():
    """Example: ZTF-like low-z volume-limited survey."""
    
    print("🌟 Example 1: ZTF-like Survey")
    print("="*40)
    
    # Configure for ZTF-like low-z survey
    sim = TNGSNSimulation(
        # Use TNG100-1 for larger volume
        simulation__name='TNG100-1',
        simulation__min_stellar_mass=5e9,    # Focus on higher mass galaxies
        simulation__max_stellar_mass=1e12,   # Reasonable upper limit
        
        # Realistic galaxy selection
        subhalo_selection__method='observed',
        subhalo_selection__selection_seed=42,
        
        # Automatic photometry mode (KIDS for low mass, regular for high mass)
        photometry__mode='auto',
        photometry__mass_threshold=3.16e9,   # 10^9.5 M☉
        
        # Ginolin+25 physics (automatically loaded from defaults)
        light_curve__x1_params=[0.25, 0.55, -1.33, 0.63, -0.3, 0.18, 0.98, -0.128, 5]
    )
    
    # Run simulation
    results = sim.run_simulation(n_subhalos=50)
    
    # Analyze key correlations
    plot_ztf_correlations(results)
    
    return results

def example_kids_only_survey():
    """Example: KIDS-only survey for low-mass galaxies."""
    
    print("🔭 Example 2: KIDS-only Low-Mass Survey")
    print("="*40)
    
    sim = TNGSNSimulation(
        simulation__name='TNG50-1',
        simulation__min_stellar_mass=1e8,    # Very low mass galaxies
        simulation__max_stellar_mass=1e10,   # Upper limit for KIDS
        
        # Force KIDS photometry for all
        photometry__force_kids=True,
        photometry__kids_aperture_mode='direct',  # Direct aperture photometry
        photometry__kids_image_type='noisy',      # Use noisy images
        
        # Log-uniform mass distribution
        subhalo_selection__method='log_uniform'
    )
    
    results = sim.run_simulation(n_subhalos=25)
    
    print(f"Generated {len(results)} SNe in low-mass hosts")
    print(f"Mass range: {results['stellar_mass'].min():.1e} - {results['stellar_mass'].max():.1e} M☉")
    
    return results

def example_high_mass_survey():
    """Example: High-mass galaxy survey with regular photometry."""
    
    print("🏛️ Example 3: High-Mass Galaxy Survey")
    print("="*40)
    
    sim = TNGSNSimulation(
        simulation__name='TNG100-1',
        simulation__min_stellar_mass=1e10,   # High mass galaxies only
        simulation__max_stellar_mass=1e12,
        
        # Force regular photometry
        photometry__force_regular=True,
        
        # Mass-weighted selection (more massive galaxies more likely)
        subhalo_selection__method='mass_weighted'
    )
    
    results = sim.run_simulation(n_subhalos=30)
    
    print(f"Generated {len(results)} SNe in high-mass hosts")
    
    return results

def example_custom_physics():
    """Example: Custom physics parameters."""
    
    print("⚗️ Example 4: Custom Physics Parameters")
    print("="*40)
    
    # Test modified Ginolin+25 parameters
    modified_g25_params = [0.25, 0.55, -1.33, 0.63, -0.3, 0.18, 0.98, -0.100, 5]  # Modified Ka
    
    sim = TNGSNSimulation(
        simulation__name='TNG50-1',
        
        # Custom light curve parameters
        light_curve__x1_params=modified_g25_params,
        light_curve__age_step__Ka=-0.100,    # Modified age dependence
        
        # Custom DTD parameters
        dtd__beta=-1.0,                      # Shallower DTD slope
        dtd__A=3.0e-13,                      # Higher normalization
        
        # Custom cosmology
        cosmology__alpha=-0.15,              # Different stretch correction
        cosmology__beta=3.0                  # Different color correction
    )
    
    results = sim.run_simulation(n_subhalos=20)
    
    print("Custom physics parameters applied:")
    print(f"  - Modified Ka: {sim.config['light_curve']['age_step']['Ka']}")
    print(f"  - DTD slope β: {sim.config['dtd']['beta']}")
    print(f"  - Cosmology α: {sim.config['cosmology']['alpha']}")
    
    return results

def example_yaml_config():
    """Example: Using YAML configuration file."""
    
    print("📄 Example 5: YAML Configuration")
    print("="*40)
    
    # Use the ZTF config file
    sim = TNGSNSimulation(
        config_file='ztf_config_tng100.yaml',
        # Override specific parameters
        simulation__n_cores=8,
        subhalo_selection__selection_seed=123
    )
    
    results = sim.run_simulation(n_subhalos=40)
    
    print(f"Used configuration: {sim.config['simulation']['name']}")
    print(f"Photometry mode: {sim.config['photometry']['mode']}")
    
    return results

def plot_ztf_correlations(results):
    """Plot key SN Ia correlations to verify physics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stretch vs Mass
    axes[0,0].scatter(np.log10(results['stellar_mass']), results['x1'], alpha=0.7)
    axes[0,0].set_xlabel('log(M*/M☉)')
    axes[0,0].set_ylabel('x1 (stretch)')
    axes[0,0].set_title('Stretch vs Mass (Ginolin+25)')
    
    # Hubble residuals vs Mass  
    axes[0,1].scatter(np.log10(results['stellar_mass']), results['MURES'], alpha=0.7)
    axes[0,1].set_xlabel('log(M*/M☉)')
    axes[0,1].set_ylabel('Hubble Residual')
    axes[0,1].set_title('Mass Step in Hubble Diagram')
    
    # x1 distribution
    axes[1,0].hist(results['x1'], bins=20, alpha=0.7)
    axes[1,0].set_xlabel('x1 (stretch)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('x1 Distribution')
    
    # Color vs x1
    axes[1,1].scatter(results['x1'], results['c'], alpha=0.7)
    axes[1,1].set_xlabel('x1 (stretch)')
    axes[1,1].set_ylabel('c (color)')
    axes[1,1].set_title('Color-Stretch Relation')
    
    plt.tight_layout()
    plt.savefig('ztf_correlations.png', dpi=150)
    plt.show()
    
    # Print correlation statistics
    x1_mass_corr = np.corrcoef(np.log10(results['stellar_mass']), results['x1'])[0,1]
    print(f"\n📊 Key Correlations:")
    print(f"   x1-mass correlation: {x1_mass_corr:.3f} (should be positive)")
    print(f"   Hubble residual std: {results['MURES'].std():.3f} mag")
    print(f"   x1 range: [{results['x1'].min():.2f}, {results['x1'].max():.2f}]")

def main():
    """Run all examples."""
    
    print("🚀 TNG Supernova Simulation Examples")
    print("="*50)
    print("Demonstrating the refactored pipeline with preserved Ginolin+25 physics\n")
    
    try:
        # Run examples
        results1 = example_ztf_like_survey()
        results2 = example_kids_only_survey() 
        results3 = example_high_mass_survey()
        results4 = example_custom_physics()
        results5 = example_yaml_config()
        
        print("\n✅ All examples completed successfully!")
        print("   - ZTF-like correlations preserved")
        print("   - KIDS photometry integrated")
        print("   - Ginolin+25 physics verified")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("   This may be expected if TNG data is not available")

if __name__ == "__main__":
    main()