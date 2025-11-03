#!/usr/bin/env python3
"""
Verify that the path resolution is working correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from snsims.simulation import TNGSNSimulation

def verify_paths():
    """Verify path resolution."""
    
    print("🔍 Verifying path resolution...")
    print("="*50)
    
    # Test with your config
    sim = TNGSNSimulation(config_file='ztf_config_tng100.yaml')
    
    print(f"Root path: {sim.root_path}")
    print(f"Root path exists: {sim.root_path.exists()}")
    
    # Check key paths
    test_paths = {
        'galmeta_file': 'TNG100-1',
        'morph_file': 'TNG100-1', 
        'kids_results_dir': 'TNG100-1'
    }
    
    for path_key, sim_name in test_paths.items():
        try:
            resolved_path = sim.get_path(path_key, simulation=sim_name)
            path_exists = Path(resolved_path).exists()
            print(f"{path_key:20}: {resolved_path}")
            print(f"{'':20}  Exists: {path_exists}")
            
            if not path_exists:
                # Suggest what the path should be
                if 'galmeta' in path_key:
                    expected = sim.root_path / f"data/{sim_name}/galmeta.csv"
                    print(f"{'':20}  Expected: {expected}")
                    print(f"{'':20}  Expected exists: {expected.exists()}")
                    
        except Exception as e:
            print(f"{path_key:20}: ERROR - {e}")
    
    # Check what files actually exist
    print(f"\n📁 Checking actual file structure...")
    data_dir = sim.root_path / "data"
    if data_dir.exists():
        print(f"Data directory: {data_dir}")
        for sim_dir in data_dir.iterdir():
            if sim_dir.is_dir():
                print(f"  Simulation: {sim_dir.name}")
                key_files = ['galmeta.csv', 'morphs_i.hdf5']
                for key_file in key_files:
                    file_path = sim_dir / key_file
                    print(f"    {key_file}: {'✅' if file_path.exists() else '❌'}")
    else:
        print(f"❌ Data directory does not exist: {data_dir}")

if __name__ == "__main__":
    verify_paths()