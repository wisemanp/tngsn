# Installation Guide for TNGSN

## Quick Installation

```bash
# Clone and install in development mode (recommended for active development)
git clone https://github.com/wisemanp/tngsn.git
cd tngsn
pip install -e ".[all]"
```

## Installation Options

### 1. Development Installation (Editable)
```bash
# Install with all optional dependencies for development
pip install -e ".[all]"

# Or just the core package
pip install -e .
```

### 2. Production Installation
```bash
# From local directory
pip install .

# From git repository
pip install git+https://github.com/wisemanp/tngsn.git
```

### 3. Minimal Installation
```bash
# Just the core dependencies
pip install git+https://github.com/wisemanp/tngsn.git
```

## Dependency Groups

- **Core**: Essential dependencies for basic functionality
- **dev**: Development tools (pytest, black, flake8, mypy)
- **docs**: Documentation generation tools (sphinx)
- **all**: Everything (core + dev + docs)

## Verification

After installation, verify everything works:

```bash
# Check installation
python -c "import tngsn; print(f'TNGSN {tngsn.__version__} installed')"

# Test command-line tools
tngsn-run --help

# Run a minimal test
python -c "
from snsims import TNGSNSimulation
sim = TNGSNSimulation()
print('Configuration loaded successfully')
print(f'Simulation: {sim.config[\"simulation\"][\"name\"]}')
"
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all required packages are installed
   ```bash
   pip install numpy pandas astropy photutils h5py tqdm pyyaml scipy matplotlib sep requests
   ```

2. **Import errors**: Make sure the package is installed properly
   ```bash
   pip install -e .
   ```

3. **Command-line tools not found**: Check your PATH and reinstall
   ```bash
   pip uninstall tngsn
   pip install -e .
   ```

### System Requirements

- Python 3.8 or higher
- Sufficient disk space for TNG data
- Memory: Recommended 8GB+ RAM for processing
- Optional: GPU for accelerated computations (future enhancement)

## Uninstallation

```bash
pip uninstall tngsn
```