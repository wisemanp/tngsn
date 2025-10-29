#!/bin/bash
# Setup script to set TNGSN_PATH environment variable

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Export the TNGSN_PATH
export TNGSN_PATH="$SCRIPT_DIR"

echo "TNGSN_PATH set to: $TNGSN_PATH"
echo "You can now run analysis scripts from any directory."
echo ""
echo "Usage examples:"
echo "  cd KIDS && python run_analysis.py"
echo "  cd KIDS && python run_analysis.py --sim TNG100-1 --output my_results.csv"
echo "  python KIDS/run_analysis.py --base-path /path/to/tngsn"