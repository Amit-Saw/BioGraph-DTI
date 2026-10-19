#!/usr/bin/env bash
# Creates .venv and installs everything.   ./setup_env.sh [cpu|cu121|cu118] [--docking]
set -e
cd "$(dirname "$0")"
BUILD="${1:-cpu}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$BUILD" = "cpu" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch --index-url "https://download.pytorch.org/whl/$BUILD"
fi
pip install -r requirements.txt
if [ "$2" = "--docking" ]; then
  pip install -r requirements-docking.txt
  echo "Also install Open Babel:  sudo apt-get install openbabel   (or: conda install -c conda-forge openbabel)"
fi
echo "Done. Next: python scripts/setup_data.py --src <your Drive folder>"
