#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if conda env list | awk '{print $1}' | grep -qx learn2branch; then
    echo "Conda environment learn2branch already exists."
else
    conda env create --file "$SCRIPT_DIR/environment.yml"
fi

# libxcrypt provides crypt.h when the legacy PySCIPOpt extension is compiled
# with the Conda GCC 7 sysroot. gh is kept in this environment so GitHub work
# for the reproduction also goes through learn2branch. --no-deps avoids mixing
# the modern conda-forge dependency stack into the legacy scientific stack.
conda install --name learn2branch --channel conda-forge --no-deps --yes \
    libxcrypt=4.4.28 gh=2.97.0

echo "Environment created. Run: conda activate learn2branch"
