#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

python reproduction/verify_environment.py
python reproduction/smoke_pipeline.py "$@"
