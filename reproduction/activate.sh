#!/usr/bin/env bash

if [[ "${CONDA_DEFAULT_ENV:-}" != "learn2branch" ]]; then
    echo "Activate the required environment first: conda activate learn2branch" >&2
    return 1 2>/dev/null || exit 1
fi

REPRO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LEARN2BRANCH_ROOT="$(cd "$REPRO_SCRIPT_DIR/.." && pwd)"
export SCIPOPTDIR="$LEARN2BRANCH_ROOT/.repro/scip"
export LD_LIBRARY_PATH="$SCIPOPTDIR/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# TensorFlow 1.12 targets CUDA 9.x, whereas the current machine has a modern
# driver/GPU stack. Empty is the verified, deterministic CPU default. A user
# with a compatible legacy CUDA runtime may set this before sourcing the file.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
