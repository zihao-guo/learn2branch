#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/activate.sh"

REPRO_DIR="$ROOT/.repro"
DOWNLOAD_DIR="$REPRO_DIR/downloads"
SOURCE_DIR="$REPRO_DIR/src"
BUILD_DIR="$REPRO_DIR/build"
INSTALL_DIR="$REPRO_DIR/scip"
JOBS="${LEARN2BRANCH_BUILD_JOBS:-4}"

mkdir -p "$DOWNLOAD_DIR" "$SOURCE_DIR" "$BUILD_DIR" "$INSTALL_DIR"

download() {
    local url="$1"
    local output="$2"
    local tls_mode="${3:-strict}"
    if [[ -f "$output" ]]; then
        return
    fi
    if [[ "$tls_mode" == "insecure" ]]; then
        curl --fail --location --retry 3 --insecure --output "$output" "$url"
    else
        curl --fail --location --retry 3 --output "$output" "$url"
    fi
}

download "https://soplex.zib.de/download/release/soplex-4.0.1.tgz" \
    "$DOWNLOAD_DIR/soplex-4.0.1.tgz"
download "https://scipopt.org/download/release/scip-6.0.1.tgz" \
    "$DOWNLOAD_DIR/scip-6.0.1.tgz"
# The upstream SVMrank certificate is expired. Integrity is enforced below by
# the pinned SHA-256 digest before the archive is ever extracted.
download "https://osmot.cs.cornell.edu/svm_rank/current/svm_rank.tar.gz" \
    "$DOWNLOAD_DIR/svm_rank.tar.gz" insecure

(cd "$DOWNLOAD_DIR" && sha256sum --check "$SCRIPT_DIR/checksums.sha256")

SOPLEX_SOURCE="$BUILD_DIR/soplex-4.0.1"
SCIP_SOURCE="$BUILD_DIR/scip-6.0.1"
[[ -d "$SOPLEX_SOURCE" ]] || tar -xzf "$DOWNLOAD_DIR/soplex-4.0.1.tgz" -C "$BUILD_DIR"
[[ -d "$SCIP_SOURCE" ]] || tar -xzf "$DOWNLOAD_DIR/scip-6.0.1.tgz" -C "$BUILD_DIR"

CC_BIN="$(command -v x86_64-conda-linux-gnu-gcc)"
CXX_BIN="$(command -v x86_64-conda-linux-gnu-g++)"

cmake -S "$SOPLEX_SOURCE" -B "$SOPLEX_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build "$SOPLEX_SOURCE/build" --parallel "$JOBS"
cmake --build "$SOPLEX_SOURCE/build" --target install --parallel "$JOBS"

if patch --dry-run --silent -d "$SCIP_SOURCE" -p1 \
        < "$ROOT/scip_patch/vanillafullstrong.patch"; then
    patch -d "$SCIP_SOURCE" -p1 < "$ROOT/scip_patch/vanillafullstrong.patch"
elif patch --dry-run --silent --reverse -d "$SCIP_SOURCE" -p1 \
        < "$ROOT/scip_patch/vanillafullstrong.patch"; then
    echo "SCIP vanilla-fullstrong patch already applied."
else
    echo "SCIP source is neither clean nor correctly patched." >&2
    exit 1
fi

cmake -S "$SCIP_SOURCE" -B "$SCIP_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DSOPLEX_DIR="$INSTALL_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build "$SCIP_SOURCE/build" --parallel "$JOBS"
cmake --build "$SCIP_SOURCE/build" --target install --parallel "$JOBS"

export SCIPOPTDIR="$INSTALL_DIR"
export LD_LIBRARY_PATH="$SCIPOPTDIR/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install --no-deps \
    "git+https://github.com/ds4dm/PySCIPOpt.git@9b88eddcfe99208e737236d39ccd6c9dd1a9bec4"
python -m pip install --no-deps \
    "git+https://github.com/jma127/pyltr.git@78fa0ebfef67d6594b8415aa5c6136e30a5e3395"

PYSVMRANK_SOURCE="$SOURCE_DIR/PySVMRank"
if [[ ! -d "$PYSVMRANK_SOURCE/.git" ]]; then
    git clone https://github.com/ds4dm/PySVMRank.git "$PYSVMRANK_SOURCE"
fi
git -C "$PYSVMRANK_SOURCE" fetch origin b7dd8598d459ddb8926467385f5658173af46752
git -C "$PYSVMRANK_SOURCE" checkout --detach b7dd8598d459ddb8926467385f5658173af46752
mkdir -p "$PYSVMRANK_SOURCE/src/c"
if [[ ! -f "$PYSVMRANK_SOURCE/src/c/svm_struct_api.c" ]]; then
    tar -xzf "$DOWNLOAD_DIR/svm_rank.tar.gz" -C "$PYSVMRANK_SOURCE/src/c"
fi
CFLAGS="${CFLAGS:-} -fcommon" python -m pip install --no-deps "$PYSVMRANK_SOURCE"
python -m pip install --no-deps --editable "$ROOT"

python "$SCRIPT_DIR/verify_environment.py"
