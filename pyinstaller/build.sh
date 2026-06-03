#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

# Build a frozen binary of quip-miner for the current platform.
# Usage: bash pyinstaller/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Key the PyInstaller cache by architecture so a future multi-arch build
# can't corrupt a shared cache. macOS ships arm64 only today.
ARCH="$(uname -m)"
export PYINSTALLER_CONFIG_DIR="$PROJECT_ROOT/build/.pyinstaller-cache-${ARCH}"
rm -rf "$PYINSTALLER_CONFIG_DIR"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

# Stamp version into boot script so --version needs zero imports
VERSION=$(python -c "
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")
sed "s/@VERSION@/${VERSION}/" "$SCRIPT_DIR/boot_miner.py" \
    > "$SCRIPT_DIR/boot_miner_stamped.py"

echo "=== Building quip-miner ${VERSION} (${ARCH}) ==="
pyinstaller "$SCRIPT_DIR/quip_miner.spec" \
    --distpath "$PROJECT_ROOT/dist" \
    --workpath "$PROJECT_ROOT/build/pyinstaller-${ARCH}"

# Smoke test: binary must print version and exit 0
echo ""
echo "=== Smoke test ==="
BINARY=$(find dist -maxdepth 1 -name 'quip-miner-*' -type f | head -1)
if [ -z "$BINARY" ]; then
    echo "ERROR: No binary found in dist/"
    exit 1
fi

"$BINARY" --version

# Data-bundle guard: load the scalecodec SCALE type-registry presets the way
# SubstrateInterface does at connect time. Catches a dropped-package-data
# regression (missing collect_data_files) here at build time rather than as a
# runtime "'NoneType' object has no attribute 'get'" on the first validator
# connect in production.
"$BINARY" selftest

ls -lh "$BINARY"
echo "OK: $BINARY"
