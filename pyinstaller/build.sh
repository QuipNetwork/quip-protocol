#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

# Build a frozen binary of quip-network-node for the current platform.
# Usage: bash pyinstaller/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Isolate PyInstaller cache per architecture to prevent corruption when
# arm64 and x86_64 jobs run concurrently on the same machine.
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
sed "s/@VERSION@/${VERSION}/" "$SCRIPT_DIR/boot_network_node.py" \
    > "$SCRIPT_DIR/boot_network_node_stamped.py"

echo "=== Building quip-network-node ${VERSION} (${ARCH}) ==="
pyinstaller "$SCRIPT_DIR/quip_network_node.spec" \
    --distpath "$PROJECT_ROOT/dist" \
    --workpath "$PROJECT_ROOT/build/pyinstaller-${ARCH}"

# Smoke test: binary must print version and exit 0
echo ""
echo "=== Smoke test ==="
BINARY=$(find dist -maxdepth 1 -name 'quip-network-node-*' -type f | head -1)
if [ -z "$BINARY" ]; then
    echo "ERROR: No binary found in dist/"
    exit 1
fi

"$BINARY" --version
ls -lh "$BINARY"
echo "OK: $BINARY"
