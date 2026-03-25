#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

# Build a frozen binary of quip-network-node for the current platform.
# Usage: bash pyinstaller/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Building quip-network-node ==="
pyinstaller "$SCRIPT_DIR/quip_network_node.spec" \
    --distpath "$PROJECT_ROOT/dist" \
    --workpath "$PROJECT_ROOT/build/pyinstaller" \
    --clean

# Smoke test: binary must print version and exit 0
echo ""
echo "=== Smoke test ==="
BINARY=$(find dist -maxdepth 1 -name 'quip-network-node-*' -type f | head -1)
if [ -z "$BINARY" ]; then
    echo "ERROR: No binary found in dist/"
    exit 1
fi

"$BINARY" --version
echo "OK: $BINARY"
