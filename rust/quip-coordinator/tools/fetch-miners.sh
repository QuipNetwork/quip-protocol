#!/usr/bin/env bash
# Fetch the miner binaries this host needs from the standalone miner repos'
# GitLab package registries, into a local bin dir the coordinator's config.toml
# points at. Miners now live in their own repos and publish per-platform
# binaries as Release/package assets; the coordinator no longer builds them.
#
# Usage: fetch-miners.sh [DEST_DIR]   (default: ./miners)
# Env:   MINERS_TAG  (default: v0.3.0) — the release tag to pull from.
#
# NOTE: this is inert until the miner repos have cut releases at MINERS_TAG.
set -euo pipefail

DEST="${1:-./miners}"
TAG="${MINERS_TAG:-v0.3.0}"
API="https://gitlab.com/api/v4"
GROUP="quip.network"

os="$(uname -s)"; machine="$(uname -m)"
case "$machine" in
  x86_64|amd64) arch=amd64 ;;
  arm64|aarch64) arch=arm64 ;;
  *) echo "unsupported arch: $machine" >&2; exit 1 ;;
esac

mkdir -p "$DEST"

# proj: URL-encoded "group/repo"; assets: space-separated asset base names.
fetch() {
  local proj="$1"; shift
  for name in "$@"; do
    local url="${API}/projects/${proj}/packages/generic/${proj##*%2F}/${TAG}/${name}"
    echo "fetch $name <- $url"
    curl --fail --location --silent --output "${DEST}/${name}" "$url"
    chmod +x "${DEST}/${name}" || true
  done
}

case "$os" in
  Linux)
    # quip-miner-cpu ships amd64+arm64; quip-miner-cuda amd64-only.
    fetch "quip.network%2Fquip-miner-cpu" "quip-cpu-sa-${arch}" "quip-cpu-gibbs-${arch}"
    if [ "$arch" = amd64 ]; then
      fetch "quip.network%2Fquip-miner-cuda" "quip-cuda-sa-amd64" "quip-cuda-gibbs-amd64" || \
        echo "note: cuda binaries optional (no GPU host)"
    fi
    ;;
  Darwin)
    # Apple Silicon: metal + cpu.
    fetch "quip.network%2Fquip-miner-metal" "quip-metal-sa-arm64" "quip-metal-gibbs-arm64"
    fetch "quip.network%2Fquip-miner-cpu" "quip-cpu-sa-arm64" "quip-cpu-gibbs-arm64"
    ;;
  *) echo "unsupported OS: $os" >&2; exit 1 ;;
esac

# The dwave miner is a Python wheel, not a native binary:
echo "dwave: pip install 'quip-miner-dwave @ git+https://gitlab.com/${GROUP}/quip-miner-dwave.git@${TAG}'"
echo "done -> $DEST"
