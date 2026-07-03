#!/bin/bash
# Entrypoint for the quip-miner Docker image (CPU / CUDA).
#
# Everything the miner does is declared in /data/config.toml (seeded
# from the image template on first run): validators, mempool, faucet,
# telemetry, identity, and backend inventory. `quip-miner --config`
# reads it and spawns/supervises every process itself — the entrypoint
# only prepares the volume and drops privileges.
#
# Host environment (not miner configuration):
#   PUID / PGID   uid/gid to run as (default 1000:1000; PUID=0 = root)
#
# Persistent state under /data:
#   config.toml      — the configuration; edit it and restart.
#   keystore.json    — hybrid sr25519 + ML-DSA-44, plaintext seed.
#                      BACK THIS UP — the chain key lives here.
set -euo pipefail

CONFIG_FILE="/data/config.toml"
TEMPLATE_FILE="/app/quip-miner.docker.toml"
# Fixed path: the template's `signer_key` value and this keygen target
# are the same by construction. Mount an existing keystore here to
# bring your own key.
KEYSTORE_FILE="/data/keystore.json"

echo "========================================"
echo "Quip Miner (substrate)  $(date)"
echo "========================================"

# ── First run: seed config + keystore ─────────────────────────────
mkdir -p /data
if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run: seeding $CONFIG_FILE from $TEMPLATE_FILE"
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
else
    echo "Using existing config at $CONFIG_FILE"
fi

if [ ! -f "$KEYSTORE_FILE" ]; then
    echo "First run: generating hybrid keystore at $KEYSTORE_FILE"
    quip-miner keygen --out "$KEYSTORE_FILE"
    # quip-miner keygen sets mode 0600 internally; re-assert for clarity.
    chmod 600 "$KEYSTORE_FILE"
else
    echo "Using existing keystore at $KEYSTORE_FILE"
fi

# ── GPU diagnostic (cuda image only — nvidia-smi absent elsewhere) ──
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "----------------------------------------"
    nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv,noheader || true
    echo "----------------------------------------"
fi

# ── Privilege drop ─────────────────────────────────────────────────
# Container starts as root so we can chown /data and generate the
# keystore with the right ownership; the miner runs as `quip`.
# PUID/PGID map the in-container user to a host uid/gid (default
# 1000:1000; PUID=0 keeps root).
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$PUID" = "0" ]; then
    echo "Privilege drop: PUID=0 — running as root (legacy mode)"
    exec quip-miner --config "$CONFIG_FILE"
fi

[ "$(id -g quip)" != "$PGID" ] && groupmod -g "$PGID" quip
[ "$(id -u quip)" != "$PUID" ] && usermod -u "$PUID" -g "$PGID" quip
chown -R quip:quip /data
chmod 600 "$KEYSTORE_FILE"
echo "Privilege drop: exec as uid=$PUID gid=$PGID"
exec gosu quip:quip quip-miner --config "$CONFIG_FILE"
