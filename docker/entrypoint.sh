#!/usr/bin/env bash
# Entrypoint for the quip-miner v0.3 image: run the quip-coordinator, which
# connects to the chain and supervises the bundled miner binaries per config.
set -euo pipefail

DATA_DIR="/data"
CONFIG_FILE="${DATA_DIR}/config.toml"
TEMPLATE_FILE="/app/config.toml"

mkdir -p "$DATA_DIR"

# Seed the persistent config from the image template on first run; after that
# the mounted /data/config.toml wins.
if [ ! -f "$CONFIG_FILE" ]; then
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    echo "entrypoint: seeded $CONFIG_FILE from image template"
fi

# Ownership for the drop-to-quip below.
chown -R quip:quip "$DATA_DIR" || true

echo "entrypoint: exec quip-coordinator --config $CONFIG_FILE"
# gosu drops root -> quip; tini (PID 1) reaps children the coordinator spawns.
exec gosu quip:quip quip-coordinator --config "$CONFIG_FILE" "$@"
