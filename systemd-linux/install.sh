#!/bin/bash
# Install quip-network-node as a systemd service on bare-metal Linux.
#
# Usage: sudo bash install.sh
#
# This script is idempotent — re-running it skips steps already completed.
# It does NOT start the service; edit /etc/quip.network/config.toml first.
set -euo pipefail

# ── Constants ────────────────────────────────────────────────────
INSTALL_DIR="/opt/quip"
CONFIG_DIR="/etc/quip.network"
DATA_DIR="/var/lib/quip.network"
LOG_DIR="/var/log/quip.network"
SERVICE_USER="quip"
SERVICE_GROUP="quip"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="$CONFIG_DIR/config.toml"
TEMPLATE_FILE="$SCRIPT_DIR/quip-node.systemd.toml"
SERVICE_FILE="$SCRIPT_DIR/quip-network-node.service"
GENESIS_SRC="$PROJECT_ROOT/genesis_block_public.json"
GENESIS_DST="$DATA_DIR/genesis_block.json"

echo "========================================"
echo "Quip Network Node — Systemd Installer"
echo "========================================"

# ── Root check ───────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo bash install.sh)"
    exit 1
fi

# ── Source check ─────────────────────────────────────────────────
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "ERROR: Cannot find pyproject.toml at $PROJECT_ROOT"
    echo "Run this script from within the quip-protocol source tree."
    exit 1
fi

# ── 1. Create system user ───────────────────────────────────────
if id "$SERVICE_USER" &>/dev/null; then
    echo "User '$SERVICE_USER' already exists — skipping"
else
    useradd --system --shell /bin/false \
        --home-dir "$DATA_DIR" --create-home "$SERVICE_USER"
    echo "Created system user '$SERVICE_USER'"
fi

# ── 2. Create directories ───────────────────────────────────────
for dir in "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$INSTALL_DIR"; do
    mkdir -p "$dir"
done
chown "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR" "$LOG_DIR"
echo "Directories: $CONFIG_DIR, $DATA_DIR, $LOG_DIR, $INSTALL_DIR"

# ── 3. Python virtual environment ───────────────────────────────
if [ -f "$INSTALL_DIR/bin/python3" ]; then
    echo "Virtual environment already exists at $INSTALL_DIR — skipping creation"
else
    echo "Creating Python virtual environment at $INSTALL_DIR..."
    python3 -m venv "$INSTALL_DIR"
fi

echo "Installing quip-protocol from source..."
"$INSTALL_DIR/bin/pip" install -U pip setuptools wheel --quiet
"$INSTALL_DIR/bin/pip" install -e "$PROJECT_ROOT" --quiet
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
echo "Installation complete: $INSTALL_DIR/bin/quip-network-node"

# ── 4. Seed configuration ───────────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
    echo "Config already exists at $CONFIG_FILE — skipping"
else
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    SECRET=$(openssl rand -hex 32)
    sed -i "s/secret = \"GENERATE_ON_FIRST_RUN\"/secret = \"$SECRET\"/" "$CONFIG_FILE"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "Config generated at $CONFIG_FILE (secret randomly generated)"
fi

# ── 5. Genesis block ────────────────────────────────────────────
if [ -f "$GENESIS_DST" ]; then
    echo "Genesis block already exists at $GENESIS_DST — skipping"
elif [ -f "$GENESIS_SRC" ]; then
    cp "$GENESIS_SRC" "$GENESIS_DST"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$GENESIS_DST"
    echo "Copied genesis block to $GENESIS_DST"
else
    echo "WARNING: Genesis block not found at $GENESIS_SRC"
fi

# ── 6. Hardware detection — CPU ──────────────────────────────────
NUM_CPUS=$(nproc)
echo "Detected $NUM_CPUS CPUs"
if grep -q '^\[cpu\]' "$CONFIG_FILE" && ! grep -q '^num_cpus' "$CONFIG_FILE"; then
    sed -i '/^\[cpu\]/a num_cpus = '"$NUM_CPUS" "$CONFIG_FILE"
    echo "Set num_cpus = $NUM_CPUS in config"
fi

# ── 7. Hardware detection — GPU (optional) ───────────────────────
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -gt 0 ]; then
        echo "Detected $NUM_GPUS NVIDIA GPU(s):"
        nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv,noheader
        if ! grep -q '^\[cuda\.' "$CONFIG_FILE"; then
            {
                echo ""
                echo "[gpu]"
                echo ""
                for ((i = 0; i < NUM_GPUS; i++)); do
                    echo "[cuda.$i]"
                    echo ""
                done
            } >> "$CONFIG_FILE"
            echo "Added GPU sections to config (uncommented and ready)"
            echo "NOTE: Remove the [cpu] section if you want GPU-only mining"
        fi
    fi
else
    echo "No NVIDIA GPUs detected (nvidia-smi not found)"
fi

# ── 8. Patch service file PYTHONPATH ─────────────────────────────
# Detect the actual Python version in the venv
PYTHON_VERSION=$("$INSTALL_DIR/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PATCHED_SERVICE=$(mktemp)
sed "s|python3\.[0-9]*|python${PYTHON_VERSION}|g" "$SERVICE_FILE" > "$PATCHED_SERVICE"

# ── 9. Install systemd service ──────────────────────────────────
cp "$PATCHED_SERVICE" /etc/systemd/system/quip-network-node.service
rm -f "$PATCHED_SERVICE"
systemctl daemon-reload
systemctl enable quip-network-node
echo "Service installed and enabled"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Installation complete"
echo "========================================"
echo ""
echo "  Config:  $CONFIG_FILE"
echo "  Data:    $DATA_DIR"
echo "  Logs:    $LOG_DIR"
echo "  Binary:  $INSTALL_DIR/bin/quip-network-node"
echo "  Service: /etc/systemd/system/quip-network-node.service"
echo ""
echo "Next steps:"
echo "  1. Edit $CONFIG_FILE"
echo "     - Set public_host to your public IP or DNS name"
echo "     - Set node_name"
echo "     - Configure peers if needed"
echo "  2. Start the service:"
echo "     sudo systemctl start quip-network-node"
echo "  3. View logs:"
echo "     journalctl -u quip-network-node -f"
