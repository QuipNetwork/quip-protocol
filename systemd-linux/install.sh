#!/bin/bash
# Install quip-miner as a systemd service on bare-metal Linux.
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
TEMPLATE_FILE="$SCRIPT_DIR/quip-miner.systemd.toml"
SERVICE_FILE="$SCRIPT_DIR/quip-miner.service"
KEYSTORE_FILE="$DATA_DIR/keystore.json"

echo "========================================"
echo "Quip Miner — Systemd Installer"
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
echo "Installation complete: $INSTALL_DIR/bin/quip-miner"

# ── 4. Seed configuration ───────────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
    echo "Config already exists at $CONFIG_FILE — skipping"
else
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE"
    echo "Config seeded at $CONFIG_FILE — edit before starting the service"
fi

# ── 5. Generate signing keystore ────────────────────────────────
if [ -f "$KEYSTORE_FILE" ]; then
    echo "Keystore already exists at $KEYSTORE_FILE — skipping"
else
    echo "Generating hybrid keystore at $KEYSTORE_FILE..."
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/bin/quip-miner" keygen --out "$KEYSTORE_FILE"
    chmod 600 "$KEYSTORE_FILE"
fi

# ── 6. Patch service file PYTHONPATH ─────────────────────────────
# Detect the actual Python version in the venv
PYTHON_VERSION=$("$INSTALL_DIR/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PATCHED_SERVICE=$(mktemp)
sed "s|python3\.[0-9]*|python${PYTHON_VERSION}|g" "$SERVICE_FILE" > "$PATCHED_SERVICE"

# ── 7. Install systemd service ──────────────────────────────────
cp "$PATCHED_SERVICE" /etc/systemd/system/quip-miner.service
rm -f "$PATCHED_SERVICE"
systemctl daemon-reload
systemctl enable quip-miner
echo "Service installed and enabled"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Installation complete"
echo "========================================"
echo ""
echo "  Config:   $CONFIG_FILE"
echo "  Data:     $DATA_DIR"
echo "  Logs:     $LOG_DIR"
echo "  Keystore: $KEYSTORE_FILE  (BACK THIS UP)"
echo "  Binary:   $INSTALL_DIR/bin/quip-miner"
echo "  Service:  /etc/systemd/system/quip-miner.service"
echo ""
echo "Next steps:"
echo "  1. Edit $CONFIG_FILE"
echo "     - Set [miner] validators = [\"wss://...\"]"
echo "     - Optionally set faucet_url, rest_port, rest_host"
echo "  2. Start the service:"
echo "     sudo systemctl start quip-miner"
echo "  3. View logs:"
echo "     journalctl -u quip-miner -f"
