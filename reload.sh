#!/bin/bash
# Reinstall the editable package from this checkout and bounce the
# systemd-managed quip-miner. Bare-metal-only helper; the Docker
# image rebuilds from source on every `docker compose up --build`.
set -e

echo "Copying source to /opt/quip/src..."
sudo rm -rf /opt/quip/src
sudo cp -r . /opt/quip/src

echo "Setting ownership and permissions..."
sudo chown -R quip:quip /opt/quip/src
sudo chmod -R o+rX /opt/quip/src

echo "Reinstalling package with CUDA support..."
sudo -u quip /opt/quip/bin/pip install -e "/opt/quip/src[cuda]"

echo "Restarting service..."
sudo systemctl restart quip-miner

echo "Checking service status..."
sleep 2
sudo systemctl status quip-miner --no-pager || true
