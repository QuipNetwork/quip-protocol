#!/bin/bash
set -e

echo "Copying genesis block..."
sudo cp genesis_block_public.json /etc/quip.network/genesis_block.json

echo "Copying source to /opt/quip/src..."
sudo rm -rf /opt/quip/src
sudo cp -r . /opt/quip/src

echo "Setting ownership and permissions..."
sudo chown -R quip:quip /opt/quip/src
sudo chmod -R o+rX /opt/quip/src

echo "Removing duplicate cupy installations..."
sudo -u quip /opt/quip/bin/pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x 2>/dev/null || true

echo "Reinstalling package..."
sudo -u quip /opt/quip/bin/pip install -e /opt/quip/src

echo "Restarting service..."
sudo systemctl restart quip-network-node

echo "Checking service status..."
sleep 2
sudo systemctl status quip-network-node --no-pager || true
