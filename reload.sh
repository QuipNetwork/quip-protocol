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

echo "Reinstalling package with CUDA support..."
sudo -u quip /opt/quip/bin/pip install -e "/opt/quip/src[cuda]"

echo "Restarting service..."
sudo systemctl restart quip-network-node

echo "Checking service status..."
sleep 2
sudo systemctl status quip-network-node --no-pager || true
