#!/bin/bash

sudo cp genesis_block_public.json /etc/quip.network/genesis_block.json
sudo cp -r . /opt/quip/src
sudo chown -R quip:quip /opt/quip/src
sudo -u quip /opt/quip/bin/pip install -e /opt/quip/src
sudo systemctl restart quip-network-node
