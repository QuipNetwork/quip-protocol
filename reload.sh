#!/bin/bash

sudo cp -r . /opt/quip/src
sudo chown -R quip:quip /opt/quip/src
sudo -u quip /opt/quip/bin/pip install -e /opt/quip/src
sudo systemctl restart quip-network-node
