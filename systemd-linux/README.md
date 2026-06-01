# Systemd (Bare-Metal Linux) Deployment

> **WARNING: These scripts are provided for convenience only and are NOT recommended.**
>
> The quip.network team does not use this setup and **will not provide support** for it.
>
> **Recommended alternatives:**
> - **[Quip Node Manager](https://gitlab.com/quip.network/quip-node-manager)** — the recommended way to run a quip node
> - **[Docker deployment](../docker/)** — for containerized setups on any Linux server
> - **[nodes.quip.network](https://gitlab.com/quip.network/nodes.quip.network)** — Docker-based deployment for remote servers
>
> If you encounter problems with this systemd setup, support will not be provided.
> We will accept suggestions and fixes via MR or PR into the project.

---

## Prerequisites

- Linux with systemd (Debian/Ubuntu, RHEL/Fedora, Arch, etc.)
- Python 3.11+
- Root access (sudo)
- `openssl` (for secret generation)
- Optional: NVIDIA drivers + CUDA toolkit (for GPU mining)

## Quick Start

```bash
cd quip-protocol
sudo bash systemd-linux/install.sh
```

Then edit the config and start the service:

```bash
sudo nano /etc/quip.network/config.toml   # set public_host, node_name, peers
sudo systemctl start quip-network-node
```

## What install.sh Does

1. Creates a `quip` system user and group
2. Creates directories: `/opt/quip`, `/etc/quip.network`, `/var/lib/quip.network`, `/var/log/quip.network`
3. Creates a Python virtual environment at `/opt/quip` and installs quip-protocol from source
4. Seeds `/etc/quip.network/config.toml` from template with a randomly generated secret
5. Copies the genesis block to `/var/lib/quip.network/genesis_block.json`
6. Detects CPU count (and optionally NVIDIA GPUs) and updates the config
7. Installs and enables the systemd service (does **not** start it)

The script is idempotent — re-running it skips steps that are already complete.

## Post-Install Configuration

Edit `/etc/quip.network/config.toml` before starting the service:

| Setting | Description |
|---------|-------------|
| `public_host` | Your public IP or DNS name (required for peers to reach you) |
| `public_port` | Public port if different from `port` (e.g., behind NAT) |
| `node_name` | Human-readable name for your node |
| `auto_mine` | Set to `true` to mine on startup |
| `peer` | Array of peers to connect to |
| `num_cpus` | Number of CPU threads for mining (under `[cpu]`) |

For GPU mining, uncomment the `[gpu]` and `[cuda.N]` sections.

## Service Management

```bash
# Start / stop / restart
sudo systemctl start quip-network-node
sudo systemctl stop quip-network-node
sudo systemctl restart quip-network-node

# Check status
sudo systemctl status quip-network-node

# View logs (live)
journalctl -u quip-network-node -f

# View recent logs
journalctl -u quip-network-node -n 100
```

## GPU Support

If `nvidia-smi` is available during installation, `install.sh` detects GPUs and adds
`[cuda.N]` sections to the config. You must have NVIDIA drivers and the CUDA toolkit
installed separately — the install script does not install GPU drivers.

To switch from CPU to GPU mining after installation, edit `/etc/quip.network/config.toml`:
uncomment the `[gpu]` section and add `[cuda.N]` entries for each GPU.

## TLS Certificates

This setup does not include automated certificate management (unlike the Docker setup
which integrates certbot). Obtain certificates via certbot, acme.sh, or your preferred
method, then set the paths in `/etc/quip.network/config.toml`:

```toml
tls_cert_file = "/etc/quip.network/certs/fullchain.pem"
tls_key_file = "/etc/quip.network/certs/privkey.pem"
```

## Uninstall

```bash
sudo systemctl stop quip-network-node
sudo systemctl disable quip-network-node
sudo rm /etc/systemd/system/quip-network-node.service
sudo systemctl daemon-reload

# Remove data (irreversible)
sudo rm -rf /opt/quip /etc/quip.network /var/lib/quip.network /var/log/quip.network
sudo userdel quip
```

## Files

| File | Description |
|------|-------------|
| `install.sh` | Installation script (run as root) |
| `quip-network-node.service` | systemd unit file |
| `quip-node.systemd.toml` | Configuration template |
