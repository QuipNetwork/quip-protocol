# Systemd (Bare-Metal Linux) Deployment

> **WARNING: These scripts are provided for convenience only and are NOT recommended.**
>
> The quip.network team does not use this setup and **will not provide support** for it.
>
> **Recommended alternatives:**
> - **[Quip Node Manager](https://gitlab.com/quip.network/quip-node-manager)** — the recommended way to run a quip miner
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
- Optional: NVIDIA drivers + CUDA toolkit (for GPU mining)

## Quick Start

```bash
cd quip-protocol
sudo bash systemd-linux/install.sh
```

Then edit the config and start the service:

```bash
sudo nano /etc/quip.network/config.toml   # set [miner] validators = ["wss://..."]
sudo systemctl start quip-miner
```

## What install.sh Does

1. Creates a `quip` system user and group
2. Creates directories: `/opt/quip`, `/etc/quip.network`, `/var/lib/quip.network`, `/var/log/quip.network`
3. Creates a Python virtual environment at `/opt/quip` and installs `quip-protocol` from source
4. Seeds `/etc/quip.network/config.toml` from the v0.2 `[miner]` template
5. Generates the hybrid signing keystore at `/var/lib/quip.network/keystore.json` (mode `0600`)
6. Installs and enables the systemd service (does **not** start it)

The script is idempotent — re-running it skips steps that are already complete.

## Post-Install Configuration

Edit `/etc/quip.network/config.toml` before starting the service:

| Setting | Description |
|---------|-------------|
| `[miner] validators` | Ordered failover list of `wss://` validator URLs (required) |
| `[miner] signer_key` | Path to the hybrid keystore (defaults to `/var/lib/quip.network/keystore.json`) |
| `[miner] faucet_url` | Optional dev-faucet URL for auto-topup on underfunded wallets |
| `[miner] rest_port` | Telemetry REST API port (`-1` disables; default) |
| `[miner] rest_host` | Telemetry REST API bind host (default `127.0.0.1`) |

To switch from CPU to GPU mining, edit `/etc/systemd/system/quip-miner.service` and
change the `ExecStart` subcommand from `cpu` to `gpu` (requires NVIDIA drivers +
CUDA toolkit on the host, and the package installed with the `[cuda]` extra).

## Service Management

```bash
# Start / stop / restart
sudo systemctl start quip-miner
sudo systemctl stop quip-miner
sudo systemctl restart quip-miner

# Check status
sudo systemctl status quip-miner

# View logs (live)
journalctl -u quip-miner -f

# View recent logs
journalctl -u quip-miner -n 100
```

## Keystore

The hybrid sr25519 + ML-DSA-44 keystore at `/var/lib/quip.network/keystore.json`
**is the chain identity**. Back it up. Loss = permanent loss of the SS58 address and
any registered solver state.

## Uninstall

```bash
sudo systemctl stop quip-miner
sudo systemctl disable quip-miner
sudo rm /etc/systemd/system/quip-miner.service
sudo systemctl daemon-reload

# Remove data (irreversible — destroys the signing keystore)
sudo rm -rf /opt/quip /etc/quip.network /var/lib/quip.network /var/log/quip.network
sudo userdel quip
```

## Files

| File | Description |
|------|-------------|
| `install.sh` | Installation script (run as root) |
| `quip-miner.service` | systemd unit file |
| `quip-miner.systemd.toml` | Configuration template |
