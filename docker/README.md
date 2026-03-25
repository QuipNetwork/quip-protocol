# Docker Deployment Guide

This guide covers Docker images for running Quip Network Nodes.

For mining rate experiments and benchmarking, see [mining_rates/README.md](mining_rates/README.md).

---

## Network Node Images

The primary Docker images for running Quip P2P network nodes:

- **CPU Node** - Uses all available CPUs (amd64 + arm64)
- **CUDA Node** - Uses all available NVIDIA GPUs (amd64 + arm64)

Available from GitLab Container Registry:
```
registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu
registry.gitlab.com/piqued/quip-protocol/quip-network-node-cuda
```

**Note:** For Apple Silicon (Metal) GPU mining, run directly on macOS without Docker. See [../CLAUDE.md](../CLAUDE.md) for native macOS setup.

## Quick Start

### 1. Create Data Directory

Create a local directory for persistent data (config, logs, trust database):

```bash
mkdir -p ~/quip-data
```

### 2. Run a Node

**CPU node:**
```bash
docker run -d --pull always --name quip-cpu \
  -v ~/quip-data:/data \
  -e QUIP_PUBLIC_HOST=myhost.example.com \
  -p 20049:20049/udp -p 20049:20049/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest
```

**CUDA node (NVIDIA):**
```bash
docker run -d --pull always --gpus all --name quip-cuda \
  -v ~/quip-data:/data \
  -e QUIP_PUBLIC_HOST=myhost.example.com \
  -p 20049:20049/udp -p 20049:20049/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cuda:latest
```

The `--pull always` flag ensures you get the latest image from Docker Hub.

### 3. View Logs

**View container output (stdout):**
```bash
docker logs -f quip-cpu
# or
docker logs -f quip-cuda
```

**View data directory contents:**
```bash
ls -la ~/quip-data/
# config.toml  - Node configuration (auto-generated)
# trust.db     - TOFU peer certificate database
```

## Environment Variables

All settings live in `/data/config.toml` (source of truth). ENV vars override the TOML only when set (non-empty). Edit the TOML directly for persistent changes.

| Variable | TOML key | TOML default | Description |
|----------|----------|--------------|-------------|
| `QUIP_LISTEN` | `listen` | `::` | Address to bind (dual-stack: accepts IPv4+IPv6) |
| `QUIP_PORT` | `port` | `20049` | Port to bind |
| `QUIP_PUBLIC_HOST` | `public_host` | (auto-detect) | Public hostname or IP for peer advertisement |
| `QUIP_PUBLIC_PORT` | `public_port` | same as `port` | Public port (if different, e.g. behind NAT) |
| `QUIP_NODE_NAME` | `node_name` | (hostname) | Human-readable node name |
| `QUIP_AUTO_MINE` | `auto_mine` | `false` | Enable auto-mining |
| `QUIP_PEERS` | `peer` | `["nodes.quip.network:20049"]` | Comma-separated peer list (TOML uses array) |
| `CERT_EMAIL` | (unset) | ACME email — enables certbot when set with a DNS domain |
| `CERT_CHALLENGE` | (unset→http) | `http` (port 80) or `dns` |
| `CERT_DNS_PLUGIN` | (unset) | cloudflare, route53, google, digitalocean, ovh, rfc2136 |
| `CERT_DNS_CREDENTIALS` | (unset) | Path to DNS credentials file (e.g. `/data/certs/cf.ini`) |
| `CERT_ACME_SERVER` | (unset→LE) | Custom ACME URL (ZeroSSL, Buypass). Default: Let's Encrypt |
| `CERT_EAB_KID` | (unset) | EAB Key ID (ZeroSSL, Buypass) |
| `CERT_EAB_HMAC_KEY` | (unset) | EAB HMAC Key (base64url) |
| `CERT_STAGING` | (unset→false) | Use Let's Encrypt staging server |

**IPv6 Support:** The default `QUIP_LISTEN=::` enables dual-stack mode, accepting both IPv4 and IPv6 connections. For IPv6-only, use `QUIP_LISTEN=::1`. For IPv4-only, use `QUIP_LISTEN=0.0.0.0`.

**Default peers:**
```
qpu-1.nodes.quip.network, cpu-1.quip.carback.us, gpu-1.quip.carback.us, gpu-2.quip.carback.us
```

## TLS Certificates (Let's Encrypt)

Certbot activates automatically when `QUIP_PUBLIC_HOST` is a DNS name and `CERT_EMAIL` is set:

```bash
docker run -d --name quip-cpu \
  -v ~/quip-data:/data \
  -e QUIP_PUBLIC_HOST=mynode.example.com \
  -e CERT_EMAIL=admin@example.com \
  -p 20049:20049/udp -p 20049:20049/tcp -p 80:80/tcp \
  registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest
```

For DNS-01 challenges (no port 80 needed), custom ACME providers, or advanced configuration, see [TLS.md](TLS.md).

## Persistent Data

Mount a volume at `/data` to persist:
- `config.toml` - Node configuration (source of truth, seeded on first run)
- `trust.db` - TOFU peer certificate database
- `certs/` - TLS certificates and certbot state

Edit `config.toml` directly for persistent changes — ENV vars only override when non-empty.

## Building Images

### Single Architecture

```bash
# Build CPU node
docker build -f Dockerfile.cpu -t quip-network-node-cpu:latest ..

# Build CUDA node
docker build -f Dockerfile.cuda -t quip-network-node-cuda:latest ..
```

### Multi-Architecture Builds with buildx

Build and push to Docker Hub:

```bash
# CPU (amd64 + arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cpu \
  -t registry.gitlab.com/piqued/quip-protocol/quip-network-node-cpu:latest \
  --push ..

# CUDA (amd64 + arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cuda \
  -t registry.gitlab.com/piqued/quip-protocol/quip-network-node-cuda:latest \
  --push ..
```

## Multi-Node Testing with Docker Compose

```bash
docker-compose up cpu-bootstrap cpu-node-2
```

## Configuration Files

Each mode has its own configuration template:
- `quip-node.cpu.toml` - CPU mode with `[cpu]` section
- `quip-node.cuda.toml` - CUDA mode with `[gpu]` backend="local"

On first run, the entrypoint copies the template to `/data/config.toml` and generates a random secret.

## Troubleshooting

**Cannot connect to peers:**
- Ensure `QUIP_PUBLIC_HOST` is set to your public IP/hostname
- Check firewall allows UDP/TCP on port 20049
- Verify the volume mount persists `/data`

**GPU not detected:**
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

---

## Additional Resources

- **Mining Rate Experiments:** [mining_rates/README.md](mining_rates/README.md)
- **AWS Deployment:** [../aws/README_AWS_DEPLOYMENT.md](../aws/README_AWS_DEPLOYMENT.md)
- **Akash Deployment:** [../akash/README_AKASH.md](../akash/README_AKASH.md)
- **Project Guide:** [../CLAUDE.md](../CLAUDE.md)
