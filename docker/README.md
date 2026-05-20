# Docker Deployment Guide

Docker images for running `quip-miner` against a Substrate validator.

For mining-rate experiments and benchmarking, see [mining_rates/README.md](mining_rates/README.md).

---

## Images

- **CPU miner** — `quip-miner-cpu` (amd64 + arm64)
- **CUDA miner** — `quip-miner-cuda` (amd64)

Available from the GitLab Container Registry:

```
registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu
registry.gitlab.com/quip.network/quip-protocol/quip-miner-cuda
```

Mutable tags:
- `:latest` — built from `main`
- `:v0.2-preview` — built from `v0.2` (in-progress)
- `:<full-sha>` — every push, pinnable

For Apple Silicon (Metal) GPU mining, run directly on macOS without Docker.

## Quick start

```bash
# Persistent /data for config + signing keystore
mkdir -p ~/quip-miner-data

docker run -d --pull always --name quip-miner \
  -v ~/quip-miner-data:/data \
  -e QUIP_VALIDATORS=wss://validator-1.quip.network/rpc,wss://validator-2.quip.network/rpc \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest
```

On first start the entrypoint:

1. Seeds `/data/config.toml` from the image template (v0.2 `[miner]` schema).
2. Generates a fresh hybrid sr25519 + ML-DSA-44 keystore at `/data/keystore.json` (mode `0600`). **Back this up — the chain key lives here.**
3. Connects to the first reachable URL in `QUIP_VALIDATORS` and starts mining PoW.

For CUDA:

```bash
docker run -d --pull always --gpus all --name quip-miner-gpu \
  -v ~/quip-miner-data:/data \
  -e QUIP_VALIDATORS=wss://validator-1.quip.network/rpc \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cuda:latest
```

View logs:

```bash
docker logs -f quip-miner
```

## Environment variables

ENV vars are converted to CLI flags at launch time. The CLI's
precedence (`CLI > TOML > defaults`) means ENV-supplied values
override the TOML, and the TOML overrides image defaults.

| Variable | CLI flag | Default |
|----------|----------|---------|
| `QUIP_VALIDATORS` | `--validator URL` (comma-split, repeated) | empty — TOML must provide |
| `QUIP_SIGNER_KEY` | `--signer-key PATH` | `/data/keystore.json` |
| `QUIP_FAUCET_URL` | `--faucet-url URL` | unset (fails fast if underfunded) |
| `QUIP_MODE` | (chooses subcommand) | `cpu` (cpu image) / `gpu` (cuda image) |
| `QUIP_MINE_MODE` | `--mode pow\|mempool\|both` | `pow` |
| `QUIP_REST_PORT` | `--rest-port N` | `-1` (telemetry disabled) |
| `QUIP_REST_HOST` | `--rest-host HOST` | `0.0.0.0` (in-container default) |
| `PUID` / `PGID` | (runtime uid/gid mapping) | `1000:1000` |

Operators can edit `/data/config.toml` directly for persistent settings; the schema is documented in [`quip-miner.example.toml`](../quip-miner.example.toml).

## File ownership (PUID/PGID)

The container drops from root to a non-root `quip` user (UID/GID 1000 by default) before launching the miner, so files under `/data` are host-readable without `sudo`. To match your host user's UID/GID:

```bash
docker run -d --name quip-miner \
  -v ~/quip-miner-data:/data \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  -e QUIP_VALIDATORS=wss://validator-1.quip.network/rpc \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest
```

`PUID=0` keeps the container running as root (legacy mode).

## Persistent data

Mount a volume at `/data`:

- `config.toml` — `[miner]` config (seeded from the image template on first run)
- `keystore.json` — hybrid keystore (auto-generated on first run; back this up)

## Telemetry REST API

Off by default. Enable per-container:

```bash
docker run -d --name quip-miner \
  -v ~/quip-miner-data:/data \
  -e QUIP_VALIDATORS=... \
  -e QUIP_REST_PORT=8086 \
  -p 8086:8086 \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest

curl http://localhost:8086/api/v1/status
```

## Building images locally

```bash
# CPU
docker build -f Dockerfile.cpu -t quip-miner-cpu:local ..

# CUDA
docker build -f Dockerfile.cuda -t quip-miner-cuda:local ..
```

Multi-arch buildx:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f Dockerfile.cpu \
  -t registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest \
  --push ..
```

## Compose

A minimal compose file lives at `docker/docker-compose.yml`:

```bash
docker compose -f docker/docker-compose.yml up cpu-miner
docker compose -f docker/docker-compose.yml up gpu-miner
```

Edit the `QUIP_VALIDATORS` env var (or the seeded `config.toml`) before starting.

---

## Additional resources

- **Mining-rate experiments:** [mining_rates/README.md](mining_rates/README.md)
- **AWS Deployment:** [../aws/README_AWS_DEPLOYMENT.md](../aws/README_AWS_DEPLOYMENT.md)
- **Akash Deployment:** [../akash/README_AKASH.md](../akash/README_AKASH.md)
- **Project Guide:** [../CLAUDE.md](../CLAUDE.md)
