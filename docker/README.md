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

## Configuration model

`/data/config.toml` is the single source of truth. It is seeded from the
image template on first run; edit it and restart the container to change
anything — validators, mode, faucet, telemetry bind, identity
(`node_name` / `public_host`), and the backend inventory. There are no
configuration env vars: the entrypoint prepares the volume, then execs
`quip-miner --config /data/config.toml`, which spawns and supervises
every process the config declares.

## Quick start

```bash
# Persistent /data for config + signing keystore
mkdir -p ~/quip-miner-data

docker run -d --pull always --name quip-miner \
  -v ~/quip-miner-data:/data \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest
```

On first start the entrypoint:

1. Seeds `/data/config.toml` from the image template (v0.2 `[miner]` schema,
   default validator `ws://127.0.0.1:9944`).
2. Generates a fresh hybrid sr25519 + ML-DSA-44 keystore at
   `/data/keystore.json` (mode `0600`). **Back this up — the chain key
   lives here.**
3. Resolves the mining subcommand(s) from the config and starts mining.

Then point it at your validators and give it a name:

```bash
$EDITOR ~/quip-miner-data/config.toml   # validators = [...], node_name = "..."
docker restart quip-miner
```

For CUDA:

```bash
docker run -d --pull always --gpus all --name quip-miner-gpu \
  -v ~/quip-miner-data:/data \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cuda:latest
```

View logs:

```bash
docker logs -f quip-miner
```

## Mode resolution

`quip-miner --config /data/config.toml` reads the config and runs
everything it declares:

1. Each declared backend group becomes one worker process. `[cpu]` →
   `quip-miner cpu`, `[gpu]`/`[cuda.N]`/`[metal]`/`[modal]` →
   `quip-miner gpu`, `[qpu]`/`[dwave]`/`[ibm]`/`[braket]`/`[pasqal]`/
   `[ionq]`/`[origin]` → `quip-miner qpu`. (The per-mode subcommands
   remain available as test/ops tooling.)
2. The telemetry aggregator is spawned when `rest_port > 0`.
3. The built-in supervisor forwards SIGTERM/SIGINT to every child and
   tears down the container if any child exits.

The templates ship with an active default section (`[cpu]` on the cpu
image, `[cuda.0]` on the cuda image), so what runs is always stated in
the config. The work source (`pow` / `mempool` / `both`) comes from the
`[miner] mode` key.

Capability is probed from the installed libraries: a config asking for
hardware the image can't run (e.g. `[cuda.0]` on the CPU image, which
lacks `cupy`) is rejected with `unsupported-mode` before any child
starts.

## Telemetry aggregation

Multi-process containers run a single `quip-miner telemetry`
aggregator alongside the per-mode mining children:

```
docker container
├── quip-miner telemetry   (binds rest_host:rest_port from config.toml)
├── quip-miner cpu         (writes telemetry-stats-cpu.json)
├── quip-miner gpu         (writes telemetry-stats-gpu.json)
└── quip-miner qpu         (writes telemetry-stats-qpu.json)
```

The aggregator globs `/data/runtime/telemetry-stats-*.json` (an
internal directory derived from the config location) on
every `/api/v1/*` request, merges them
(`shared.stats_snapshot.merge_snapshots`), and serves the unified
view to the indexer. Counters sum across modes; `miners[]` is
unioned; `node_id` / `ss58_address` / `descriptor` take the first
non-empty value; per-mode breakdown is exposed under the
`modes` key for operators investigating per-backend behaviour.

Each worker child skips its in-process REST sibling (internal glue set
by the supervisor). The indexer still sees one endpoint per container —
the aggregator hides the multi-process fan-out.

Expose the API by publishing the configured port:

```bash
docker run -d --name quip-miner \
  -v ~/quip-miner-data:/data \
  -p 8086:8086 \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest

curl http://localhost:8086/api/v1/status
```

## Environment variables

Miner configuration is **not** env-driven — it lives in
`/data/config.toml` (schema documented in
[`quip-miner.example.toml`](../quip-miner.example.toml)). The only env
vars are host properties:

| Variable | Role | Default |
|----------|------|---------|
| `PUID` / `PGID` | runtime uid/gid mapping for `/data` ownership | `1000:1000` |

## File ownership (PUID/PGID)

The container drops from root to a non-root `quip` user (UID/GID 1000 by default) before launching the miner, so files under `/data` are host-readable without `sudo`. To match your host user's UID/GID:

```bash
docker run -d --name quip-miner \
  -v ~/quip-miner-data:/data \
  -e PUID=$(id -u) -e PGID=$(id -g) \
  registry.gitlab.com/quip.network/quip-protocol/quip-miner-cpu:latest
```

`PUID=0` keeps the container running as root (legacy mode).

## Persistent data

Mount a volume at `/data`:

- `config.toml` — the configuration (seeded from the image template on first run)
- `keystore.json` — hybrid keystore (auto-generated on first run; back this up)

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

Edit the seeded `config.toml` in the service's volume to configure it.

---

## Additional resources

- **Mining-rate experiments:** [mining_rates/README.md](mining_rates/README.md)
- **AWS Deployment:** [../aws/README_AWS_DEPLOYMENT.md](../aws/README_AWS_DEPLOYMENT.md)
- **Akash Deployment:** [../akash/README_AKASH.md](../akash/README_AKASH.md)
- **Project Guide:** [../CLAUDE.md](../CLAUDE.md)
