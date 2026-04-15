# Akash Network Mining Deployment

Deploy QUIP miners on the Akash Network decentralized cloud.

## Quick Start

Use the standalone Akash deployer tool: [gitlab.com/carback1/akash-deployer](https://gitlab.com/carback1/akash-deployer)

The deployer handles wallet connection (Keplr), SDL generation, bid selection, lease management, and IPFS result collection through a browser-based interface.

## Docker Images

The same images work on both Akash and AWS:

- **CPU**: `carback1/quip-protocol-cpu-miner:latest`
- **CUDA**: `carback1/quip-protocol-cuda-miner:latest`

Build locally:

```bash
./minertest/build_images.sh
```

## Manual CLI Deploy

If deploying without the web tool, create an SDL file and use the `akash` CLI:

```bash
# Install akash CLI: https://docs.akash.network/guides/cli

# Create deployment
akash tx deployment create deploy.yaml --from $WALLET --chain-id akashnet-2

# List bids
akash query market bid list --owner $WALLET --dseq $DSEQ

# Accept bid
akash tx market lease create --from $WALLET --dseq $DSEQ --provider $PROVIDER --gseq 1 --oseq 1

# Send manifest
akash provider send-manifest deploy.yaml --from $WALLET --provider $PROVIDER --dseq $DSEQ

# Check status
akash provider lease-status --from $WALLET --provider $PROVIDER --dseq $DSEQ
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MINER_TYPE` | `cpu` | `cpu` or `cuda` |
| `MINING_DURATION` | `90m` | Duration string (e.g., `5m`, `1h`, `90m`) |
| `DIFFICULTY_ENERGY` | `-14900` | Energy threshold for mining |
| `MIN_DIVERSITY` | `0.1` | Minimum solution diversity |
| `MIN_SOLUTIONS` | `5` | Minimum solutions per block |
| `TOPOLOGY_FILE` | `...advantage2_system1_13.json.gz` | Ising topology file |
| `GPU_DEVICE` | `0` | GPU device index (CUDA only) |
| `IPFS_API_URL` | | IPFS node URL for result upload |
| `IPFS_API_KEY` | | IPFS API authentication key |
