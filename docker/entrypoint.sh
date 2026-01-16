#!/bin/bash
# Entrypoint script for Quip Network Node (CPU and GPU)
# Generates secret on first run, auto-detects all CPUs/GPUs
set -e

echo "========================================"
echo "Quip Protocol Network Node"
echo "========================================"
echo "Start time: $(date)"
echo "Mode: $QUIP_MODE"

CONFIG_FILE="/data/config.toml"
TEMPLATE_FILE="/app/quip-node.docker.toml"

# Generate config with secret on first run
if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run detected - generating config with new secret..."
    mkdir -p /data

    # Generate random 64-char hex secret
    SECRET=$(openssl rand -hex 32)

    # Copy template and set secret
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    sed -i "s/secret = \"GENERATE_ON_FIRST_RUN\"/secret = \"$SECRET\"/" "$CONFIG_FILE"

    echo "Config generated at $CONFIG_FILE"
    echo "Secret has been randomly generated and saved."
else
    echo "Using existing config at $CONFIG_FILE"
fi

# Auto-detect resources based on mode
if [ "$QUIP_MODE" = "gpu" ]; then
    echo "----------------------------------------"
    echo "GPU Mode - Detecting NVIDIA GPUs..."
    nvidia-smi 2>&1 || { echo "ERROR: nvidia-smi not available - no GPUs detected. Run with --gpus all"; exit 1; }
    echo "----------------------------------------"

    NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo "Detected GPUs: $NUM_GPUS"

    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: No GPUs detected. Make sure to run with --gpus all"
        exit 1
    fi

    # Build device arguments for all detected GPUs
    DEVICE_ARGS=""
    for ((i=0; i<NUM_GPUS; i++)); do
        DEVICE_ARGS="$DEVICE_ARGS --device $i"
    done
    echo "GPU devices: 0-$((NUM_GPUS-1))"
    MODE_ARGS="gpu --gpu-backend local $DEVICE_ARGS"

else
    # CPU mode (default)
    NUM_CPUS=$(nproc)
    echo "CPU Mode - Detected CPUs: $NUM_CPUS"
    MODE_ARGS="cpu --num-cpus $NUM_CPUS"
fi

# Parse peer arguments from comma-separated list (with defaults)
DEFAULT_PEERS="qpu-1.nodes.quip.network,cpu-1.quip.carback.us,gpu-1.quip.carback.us,gpu-2.quip.carback.us"
QUIP_PEERS="${QUIP_PEERS:-$DEFAULT_PEERS}"

PEERS_ARG=""
echo "Peers: $QUIP_PEERS"
IFS=',' read -ra PEER_ARRAY <<< "$QUIP_PEERS"
for peer in "${PEER_ARRAY[@]}"; do
    peer=$(echo "$peer" | xargs)  # Trim whitespace
    if [ -n "$peer" ]; then
        PEERS_ARG="$PEERS_ARG --peer $peer"
    fi
done

# Set port (default 20049)
PORT="${QUIP_PORT:-20049}"
echo "Port: $PORT"

# Set listen address (default 0.0.0.0)
LISTEN="${QUIP_LISTEN:-0.0.0.0}"
echo "Listen: $LISTEN"

# Build optional arguments
PUBLIC_HOST_ARG=""
if [ -n "$QUIP_PUBLIC_HOST" ]; then
    PUBLIC_HOST_ARG="--public-host $QUIP_PUBLIC_HOST"
    echo "Public Host: $QUIP_PUBLIC_HOST"
else
    echo "Warning: QUIP_PUBLIC_HOST not set - node may not be reachable by peers"
fi

NODE_NAME_ARG=""
if [ -n "$QUIP_NODE_NAME" ]; then
    NODE_NAME_ARG="--node-name $QUIP_NODE_NAME"
    echo "Node Name: $QUIP_NODE_NAME"
fi

# Auto-mine setting (default: false)
AUTO_MINE_ARG="--no-auto-mine"
if [ "$QUIP_AUTO_MINE" = "true" ]; then
    AUTO_MINE_ARG="--auto-mine"
    echo "Auto-mine: enabled"
else
    echo "Auto-mine: disabled"
fi

echo "========================================"
echo "Starting Quip Network Node..."
echo "========================================"

# Construct and execute the command
CMD="quip-network-node --config $CONFIG_FILE $MODE_ARGS \
    --listen $LISTEN \
    --port $PORT \
    --genesis-config genesis_block_public.json \
    $PUBLIC_HOST_ARG \
    $NODE_NAME_ARG \
    $AUTO_MINE_ARG \
    $PEERS_ARG"

echo "Command: $CMD"
echo "----------------------------------------"

# Execute the network node (replace shell process for proper signal handling)
exec $CMD
