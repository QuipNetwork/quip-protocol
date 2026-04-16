#!/bin/bash
# EC2 user-data script for H100 GPU mining
# Pulls Docker image, launches one container per GPU, auto-terminates
set -e

# Configuration (override via launch template or env injection)
MINING_DURATION="${MINING_DURATION:-90m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1.json.gz}"
DOCKER_IMAGE="${DOCKER_IMAGE:-carback1/quip-protocol-cuda-miner:latest}"
AUTO_TERMINATE="${AUTO_TERMINATE:-true}"

# IPFS configuration (required for result upload)
IPFS_NODE="${IPFS_NODE:-}"
IPFS_API_KEY="${IPFS_API_KEY:-}"

echo "========================================="
echo "Quip Mining - EC2 Startup"
echo "========================================="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Region: $(curl -s http://169.254.169.254/latest/meta-data/placement/region)"
echo "Duration: $MINING_DURATION"
echo "Difficulty: $DIFFICULTY_ENERGY"
echo "Image: $DOCKER_IMAGE"
date

# Install NVIDIA Docker runtime
apt-get update -y
apt-get install -y docker.io curl jq
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update -y
apt-get install -y nvidia-docker2
systemctl restart docker

# Verify GPU access
echo "GPU inventory:"
nvidia-smi -L
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Pull Docker image
echo "Pulling $DOCKER_IMAGE..."
docker pull "$DOCKER_IMAGE"

# Get instance metadata for unique deployment IDs
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Launch one container per GPU
echo "Launching $NUM_GPUS containers (one per GPU)..."
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  Starting container for GPU $GPU_ID..."
    docker run -d \
        --name "miner-gpu${GPU_ID}" \
        --gpus "\"device=${GPU_ID}\"" \
        -v "/tmp/output/gpu${GPU_ID}:/output" \
        -e MINER_TYPE=cuda \
        -e GPU_DEVICE=0 \
        -e MINING_DURATION="$MINING_DURATION" \
        -e DIFFICULTY_ENERGY="$DIFFICULTY_ENERGY" \
        -e MIN_DIVERSITY="$MIN_DIVERSITY" \
        -e MIN_SOLUTIONS="$MIN_SOLUTIONS" \
        -e TOPOLOGY_FILE="$TOPOLOGY_FILE" \
        -e DEPLOYMENT_ID="${INSTANCE_ID}-gpu${GPU_ID}" \
        -e IPFS_NODE="$IPFS_NODE" \
        -e IPFS_API_KEY="$IPFS_API_KEY" \
        "$DOCKER_IMAGE"
done

echo "All $NUM_GPUS containers launched."
echo "Waiting for completion..."

# Wait for all containers to finish
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Waiting for miner-gpu${GPU_ID}..."
    docker wait "miner-gpu${GPU_ID}" || true
    EXIT_CODE=$(docker inspect "miner-gpu${GPU_ID}" --format='{{.State.ExitCode}}' 2>/dev/null || echo "unknown")
    echo "  miner-gpu${GPU_ID} exited with code: $EXIT_CODE"
done

echo "========================================="
echo "All mining containers completed."
date
echo "========================================="

# List results
echo "Results:"
find /tmp/output -name "*.json" -type f 2>/dev/null || echo "  (no results found)"

# Auto-terminate
if [ "$AUTO_TERMINATE" = "true" ]; then
    REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
    echo "Auto-terminating in 60 seconds..."
    sleep 60
    aws ec2 terminate-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION" 2>/dev/null || true
fi
