#!/bin/bash
# User-data script for CUDA GPU mining EC2 instances
# This script runs on instance startup and performs the mining experiment
set -e

# ====================
# Configuration
# ====================
S3_BUCKET="${S3_BUCKET:-REPLACE_WITH_S3_BUCKET}"
MINING_DURATION="${MINING_DURATION:-30m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"
GPU_DEVICE="${GPU_DEVICE:-0}"
ECR_REGISTRY="${ECR_REGISTRY:-}"
AUTO_TERMINATE="${AUTO_TERMINATE:-true}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/your-repo/quip-protocol.git}"

# ====================
# System Setup
# ====================
echo "Starting CUDA GPU mining setup..."
date

# Update package lists
apt-get update -y

# Install required packages
apt-get install -y \
    docker.io \
    awscli \
    curl \
    git \
    jq \
    software-properties-common

# ====================
# NVIDIA Docker Runtime
# ====================
echo "Installing NVIDIA Docker runtime..."

# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
apt-get update -y
apt-get install -y nvidia-docker2

# Restart Docker to load NVIDIA runtime
systemctl restart docker
systemctl enable docker

# Verify NVIDIA GPU is accessible
echo "Verifying GPU access..."
nvidia-smi

# Test NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 nvidia-smi

# ====================
# Instance Metadata
# ====================
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

echo "Instance ID: $INSTANCE_ID"
echo "Region: $REGION"
echo "Availability Zone: $AZ"

# ====================
# Docker Image Setup
# ====================
if [ -n "$ECR_REGISTRY" ]; then
    echo "Pulling Docker image from ECR..."
    aws ecr get-login-password --region $REGION | \
        docker login --username AWS --password-stdin $ECR_REGISTRY
    docker pull ${ECR_REGISTRY}/cuda-miner:latest
    IMAGE=${ECR_REGISTRY}/cuda-miner:latest
else
    echo "Building Docker image from source..."
    cd /tmp
    git clone $GITHUB_REPO || { echo "Failed to clone repository"; exit 1; }
    cd quip-protocol
    docker build -f docker/Dockerfile.cuda -t quip-protocol/cuda-miner:latest .
    IMAGE=quip-protocol/cuda-miner:latest
fi

# ====================
# Run Mining Container
# ====================
echo "Starting CUDA mining container..."
echo "Configuration:"
echo "  Duration: $MINING_DURATION"
echo "  Difficulty Energy: $DIFFICULTY_ENERGY"
echo "  Min Diversity: $MIN_DIVERSITY"
echo "  Min Solutions: $MIN_SOLUTIONS"
echo "  Topology: $TOPOLOGY_FILE"
echo "  GPU Device: $GPU_DEVICE"
echo "  S3 Bucket: $S3_BUCKET"

mkdir -p /tmp/output

docker run --rm \
    --gpus all \
    -v /tmp/output:/output \
    -e MINING_DURATION=$MINING_DURATION \
    -e DIFFICULTY_ENERGY=$DIFFICULTY_ENERGY \
    -e MIN_DIVERSITY=$MIN_DIVERSITY \
    -e MIN_SOLUTIONS=$MIN_SOLUTIONS \
    -e TOPOLOGY_FILE=$TOPOLOGY_FILE \
    -e GPU_DEVICE=$GPU_DEVICE \
    -e S3_BUCKET=$S3_BUCKET \
    -e AWS_DEFAULT_REGION=$REGION \
    $IMAGE

EXIT_CODE=$?

echo "Mining container exited with code: $EXIT_CODE"
date

# ====================
# GPU Cleanup
# ====================
echo "Cleaning up GPU resources..."
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 sh -c "python3 -c 'import cupy; cupy.cuda.runtime.deviceReset()'" || true

# ====================
# Cleanup & Termination
# ====================
if [ "$AUTO_TERMINATE" == "true" ]; then
    echo "Auto-termination enabled. Terminating instance in 60 seconds..."
    sleep 60
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION || true
else
    echo "Auto-termination disabled. Instance will remain running."
fi

exit $EXIT_CODE
