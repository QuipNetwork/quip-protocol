#!/bin/bash
# User-data script for CPU mining EC2 instances
# This script runs on instance startup and performs the mining experiment
set -e

# ====================
# Configuration
# ====================
# These variables should be replaced with actual values before launch
# or passed via EC2 instance tags

S3_BUCKET="${S3_BUCKET:-REPLACE_WITH_S3_BUCKET}"
MINING_DURATION="${MINING_DURATION:-30m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"
ECR_REGISTRY="${ECR_REGISTRY:-}"  # Optional ECR registry
AUTO_TERMINATE="${AUTO_TERMINATE:-true}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/your-repo/quip-protocol.git}"

# ====================
# System Setup
# ====================
echo "Starting CPU mining setup..."
date

# Update package lists
apt-get update -y

# Install required packages
apt-get install -y \
    docker.io \
    awscli \
    curl \
    git \
    jq

# Start and enable Docker
systemctl start docker
systemctl enable docker

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
    docker pull ${ECR_REGISTRY}/cpu-miner:latest
    IMAGE=${ECR_REGISTRY}/cpu-miner:latest
else
    echo "Building Docker image from source..."
    cd /tmp
    git clone $GITHUB_REPO || { echo "Failed to clone repository"; exit 1; }
    cd quip-protocol
    docker build -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:latest .
    IMAGE=quip-protocol/cpu-miner:latest
fi

# ====================
# Run Mining Container
# ====================
echo "Starting mining container..."
echo "Configuration:"
echo "  Duration: $MINING_DURATION"
echo "  Difficulty Energy: $DIFFICULTY_ENERGY"
echo "  Min Diversity: $MIN_DIVERSITY"
echo "  Min Solutions: $MIN_SOLUTIONS"
echo "  Topology: $TOPOLOGY_FILE"
echo "  S3 Bucket: $S3_BUCKET"

mkdir -p /tmp/output

docker run --rm \
    -v /tmp/output:/output \
    -e MINING_DURATION=$MINING_DURATION \
    -e DIFFICULTY_ENERGY=$DIFFICULTY_ENERGY \
    -e MIN_DIVERSITY=$MIN_DIVERSITY \
    -e MIN_SOLUTIONS=$MIN_SOLUTIONS \
    -e TOPOLOGY_FILE=$TOPOLOGY_FILE \
    -e S3_BUCKET=$S3_BUCKET \
    -e AWS_DEFAULT_REGION=$REGION \
    $IMAGE

EXIT_CODE=$?

echo "Mining container exited with code: $EXIT_CODE"
date

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
