# Docker Deployment Guide

This guide covers building, testing, and deploying Dockerized mining experiments for the Quip Protocol.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Building Docker Images](#building-docker-images)
4. [Testing Locally](#testing-locally)
5. [AWS Deployment](#aws-deployment)
6. [Akash Network Deployment](#akash-network-deployment)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Quip Protocol provides three Docker images for mining rate comparison experiments:

- **CPU Miner** - Simulated annealing on CPU (Debian-based, amd64)
- **CUDA Miner** - GPU-accelerated mining with NVIDIA CUDA 13.x (Ubuntu-based, amd64)
- **Metal Miner** - Apple Silicon GPU mining (arm64, limited Docker support)

Each image runs the `compare_mining_rates.py` tool with configurable parameters for duration, difficulty, diversity, solutions, and topology.

### Key Features

✅ **Parameterized via Environment Variables** - No code changes needed
✅ **Automatic S3 Upload** - Results synced to S3 automatically
✅ **Auto-Termination** - Optional instance shutdown after completion
✅ **Volume Mounts** - Easy access to results and topology files
✅ **Production Ready** - Includes logging, error handling, cleanup

---

## Quick Start

### Build All Images

```bash
# From repository root
cd /path/to/quip-protocol

# Build all three images
./docker/build_all.sh
```

### Run Automated Tests

```bash
# Quick 10-second tests for all images
./docker/test_local.sh

# Custom test duration
TEST_DURATION=30s ./docker/test_local.sh
```

### Run Individual Miner

```bash
# CPU miner (30-minute run)
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=30m \
  -e DIFFICULTY_ENERGY=-14900 \
  quip-protocol/cpu-miner:latest

# Check results
cat docker/output/cpu_*.json | jq '.results'
```

---

## Building Docker Images

### Build Individual Images

**CPU Miner:**
```bash
docker build -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:latest .
```

**CUDA Miner:**
```bash
docker build -f docker/Dockerfile.cuda -t quip-protocol/cuda-miner:latest .
```

**Metal Miner (macOS only):**
```bash
docker build --platform linux/arm64 -f docker/Dockerfile.metal -t quip-protocol/metal-miner:latest .
```

### Build All Images with Versioning

```bash
# Build with version tag
VERSION=v1.0.0 ./docker/build_all.sh

# Push to ECR (optional)
ECR_REGISTRY=123456.dkr.ecr.us-east-1.amazonaws.com \
PUSH_TO_REGISTRY=1 \
./docker/build_all.sh
```

### Build Options

**Custom base image:**
```bash
# Modify Dockerfile.cpu to use different Python version
# FROM python:3.12-slim-bookworm
docker build -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:py312 .
```

**Build with no cache (clean rebuild):**
```bash
docker build --no-cache -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:latest .
```

---

## Testing Locally

### Automated Testing (Recommended)

The `test_local.sh` script runs quick validation tests on all images:

```bash
# Default: 10-second test runs
./docker/test_local.sh

# Custom test parameters
TEST_DURATION=30s \
TEST_DIFFICULTY=-4100 \
TEST_DIVERSITY=0.15 \
./docker/test_local.sh
```

**What it tests:**
- ✅ Image builds successfully
- ✅ Container starts and runs
- ✅ Mining completes without errors
- ✅ Output JSON file is created
- ✅ Log file is created
- ✅ JSON contains expected fields

### Manual Testing

#### 1. Test CPU Miner

```bash
# Create output directory
mkdir -p docker/output

# Run 10-second test
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=10s \
  -e DIFFICULTY_ENERGY=-14900 \
  -e MIN_DIVERSITY=0.1 \
  -e MIN_SOLUTIONS=5 \
  -e TOPOLOGY_FILE=dwave_topologies/topologies/advantage2_system1_7.json.gz \
  quip-protocol/cpu-miner:latest

# Verify output
ls -lh docker/output/
cat docker/output/cpu_*.json | jq '.results'
cat docker/output/cpu_*.log
```

**Expected output:**
```json
{
  "config": {
    "miner_type": "cpu",
    "difficulty_energy": -14900.0,
    "duration": "10s",
    "topology": "advantage2_system1_7"
  },
  "results": {
    "total_attempts": 50,
    "blocks_found": 0,
    "success_rate": 0.0,
    "blocks_per_minute": 0.0,
    "avg_energy": -14850.2,
    "avg_diversity": 0.14,
    "avg_solutions": 6.5
  }
}
```

#### 2. Test CUDA Miner

**Prerequisites:**
- NVIDIA GPU with CUDA 13.x support
- NVIDIA Docker runtime installed
- GPU accessible from Docker

**Verify GPU access:**
```bash
# Check GPU
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 nvidia-smi
```

**Run test:**
```bash
docker run --rm \
  --gpus all \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=10s \
  -e DIFFICULTY_ENERGY=-14900 \
  -e GPU_DEVICE=0 \
  quip-protocol/cuda-miner:latest

# Check output
cat docker/output/cuda_*.json | jq '.results'
```

**No GPU available?**
```bash
# You can still build the image to test the build process
docker build -f docker/Dockerfile.cuda -t quip-protocol/cuda-miner:latest .
echo "Build successful! Runtime testing requires NVIDIA GPU."
```

#### 3. Test Metal Miner

**Option A: Docker (Limited Support)**

```bash
# macOS with Apple Silicon only
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=10s \
  -e DIFFICULTY_ENERGY=-14900 \
  quip-protocol/metal-miner:latest
```

⚠️ **Note:** Metal framework has limited Docker support. Use Option B for production.

**Option B: Direct Execution (Recommended)**

```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install \
  "pyobjc-framework-Metal" \
  "pyobjc-framework-MetalPerformanceShaders" \
  "dwave-ocean-sdk>=6.0.0" \
  "numpy>=1.24.0" \
  "pandas>=2.0.0" \
  "click>=8.1.7"

# Run test
python tools/compare_mining_rates.py \
  --miner-type metal \
  --difficulty-energy -14900 \
  --duration 10s \
  --min-diversity 0.1 \
  --min-solutions 5 \
  -o docker/output/metal_test.json \
  2>&1 | tee docker/output/metal_test.log

# Check output
cat docker/output/metal_test.json | jq '.results'
```

### Testing with Docker Compose

**Create environment file:**
```bash
cat > docker/.env <<EOF
MINING_DURATION=30s
DIFFICULTY_ENERGY=-14900
MIN_DIVERSITY=0.1
MIN_SOLUTIONS=5
TOPOLOGY_FILE=dwave_topologies/topologies/advantage2_system1_7.json.gz
EOF
```

**Run services:**
```bash
cd docker

# Test CPU miner
docker-compose up cpu-miner

# Test CUDA miner (requires GPU)
docker-compose up cuda-miner

# Test Metal miner (macOS only)
docker-compose -f docker-compose.mac.yml up metal-miner
```

### Verifying Output Files

After any test run, verify the output files:

```bash
# List output files
ls -lh docker/output/

# Validate JSON structure
cat docker/output/cpu_*.json | jq '.'

# Check required fields
cat docker/output/cpu_*.json | jq '{
  miner_type: .config.miner_type,
  difficulty: .config.difficulty_energy,
  duration: .config.duration,
  attempts: .results.total_attempts,
  blocks: .results.blocks_found,
  success_rate: .results.success_rate,
  blocks_per_min: .results.blocks_per_minute
}'

# View log file
cat docker/output/cpu_*.log | tail -50
```

### Testing Different Configurations

**Different topologies:**
```bash
# Small topology (Z(8,2))
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e TOPOLOGY_FILE=dwave_topologies/topologies/zephyr_z8_t2.json.gz \
  -e DIFFICULTY_ENERGY=-2700 \
  quip-protocol/cpu-miner:latest

# Large topology (Z(11,4))
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e TOPOLOGY_FILE=dwave_topologies/topologies/zephyr_z11_t4.json.gz \
  -e DIFFICULTY_ENERGY=-15000 \
  quip-protocol/cpu-miner:latest
```

**Different difficulty levels:**
```bash
# Easy difficulty
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e DIFFICULTY_ENERGY=-4100 \
  -e MINING_DURATION=5m \
  quip-protocol/cpu-miner:latest

# Hard difficulty
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e DIFFICULTY_ENERGY=-15450 \
  -e MINING_DURATION=5m \
  quip-protocol/cpu-miner:latest
```

**Multiple parallel tests:**
```bash
# Run 3 CPU miners in parallel with different difficulties
for DIFF in -4100 -14900 -15450; do
  docker run -d \
    --name cpu-miner-diff${DIFF} \
    -v $(pwd)/docker/output:/output \
    -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
    -e DIFFICULTY_ENERGY=$DIFF \
    -e MINING_DURATION=10m \
    quip-protocol/cpu-miner:latest
done

# Monitor progress
docker ps

# View logs
docker logs -f cpu-miner-diff-4100

# Clean up
docker rm -f cpu-miner-diff-4100 cpu-miner-diff-14900 cpu-miner-diff-15450
```

### Testing S3 Integration

**Test S3 upload (requires AWS credentials):**

```bash
# Set up AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Or mount AWS credentials
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -v ~/.aws:/root/.aws:ro \
  -e S3_BUCKET=your-test-bucket \
  -e MINING_DURATION=10s \
  quip-protocol/cpu-miner:latest

# Verify upload
aws s3 ls s3://your-test-bucket/cpu/
```

---

## AWS Deployment

Once local testing is complete, deploy to AWS for large-scale experiments.

### Prerequisites

1. **AWS CLI configured** with appropriate credentials
2. **Docker images** built and optionally pushed to ECR
3. **S3 bucket** and IAM roles set up

### Setup S3 and IAM

```bash
cd aws

# Create S3 bucket and IAM roles
./setup_s3_output.sh

# Save the output
export EXPERIMENT_ID=exp_20250119_120000
export S3_BUCKET=quip-mining-results-exp_20250119_120000
export IAM_ROLE_NAME=QuipMiningEC2Role-exp_20250119_120000
```

### Launch Fleet

**CPU Fleet (100 instances, 30 minutes):**
```bash
export FLEET_SIZE=100
export MINING_DURATION=30m
export DIFFICULTY_ENERGY=-14900
export USE_SPOT=true

./launch_cpu_fleet.sh
```

**CUDA Fleet (100 instances, 30 minutes):**
```bash
export FLEET_SIZE=100
export INSTANCE_TYPE=g4dn.xlarge  # or g5.xlarge for better performance
export MINING_DURATION=30m
export USE_SPOT=true

./launch_cuda_fleet.sh
```

**Metal Fleet (10 instances, 24 hours):**
```bash
export FLEET_SIZE=10
export INSTANCE_TYPE=mac2-m2.metal
export MINING_DURATION=24h

./launch_metal_fleet.sh
```

### Monitor Progress

```bash
# List instances
aws ec2 describe-instances \
  --filters Name=tag:Experiment,Values=$EXPERIMENT_ID \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' \
  --output table

# Watch S3 uploads
watch -n 30 "aws s3 ls s3://$S3_BUCKET/cpu/ | wc -l"
```

### Collect Results

```bash
# Download and analyze results
./collect_results.sh

# View summary
cat results/summary_${EXPERIMENT_ID}.txt

# View CSV
cat results/results_${EXPERIMENT_ID}.csv
```

### Terminate Fleet

```bash
# Terminate all instances
./terminate_fleet.sh
```

### Complete Documentation

For detailed AWS deployment instructions, see:
- [aws/README_AWS_DEPLOYMENT.md](aws/README_AWS_DEPLOYMENT.md) - Complete AWS guide
- [docker/README.md](docker/README.md) - Docker-specific documentation

---

## Akash Network Deployment

**Akash** is a decentralized cloud computing marketplace offering 2-5x cost savings compared to AWS. It's perfect for 90-minute mining experiments with automatic result retrieval via HTTP.

### Why Akash?

✅ **70% cheaper** than AWS for GPU workloads
✅ **Per-second billing** - no hourly minimums
✅ **HTTP result retrieval** - download logs and JSON directly
✅ **Decentralized** - no single point of failure
✅ **Easy deployment** - simple YAML configuration

### Quick Start

**1. Install Akash CLI:**
```bash
# macOS
brew tap ovrclk/tap
brew install akash-provider-services

# Linux
curl -sSfL https://raw.githubusercontent.com/akash-network/node/master/install.sh | sh
```

**2. Setup wallet and fund with AKT:**
```bash
# Create wallet
akash keys add default

# Fund wallet (buy AKT on exchanges or use testnet faucet)
# Minimum recommended: 10 AKT for testing
```

**3. Build and push Docker images:**
```bash
# Set your container registry
export REGISTRY=ghcr.io/your-username

# Build Akash-optimized images
./akash/build_akash_images.sh

# Push to registry
docker push $REGISTRY/quip-protocol-cpu-miner:latest
docker push $REGISTRY/quip-protocol-cuda-miner:latest
```

**4. Update SDL files with your registry:**
Edit `akash/deploy-cpu.yaml` and `akash/deploy-cuda.yaml`:
```yaml
services:
  cpu-miner:
    image: ghcr.io/your-username/quip-protocol-cpu-miner:latest
```

**5. Deploy to Akash:**
```bash
# Deploy 10 CPU instances
export MINER_TYPE=cpu
export FLEET_SIZE=10
./akash/deploy.sh

# Or deploy CUDA GPU instances
export MINER_TYPE=cuda
export FLEET_SIZE=10
./akash/deploy.sh
```

**6. Accept bids (follow on-screen instructions)**

**7. Wait 90 minutes for mining to complete**

**8. Retrieve results:**
```bash
# Download all results via HTTP
./akash/retrieve_results.sh akash/deployments_cpu_20250119_120000.txt

# Results saved to: ./akash_results/
# - deployment_<id>.json (mining results)
# - deployment_<id>.log (console logs)
# - summary.txt (aggregated statistics)
```

**9. Close deployments to stop billing:**
```bash
./akash/close_deployments.sh akash/deployments_cpu_20250119_120000.txt
```

### Cost Comparison (90-minute experiment)

| Configuration | Akash | AWS Spot | Savings |
|---------------|-------|----------|---------|
| 10 CPU instances | ~$0.30 | ~$0.63 | 52% |
| 10 CUDA GPU instances | ~$2.25 | ~$7.89 | 71% |

### Key Differences from AWS

**Akash:**
- Results available via HTTP endpoint (no S3 needed)
- Marketplace bidding model (wait for providers)
- Variable hardware specs
- Lower cost but less guaranteed uptime

**AWS:**
- Results via S3 or SSH
- Instant provisioning
- Consistent hardware
- Higher cost but enterprise-grade reliability

### Complete Documentation

For detailed Akash deployment instructions, see:
- **[akash/README_AKASH.md](akash/README_AKASH.md)** - Complete Akash guide with troubleshooting

---

## Configuration Reference

### Environment Variables

All Docker images support these environment variables:

| Variable | Description | Default | Examples |
|----------|-------------|---------|----------|
| `MINING_DURATION` | Mining duration | `30m` | `10s`, `5m`, `2h`, `1d`, `1w` |
| `DIFFICULTY_ENERGY` | Energy threshold | `-14900` | `-4100`, `-15450` |
| `MIN_DIVERSITY` | Solution diversity (0.0-1.0) | `0.1` | `0.15`, `0.2` |
| `MIN_SOLUTIONS` | Minimum solutions required | `5` | `3`, `10` |
| `TOPOLOGY_FILE` | Topology file path | `dwave_topologies/topologies/advantage2_system1_7.json.gz` | See topologies below |
| `GPU_DEVICE` | CUDA GPU device ID | `0` | `1`, `2` |
| `S3_BUCKET` | S3 bucket for uploads | (empty) | `quip-results-exp123` |
| `OUTPUT_FILE` | Output file path | (auto-generated) | `/output/custom.json` |
| `AWS_DEFAULT_REGION` | AWS region for S3 | (from credentials) | `us-east-1` |

### Available Topologies

| Topology | File | Nodes | Edges | Difficulty Range |
|----------|------|-------|-------|------------------|
| Z(8,2) | `zephyr_z8_t2.json.gz` | 1,088 | 6,068 | -2869 to -2677 |
| Z(9,2) | `zephyr_z9_t2.json.gz` | 1,368 | 7,692 | -4100 to -3870 |
| Z(10,2) | `zephyr_z10_t2.json.gz` | 1,680 | 9,508 | -5470 to -5200 |
| Z(11,4) | `zephyr_z11_t4.json.gz` | 4,048 | 38,520 | -15170 to -14158 |
| Advantage2 | `advantage2_system1_7.json.gz` | 4,593 | 41,796 | -15170 to -14158 |

### Volume Mounts

**Required mounts:**
```bash
-v $(pwd)/docker/output:/output                          # Results directory
-v $(pwd)/dwave_topologies:/app/dwave_topologies:ro      # Topology files (read-only)
```

**Optional mounts:**
```bash
-v ~/.aws:/root/.aws:ro                                  # AWS credentials (for S3)
-v $(pwd)/custom_topologies:/app/custom_topologies:ro    # Custom topology directory
```

### Output Files

Each run generates two files:

**JSON Results:** `{miner_type}_{instance_id}_{timestamp}.json`
```json
{
  "config": {
    "miner_type": "cpu",
    "difficulty_energy": -14900.0,
    "duration": "30m",
    "min_diversity": 0.1,
    "min_solutions": 5,
    "topology": "advantage2_system1_7"
  },
  "results": {
    "total_attempts": 1234,
    "blocks_found": 45,
    "success_rate": 0.0365,
    "blocks_per_minute": 1.5,
    "avg_energy": -14920.3,
    "avg_diversity": 0.187,
    "avg_solutions": 8.2
  },
  "energy_distribution": [...],
  "timestamps": [...]
}
```

**Log File:** `{miner_type}_{instance_id}_{timestamp}.log`
- Configuration parameters
- Real-time progress
- Error messages
- Completion status

---

## Troubleshooting

### Build Issues

**Error: "failed to solve with frontend dockerfile.v0"**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1
docker build -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:latest .
```

**Error: "no space left on device"**
```bash
# Clean up Docker resources
docker system prune -a

# Check disk space
df -h
```

**Error: "Cannot find Python 3.13"**
```bash
# For Dockerfile.cuda, ensure deadsnakes PPA is accessible
# Check your network connection or use a different base image
```

### Runtime Issues

**Error: "Cannot connect to Docker daemon"**
```bash
# Start Docker service
sudo systemctl start docker  # Linux
# Or open Docker Desktop on macOS/Windows

# Verify
docker ps
```

**Error: "CUDA not available" or "nvidia-smi not found"**
```bash
# Install NVIDIA Docker runtime (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 nvidia-smi
```

**Error: "permission denied" writing to /output**
```bash
# Create output directory with correct permissions
mkdir -p docker/output
chmod 777 docker/output

# Or run container with user permissions
docker run --rm \
  --user $(id -u):$(id -g) \
  -v $(pwd)/docker/output:/output \
  quip-protocol/cpu-miner:latest
```

**Error: "No such file or directory: dwave_topologies"**
```bash
# Ensure you're in repository root
cd /path/to/quip-protocol

# Verify topology files exist
ls -la dwave_topologies/topologies/

# Check volume mount is correct
docker run --rm \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  quip-protocol/cpu-miner:latest \
  ls -la /app/dwave_topologies/topologies/
```

**Error: Metal miner fails in Docker**
```bash
# Expected - Metal framework has limited Docker support
# Use direct execution on macOS instead:
python tools/compare_mining_rates.py --miner-type metal ...
```

### S3 Upload Issues

**Error: "Access Denied" when uploading to S3**
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check credentials are mounted
docker run --rm \
  -v ~/.aws:/root/.aws:ro \
  quip-protocol/cpu-miner:latest \
  cat /root/.aws/credentials

# Or use IAM role on EC2 (preferred)
```

**Error: "NoSuchBucket"**
```bash
# Create bucket first
aws s3 mb s3://your-bucket-name

# Verify bucket exists
aws s3 ls
```

### Performance Issues

**CPU miner is too slow**
```bash
# Reduce thread count
docker run --rm \
  -e OMP_NUM_THREADS=2 \
  quip-protocol/cpu-miner:latest

# Or use smaller topology
docker run --rm \
  -e TOPOLOGY_FILE=dwave_topologies/topologies/zephyr_z8_t2.json.gz \
  -e DIFFICULTY_ENERGY=-2700 \
  quip-protocol/cpu-miner:latest
```

**CUDA out of memory**
```bash
# Use smaller topology
docker run --rm \
  --gpus all \
  -e TOPOLOGY_FILE=dwave_topologies/topologies/zephyr_z9_t2.json.gz \
  quip-protocol/cuda-miner:latest

# Or use GPU with more memory (g5.xlarge instead of g4dn.xlarge)
```

---

## Pre-Deployment Checklist

Before deploying to AWS, verify:

- [ ] All Docker images build successfully
- [ ] CPU miner produces valid JSON output
- [ ] CUDA miner works with GPU (if testing locally)
- [ ] Output files contain expected fields
- [ ] Log files show mining progress
- [ ] Topology files are accessible in containers
- [ ] S3 upload works (if configured)
- [ ] Tested with multiple difficulty levels
- [ ] Tested with different topologies
- [ ] No errors in logs

---

## Example Testing Workflow

```bash
# 1. Build all images
./docker/build_all.sh

# 2. Run quick automated tests
./docker/test_local.sh

# 3. Test CPU with longer duration
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=5m \
  -e DIFFICULTY_ENERGY=-14900 \
  quip-protocol/cpu-miner:latest

# 4. Verify output
cat docker/output/cpu_*.json | jq '.results'
cat docker/output/cpu_*.log | grep -E "(blocks_found|success_rate)"

# 5. Test CUDA (if GPU available)
docker run --rm \
  --gpus all \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=5m \
  -e DIFFICULTY_ENERGY=-14900 \
  quip-protocol/cuda-miner:latest

# 6. Compare performance
echo "CPU performance:"
cat docker/output/cpu_*.json | jq '.results.blocks_per_minute'
echo "CUDA performance:"
cat docker/output/cuda_*.json | jq '.results.blocks_per_minute'

# 7. Clean up
rm -rf docker/output/*

# 8. Ready for AWS deployment!
echo "All tests passed! Ready to deploy to AWS."
```

---

## Additional Resources

- **Docker Documentation**: [docker/README.md](docker/README.md)
- **AWS Deployment**: [aws/README_AWS_DEPLOYMENT.md](aws/README_AWS_DEPLOYMENT.md)
- **Mining Tool**: [tools/compare_mining_rates.py](tools/compare_mining_rates.py)
- **Project Guide**: [CLAUDE.md](CLAUDE.md)

For issues or questions, please open a GitHub issue.
