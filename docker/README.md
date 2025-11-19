# Quip Protocol - Docker Deployment Guide

This directory contains Docker configurations for running mining rate comparison experiments across different hardware types (CPU, CUDA GPU, Apple Silicon Metal GPU).

## Overview

Three Docker images are provided:
- **CPU Miner** (`Dockerfile.cpu`) - Simulated annealing on CPU
- **CUDA Miner** (`Dockerfile.cuda`) - GPU-accelerated mining with NVIDIA CUDA 13.x
- **Metal Miner** (`Dockerfile.metal`) - Apple Silicon GPU mining (limited Docker support)

## Prerequisites

### General Requirements
- Docker 20.10+ with BuildKit enabled
- Docker Compose 2.0+ (optional, for multi-container testing)
- At least 4GB free disk space for images
- Internet connection for downloading base images

### Hardware-Specific Requirements

**For CPU Mining:**
- Any modern x86_64/amd64 CPU
- 2+ CPU cores recommended
- 2GB+ RAM

**For CUDA Mining:**
- NVIDIA GPU with CUDA 13.x support (Compute Capability 6.0+)
- NVIDIA Docker runtime installed
- NVIDIA Driver 450.80.02+ (Linux) or 452.39+ (Windows)
- 4GB+ GPU memory recommended

**For Metal Mining:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Docker Desktop for Mac 4.0+
- **Note:** Metal framework access from Docker is limited. For production deployment, run directly on macOS (see [AWS Deployment Guide](../aws/README_AWS_DEPLOYMENT.md))

### Installing NVIDIA Docker Runtime (CUDA only)

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 nvidia-smi
```

**Amazon Linux 2:**
```bash
# Install nvidia-docker2
sudo yum install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### 1. Build All Images

```bash
# From repository root
cd /path/to/quip-protocol
./docker/build_all.sh
```

This builds:
- `quip-protocol/cpu-miner:latest`
- `quip-protocol/cuda-miner:latest`
- `quip-protocol/metal-miner:latest`

### 2. Test Locally

```bash
# Run quick 10-second tests
./docker/test_local.sh

# Custom test duration
TEST_DURATION=30s ./docker/test_local.sh
```

### 3. Run Individual Miners

**CPU Miner:**
```bash
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=30m \
  -e DIFFICULTY_ENERGY=-14900 \
  -e MIN_DIVERSITY=0.1 \
  -e MIN_SOLUTIONS=5 \
  quip-protocol/cpu-miner:latest
```

**CUDA Miner:**
```bash
docker run --rm \
  --gpus all \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=30m \
  -e DIFFICULTY_ENERGY=-14900 \
  -e MIN_DIVERSITY=0.1 \
  -e MIN_SOLUTIONS=5 \
  -e GPU_DEVICE=0 \
  quip-protocol/cuda-miner:latest
```

**Metal Miner (macOS only):**
```bash
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/dwave_topologies:/app/dwave_topologies:ro \
  -e MINING_DURATION=30m \
  -e DIFFICULTY_ENERGY=-14900 \
  -e MIN_DIVERSITY=0.1 \
  -e MIN_SOLUTIONS=5 \
  quip-protocol/metal-miner:latest
```

### 4. Using Docker Compose

**CPU and CUDA miners:**
```bash
cd docker
MINING_DURATION=30m DIFFICULTY_ENERGY=-14900 docker-compose up
```

**Metal miner (macOS):**
```bash
cd docker
MINING_DURATION=30m DIFFICULTY_ENERGY=-14900 docker-compose -f docker-compose.mac.yml up
```

## Configuration

### Environment Variables

All miners support the following environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MINING_DURATION` | Mining duration | `30m` | `10s`, `5m`, `2h`, `1d`, `1w` |
| `DIFFICULTY_ENERGY` | Energy threshold | `-14900` | `-4100`, `-15450` |
| `MIN_DIVERSITY` | Solution diversity (0.0-1.0) | `0.1` | `0.15`, `0.2` |
| `MIN_SOLUTIONS` | Minimum solutions required | `5` | `3`, `10` |
| `TOPOLOGY_FILE` | Topology graph file path | `dwave_topologies/topologies/advantage2_system1_7.json.gz` | `dwave_topologies/topologies/zephyr_z9_t2.json.gz` |
| `GPU_DEVICE` | CUDA GPU device ID (CUDA only) | `0` | `1`, `2` |
| `S3_BUCKET` | S3 bucket for result uploads | (empty) | `quip-mining-results-exp123` |
| `OUTPUT_FILE` | Override output file path | (auto-generated) | `/output/custom_name.json` |

### Available Topologies

The project includes several pre-configured topologies in `dwave_topologies/topologies/`:

| Topology | Nodes | Edges | Avg Degree | Difficulty Range | Use Case |
|----------|-------|-------|------------|------------------|----------|
| Z(8,2) | 1,088 | 6,068 | 11.15 | -2869 to -2677 | Small-scale testing |
| Z(9,2) | 1,368 | 7,692 | 11.24 | -4100 to -3870 | **Default** - balanced |
| Z(10,2) | 1,680 | 9,508 | 11.32 | -5470 to -5200 | Medium-scale |
| Z(11,4) | 4,048 | 38,520 | 19.03 | -15170 to -14158 | Large-scale (88% QPU) |
| Advantage2 | 4,593 | 41,796 | 18.20 | -15170 to -14158 | Real hardware topology |

## Output Files

Each run generates two files in the `/output` directory:

### JSON Results File
**Format:** `{miner_type}_{instance_id}_{timestamp}.json`

Example:
```json
{
  "config": {
    "miner_type": "cpu",
    "difficulty_energy": -14900.0,
    "duration": "30m",
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

### Log File
**Format:** `{miner_type}_{instance_id}_{timestamp}.log`

Contains:
- Configuration parameters
- Real-time mining progress
- Error messages (if any)
- Completion status

## Advanced Usage

### Push Images to ECR (AWS)

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
ECR_REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com \
PUSH_TO_REGISTRY=1 \
./docker/build_all.sh
```

### Multi-GPU CUDA Setup

```bash
# Use specific GPU
docker run --rm --gpus '"device=1"' \
  -e GPU_DEVICE=1 \
  quip-protocol/cuda-miner:latest

# Use multiple GPUs (run multiple containers)
for GPU_ID in 0 1 2 3; do
  docker run -d \
    --gpus "\"device=$GPU_ID\"" \
    -e GPU_DEVICE=$GPU_ID \
    -v $(pwd)/docker/output:/output \
    quip-protocol/cuda-miner:latest
done
```

### Custom Topology

```bash
# Mount custom topology directory
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v $(pwd)/custom_topologies:/app/custom_topologies:ro \
  -e TOPOLOGY_FILE=custom_topologies/my_topology.json.gz \
  quip-protocol/cpu-miner:latest
```

### S3 Integration

```bash
# Auto-upload results to S3
docker run --rm \
  -v $(pwd)/docker/output:/output \
  -v ~/.aws:/root/.aws:ro \
  -e S3_BUCKET=my-mining-results \
  quip-protocol/cpu-miner:latest
```

## Troubleshooting

### CUDA Issues

**Error: "could not select device driver"**
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon.json
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}
```

**Error: "CUDA out of memory"**
- Reduce batch size or use smaller topology
- Close other GPU applications
- Use smaller `num_sweeps` parameter

### Metal Issues

**Error: "Metal device not found"**
- Metal framework access from Docker is limited
- For production, use direct deployment: `aws/user_data_metal.sh`
- Test on macOS host without Docker first

### Build Issues

**Error: "failed to solve with frontend dockerfile.v0"**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1
docker build ...
```

**Error: "no space left on device"**
```bash
# Clean up Docker resources
docker system prune -a
```

### Permission Issues

**Error: "permission denied" when writing to /output**
```bash
# Create output directory with correct permissions
mkdir -p docker/output
chmod 777 docker/output
```

## Performance Benchmarks

Approximate mining rates at difficulty -14900 (Z(9,2) topology):

| Hardware | Blocks/Minute | Cost/Hour (AWS) | $/Block |
|----------|---------------|-----------------|---------|
| t3.medium (2 vCPU) | 0.5 | $0.0416 | $0.0014 |
| g4dn.xlarge (T4) | 12.0 | $0.526 | $0.0007 |
| g5.xlarge (A10G) | 25.0 | $1.006 | $0.0007 |
| mac2-m2.metal (M2) | 8.0 | $0.6695 | $0.0014 |

*Note: Rates vary significantly based on difficulty and topology size*

## Next Steps

- **Local Testing:** Run `./docker/test_local.sh` to validate setup
- **AWS Deployment:** See [AWS Deployment Guide](../aws/README_AWS_DEPLOYMENT.md) for large-scale testing
- **Custom Topologies:** See `tools/analyze_topology_sizes.py` for topology analysis

## Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation:** See root `README.md` and `CLAUDE.md`
- **AWS Deployment:** See `aws/README_AWS_DEPLOYMENT.md`
