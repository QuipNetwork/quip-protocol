# Quip Protocol - Docker Deployment Guide

This directory contains Docker configurations for running mining rate comparison experiments across different hardware types (CPU, CUDA GPU).

## Overview

Two Docker images are provided:
- **CPU Miner** (`Dockerfile.cpu`) - Simulated annealing on CPU
- **CUDA Miner** (`Dockerfile.cuda`) - GPU-accelerated mining with NVIDIA CUDA

**Note:** For Apple Silicon (Metal) GPU mining, run directly on macOS without Docker. See [../../CLAUDE.md](../../CLAUDE.md) for native macOS setup.

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
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- NVIDIA Docker runtime installed
- NVIDIA Driver 450.80.02+ (Linux) or 452.39+ (Windows)
- 4GB+ GPU memory recommended

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
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
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
./docker/mining_rates/build_all.sh
```

This builds:
- `quip-protocol/cpu-miner:latest`
- `quip-protocol/cuda-miner:latest`

### 2. Test Locally

```bash
# Run quick 10-second tests
./docker/mining_rates/test_local.sh

# Custom test duration
TEST_DURATION=30s ./docker/mining_rates/test_local.sh
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

### 4. Using Docker Compose

```bash
cd docker
MINING_DURATION=30m DIFFICULTY_ENERGY=-14900 docker-compose up
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
./docker/mining_rates/build_all.sh
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
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon.json
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}
```

**Error: "CUDA out of memory"**
- Reduce batch size or use smaller topology
- Close other GPU applications
- Use smaller `num_sweeps` parameter

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

*Note: Rates vary significantly based on difficulty and topology size*

## Next Steps

- **Local Testing:** Run `./docker/mining_rates/test_local.sh` to validate setup
- **AWS Deployment:** See [AWS Deployment Guide](../../aws/README_AWS_DEPLOYMENT.md) for large-scale testing
- **Custom Topologies:** See `tools/analyze_topology_sizes.py` for topology analysis

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
```

### Complete Documentation

For detailed AWS deployment instructions, see:
- [aws/README_AWS_DEPLOYMENT.md](../../aws/README_AWS_DEPLOYMENT.md)

---

## Akash Network Deployment

**Akash** is a decentralized cloud computing marketplace offering 2-5x cost savings compared to AWS.

### Why Akash?

- **70% cheaper** than AWS for GPU workloads
- **Per-second billing** - no hourly minimums
- **HTTP result retrieval** - download logs and JSON directly
- **Decentralized** - no single point of failure

### Quick Start

**1. Install Akash CLI:**
```bash
# macOS
brew tap ovrclk/tap
brew install akash-provider-services

# Linux
curl -sSfL https://raw.githubusercontent.com/akash-network/node/master/install.sh | sh
```

**2. Setup wallet and fund with AKT**

**3. Build and push Docker images:**
```bash
export REGISTRY=ghcr.io/your-username
./akash/build_akash_images.sh
docker push $REGISTRY/quip-protocol-cpu-miner:latest
```

**4. Deploy to Akash:**
```bash
export MINER_TYPE=cpu
export FLEET_SIZE=10
./akash/deploy.sh
```

**5. Retrieve results:**
```bash
./akash/retrieve_results.sh akash/deployments_cpu_20250119_120000.txt
```

### Cost Comparison (90-minute experiment)

| Configuration | Akash | AWS Spot | Savings |
|---------------|-------|----------|---------|
| 10 CPU instances | ~$0.30 | ~$0.63 | 52% |
| 10 CUDA GPU instances | ~$2.25 | ~$7.89 | 71% |

### Complete Documentation

For detailed Akash deployment instructions, see:
- [akash/README_AKASH.md](../../akash/README_AKASH.md)

---

## Extended Testing

### Testing Different Topologies

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

### Multiple Parallel Tests

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

# Clean up
docker rm -f cpu-miner-diff-4100 cpu-miner-diff-14900 cpu-miner-diff-15450
```

### Pre-Deployment Checklist

Before deploying to AWS, verify:

- [ ] All Docker images build successfully
- [ ] CPU miner produces valid JSON output
- [ ] CUDA miner works with GPU (if testing locally)
- [ ] Output files contain expected fields
- [ ] Log files show mining progress
- [ ] Topology files are accessible in containers
- [ ] S3 upload works (if configured)

---

## Support

- **Issues:** [GitLab Issues](https://gitlab.com/piqued/quip-protocol/-/issues)
- **Documentation:** See root `README.md` and `CLAUDE.md`
- **AWS Deployment:** See `aws/README_AWS_DEPLOYMENT.md`
- **Akash Deployment:** See `akash/README_AKASH.md`
