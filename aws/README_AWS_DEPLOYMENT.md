# AWS Deployment Guide - Quip Protocol Mining Experiments

This guide provides comprehensive instructions for deploying large-scale mining rate comparison experiments across ~100 AWS EC2 instances.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Cost Estimates](#cost-estimates)
4. [Quick Start](#quick-start)
5. [Detailed Deployment Steps](#detailed-deployment-steps)
6. [Monitoring and Management](#monitoring-and-management)
7. [Results Collection](#results-collection)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

The AWS deployment infrastructure enables running parallel mining experiments across three hardware types:

- **CPU Mining**: General-purpose EC2 instances (t3.medium)
- **CUDA Mining**: GPU-accelerated EC2 instances (g4dn.xlarge with NVIDIA T4)
- **Metal Mining**: macOS EC2 instances with Apple Silicon (mac2-m2.metal)

Each deployment uses Docker containers (CPU/CUDA) or native execution (Metal) to run the `compare_mining_rates.py` tool, with automatic result uploads to S3.

### Key Features

- **Automated Deployment**: Launch scripts handle all infrastructure setup
- **S3 Integration**: Automatic result uploads with organized folder structure
- **Spot Instance Support**: 50-70% cost savings for CPU and CUDA miners
- **IAM Role Management**: Secure S3 access without embedded credentials
- **Auto-Termination**: Optional instance shutdown after experiment completion
- **Result Aggregation**: Automated collection and analysis tools

---

## Prerequisites

### Required Tools

1. **AWS CLI 2.x+**
   ```bash
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install

   # Configure credentials
   aws configure
   ```

2. **jq** (for JSON processing)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install jq

   # macOS
   brew install jq
   ```

3. **bc** (for cost calculations)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install bc

   # macOS
   brew install bc
   ```

### AWS Account Setup

1. **IAM Permissions**: Ensure your AWS user has permissions for:
   - EC2 (launch, describe, terminate instances)
   - S3 (create buckets, put/get objects)
   - IAM (create roles, policies, instance profiles)
   - VPC (security groups, if needed)

2. **Service Limits**: Check and request limit increases if needed:
   - EC2 vCPU limits (CPU instances)
   - EC2 GPU instance limits (g4dn, g5)
   - Mac instance allocation (can take 30+ minutes)
   - Spot instance limits

3. **SSH Key Pair** (optional, for debugging):
   ```bash
   aws ec2 create-key-pair \
     --key-name quip-mining-key \
     --query 'KeyMaterial' \
     --output text > ~/.ssh/quip-mining-key.pem
   chmod 400 ~/.ssh/quip-mining-key.pem
   ```

### Docker Images (Optional)

For faster deployment, push Docker images to Amazon ECR:

```bash
# Create ECR repositories
aws ecr create-repository --repository-name cpu-miner
aws ecr create-repository --repository-name cuda-miner

# Build and push images
./docker/build_all.sh
ECR_REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com \
PUSH_TO_REGISTRY=1 \
./docker/build_all.sh
```

---

## Cost Estimates

### Instance Pricing (us-east-1, on-demand rates)

| Miner Type | Instance Type | GPU | vCPU | RAM | Hourly Cost | Spot Cost (est.) |
|------------|---------------|-----|------|-----|-------------|------------------|
| **CPU** | t3.medium | - | 2 | 4GB | $0.0416 | $0.0125 (70% off) |
| **CUDA** | g4dn.xlarge | T4 | 4 | 16GB | $0.526 | $0.1578 (70% off) |
| **CUDA** | g5.xlarge | A10G | 4 | 16GB | $1.006 | $0.302 (70% off) |
| **Metal** | mac2-m2.metal | M2 (10 cores) | 8 | 24GB | $0.6695 | N/A (no spot) |
| **Metal** | mac2-m2pro.metal | M2 Pro (19 cores) | 12 | 32GB | $1.0835 | N/A (no spot) |

### Fleet Cost Examples

**100 Instances, 30-minute Run:**

| Configuration | On-Demand | Spot (70% savings) |
|---------------|-----------|-------------------|
| 100x CPU (t3.medium) | $2.08 | $0.63 |
| 100x CUDA (g4dn.xlarge) | $26.30 | $7.89 |
| 100x CUDA (g5.xlarge) | $50.30 | $15.10 |
| 10x Metal (mac2-m2.metal) | $1,606.80 | N/A (24hr minimum) |

**Important Notes:**
- **Metal instances**: 24-hour minimum billing period
- **Spot instances**: Can be interrupted with 2-minute warning
- **Data transfer**: S3 uploads incur minimal egress costs (<$0.01/GB)
- **Storage**: S3 storage ~$0.023/GB/month (minimal for JSON results)

---

## Quick Start

### 1. Setup S3 Bucket and IAM Roles

```bash
cd /path/to/quip-protocol/aws

# Set experiment ID
export EXPERIMENT_ID=exp_$(date +%Y%m%d_%H%M%S)

# Create S3 bucket and IAM roles
./setup_s3_output.sh
```

**Output:** Creates S3 bucket and IAM role, displays configuration to export.

### 2. Launch CPU Fleet (100 instances, 30 minutes)

```bash
# Configure
export S3_BUCKET=quip-mining-results-${EXPERIMENT_ID}
export IAM_ROLE_NAME=QuipMiningEC2Role-${EXPERIMENT_ID}
export FLEET_SIZE=100
export MINING_DURATION=30m
export DIFFICULTY_ENERGY=-14900

# Launch
./launch_cpu_fleet.sh
```

### 3. Launch CUDA Fleet (100 instances, 30 minutes)

```bash
# Configure
export FLEET_SIZE=100
export INSTANCE_TYPE=g4dn.xlarge  # or g5.xlarge
export USE_SPOT=true

# Launch
./launch_cuda_fleet.sh
```

### 4. Launch Metal Fleet (10 instances, 24 hours)

**⚠ Warning**: Mac instances have 24-hour minimum billing!

```bash
# Configure
export FLEET_SIZE=10  # Smaller fleet due to cost
export MINING_DURATION=24h  # Full 24 hours to maximize value

# Launch (will prompt for confirmation)
./launch_metal_fleet.sh
```

### 5. Monitor Progress

```bash
# List running instances
aws ec2 describe-instances \
  --filters Name=tag:Experiment,Values=$EXPERIMENT_ID \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`MinerType`].Value|[0]]' \
  --output table

# Check S3 results in real-time
watch -n 30 "aws s3 ls s3://$S3_BUCKET/cpu/ | wc -l"
```

### 6. Collect Results

```bash
# Download all results
export S3_BUCKET=quip-mining-results-${EXPERIMENT_ID}
./collect_results.sh

# View summary
cat results/summary_${EXPERIMENT_ID}.txt
```

### 7. Terminate Fleet

```bash
# Terminate all instances
export EXPERIMENT_ID=exp_20250118_120000  # Your experiment ID
./terminate_fleet.sh
```

---

## Detailed Deployment Steps

### Step 1: S3 Bucket Setup

The `setup_s3_output.sh` script creates:

- **S3 Bucket**: Named `quip-mining-results-{experiment_id}`
- **Folder Structure**: `/cpu/`, `/cuda/`, `/metal/`
- **Versioning**: Enabled for data safety
- **Lifecycle Policy**: Transition to Glacier after 90 days
- **IAM Policy**: Grant EC2 instances S3 write access
- **IAM Role**: `QuipMiningEC2Role-{experiment_id}`

**Usage:**
```bash
# Basic
./setup_s3_output.sh

# Custom bucket name
BUCKET_NAME=my-custom-bucket ./setup_s3_output.sh

# Different region
AWS_REGION=us-west-2 ./setup_s3_output.sh
```

**Save these outputs:**
```bash
export EXPERIMENT_ID=exp_20250118_120000
export S3_BUCKET=quip-mining-results-exp_20250118_120000
export IAM_ROLE_NAME=QuipMiningEC2Role-exp_20250118_120000
export AWS_REGION=us-east-1
```

### Step 2: Launch CPU Fleet

**Configuration Options:**

```bash
# Fleet parameters
export FLEET_SIZE=100              # Number of instances
export INSTANCE_TYPE=t3.medium     # Instance type
export USE_SPOT=true               # Use spot instances
export SPOT_MAX_PRICE=0.10         # Max price (on-demand is $0.0416)

# Mining parameters
export MINING_DURATION=30m
export DIFFICULTY_ENERGY=-14900
export MIN_DIVERSITY=0.1
export MIN_SOLUTIONS=5
export TOPOLOGY_FILE=dwave_topologies/topologies/advantage2_system1_7.json.gz

# Advanced
export AMI_ID=ami-0c55b159cbfafe1f0  # Custom AMI (optional)
export KEY_NAME=quip-mining-key      # SSH key for debugging
export ECR_REGISTRY=123456.dkr.ecr.us-east-1.amazonaws.com  # Use ECR images
```

**Launch:**
```bash
./launch_cpu_fleet.sh
```

**What it does:**
1. Auto-detects Ubuntu 22.04 LTS AMI
2. Creates security group
3. Launches instances with user-data script
4. Instances automatically:
   - Install Docker
   - Pull/build Docker image
   - Run mining experiment
   - Upload results to S3
   - Self-terminate (if AUTO_TERMINATE=true)

### Step 3: Launch CUDA Fleet

**Configuration:**

```bash
# Choose GPU type
export INSTANCE_TYPE=g4dn.xlarge    # NVIDIA T4 (cheaper, good performance)
# OR
export INSTANCE_TYPE=g5.xlarge      # NVIDIA A10G (2x faster, 2x cost)

# Spot pricing
export USE_SPOT=true
export SPOT_MAX_PRICE=0.30          # g4dn on-demand is $0.526
```

**Launch:**
```bash
./launch_cuda_fleet.sh
```

**What it does:**
1. Auto-detects Deep Learning AMI (includes NVIDIA drivers)
2. Installs NVIDIA Docker runtime
3. Verifies GPU access with `nvidia-smi`
4. Runs CUDA mining container with `--gpus all` flag

**GPU Selection:**
- **g4dn.xlarge**: Best value (~12 blocks/min at difficulty -14900)
- **g5.xlarge**: 2x performance (~25 blocks/min), better for difficult topologies

### Step 4: Launch Metal Fleet

**⚠ IMPORTANT: Mac Instance Considerations**

1. **24-Hour Minimum Billing**: You pay for 24 hours even if you use 1 hour
2. **Allocation Time**: AWS can take 30+ minutes to provision dedicated hosts
3. **No Spot Instances**: Only on-demand pricing available
4. **Cost**: 10 instances = $160-260/day depending on chip

**Strategy:** Run multiple experiments sequentially over 24 hours to maximize value.

**Configuration:**

```bash
# Choose chip type
export INSTANCE_TYPE=mac2-m2.metal      # M2, 10 GPU cores, $0.6695/hr
# OR
export INSTANCE_TYPE=mac2-m2pro.metal   # M2 Pro, 19 GPU cores, $1.0835/hr

# Smaller fleet recommended
export FLEET_SIZE=10

# Long duration to maximize value
export MINING_DURATION=24h
```

**Launch:**
```bash
./launch_metal_fleet.sh
```

**What it does:**
1. Allocates dedicated host (required for Mac instances)
2. Waits for host to become available
3. Launches instances on dedicated host
4. Runs mining **directly on macOS** (no Docker)
5. Installs Homebrew, Python 3.13, Metal frameworks
6. Does NOT auto-terminate (manual termination after 24hr)

**Maximizing Value:**

Edit `user_data_metal.sh` to run multiple experiments:

```bash
# Run multiple difficulty levels
for DIFF in -4100 -14900 -15450; do
    python tools/compare_mining_rates.py \
        --miner-type metal \
        --difficulty-energy $DIFF \
        --duration 6h \
        -o /tmp/output/metal_diff${DIFF}.json
    aws s3 cp /tmp/output/metal_diff${DIFF}.json s3://$S3_BUCKET/metal/
done
```

---

## Monitoring and Management

### Real-Time Monitoring

**List All Instances:**
```bash
aws ec2 describe-instances \
  --filters Name=tag:Experiment,Values=$EXPERIMENT_ID \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress,Tags[?Key==`MinerType`].Value|[0]]' \
  --output table
```

**Watch S3 Upload Progress:**
```bash
# Count files uploaded
watch -n 30 '
  echo "CPU: $(aws s3 ls s3://$S3_BUCKET/cpu/ | wc -l)"
  echo "CUDA: $(aws s3 ls s3://$S3_BUCKET/cuda/ | wc -l)"
  echo "Metal: $(aws s3 ls s3://$S3_BUCKET/metal/ | wc -l)"
'
```

**Check Instance Logs (if you have SSH access):**
```bash
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids i-1234567890abcdef0 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# SSH to instance
ssh -i ~/.ssh/quip-mining-key.pem ubuntu@$INSTANCE_IP

# View Docker logs
docker logs $(docker ps -q)
```

**CloudWatch Metrics:**
```bash
# CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-1234567890abcdef0 \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

### Spot Instance Interruptions

Spot instances can be interrupted with 2-minute warning. Check spot termination notices:

```bash
# From inside instance
curl -s http://169.254.169.254/latest/meta-data/spot/instance-action
```

**Handling Interruptions:**
- Results are synced to S3 every 5 minutes (if configured)
- Containers handle SIGTERM gracefully
- Relaunch interrupted instances with same config

---

## Results Collection

### Automated Collection

```bash
# Download all results to local directory
export S3_BUCKET=quip-mining-results-exp_20250118_120000
./collect_results.sh
```

**Output:**
- `results/cpu/*.json` - Individual CPU results
- `results/cuda/*.json` - Individual CUDA results
- `results/metal/*.json` - Individual Metal results
- `results/summary_exp_20250118_120000.txt` - Aggregated statistics
- `results/results_exp_20250118_120000.csv` - CSV export

### Manual Collection

```bash
# Download specific miner type
aws s3 sync s3://$S3_BUCKET/cpu/ ./results/cpu/

# Download single file
aws s3 cp s3://$S3_BUCKET/cpu/cpu_i-123456_20250118.json ./
```

### Analysis with jq

```bash
# Extract all success rates
cat results/cpu/*.json | jq '.results.success_rate'

# Average blocks per minute
cat results/cuda/*.json | jq -s 'map(.results.blocks_per_minute) | add / length'

# Filter experiments with high success rate
cat results/*/*.json | jq 'select(.results.success_rate > 0.05)'
```

---

## Troubleshooting

### Common Issues

**1. "Insufficient capacity" for GPU instances**
```
Error: InsufficientInstanceCapacity
```
**Solution:**
- Try different availability zones
- Use spot instances
- Request limit increase
- Try different GPU type (g4dn vs g5)

**2. Mac instance allocation timeout**
```
Waiting for host to become available... (30+ minutes)
```
**Solution:**
- This is normal - Mac hosts take time to provision
- Try different region (us-east-1, us-west-2 typically faster)
- Check AWS Service Health Dashboard

**3. Docker build fails on instance**
```
Error: Failed to clone repository
```
**Solution:**
- Use ECR to pre-push images
- Check GitHub repo URL in user-data script
- Ensure security group allows outbound HTTPS

**4. S3 upload permission denied**
```
Error: Access Denied (Service: Amazon S3)
```
**Solution:**
- Verify IAM role attached to instance
- Check IAM policy grants s3:PutObject
- Ensure S3_BUCKET environment variable is correct

**5. CUDA out of memory**
```
CuPy: CUDADriverError: CUDA_ERROR_OUT_OF_MEMORY
```
**Solution:**
- Use smaller topology
- Reduce num_sweeps parameter
- Use larger GPU (g5.xlarge instead of g4dn.xlarge)

### Debug Commands

**Check instance user-data execution:**
```bash
# SSH to instance
ssh -i ~/.ssh/quip-mining-key.pem ubuntu@$INSTANCE_IP

# View cloud-init logs
sudo tail -f /var/log/cloud-init-output.log

# Check Docker status
sudo systemctl status docker
docker ps -a
```

**Verify IAM role:**
```bash
# From inside instance
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

**Test S3 access:**
```bash
# From inside instance
aws s3 ls s3://$S3_BUCKET/
```

---

## Best Practices

### Cost Optimization

1. **Use Spot Instances** for CPU and CUDA (50-70% savings)
2. **Right-size instances**: Don't overprovision
3. **Auto-terminate** instances after experiments
4. **Mac instances**: Queue multiple experiments over 24 hours
5. **Monitor spending**: Set up AWS Budgets alerts

### Reliability

1. **Use multiple AZs** for larger fleets
2. **Enable S3 versioning** (already configured)
3. **Tag resources** with Experiment ID for tracking
4. **Save outputs** before terminating instances
5. **Test with small fleet** (10 instances) before scaling to 100

### Performance

1. **Pre-push Docker images to ECR** for faster startup
2. **Use larger EBS volumes** for CUDA (faster Docker builds)
3. **Choose appropriate instance types**:
   - CPU: t3.medium (sufficient for SA)
   - CUDA: g4dn.xlarge (best value) or g5.xlarge (performance)
   - Metal: mac2-m2pro.metal (if budget allows)

### Security

1. **Use IAM roles** instead of embedding AWS credentials
2. **Restrict security groups** (only allow outbound HTTPS/S3)
3. **Use temporary SSH keys** for debugging
4. **Enable CloudTrail** for audit logging
5. **Encrypt S3 bucket** (optional):
   ```bash
   aws s3api put-bucket-encryption \
     --bucket $S3_BUCKET \
     --server-side-encryption-configuration \
     '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
   ```

---

## Example Workflows

### Workflow 1: Quick CPU Benchmark (10 instances, 5 minutes)

```bash
export EXPERIMENT_ID=quick_test_$(date +%Y%m%d_%H%M%S)
./setup_s3_output.sh

export FLEET_SIZE=10
export MINING_DURATION=5m
export USE_SPOT=true
./launch_cpu_fleet.sh

# Wait 10 minutes
sleep 600

./collect_results.sh
./terminate_fleet.sh
```

### Workflow 2: Full Comparison (100 of each type, 30 minutes)

```bash
export EXPERIMENT_ID=full_comparison_$(date +%Y%m%d_%H%M%S)
./setup_s3_output.sh

# Save configuration
export FLEET_SIZE=100
export MINING_DURATION=30m
export S3_BUCKET=quip-mining-results-${EXPERIMENT_ID}
export IAM_ROLE_NAME=QuipMiningEC2Role-${EXPERIMENT_ID}

# Launch all fleets
./launch_cpu_fleet.sh
./launch_cuda_fleet.sh

# Metal with smaller fleet (cost consideration)
export FLEET_SIZE=10
export MINING_DURATION=24h
./launch_metal_fleet.sh

# Monitor progress
watch -n 60 "aws s3 ls s3://$S3_BUCKET/cpu/ | wc -l"

# After 30+ minutes for CPU/CUDA, 24 hours for Metal
./collect_results.sh
./terminate_fleet.sh  # This will keep Mac instances running
```

### Workflow 3: Testing Different Topologies

```bash
# Run separate experiments for each topology
for TOPO in "Z(8,2)" "Z(9,2)" "Z(10,2)"; do
    export EXPERIMENT_ID=topo_${TOPO//(/_}_$(date +%Y%m%d_%H%M%S)
    export TOPOLOGY_FILE=dwave_topologies/topologies/zephyr_${TOPO//(/_}.json.gz

    ./setup_s3_output.sh
    FLEET_SIZE=50 ./launch_cpu_fleet.sh

    sleep 1800  # Wait 30 minutes
    ./collect_results.sh
    ./terminate_fleet.sh
done
```

---

## Support and Resources

- **Documentation**: See [../docker/README.md](../docker/README.md) for Docker details
- **Tool Reference**: See [../tools/README.md](../tools/README.md) for mining comparison tool
- **AWS Pricing**: https://aws.amazon.com/ec2/pricing/
- **AWS Service Limits**: https://console.aws.amazon.com/servicequotas/

---

## Summary

This AWS deployment infrastructure provides a scalable, cost-effective way to run large-scale mining experiments:

✅ **Automated deployment** with single-command fleet launches
✅ **Cost optimized** with spot instances and auto-termination
✅ **S3 integration** for reliable result storage
✅ **Multi-hardware** support (CPU, NVIDIA GPU, Apple Silicon)
✅ **Production ready** with IAM roles, security groups, error handling

For questions or issues, please open an issue on GitHub.
