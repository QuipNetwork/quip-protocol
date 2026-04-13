# AWS GPU Mining Deployment

Deploy QUIP CUDA miners on AWS EC2 with H100 GPUs. Results are uploaded to IPFS by the container entrypoint.

## Prerequisites

- AWS CLI configured: `aws configure --profile postquant`
- P-instance vCPU quota >= 192 per p5.48xlarge instance (check below)
- Docker images published: `carback1/quip-protocol-cuda-miner:latest`
- IPFS API credentials (URL + API key)

## Check GPU Quota

```bash
# P-instance Spot vCPU quota
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-7212CCBC \
  --region us-west-2 --profile postquant \
  --query 'Quota.Value' --output text

# P-instance On-Demand vCPU quota
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-417A185B \
  --region us-west-2 --profile postquant \
  --query 'Quota.Value' --output text

# Request increase (1536 vCPUs = 8x p5.48xlarge)
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-7212CCBC \
  --desired-value 1536 --region us-west-2 --profile postquant
```

## Check Spot Pricing

```bash
# Compare regions
for region in us-west-2 us-east-1 us-east-2; do
  echo "=== $region ==="
  aws ec2 describe-spot-price-history \
    --instance-types p5.48xlarge \
    --product-descriptions "Linux/UNIX" \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --region $region --profile postquant \
    --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice]' \
    --output table
done
```

## Launch

### 1. Find the Deep Learning AMI

```bash
REGION=us-west-2
AMI_ID=$(aws ec2 describe-images \
  --region $REGION --profile postquant \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)
echo "AMI: $AMI_ID"
```

### 2. Create security group (outbound only)

```bash
SG_ID=$(aws ec2 create-security-group \
  --region $REGION --profile postquant \
  --group-name "quip-mining-$(date +%Y%m%d)" \
  --description "Quip mining - outbound only" \
  --query 'GroupId' --output text)
echo "Security Group: $SG_ID"
```

### 3. Encode user-data

Edit `minertest/aws-userdata.sh` to set your IPFS credentials, then:

```bash
USER_DATA=$(base64 < minertest/aws-userdata.sh)
```

Or inject env vars at launch time by prepending them to the script:

```bash
cat > /tmp/userdata.sh << 'OUTER'
#!/bin/bash
export IPFS_API_URL="https://your-ipfs-node.example.com"
export IPFS_API_KEY="your-api-key"
export MINING_DURATION="90m"
export DIFFICULTY_ENERGY="-14900"
OUTER
cat minertest/aws-userdata.sh >> /tmp/userdata.sh
USER_DATA=$(base64 < /tmp/userdata.sh)
```

### 4. Launch spot instances

```bash
# 8x p5.48xlarge = 64 H100 GPUs (8 GPUs per instance, 8 containers per instance)
FLEET_SIZE=8

aws ec2 run-instances \
  --region $REGION --profile postquant \
  --image-id $AMI_ID \
  --instance-type p5.48xlarge \
  --count $FLEET_SIZE \
  --security-group-ids $SG_ID \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --user-data "$USER_DATA" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=quip-miner},{Key=Purpose,Value=mining-test}]" \
  --query 'Instances[].InstanceId' --output text
```

**Cost estimate (8x p5.48xlarge, 90 min):**

| Region | Spot/hr (cheapest AZ) | Total 90 min |
|---|---|---|
| us-west-2 | ~$5.70/instance | ~$68 |
| us-east-2 | ~$12.15/instance | ~$146 |
| us-east-1 | ~$16.30/instance | ~$196 |

## Monitor

```bash
# Instance status
aws ec2 describe-instances \
  --region $REGION --profile postquant \
  --filters "Name=tag:Name,Values=quip-miner" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]' \
  --output table

# Watch for completion (instances auto-terminate)
watch -n 30 'aws ec2 describe-instances \
  --region us-west-2 --profile postquant \
  --filters "Name=tag:Name,Values=quip-miner" \
  --query "Reservations[].Instances[].[InstanceId,State.Name]" \
  --output table'
```

## Terminate (manual, if auto-terminate fails)

```bash
INSTANCE_IDS=$(aws ec2 describe-instances \
  --region $REGION --profile postquant \
  --filters "Name=tag:Name,Values=quip-miner" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

aws ec2 terminate-instances \
  --region $REGION --profile postquant \
  --instance-ids $INSTANCE_IDS

# Clean up security group
aws ec2 delete-security-group \
  --region $REGION --profile postquant \
  --group-id $SG_ID
```

## Instance Types Reference

| Type | GPUs | GPU Model | vCPUs | On-Demand/hr | Use case |
|---|---|---|---|---|---|
| p5.48xlarge | 8 | H100 80GB | 192 | ~$98 | Production benchmarks |
| p4d.24xlarge | 8 | A100 40GB | 96 | ~$33 | Budget alternative |
| g5.xlarge | 1 | A10G 24GB | 4 | ~$1 | Quick smoke tests |
| g6.xlarge | 1 | L4 24GB | 4 | ~$0.80 | Cheapest GPU option |
