#!/bin/bash
# Master script to launch all mining fleets for performance comparison
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Fleet sizes
H100_COUNT="${H100_COUNT:-60}"           # 60 H100 GPUs
CPU_COUNT="${CPU_COUNT:-45}"             # 45 CPU instances
MAC_COUNT="${MAC_COUNT:-10}"             # 10 Mac Metal instances (will reuse if available)

# Instance types
H100_INSTANCE_TYPE="${H100_INSTANCE_TYPE:-p5.48xlarge}"  # 8x H100 per instance
CPU_INSTANCE_TYPE="${CPU_INSTANCE_TYPE:-c6a.2xlarge}"
MAC_INSTANCE_TYPE="${MAC_INSTANCE_TYPE:-mac-m4.metal}"

# Mining parameters
MINING_DURATION="${MINING_DURATION:-5s}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"

# S3 and IAM
S3_BUCKET="${S3_BUCKET:-quip-mining-results-${EXPERIMENT_ID}}"
IAM_ROLE_NAME="QuipMiningEC2Role-${EXPERIMENT_ID}"

# Use spot instances for cost savings (not available for Mac)
USE_SPOT="${USE_SPOT:-true}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quip Protocol - Master Fleet Launch${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Experiment Configuration:${NC}"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Region: $AWS_REGION"
echo "  Duration: $MINING_DURATION"
echo "  Difficulty: $DIFFICULTY_ENERGY"
echo ""
echo -e "${GREEN}Fleet Configuration:${NC}"
echo "  H100 GPUs: $H100_COUNT ($H100_INSTANCE_TYPE)"
echo "  CPU Instances: $CPU_COUNT ($CPU_INSTANCE_TYPE)"
echo "  Mac Metal: $MAC_COUNT ($MAC_INSTANCE_TYPE - will reuse existing if available)"
echo "  QPU: Run locally with: python tools/compare_mining_rates.py --miner-type qpu ..."
echo ""
echo -e "${GREEN}Storage:${NC}"
echo "  S3 Bucket: $S3_BUCKET"
echo "  IAM Role: $IAM_ROLE_NAME"
echo ""

# Calculate number of p5.48xlarge instances needed (8 GPUs each)
P5_INSTANCES=$(echo "($H100_COUNT + 7) / 8" | bc)

# Confirmation
echo -e "${YELLOW}⚠️  Cost Estimate (on-demand pricing):${NC}"
echo "  H100 (p5.48xlarge): ~\$98/hr × $P5_INSTANCES instances = ~\$$(echo "$P5_INSTANCES * 98" | bc)/hr"
if [ "$USE_SPOT" == "true" ]; then
    echo "    With spot: ~\$$(echo "$P5_INSTANCES * 98 * 0.5" | bc)/hr (50% savings estimate)"
fi
echo "  CPU (c6a.2xlarge): ~\$0.34/hr × $CPU_COUNT = ~\$$(echo "$CPU_COUNT * 0.34" | bc)/hr"
if [ "$USE_SPOT" == "true" ]; then
    echo "    With spot: ~\$$(echo "$CPU_COUNT * 0.34 * 0.3" | bc)/hr (70% savings estimate)"
fi
echo "  Mac (mac2-m2.metal): ~\$1.67/hr × $MAC_COUNT = ~\$$(echo "$MAC_COUNT * 1.67" | bc)/hr"
echo -e "${RED}    ⚠️  Mac instances: 24-hour minimum billing (~\$$(echo "$MAC_COUNT * 1.67 * 24" | bc) total)${NC}"
echo ""
echo -e "${YELLOW}  For $MINING_DURATION run:${NC}"
echo "    H100+CPU: ~\$$(echo "($P5_INSTANCES * 98 * 0.5 + $CPU_COUNT * 0.34 * 0.3) * 0.083" | bc) (5min with spot)"
echo "    Mac: Already paid for 24h if running"
echo ""

read -p "Continue with deployment? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Step 1: Setup S3 and IAM${NC}"
echo -e "${BLUE}========================================${NC}"

# Setup S3 bucket and IAM role
EXPERIMENT_ID="$EXPERIMENT_ID" \
S3_BUCKET="$S3_BUCKET" \
IAM_ROLE_NAME="$IAM_ROLE_NAME" \
AWS_REGION="$AWS_REGION" \
./aws/setup_s3_output.sh

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Step 2: Launch H100 CUDA Fleet${NC}"
echo -e "${BLUE}========================================${NC}"

EXPERIMENT_ID="$EXPERIMENT_ID" \
FLEET_SIZE="$P5_INSTANCES" \
INSTANCE_TYPE="$H100_INSTANCE_TYPE" \
AWS_REGION="$AWS_REGION" \
USE_SPOT="$USE_SPOT" \
MINING_DURATION="$MINING_DURATION" \
DIFFICULTY_ENERGY="$DIFFICULTY_ENERGY" \
MIN_DIVERSITY="$MIN_DIVERSITY" \
MIN_SOLUTIONS="$MIN_SOLUTIONS" \
TOPOLOGY_FILE="$TOPOLOGY_FILE" \
S3_BUCKET="$S3_BUCKET" \
IAM_ROLE_NAME="$IAM_ROLE_NAME" \
./aws/launch_cuda_fleet.sh

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Step 3: Launch CPU Fleet${NC}"
echo -e "${BLUE}========================================${NC}"

EXPERIMENT_ID="$EXPERIMENT_ID" \
FLEET_SIZE="$CPU_COUNT" \
INSTANCE_TYPE="$CPU_INSTANCE_TYPE" \
AWS_REGION="$AWS_REGION" \
USE_SPOT="$USE_SPOT" \
MINING_DURATION="$MINING_DURATION" \
DIFFICULTY_ENERGY="$DIFFICULTY_ENERGY" \
MIN_DIVERSITY="$MIN_DIVERSITY" \
MIN_SOLUTIONS="$MIN_SOLUTIONS" \
TOPOLOGY_FILE="$TOPOLOGY_FILE" \
S3_BUCKET="$S3_BUCKET" \
IAM_ROLE_NAME="$IAM_ROLE_NAME" \
./aws/launch_cpu_fleet.sh

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Step 4: Mac Metal Fleet (Reuse or Launch)${NC}"
echo -e "${BLUE}========================================${NC}"

# Check for existing Mac instances
EXISTING_MAC_INSTANCES=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --filters "Name=instance-type,Values=mac-m4.metal,mac-m4pro.metal,mac2-m2.metal,mac2-m2pro.metal,mac1.metal" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text 2>/dev/null || echo "")

EXISTING_COUNT=$(echo "$EXISTING_MAC_INSTANCES" | wc -w | tr -d ' ')

if [ "$EXISTING_COUNT" -ge "$MAC_COUNT" ]; then
    echo -e "${GREEN}✓ Found $EXISTING_COUNT existing Mac instances (need $MAC_COUNT)${NC}"
    echo -e "${GREEN}  Reusing existing instances to avoid 24h billing cycle${NC}"
    echo -e "${BLUE}  Instances: $EXISTING_MAC_INSTANCES${NC}"

    # Trigger mining on existing instances via SSM or user-data update
    echo -e "${YELLOW}  You'll need to SSH into these instances and run mining manually:${NC}"
    for INSTANCE_ID in $EXISTING_MAC_INSTANCES; do
        IP=$(aws ec2 describe-instances \
            --region "$AWS_REGION" \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[].Instances[].PublicIpAddress' \
            --output text)
        echo "    ssh ec2-user@$IP"
        echo "    cd /tmp && git clone <your-repo> && cd quip-protocol"
        echo "    python3 tools/compare_mining_rates.py --miner-type metal \\"
        echo "      --difficulty-energy $DIFFICULTY_ENERGY --duration $MINING_DURATION \\"
        echo "      --output /tmp/metal_output.json"
        echo "    aws s3 cp /tmp/metal_output.json s3://$S3_BUCKET/metal/"
        echo ""
    done
else
    echo -e "${YELLOW}⚠️  Found $EXISTING_COUNT Mac instances, need $MAC_COUNT${NC}"
    echo -e "${YELLOW}  Launching $(echo "$MAC_COUNT - $EXISTING_COUNT" | bc) new Mac instances${NC}"
    echo -e "${RED}  ⚠️  This will incur 24-hour minimum billing!${NC}"

    read -p "Continue with Mac launch? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        EXPERIMENT_ID="$EXPERIMENT_ID" \
        FLEET_SIZE="$MAC_COUNT" \
        INSTANCE_TYPE="$MAC_INSTANCE_TYPE" \
        AWS_REGION="$AWS_REGION" \
        MINING_DURATION="$MINING_DURATION" \
        DIFFICULTY_ENERGY="$DIFFICULTY_ENERGY" \
        MIN_DIVERSITY="$MIN_DIVERSITY" \
        MIN_SOLUTIONS="$MIN_SOLUTIONS" \
        TOPOLOGY_FILE="$TOPOLOGY_FILE" \
        S3_BUCKET="$S3_BUCKET" \
        IAM_ROLE_NAME="$IAM_ROLE_NAME" \
        ./aws/launch_metal_fleet.sh
    else
        echo -e "${YELLOW}  Skipped Mac launch${NC}"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ FLEET LAUNCH COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Deployment Summary:${NC}"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  H100 Instances: $P5_INSTANCES × $H100_INSTANCE_TYPE (~$H100_COUNT GPUs)"
echo "  CPU Instances: $CPU_COUNT × $CPU_INSTANCE_TYPE"
echo "  Mac Instances: $EXISTING_COUNT existing + launched"
echo "  Mining Duration: $MINING_DURATION"
echo "  S3 Bucket: s3://$S3_BUCKET"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1️⃣  Run QPU locally (requires D-Wave credentials):"
echo "   source venv/bin/activate"
echo "   python tools/compare_mining_rates.py --miner-type qpu \\"
echo "     --difficulty-energy $DIFFICULTY_ENERGY \\"
echo "     --duration $MINING_DURATION \\"
echo "     --topology $TOPOLOGY_FILE \\"
echo "     --output docker/output/qpu_results.json"
echo "   aws s3 cp docker/output/qpu_results.json s3://$S3_BUCKET/qpu/"
echo ""
echo "2️⃣  Monitor instances:"
echo "   aws ec2 describe-instances --region $AWS_REGION \\"
echo "     --filters Name=tag:Experiment,Values=$EXPERIMENT_ID \\"
echo "     --query 'Reservations[].Instances[].[Tags[?Key==\`MinerType\`].Value|[0],InstanceId,State.Name,InstanceType]' \\"
echo "     --output table"
echo ""
echo "3️⃣  Monitor S3 results:"
echo "   watch -n 10 'aws s3 ls s3://$S3_BUCKET/ --recursive | tail -20'"
echo ""
echo "4️⃣  Collect and visualize results (after mining completes):"
echo "   EXPERIMENT_ID=$EXPERIMENT_ID S3_BUCKET=$S3_BUCKET ./aws/collect_results.sh"
echo ""
echo "5️⃣  Terminate instances (KEEP Mac instances for 24h!):"
echo "   EXPERIMENT_ID=$EXPERIMENT_ID ./aws/terminate_fleet.sh"
echo ""
echo -e "${GREEN}Results will be available at:${NC}"
echo "  s3://$S3_BUCKET/cpu/"
echo "  s3://$S3_BUCKET/cuda/"
echo "  s3://$S3_BUCKET/metal/"
echo "  s3://$S3_BUCKET/qpu/"
