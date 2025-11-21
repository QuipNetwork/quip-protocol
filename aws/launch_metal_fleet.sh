#!/bin/bash
# Launch Metal (macOS) mining fleet on AWS EC2 Mac instances
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
FLEET_SIZE="${FLEET_SIZE:-10}"  # Note: Mac instances are expensive, default smaller
INSTANCE_TYPE="${INSTANCE_TYPE:-mac-m4.metal}"  # M4 with GPU cores
AWS_REGION="${AWS_REGION:-us-east-1}"
AMI_ID="${AMI_ID:-}"  # Auto-detect macOS AMI
KEY_NAME="${KEY_NAME:-}"  # SSH key name (recommended for debugging)
SECURITY_GROUP="${SECURITY_GROUP:-}"  # Auto-create if not specified
IAM_ROLE_NAME="${IAM_ROLE_NAME:-QuipMiningEC2Role-${EXPERIMENT_ID}}"
S3_BUCKET="${S3_BUCKET:-quip-mining-results-${EXPERIMENT_ID}}"
DEDICATED_HOST="${DEDICATED_HOST:-}"  # Mac instances require dedicated hosts

# Mining parameters
MINING_DURATION="${MINING_DURATION:-24h}"  # Default to 24h (minimum billing period)
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Launching Metal (macOS) Mining Fleet${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}⚠ IMPORTANT: Mac instances have 24-hour minimum billing${NC}"
echo -e "${YELLOW}   Consider running multiple experiments over 24h${NC}"
echo ""
echo "Experiment ID: $EXPERIMENT_ID"
echo "Fleet Size: $FLEET_SIZE instances"
echo "Instance Type: $INSTANCE_TYPE"
echo "Region: $AWS_REGION"
echo "Mining Duration: $MINING_DURATION"
echo "Difficulty: $DIFFICULTY_ENERGY"
echo "S3 Bucket: $S3_BUCKET"
echo ""

# Instance type pricing
case $INSTANCE_TYPE in
    mac1.metal)
        HOURLY_COST=0.5465
        GPU_CORES=8
        CHIP="M1"
        ;;
    mac2-m2.metal)
        HOURLY_COST=0.6695
        GPU_CORES=10
        CHIP="M2"
        ;;
    mac2-m2pro.metal)
        HOURLY_COST=1.0835
        GPU_CORES=19
        CHIP="M2 Pro"
        ;;
    mac-m4.metal)
        HOURLY_COST=0.90
        GPU_CORES=10
        CHIP="M4"
        ;;
    mac-m4pro.metal)
        HOURLY_COST=1.20
        GPU_CORES=16
        CHIP="M4 Pro"
        ;;
    *)
        HOURLY_COST=0.90
        GPU_CORES=10
        CHIP="M4"
        ;;
esac

DAILY_COST=$(echo "$HOURLY_COST * 24 * $FLEET_SIZE" | bc)

echo -e "${YELLOW}Cost Estimate:${NC}"
echo "  Instance Type: $INSTANCE_TYPE ($CHIP, $GPU_CORES GPU cores)"
echo "  Hourly Cost: \$$HOURLY_COST per instance"
echo "  Fleet Cost: \$$(echo "$HOURLY_COST * $FLEET_SIZE" | bc)/hr"
echo "  24-hour minimum: \$$DAILY_COST total"
echo ""

read -p "Continue with launch? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Launch cancelled."
    exit 0
fi

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found${NC}"
    exit 1
fi

# Auto-detect macOS AMI
if [ -z "$AMI_ID" ]; then
    echo -e "${GREEN}Auto-detecting macOS AMI...${NC}"
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners amazon \
        --filters "Name=name,Values=amzn-ec2-macos-*" \
        "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
    echo -e "${GREEN}Using AMI: $AMI_ID${NC}"
fi

# Create security group if not specified
if [ -z "$SECURITY_GROUP" ]; then
    echo -e "${GREEN}Creating security group...${NC}"
    SG_NAME="quip-mining-metal-${EXPERIMENT_ID}"

    VPC_ID=$(aws ec2 describe-vpcs \
        --region "$AWS_REGION" \
        --filters "Name=isDefault,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)

    SECURITY_GROUP=$(aws ec2 create-security-group \
        --region "$AWS_REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for Quip Metal mining experiment $EXPERIMENT_ID" \
        --vpc-id "$VPC_ID" \
        --output text 2>&1 | grep -o 'sg-[a-z0-9]*' | head -n1 || true)

    if [ -n "$SECURITY_GROUP" ]; then
        # Allow SSH (for debugging)
        aws ec2 authorize-security-group-ingress \
            --region "$AWS_REGION" \
            --group-id "$SECURITY_GROUP" \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 2>&1 | grep -v "already exists" || true

        # Allow outbound traffic
        aws ec2 authorize-security-group-egress \
            --region "$AWS_REGION" \
            --group-id "$SECURITY_GROUP" \
            --protocol -1 \
            --cidr 0.0.0.0/0 2>&1 | grep -v "already exists" || true

        echo -e "${GREEN}✓ Security group created: $SECURITY_GROUP${NC}"
    fi
fi

# Allocate dedicated host (required for Mac instances)
echo -e "${GREEN}Allocating dedicated host for Mac instances...${NC}"
echo -e "${YELLOW}Note: This may take 30+ minutes for AWS to provision${NC}"

DEDICATED_HOST=$(aws ec2 allocate-hosts \
    --region "$AWS_REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --availability-zone "${AWS_REGION}a" \
    --quantity 1 \
    --tag-specifications "ResourceType=dedicated-host,Tags=[{Key=Name,Value=quip-metal-host-$EXPERIMENT_ID},{Key=Experiment,Value=$EXPERIMENT_ID}]" \
    --query 'HostIds[0]' \
    --output text)

echo -e "${GREEN}✓ Dedicated host allocated: $DEDICATED_HOST${NC}"
echo -e "${YELLOW}Waiting for host to become available...${NC}"

# Wait for host to be available
while true; do
    HOST_STATE=$(aws ec2 describe-hosts \
        --region "$AWS_REGION" \
        --host-ids "$DEDICATED_HOST" \
        --query 'Hosts[0].State' \
        --output text)

    if [ "$HOST_STATE" == "available" ]; then
        echo -e "${GREEN}✓ Host is now available${NC}"
        break
    fi

    echo "Host state: $HOST_STATE (waiting...)"
    sleep 30
done

# Generate user-data script (runs directly on macOS, no Docker)
echo -e "${GREEN}Generating user-data script...${NC}"
USER_DATA=$(cat <<'EOF'
#!/bin/bash
set -e

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.13
brew install python@3.13 git

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Clone repository
cd /tmp
git clone https://github.com/your-repo/quip-protocol.git || exit 1
cd quip-protocol

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install \
    "pyobjc-framework-Metal" \
    "pyobjc-framework-MetalPerformanceShaders" \
    "dwave-ocean-sdk>=6.0.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.6.0" \
    "pandas>=2.0.0" \
    "seaborn>=0.12.0" \
    "python-dotenv>=1.0.0" \
    "click>=8.1.7" \
    "hashsigs>=0.0.2" \
    "blake3>=1.0.5"

# Install AWS CLI
brew install awscli

# Create output directory
mkdir -p /tmp/output

# Run mining comparison
python tools/compare_mining_rates.py \
    --miner-type metal \
    --difficulty-energy {{DIFFICULTY_ENERGY}} \
    --duration {{MINING_DURATION}} \
    --min-diversity {{MIN_DIVERSITY}} \
    --min-solutions {{MIN_SOLUTIONS}} \
    --topology {{TOPOLOGY_FILE}} \
    -o /tmp/output/metal_${INSTANCE_ID}_$(date +%Y%m%d_%H%M%S).json \
    2>&1 | tee /tmp/output/metal_${INSTANCE_ID}_$(date +%Y%m%d_%H%M%S).log

# Sync results to S3
aws s3 cp /tmp/output/ s3://{{S3_BUCKET}}/metal/ --recursive

# Note: Do NOT auto-terminate (24-hour minimum billing)
# Instance should be manually terminated after 24 hours
EOF
)

# Replace placeholders
USER_DATA="${USER_DATA//\{\{DIFFICULTY_ENERGY\}\}/$DIFFICULTY_ENERGY}"
USER_DATA="${USER_DATA//\{\{MINING_DURATION\}\}/$MINING_DURATION}"
USER_DATA="${USER_DATA//\{\{MIN_DIVERSITY\}\}/$MIN_DIVERSITY}"
USER_DATA="${USER_DATA//\{\{MIN_SOLUTIONS\}\}/$MIN_SOLUTIONS}"
USER_DATA="${USER_DATA//\{\{TOPOLOGY_FILE\}\}/$TOPOLOGY_FILE}"
USER_DATA="${USER_DATA//\{\{S3_BUCKET\}\}/$S3_BUCKET}"

# Encode user-data
USER_DATA_ENCODED=$(echo "$USER_DATA" | base64)

# Launch instances on dedicated host
echo -e "${GREEN}Launching $FLEET_SIZE Mac instances...${NC}"

INSTANCE_IDS=$(aws ec2 run-instances \
    --region "$AWS_REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$IAM_ROLE_NAME" \
    --user-data "$USER_DATA" \
    --placement "Tenancy=host,HostId=$DEDICATED_HOST" \
    --count "$FLEET_SIZE" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=quip-metal-miner-$EXPERIMENT_ID},{Key=Experiment,Value=$EXPERIMENT_ID},{Key=MinerType,Value=metal}]" \
    --query 'Instances[].InstanceId' \
    --output text)

echo -e "${GREEN}✓ Launched instances: $INSTANCE_IDS${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Metal Fleet Launch Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${RED}⚠ IMPORTANT REMINDERS:${NC}"
echo "1. Mac instances billed for minimum 24 hours"
echo "2. Instances will NOT auto-terminate"
echo "3. Manually terminate after 24 hours to avoid extra charges"
echo "4. Consider queuing multiple experiments to maximize value"
echo ""
echo -e "${YELLOW}Cost Summary:${NC}"
echo "  Hourly: \$$(echo "$HOURLY_COST * $FLEET_SIZE" | bc)"
echo "  24-hour minimum: \$$DAILY_COST"
echo ""
echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "# List instances:"
echo "aws ec2 describe-instances --region $AWS_REGION --filters Name=tag:Experiment,Values=$EXPERIMENT_ID --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output table"
echo ""
echo "# SSH to instance:"
echo "ssh -i ~/.ssh/$KEY_NAME.pem ec2-user@<public-ip>"
echo ""
echo "# Check S3 results:"
echo "aws s3 ls s3://$S3_BUCKET/metal/"
echo ""
echo "# Terminate after 24 hours:"
echo "aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_IDS"
echo "aws ec2 release-hosts --region $AWS_REGION --host-ids $DEDICATED_HOST"
echo ""
echo -e "${BLUE}Results will be saved to:${NC} s3://$S3_BUCKET/metal/"
