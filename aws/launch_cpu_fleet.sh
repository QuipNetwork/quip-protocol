#!/bin/bash
# Launch CPU mining fleet on AWS EC2
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
FLEET_SIZE="${FLEET_SIZE:-100}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AMI_ID="${AMI_ID:-}"  # Auto-detect if not specified
KEY_NAME="${KEY_NAME:-}"  # SSH key name (optional)
SECURITY_GROUP="${SECURITY_GROUP:-}"  # Auto-create if not specified
IAM_ROLE_NAME="${IAM_ROLE_NAME:-QuipMiningEC2Role-${EXPERIMENT_ID}}"
S3_BUCKET="${S3_BUCKET:-quip-mining-results-${EXPERIMENT_ID}}"
ECR_REGISTRY="${ECR_REGISTRY:-}"  # Optional: use ECR for Docker images
USE_SPOT="${USE_SPOT:-true}"  # Use spot instances for cost savings
SPOT_MAX_PRICE="${SPOT_MAX_PRICE:-0.10}"  # Max price per hour

# Mining parameters
MINING_DURATION="${MINING_DURATION:-30m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Launching CPU Mining Fleet${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Fleet Size: $FLEET_SIZE instances"
echo "Instance Type: $INSTANCE_TYPE"
echo "Region: $AWS_REGION"
echo "Use Spot: $USE_SPOT"
echo "Mining Duration: $MINING_DURATION"
echo "Difficulty: $DIFFICULTY_ENERGY"
echo "S3 Bucket: $S3_BUCKET"
echo ""

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found${NC}"
    exit 1
fi

# Auto-detect AMI if not specified (Ubuntu 22.04 LTS)
if [ -z "$AMI_ID" ]; then
    echo -e "${GREEN}Auto-detecting Ubuntu 22.04 LTS AMI...${NC}"
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
    echo -e "${GREEN}Using AMI: $AMI_ID${NC}"
fi

# Create security group if not specified
if [ -z "$SECURITY_GROUP" ]; then
    echo -e "${GREEN}Creating security group...${NC}"
    SG_NAME="quip-mining-cpu-${EXPERIMENT_ID}"

    VPC_ID=$(aws ec2 describe-vpcs \
        --region "$AWS_REGION" \
        --filters "Name=isDefault,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)

    SECURITY_GROUP=$(aws ec2 create-security-group \
        --region "$AWS_REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for Quip CPU mining experiment $EXPERIMENT_ID" \
        --vpc-id "$VPC_ID" \
        --output text 2>&1 | grep -o 'sg-[a-z0-9]*' || true)

    if [ -n "$SECURITY_GROUP" ]; then
        # Allow outbound traffic (needed for Docker pull, S3 upload)
        aws ec2 authorize-security-group-egress \
            --region "$AWS_REGION" \
            --group-id "$SECURITY_GROUP" \
            --protocol -1 \
            --cidr 0.0.0.0/0 2>&1 | grep -v "already exists" || true

        echo -e "${GREEN}✓ Security group created: $SECURITY_GROUP${NC}"
    fi
fi

# Generate user-data script
echo -e "${GREEN}Generating user-data script...${NC}"
USER_DATA=$(cat <<'EOF'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get install -y docker.io awscli curl

# Start Docker
systemctl start docker
systemctl enable docker

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Pull Docker image
if [ -n "{{ECR_REGISTRY}}" ]; then
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin {{ECR_REGISTRY}}
    docker pull {{ECR_REGISTRY}}/cpu-miner:latest
    IMAGE={{ECR_REGISTRY}}/cpu-miner:latest
else
    # Build image locally (fallback)
    cd /tmp
    git clone https://github.com/your-repo/quip-protocol.git || exit 1
    cd quip-protocol
    docker build -f docker/Dockerfile.cpu -t quip-protocol/cpu-miner:latest .
    IMAGE=quip-protocol/cpu-miner:latest
fi

# Run mining container
docker run --rm \
    -v /tmp/output:/output \
    -e MINING_DURATION={{MINING_DURATION}} \
    -e DIFFICULTY_ENERGY={{DIFFICULTY_ENERGY}} \
    -e MIN_DIVERSITY={{MIN_DIVERSITY}} \
    -e MIN_SOLUTIONS={{MIN_SOLUTIONS}} \
    -e TOPOLOGY_FILE={{TOPOLOGY_FILE}} \
    -e S3_BUCKET={{S3_BUCKET}} \
    -e AWS_DEFAULT_REGION=$REGION \
    $IMAGE

# Terminate instance after completion (optional)
if [ "{{AUTO_TERMINATE}}" == "true" ]; then
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
fi
EOF
)

# Replace placeholders
USER_DATA="${USER_DATA//\{\{ECR_REGISTRY\}\}/$ECR_REGISTRY}"
USER_DATA="${USER_DATA//\{\{MINING_DURATION\}\}/$MINING_DURATION}"
USER_DATA="${USER_DATA//\{\{DIFFICULTY_ENERGY\}\}/$DIFFICULTY_ENERGY}"
USER_DATA="${USER_DATA//\{\{MIN_DIVERSITY\}\}/$MIN_DIVERSITY}"
USER_DATA="${USER_DATA//\{\{MIN_SOLUTIONS\}\}/$MIN_SOLUTIONS}"
USER_DATA="${USER_DATA//\{\{TOPOLOGY_FILE\}\}/$TOPOLOGY_FILE}"
USER_DATA="${USER_DATA//\{\{S3_BUCKET\}\}/$S3_BUCKET}"
USER_DATA="${USER_DATA//\{\{AUTO_TERMINATE\}\}/true}"

# Encode user-data
USER_DATA_ENCODED=$(echo "$USER_DATA" | base64 -w 0 2>/dev/null || echo "$USER_DATA" | base64)

# Launch instances
echo -e "${GREEN}Launching $FLEET_SIZE instances...${NC}"

if [ "$USE_SPOT" == "true" ]; then
    # Launch spot instances
    echo -e "${BLUE}Using spot instances (max price: \$$SPOT_MAX_PRICE/hr)${NC}"

    SPOT_REQUEST=$(aws ec2 request-spot-instances \
        --region "$AWS_REGION" \
        --spot-price "$SPOT_MAX_PRICE" \
        --instance-count "$FLEET_SIZE" \
        --type "one-time" \
        --launch-specification "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"KeyName\": \"$KEY_NAME\",
            \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
            \"IamInstanceProfile\": {
                \"Name\": \"$IAM_ROLE_NAME\"
            },
            \"UserData\": \"$USER_DATA_ENCODED\",
            \"BlockDeviceMappings\": [{
                \"DeviceName\": \"/dev/sda1\",
                \"Ebs\": {
                    \"VolumeSize\": 20,
                    \"VolumeType\": \"gp3\",
                    \"DeleteOnTermination\": true
                }
            }]
        }")

    SPOT_REQUEST_IDS=$(echo "$SPOT_REQUEST" | jq -r '.SpotInstanceRequests[].SpotInstanceRequestId' | tr '\n' ' ')
    echo -e "${GREEN}✓ Spot request IDs: $SPOT_REQUEST_IDS${NC}"

else
    # Launch on-demand instances
    echo -e "${BLUE}Using on-demand instances${NC}"

    INSTANCE_IDS=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --iam-instance-profile "Name=$IAM_ROLE_NAME" \
        --user-data "$USER_DATA" \
        --count "$FLEET_SIZE" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":20,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=quip-cpu-miner-$EXPERIMENT_ID},{Key=Experiment,Value=$EXPERIMENT_ID},{Key=MinerType,Value=cpu}]" \
        --query 'Instances[].InstanceId' \
        --output text)

    echo -e "${GREEN}✓ Launched instances: $INSTANCE_IDS${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Fleet Launch Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "# List instances:"
echo "aws ec2 describe-instances --region $AWS_REGION --filters Name=tag:Experiment,Values=$EXPERIMENT_ID --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output table"
echo ""
echo "# Check S3 results:"
echo "aws s3 ls s3://$S3_BUCKET/cpu/"
echo ""
echo "# Terminate all instances:"
echo "./aws/terminate_fleet.sh"
echo ""
echo -e "${BLUE}Results will be saved to:${NC} s3://$S3_BUCKET/cpu/"
