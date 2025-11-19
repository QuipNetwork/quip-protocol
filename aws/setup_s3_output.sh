#!/bin/bash
# Setup S3 bucket for mining results storage
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
BUCKET_NAME="${BUCKET_NAME:-quip-mining-results-${EXPERIMENT_ID}}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Setting up S3 Bucket for Mining Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Bucket Name: $BUCKET_NAME"
echo "AWS Region: $AWS_REGION"
echo ""

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install it first.${NC}"
    echo "Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured.${NC}"
    echo "Run: aws configure"
    exit 1
fi

# Create S3 bucket
echo -e "${GREEN}Creating S3 bucket: $BUCKET_NAME${NC}"
if [ "$AWS_REGION" == "us-east-1" ]; then
    aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION"
else
    aws s3 mb "s3://$BUCKET_NAME" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bucket created successfully${NC}"
else
    echo -e "${YELLOW}Warning: Bucket may already exist or creation failed${NC}"
fi
echo ""

# Create folder structure
echo -e "${GREEN}Creating folder structure...${NC}"
for MINER_TYPE in cpu cuda metal; do
    echo "  - ${MINER_TYPE}/"
    aws s3api put-object --bucket "$BUCKET_NAME" --key "${MINER_TYPE}/"
done
echo ""

# Enable versioning (optional but recommended)
echo -e "${GREEN}Enabling versioning...${NC}"
aws s3api put-bucket-versioning \
  --bucket "$BUCKET_NAME" \
  --versioning-configuration Status=Enabled
echo -e "${GREEN}✓ Versioning enabled${NC}"
echo ""

# Set lifecycle policy (transition to Glacier after 90 days)
echo -e "${GREEN}Configuring lifecycle policy...${NC}"
cat > /tmp/lifecycle-policy.json <<EOF
{
  "Rules": [
    {
      "Id": "ArchiveOldResults",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET_NAME" \
  --lifecycle-configuration file:///tmp/lifecycle-policy.json
rm /tmp/lifecycle-policy.json

echo -e "${GREEN}✓ Lifecycle policy configured${NC}"
echo ""

# Create IAM policy for EC2 instances
echo -e "${GREEN}Creating IAM policy for EC2 access...${NC}"
IAM_POLICY_NAME="QuipMiningS3Access-${EXPERIMENT_ID}"

cat > /tmp/s3-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name "$IAM_POLICY_NAME" \
  --policy-document file:///tmp/s3-policy.json \
  2>&1 | grep -v "EntityAlreadyExists" || true

POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='$IAM_POLICY_NAME'].Arn" --output text)
rm /tmp/s3-policy.json

if [ -n "$POLICY_ARN" ]; then
    echo -e "${GREEN}✓ IAM policy created: $POLICY_ARN${NC}"
else
    echo -e "${YELLOW}Warning: Could not retrieve policy ARN${NC}"
fi
echo ""

# Create IAM role for EC2 instances
echo -e "${GREEN}Creating IAM role for EC2 instances...${NC}"
IAM_ROLE_NAME="QuipMiningEC2Role-${EXPERIMENT_ID}"

cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name "$IAM_ROLE_NAME" \
  --assume-role-policy-document file:///tmp/trust-policy.json \
  2>&1 | grep -v "EntityAlreadyExists" || true
rm /tmp/trust-policy.json

# Attach policy to role
if [ -n "$POLICY_ARN" ]; then
    aws iam attach-role-policy \
      --role-name "$IAM_ROLE_NAME" \
      --policy-arn "$POLICY_ARN" 2>&1 | grep -v "already exists" || true
fi

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name "$IAM_ROLE_NAME" \
  2>&1 | grep -v "EntityAlreadyExists" || true

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name "$IAM_ROLE_NAME" \
  --role-name "$IAM_ROLE_NAME" \
  2>&1 | grep -v "LimitExceeded" || true

echo -e "${GREEN}✓ IAM role created: $IAM_ROLE_NAME${NC}"
echo ""

# Enable S3 event notifications (optional)
echo -e "${GREEN}Setting up event notifications...${NC}"
cat > /tmp/notification-config.json <<EOF
{
  "TopicConfigurations": []
}
EOF

aws s3api put-bucket-notification-configuration \
  --bucket "$BUCKET_NAME" \
  --notification-configuration file:///tmp/notification-config.json
rm /tmp/notification-config.json

echo -e "${GREEN}✓ Event notifications configured${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}S3 Bucket:${NC} s3://$BUCKET_NAME"
echo -e "${GREEN}Region:${NC} $AWS_REGION"
echo -e "${GREEN}IAM Policy:${NC} $IAM_POLICY_NAME"
echo -e "${GREEN}IAM Role:${NC} $IAM_ROLE_NAME"
echo ""
echo -e "${BLUE}Folder Structure:${NC}"
echo "  s3://$BUCKET_NAME/cpu/"
echo "  s3://$BUCKET_NAME/cuda/"
echo "  s3://$BUCKET_NAME/metal/"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Use IAM role when launching EC2 instances"
echo "2. Set S3_BUCKET=$BUCKET_NAME in Docker containers"
echo "3. Results will be automatically synced to S3"
echo ""
echo -e "${YELLOW}Save these values for later use:${NC}"
echo "export EXPERIMENT_ID=$EXPERIMENT_ID"
echo "export S3_BUCKET=$BUCKET_NAME"
echo "export IAM_ROLE_NAME=$IAM_ROLE_NAME"
echo "export AWS_REGION=$AWS_REGION"
