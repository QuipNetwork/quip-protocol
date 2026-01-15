#!/bin/bash
# Terminate all mining fleet instances for an experiment
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
MINER_TYPES="${MINER_TYPES:-cpu cuda metal}"  # Space-separated list
DRY_RUN="${DRY_RUN:-false}"  # Set to true for safety check
RELEASE_HOSTS="${RELEASE_HOSTS:-true}"  # Release Mac dedicated hosts

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Terminating Mining Fleet${NC}"
echo -e "${BLUE}========================================${NC}"

# Validate inputs
if [ -z "$EXPERIMENT_ID" ]; then
    echo -e "${YELLOW}Warning: EXPERIMENT_ID not specified${NC}"
    echo "This will terminate ALL Quip mining instances in the region!"
    echo ""
    read -p "Continue? (yes/NO) " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Termination cancelled."
        exit 0
    fi
    FILTER_TAG=""
else
    echo "Experiment ID: $EXPERIMENT_ID"
    FILTER_TAG="--filters Name=tag:Experiment,Values=$EXPERIMENT_ID"
fi

echo "AWS Region: $AWS_REGION"
echo "Miner Types: $MINER_TYPES"
echo "Dry Run: $DRY_RUN"
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found${NC}"
    exit 1
fi

# ====================
# Find Instances
# ====================
echo -e "${GREEN}Finding instances...${NC}"

ALL_INSTANCE_IDS=""
INSTANCE_COUNTS=""

for MINER_TYPE in $MINER_TYPES; do
    echo -e "${BLUE}Searching for $MINER_TYPE instances...${NC}"

    INSTANCE_IDS=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        $FILTER_TAG \
        --filters "Name=tag:MinerType,Values=$MINER_TYPE" "Name=instance-state-name,Values=running,pending,stopped,stopping" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)

    if [ -n "$INSTANCE_IDS" ]; then
        COUNT=$(echo "$INSTANCE_IDS" | wc -w)
        echo -e "${GREEN}  Found $COUNT $MINER_TYPE instances${NC}"
        ALL_INSTANCE_IDS="$ALL_INSTANCE_IDS $INSTANCE_IDS"
        INSTANCE_COUNTS="$INSTANCE_COUNTS\n  $MINER_TYPE: $COUNT"
    else
        echo -e "${YELLOW}  No $MINER_TYPE instances found${NC}"
    fi
done

# Trim whitespace
ALL_INSTANCE_IDS=$(echo "$ALL_INSTANCE_IDS" | xargs)

if [ -z "$ALL_INSTANCE_IDS" ]; then
    echo ""
    echo -e "${YELLOW}No instances found to terminate${NC}"
    exit 0
fi

TOTAL_COUNT=$(echo "$ALL_INSTANCE_IDS" | wc -w)

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Instances to Terminate:${NC}"
echo -e "$INSTANCE_COUNTS"
echo -e "${YELLOW}Total: $TOTAL_COUNT instances${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Display instance details
echo -e "${BLUE}Instance Details:${NC}"
aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --instance-ids $ALL_INSTANCE_IDS \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`].Value|[0],Tags[?Key==`MinerType`].Value|[0]]' \
    --output table

echo ""

# ====================
# Confirmation
# ====================
if [ "$DRY_RUN" == "true" ]; then
    echo -e "${YELLOW}DRY RUN MODE: No instances will be terminated${NC}"
    echo "Instance IDs: $ALL_INSTANCE_IDS"
    exit 0
fi

echo -e "${RED}⚠ WARNING: This will terminate $TOTAL_COUNT instances${NC}"
echo ""
read -p "Are you sure you want to proceed? (yes/NO) " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Termination cancelled."
    exit 0
fi

# ====================
# Terminate Instances
# ====================
echo -e "${GREEN}Terminating instances...${NC}"

aws ec2 terminate-instances \
    --region "$AWS_REGION" \
    --instance-ids $ALL_INSTANCE_IDS

echo -e "${GREEN}✓ Termination initiated for $TOTAL_COUNT instances${NC}"

# ====================
# Wait for Termination
# ====================
echo ""
echo -e "${BLUE}Waiting for instances to terminate...${NC}"

aws ec2 wait instance-terminated \
    --region "$AWS_REGION" \
    --instance-ids $ALL_INSTANCE_IDS

echo -e "${GREEN}✓ All instances terminated${NC}"

# ====================
# Release Mac Dedicated Hosts
# ====================
if [ "$RELEASE_HOSTS" == "true" ]; then
    echo ""
    echo -e "${GREEN}Checking for dedicated hosts to release...${NC}"

    if [ -n "$FILTER_TAG" ]; then
        HOST_IDS=$(aws ec2 describe-hosts \
            --region "$AWS_REGION" \
            --filter "Name=tag:Experiment,Values=$EXPERIMENT_ID" \
            --query 'Hosts[?State==`available`].HostId' \
            --output text)
    else
        # Don't auto-release hosts without experiment ID (too dangerous)
        echo -e "${YELLOW}Skipping host release (no EXPERIMENT_ID specified)${NC}"
        HOST_IDS=""
    fi

    if [ -n "$HOST_IDS" ]; then
        HOST_COUNT=$(echo "$HOST_IDS" | wc -w)
        echo -e "${YELLOW}Found $HOST_COUNT dedicated hosts to release${NC}"

        read -p "Release dedicated hosts? (yes/NO) " -r
        echo

        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            aws ec2 release-hosts \
                --region "$AWS_REGION" \
                --host-ids $HOST_IDS

            echo -e "${GREEN}✓ Released $HOST_COUNT dedicated hosts${NC}"
        else
            echo -e "${YELLOW}Dedicated hosts not released${NC}"
            echo "To release manually:"
            echo "  aws ec2 release-hosts --region $AWS_REGION --host-ids $HOST_IDS"
        fi
    else
        echo -e "${BLUE}No dedicated hosts to release${NC}"
    fi
fi

# ====================
# Cleanup Security Groups
# ====================
echo ""
echo -e "${GREEN}Checking for security groups to clean up...${NC}"

if [ -n "$EXPERIMENT_ID" ]; then
    SG_IDS=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --filters "Name=tag:Experiment,Values=$EXPERIMENT_ID" \
        --query 'SecurityGroups[].GroupId' \
        --output text)

    if [ -n "$SG_IDS" ]; then
        SG_COUNT=$(echo "$SG_IDS" | wc -w)
        echo -e "${YELLOW}Found $SG_COUNT security groups for this experiment${NC}"

        read -p "Delete security groups? (yes/NO) " -r
        echo

        if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            for SG_ID in $SG_IDS; do
                aws ec2 delete-security-group \
                    --region "$AWS_REGION" \
                    --group-id "$SG_ID" 2>&1 | grep -v "DependencyViolation" || true
            done

            echo -e "${GREEN}✓ Security groups cleaned up${NC}"
        else
            echo -e "${YELLOW}Security groups not deleted${NC}"
        fi
    fi
fi

# ====================
# Summary
# ====================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Fleet Termination Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  Instances terminated: $TOTAL_COUNT"
echo "  Region: $AWS_REGION"
if [ -n "$EXPERIMENT_ID" ]; then
    echo "  Experiment ID: $EXPERIMENT_ID"
fi
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Verify termination: aws ec2 describe-instances --region $AWS_REGION $FILTER_TAG"
echo "2. Check for lingering resources (EBS volumes, Elastic IPs, etc.)"
echo "3. Review AWS billing to confirm charges stopped"
echo "4. Collect results if not already done: ./aws/collect_results.sh"
