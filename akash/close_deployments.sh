#!/bin/bash
# Close Akash deployments to stop billing
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DEPLOYMENT_LIST="${1:-}"
WALLET_NAME="${WALLET_NAME:-default}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Closing Akash Deployments${NC}"
echo -e "${BLUE}========================================${NC}"

if [ -z "$DEPLOYMENT_LIST" ]; then
    echo -e "${RED}Error: Please provide deployment list file${NC}"
    echo "Usage: $0 <deployment-list-file>"
    exit 1
fi

if [ ! -f "$DEPLOYMENT_LIST" ]; then
    echo -e "${RED}Error: Deployment list file not found: $DEPLOYMENT_LIST${NC}"
    exit 1
fi

# Read deployment IDs
DEPLOYMENT_IDS=($(cat "$DEPLOYMENT_LIST"))
echo "Found ${#DEPLOYMENT_IDS[@]} deployments to close"
echo ""

# Confirmation
echo -e "${YELLOW}⚠ This will close all deployments and stop billing${NC}"
read -p "Continue? (yes/NO) " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Close each deployment
SUCCESS_COUNT=0
FAIL_COUNT=0

for DSEQ in "${DEPLOYMENT_IDS[@]}"; do
    echo -e "${BLUE}Closing deployment: $DSEQ${NC}"

    akash tx deployment close \
        --dseq $DSEQ \
        --from $WALLET_NAME \
        --node https://rpc.akashnet.net:443 \
        --chain-id akashnet-2 \
        --fees 5000uakt \
        --gas auto \
        --gas-adjustment 1.5 \
        -y 2>&1 | grep -E "(code|txhash)" || true

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Deployment $DSEQ closed${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Failed to close deployment $DSEQ${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo ""
    sleep 1
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Closure Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total deployments: ${#DEPLOYMENT_IDS[@]}"
echo -e "${GREEN}Successfully closed: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo -e "${GREEN}All billing has been stopped for closed deployments${NC}"
