#!/bin/bash
# Deploy mining workload to Akash Network
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
MINER_TYPE="${MINER_TYPE:-cpu}"  # cpu or cuda
WALLET_NAME="${WALLET_NAME:-default}"
DEPLOYMENT_FILE="akash/deploy-${MINER_TYPE}.yaml"
FLEET_SIZE="${FLEET_SIZE:-10}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deploying to Akash Network${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Miner Type: $MINER_TYPE"
echo "Wallet: $WALLET_NAME"
echo "Deployment File: $DEPLOYMENT_FILE"
echo "Fleet Size: $FLEET_SIZE"
echo ""

# Check prerequisites
if ! command -v akash &> /dev/null; then
    echo -e "${RED}Error: Akash CLI not found${NC}"
    echo "Install from: https://docs.akash.network/guides/cli/installation"
    exit 1
fi

if [ ! -f "$DEPLOYMENT_FILE" ]; then
    echo -e "${RED}Error: Deployment file not found: $DEPLOYMENT_FILE${NC}"
    exit 1
fi

# Check wallet
echo -e "${GREEN}Checking wallet balance...${NC}"
BALANCE=$(akash query bank balances $(akash keys show $WALLET_NAME -a) --node https://rpc.akashnet.net:443 -o json | jq -r '.balances[] | select(.denom=="uakt") | .amount')

if [ -z "$BALANCE" ] || [ "$BALANCE" -eq 0 ]; then
    echo -e "${RED}Error: Insufficient AKT balance${NC}"
    echo "Please fund your wallet: $(akash keys show $WALLET_NAME -a)"
    exit 1
fi

echo -e "${GREEN}✓ Balance: $BALANCE uAKT (~$(echo "scale=2; $BALANCE/1000000" | bc) AKT)${NC}"
echo ""

# Deploy multiple instances
echo -e "${GREEN}Deploying $FLEET_SIZE instances...${NC}"
DEPLOYMENT_IDS=()

for i in $(seq 1 $FLEET_SIZE); do
    echo -e "${BLUE}Deploying instance $i/$FLEET_SIZE...${NC}"

    # Create deployment
    DEPLOY_OUTPUT=$(akash tx deployment create $DEPLOYMENT_FILE \
        --from $WALLET_NAME \
        --node https://rpc.akashnet.net:443 \
        --chain-id akashnet-2 \
        --fees 5000uakt \
        --gas auto \
        --gas-adjustment 1.5 \
        -y 2>&1)

    # Extract deployment ID from output
    DSEQ=$(echo "$DEPLOY_OUTPUT" | grep -oP 'deployment/\K[0-9]+' | head -1)

    if [ -z "$DSEQ" ]; then
        echo -e "${RED}✗ Failed to create deployment $i${NC}"
        continue
    fi

    echo -e "${GREEN}✓ Deployment $i created: DSEQ=$DSEQ${NC}"
    DEPLOYMENT_IDS+=("$DSEQ")

    # Wait a bit between deployments
    sleep 2
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployments Created!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Deployment IDs:${NC}"
for DSEQ in "${DEPLOYMENT_IDS[@]}"; do
    echo "  - $DSEQ"
done

# Save deployment IDs to file
DEPLOYMENT_LIST="akash/deployments_${MINER_TYPE}_$(date +%Y%m%d_%H%M%S).txt"
printf "%s\n" "${DEPLOYMENT_IDS[@]}" > "$DEPLOYMENT_LIST"
echo ""
echo -e "${GREEN}Deployment IDs saved to: $DEPLOYMENT_LIST${NC}"

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Accept bids for each deployment:"
for DSEQ in "${DEPLOYMENT_IDS[@]}"; do
    echo "   akash tx market lease create --dseq $DSEQ --from $WALLET_NAME --node https://rpc.akashnet.net:443 --chain-id akashnet-2 --fees 5000uakt -y"
done
echo ""
echo "2. Monitor deployments:"
echo "   ./akash/monitor.sh $DEPLOYMENT_LIST"
echo ""
echo "3. Retrieve results after 90 minutes:"
echo "   ./akash/retrieve_results.sh $DEPLOYMENT_LIST"
echo ""
echo "4. Close deployments when done:"
echo "   ./akash/close_deployments.sh $DEPLOYMENT_LIST"
