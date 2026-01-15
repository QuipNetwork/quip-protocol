#!/bin/bash
# Retrieve mining results from Akash deployments
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DEPLOYMENT_LIST="${1:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./akash_results}"
WALLET_NAME="${WALLET_NAME:-default}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Retrieving Results from Akash${NC}"
echo -e "${BLUE}========================================${NC}"

if [ -z "$DEPLOYMENT_LIST" ]; then
    echo -e "${RED}Error: Please provide deployment list file${NC}"
    echo "Usage: $0 <deployment-list-file>"
    echo ""
    echo "Example:"
    echo "  $0 akash/deployments_cpu_20250119.txt"
    exit 1
fi

if [ ! -f "$DEPLOYMENT_LIST" ]; then
    echo -e "${RED}Error: Deployment list file not found: $DEPLOYMENT_LIST${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Read deployment IDs
DEPLOYMENT_IDS=($(cat "$DEPLOYMENT_LIST"))
echo "Found ${#DEPLOYMENT_IDS[@]} deployments"
echo ""

# Function to get deployment URI
get_deployment_uri() {
    local DSEQ=$1
    local WALLET_ADDR=$(akash keys show $WALLET_NAME -a)

    # Query lease status
    LEASE_INFO=$(akash query market lease list \
        --owner $WALLET_ADDR \
        --dseq $DSEQ \
        --node https://rpc.akashnet.net:443 \
        -o json 2>/dev/null)

    if [ -z "$LEASE_INFO" ]; then
        echo ""
        return 1
    fi

    # Get provider and extract URI from lease status
    PROVIDER=$(echo "$LEASE_INFO" | jq -r '.leases[0].lease.lease_id.provider // empty')

    if [ -z "$PROVIDER" ]; then
        echo ""
        return 1
    fi

    # Query service status to get URI
    SERVICE_STATUS=$(akash provider lease-status \
        --dseq $DSEQ \
        --from $WALLET_NAME \
        --provider $PROVIDER \
        --node https://rpc.akashnet.net:443 2>/dev/null)

    # Extract URI (provider-specific format)
    URI=$(echo "$SERVICE_STATUS" | jq -r '.services.cpu-miner.uris[0] // .services["cuda-miner"].uris[0] // empty' 2>/dev/null)

    echo "$URI"
    return 0
}

# Retrieve results from each deployment
SUCCESS_COUNT=0
FAIL_COUNT=0

for DSEQ in "${DEPLOYMENT_IDS[@]}"; do
    echo -e "${BLUE}Processing deployment: $DSEQ${NC}"

    # Get deployment URI
    URI=$(get_deployment_uri $DSEQ)

    if [ -z "$URI" ]; then
        echo -e "${YELLOW}⚠ Could not get URI for deployment $DSEQ (may be pending or closed)${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        continue
    fi

    echo -e "${GREEN}  URI: $URI${NC}"

    # Check if mining is complete
    STATUS=$(curl -s --max-time 10 "$URI/status.txt" 2>/dev/null || echo "")

    if [[ "$STATUS" == *"COMPLETED"* ]]; then
        echo -e "${GREEN}  ✓ Mining completed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Mining may still be running or failed${NC}"
    fi

    # Download JSON results
    echo -e "  Downloading JSON..."
    curl -s --max-time 30 "$URI/latest.json" -o "$OUTPUT_DIR/deployment_${DSEQ}.json"

    if [ $? -eq 0 ] && [ -s "$OUTPUT_DIR/deployment_${DSEQ}.json" ]; then
        echo -e "${GREEN}  ✓ JSON downloaded: deployment_${DSEQ}.json${NC}"
    else
        echo -e "${RED}  ✗ Failed to download JSON${NC}"
        rm -f "$OUTPUT_DIR/deployment_${DSEQ}.json"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        continue
    fi

    # Download log file
    echo -e "  Downloading log..."
    curl -s --max-time 30 "$URI/latest.log" -o "$OUTPUT_DIR/deployment_${DSEQ}.log"

    if [ $? -eq 0 ] && [ -s "$OUTPUT_DIR/deployment_${DSEQ}.log" ]; then
        echo -e "${GREEN}  ✓ Log downloaded: deployment_${DSEQ}.log${NC}"
    else
        echo -e "${YELLOW}  ⚠ Log download failed (non-critical)${NC}"
    fi

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Retrieval Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total deployments: ${#DEPLOYMENT_IDS[@]}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo -e "${YELLOW}Results saved to: $OUTPUT_DIR${NC}"
echo ""

# Generate summary
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}Generating summary...${NC}"

    SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
    cat > "$SUMMARY_FILE" <<EOF
Akash Mining Results Summary
=============================
Retrieved: $(date)
Total deployments: ${#DEPLOYMENT_IDS[@]}
Successful: $SUCCESS_COUNT
Failed: $FAIL_COUNT

EOF

    # Extract key metrics from each JSON
    for JSON_FILE in "$OUTPUT_DIR"/deployment_*.json; do
        if [ -f "$JSON_FILE" ]; then
            DSEQ=$(basename "$JSON_FILE" .json | sed 's/deployment_//')
            BLOCKS=$(jq -r '.results.blocks_found // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
            RATE=$(jq -r '.results.blocks_per_minute // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")

            echo "Deployment $DSEQ: $BLOCKS blocks, $RATE blocks/min" >> "$SUMMARY_FILE"
        fi
    done

    echo -e "${GREEN}✓ Summary saved to: $SUMMARY_FILE${NC}"
    cat "$SUMMARY_FILE"
fi

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review results in: $OUTPUT_DIR"
echo "2. Analyze with: cat $OUTPUT_DIR/deployment_*.json | jq '.results'"
echo "3. Close deployments to stop billing: ./akash/close_deployments.sh $DEPLOYMENT_LIST"
