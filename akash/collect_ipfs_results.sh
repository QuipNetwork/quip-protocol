#!/bin/bash
# Collect mining results from IPFS
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
IPFS_GATEWAY="${IPFS_GATEWAY:-https://ipfs.io}"
LOCAL_IPFS="${LOCAL_IPFS:-https://carback-ipfs.ngrok.io}"
OUTPUT_DIR="${OUTPUT_DIR:-./ipfs_results}"
MANIFEST_CIDS_FILE="${1:-}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Collecting Results from IPFS${NC}"
echo -e "${BLUE}========================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to fetch from IPFS
fetch_ipfs() {
    local cid=$1
    local output_file=$2
    local gateway=${3:-$IPFS_GATEWAY}

    echo -e "${BLUE}Fetching CID: $cid${NC}"

    # Try local IPFS node first, then fallback to public gateway
    if ! curl -f -s -m 30 "${LOCAL_IPFS}/ipfs/${cid}" -o "$output_file" 2>/dev/null; then
        echo -e "${YELLOW}  Local IPFS failed, trying public gateway...${NC}"
        if ! curl -f -s -m 60 "${gateway}/ipfs/${cid}" -o "$output_file" 2>/dev/null; then
            echo -e "${RED}  ✗ Failed to fetch from IPFS${NC}"
            return 1
        fi
    fi

    echo -e "${GREEN}  ✓ Downloaded: $(basename $output_file)${NC}"
    return 0
}

# Usage instructions
if [ -z "$MANIFEST_CIDS_FILE" ]; then
    echo -e "${YELLOW}Usage: $0 <manifest-cids-file>${NC}"
    echo ""
    echo "Provide a file with one manifest CID per line, OR pass CIDs as arguments"
    echo ""
    echo "Examples:"
    echo "  # From file:"
    echo "  echo 'QmXXX...' > manifest_cids.txt"
    echo "  $0 manifest_cids.txt"
    echo ""
    echo "  # Direct CID:"
    echo "  MANIFEST_CID=QmXXX... $0"
    echo ""
    echo "  # Multiple CIDs (space-separated):"
    echo "  $0 'QmXXX QmYYY QmZZZ'"
    exit 1
fi

# Parse input
if [ -f "$MANIFEST_CIDS_FILE" ]; then
    # File with CIDs (one per line)
    MANIFEST_CIDS=($(cat "$MANIFEST_CIDS_FILE"))
elif [ -n "$MANIFEST_CID" ]; then
    # Single CID from env var
    MANIFEST_CIDS=("$MANIFEST_CID")
else
    # Space-separated CIDs as argument
    MANIFEST_CIDS=($MANIFEST_CIDS_FILE)
fi

echo "Found ${#MANIFEST_CIDS[@]} manifest(s) to process"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

# Process each manifest
for MANIFEST_CID in "${MANIFEST_CIDS[@]}"; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Processing Manifest: $MANIFEST_CID${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Download manifest
    MANIFEST_FILE="${OUTPUT_DIR}/manifest_${MANIFEST_CID}.json"

    if ! fetch_ipfs "$MANIFEST_CID" "$MANIFEST_FILE"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        continue
    fi

    # Parse manifest
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq not found. Please install jq.${NC}"
        exit 1
    fi

    DEPLOYMENT_ID=$(jq -r '.deployment_id // "unknown"' "$MANIFEST_FILE")
    MINER_TYPE=$(jq -r '.miner_type // "unknown"' "$MANIFEST_FILE")
    UPDATE_MODE=$(jq -r '.update_mode // "sa"' "$MANIFEST_FILE")
    JSON_CID=$(jq -r '.results.json_cid // ""' "$MANIFEST_FILE")
    LOG_CID=$(jq -r '.results.log_cid // ""' "$MANIFEST_FILE")

    echo -e "${GREEN}Deployment: $DEPLOYMENT_ID ($MINER_TYPE, $UPDATE_MODE)${NC}"

    # Create deployment directory (include update_mode for CUDA)
    if [ "$MINER_TYPE" = "cuda" ]; then
        DEPLOYMENT_DIR="${OUTPUT_DIR}/${MINER_TYPE}_${UPDATE_MODE}/${DEPLOYMENT_ID}"
    else
        DEPLOYMENT_DIR="${OUTPUT_DIR}/${MINER_TYPE}/${DEPLOYMENT_ID}"
    fi
    mkdir -p "$DEPLOYMENT_DIR"

    # Download JSON results
    if [ -n "$JSON_CID" ] && [ "$JSON_CID" != "null" ]; then
        JSON_FILE="${DEPLOYMENT_DIR}/results.json"
        if fetch_ipfs "$JSON_CID" "$JSON_FILE"; then
            # Extract key metrics
            BLOCKS=$(jq -r '.results.blocks_found // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
            RATE=$(jq -r '.results.blocks_per_minute // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
            echo -e "${GREEN}  Results: $BLOCKS blocks, $RATE blocks/min${NC}"
        fi
    fi

    # Download log file
    if [ -n "$LOG_CID" ] && [ "$LOG_CID" != "null" ]; then
        LOG_FILE="${DEPLOYMENT_DIR}/mining.log"
        fetch_ipfs "$LOG_CID" "$LOG_FILE"
    fi

    # Copy manifest to deployment directory
    cp "$MANIFEST_FILE" "${DEPLOYMENT_DIR}/manifest.json"

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

# Generate summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Collection Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total manifests: ${#MANIFEST_CIDS[@]}"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo -e "${YELLOW}Results saved to: $OUTPUT_DIR${NC}"
echo ""

# Generate summary report
if [ $SUCCESS_COUNT -gt 0 ]; then
    SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

    cat > "$SUMMARY_FILE" <<EOF
IPFS Mining Results Summary
============================
Retrieved: $(date)
Total manifests: ${#MANIFEST_CIDS[@]}
Successful: $SUCCESS_COUNT
Failed: $FAIL_COUNT

Deployments:
EOF

    # Add details for each deployment
    for MINER_TYPE in cpu cuda_sa cuda_gibbs cuda metal; do
        if [ -d "$OUTPUT_DIR/$MINER_TYPE" ]; then
            echo "" >> "$SUMMARY_FILE"
            echo "$MINER_TYPE Miners:" >> "$SUMMARY_FILE"
            echo "-------------" >> "$SUMMARY_FILE"

            for DEPLOYMENT_DIR in "$OUTPUT_DIR/$MINER_TYPE"/*; do
                if [ -d "$DEPLOYMENT_DIR" ]; then
                    DEPLOYMENT_ID=$(basename "$DEPLOYMENT_DIR")
                    JSON_FILE="$DEPLOYMENT_DIR/results.json"

                    if [ -f "$JSON_FILE" ]; then
                        BLOCKS=$(jq -r '.results.blocks_found // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
                        RATE=$(jq -r '.results.blocks_per_minute // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
                        echo "  $DEPLOYMENT_ID: $BLOCKS blocks, $RATE blocks/min" >> "$SUMMARY_FILE"
                    fi
                fi
            done
        fi
    done

    echo "" >> "$SUMMARY_FILE"
    echo "File Structure:" >> "$SUMMARY_FILE"
    echo "--------------" >> "$SUMMARY_FILE"
    tree -L 3 "$OUTPUT_DIR" >> "$SUMMARY_FILE" 2>/dev/null || \
        find "$OUTPUT_DIR" -type f >> "$SUMMARY_FILE"

    echo -e "${GREEN}✓ Summary saved to: $SUMMARY_FILE${NC}"
    cat "$SUMMARY_FILE"
fi

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review results in: $OUTPUT_DIR"
echo "2. Analyze with: cat $OUTPUT_DIR/*/*/results.json | jq '.results'"
echo "3. View logs: cat $OUTPUT_DIR/*/*/mining.log"
