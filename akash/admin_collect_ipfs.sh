#!/bin/bash
# Admin script to collect all mining results from IPFS server
# Uses admin credentials to list and retrieve all uploaded manifests
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
IPFS_NODE="${IPFS_NODE:-https://carback-ipfs.ngrok.io}"
IPFS_API_KEY="${IPFS_API_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./ipfs_results}"
IPFS_GATEWAY="${IPFS_GATEWAY:-https://ipfs.io}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Admin IPFS Results Collection${NC}"
echo -e "${BLUE}========================================${NC}"

# Check credentials
if [ -z "$IPFS_API_KEY" ]; then
    echo -e "${YELLOW}Usage: IPFS_API_KEY=your-key $0${NC}"
    echo ""
    echo "Environment variables:"
    echo "  IPFS_API_KEY    - API key for IPFS node (X-API-Key header)"
    echo "  IPFS_NODE       - IPFS node URL (default: https://carback-ipfs.ngrok.io)"
    echo "  OUTPUT_DIR      - Output directory (default: ./ipfs_results)"
    echo ""
    echo "Example:"
    echo "  IPFS_API_KEY=abc123 $0"
    exit 1
fi

echo "IPFS Node: $IPFS_NODE"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to make authenticated API calls
ipfs_api_call() {
    local endpoint=$1
    shift
    local url="${IPFS_NODE}${endpoint}"

    curl -s -H "X-API-Key: ${IPFS_API_KEY}" "$@" "$url"
}

# Function to fetch from IPFS
fetch_ipfs() {
    local cid=$1
    local output_file=$2

    echo -e "${BLUE}Fetching CID: $cid${NC}"

    # Try local IPFS node first (with auth), then fallback to public gateway
    if ! ipfs_api_call "/ipfs/${cid}" -f -m 30 -o "$output_file" 2>/dev/null; then
        echo -e "${YELLOW}  Local IPFS failed, trying public gateway...${NC}"
        if ! curl -f -s -m 60 "${IPFS_GATEWAY}/ipfs/${cid}" -o "$output_file" 2>/dev/null; then
            echo -e "${RED}  ✗ Failed to fetch from IPFS${NC}"
            return 1
        fi
    fi

    echo -e "${GREEN}  ✓ Downloaded: $(basename $output_file)${NC}"
    return 0
}

# Get list of all pinned files (manifests)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fetching Pinned Manifests${NC}"
echo -e "${BLUE}========================================${NC}"

# List all pins
PINS_JSON=$(ipfs_api_call "/api/v0/pin/ls?type=recursive" -X POST 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to connect to IPFS node${NC}"
    echo "Please check:"
    echo "  1. IPFS node is running: $IPFS_NODE"
    echo "  2. Admin credentials are correct"
    echo "  3. Network connectivity"
    exit 1
fi

# Extract CIDs from response
# IPFS API returns JSON like: {"Keys":{"QmXXX":{"Type":"recursive"},...}}
MANIFEST_CIDS=$(echo "$PINS_JSON" | grep -o 'Qm[a-zA-Z0-9]\{44\}' || true)

if [ -z "$MANIFEST_CIDS" ]; then
    echo -e "${YELLOW}No pinned files found on IPFS node${NC}"
    echo ""
    echo "This could mean:"
    echo "  1. No mining results have been uploaded yet"
    echo "  2. Files were uploaded without pinning (IPFS_PIN=false)"
    echo "  3. API key doesn't have permission to list pins"
    exit 0
fi

# Convert to array
MANIFEST_CIDS_ARRAY=($MANIFEST_CIDS)
echo "Found ${#MANIFEST_CIDS_ARRAY[@]} pinned CID(s)"
echo ""

# Try to identify which CIDs are manifests by fetching and checking
MANIFEST_LIST=()
TEMP_DIR="${OUTPUT_DIR}/.temp"
mkdir -p "$TEMP_DIR"

echo -e "${BLUE}Identifying manifest files...${NC}"
for CID in "${MANIFEST_CIDS_ARRAY[@]}"; do
    TEMP_FILE="${TEMP_DIR}/${CID}.json"

    # Try to fetch as JSON
    if fetch_ipfs "$CID" "$TEMP_FILE" >/dev/null 2>&1; then
        # Check if it's a manifest (has deployment_id and results fields)
        if command -v jq &> /dev/null; then
            if jq -e '.deployment_id and .results' "$TEMP_FILE" >/dev/null 2>&1; then
                MANIFEST_LIST+=("$CID")
                echo -e "${GREEN}  ✓ Manifest: $CID${NC}"
            fi
        else
            # Without jq, check for basic manifest structure
            if grep -q '"deployment_id"' "$TEMP_FILE" && grep -q '"results"' "$TEMP_FILE"; then
                MANIFEST_LIST+=("$CID")
                echo -e "${GREEN}  ✓ Manifest: $CID${NC}"
            fi
        fi
    fi
done

echo ""
echo "Found ${#MANIFEST_LIST[@]} manifest file(s)"
echo ""

if [ ${#MANIFEST_LIST[@]} -eq 0 ]; then
    echo -e "${YELLOW}No manifest files found among pinned CIDs${NC}"
    echo "Pinned CIDs might be individual result files rather than manifests."
    exit 0
fi

# Process each manifest
SUCCESS_COUNT=0
FAIL_COUNT=0

for MANIFEST_CID in "${MANIFEST_LIST[@]}"; do
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
    TIMESTAMP=$(jq -r '.timestamp // "unknown"' "$MANIFEST_FILE")
    JSON_CID=$(jq -r '.results.json_cid // ""' "$MANIFEST_FILE")
    LOG_CID=$(jq -r '.results.log_cid // ""' "$MANIFEST_FILE")

    echo -e "${GREEN}Deployment: $DEPLOYMENT_ID ($MINER_TYPE)${NC}"
    echo -e "${GREEN}Timestamp: $TIMESTAMP${NC}"

    # Create deployment directory
    DEPLOYMENT_DIR="${OUTPUT_DIR}/${MINER_TYPE}/${DEPLOYMENT_ID}"
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

# Cleanup temp directory
rm -rf "$TEMP_DIR"

# Generate summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Collection Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total manifests found: ${#MANIFEST_LIST[@]}"
echo -e "${GREEN}Successfully collected: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo -e "${YELLOW}Results saved to: $OUTPUT_DIR${NC}"
echo ""

# Generate summary report
if [ $SUCCESS_COUNT -gt 0 ]; then
    SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

    cat > "$SUMMARY_FILE" <<EOF
IPFS Mining Results Summary (Admin Collection)
===============================================
Retrieved: $(date)
IPFS Node: $IPFS_NODE
Total manifests: ${#MANIFEST_LIST[@]}
Successful: $SUCCESS_COUNT
Failed: $FAIL_COUNT

Deployments:
EOF

    # Add details for each deployment
    for MINER_TYPE in cpu cuda cuda_sa cuda_gibbs metal; do
        if [ -d "$OUTPUT_DIR/$MINER_TYPE" ]; then
            echo "" >> "$SUMMARY_FILE"
            echo "$MINER_TYPE Miners:" >> "$SUMMARY_FILE"
            echo "-------------" >> "$SUMMARY_FILE"

            for DEPLOYMENT_DIR in "$OUTPUT_DIR/$MINER_TYPE"/*; do
                if [ -d "$DEPLOYMENT_DIR" ]; then
                    DEPLOYMENT_ID=$(basename "$DEPLOYMENT_DIR")
                    JSON_FILE="$DEPLOYMENT_DIR/results.json"
                    MANIFEST_FILE="$DEPLOYMENT_DIR/manifest.json"

                    if [ -f "$JSON_FILE" ]; then
                        BLOCKS=$(jq -r '.results.blocks_found // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
                        RATE=$(jq -r '.results.blocks_per_minute // 0' "$JSON_FILE" 2>/dev/null || echo "N/A")
                        TIMESTAMP=$(jq -r '.timestamp // "unknown"' "$MANIFEST_FILE" 2>/dev/null || echo "N/A")
                        echo "  $DEPLOYMENT_ID [$TIMESTAMP]: $BLOCKS blocks, $RATE blocks/min" >> "$SUMMARY_FILE"
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
echo "4. Generate plots: python tools/compare_mining_rates.py --analyze $OUTPUT_DIR"
