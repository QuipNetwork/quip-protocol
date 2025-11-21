#!/bin/bash
# Collect and analyze mining results from S3
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MINER_TYPES="${MINER_TYPES:-cpu cuda metal}"  # Space-separated list

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Collecting Mining Results from S3${NC}"
echo -e "${BLUE}========================================${NC}"

# Validate inputs
if [ -z "$S3_BUCKET" ]; then
    echo -e "${RED}Error: S3_BUCKET not specified${NC}"
    echo "Usage: S3_BUCKET=your-bucket-name ./aws/collect_results.sh"
    exit 1
fi

echo "S3 Bucket: s3://$S3_BUCKET"
echo "Output Directory: $OUTPUT_DIR"
echo "Miner Types: $MINER_TYPES"
echo "AWS Region: $AWS_REGION"
echo ""

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ====================
# Download Results
# ====================
echo -e "${GREEN}Downloading results from S3...${NC}"

for MINER_TYPE in $MINER_TYPES; do
    echo -e "${BLUE}Downloading ${MINER_TYPE} results...${NC}"

    MINER_DIR="$OUTPUT_DIR/$MINER_TYPE"
    mkdir -p "$MINER_DIR"

    # Download all JSON and log files
    aws s3 sync \
        "s3://$S3_BUCKET/$MINER_TYPE/" \
        "$MINER_DIR/" \
        --region "$AWS_REGION" \
        --exclude "*" \
        --include "*.json" \
        --include "*.log"

    # Count files
    JSON_COUNT=$(find "$MINER_DIR" -name "*.json" | wc -l)
    LOG_COUNT=$(find "$MINER_DIR" -name "*.log" | wc -l)

    echo -e "${GREEN}  ✓ Downloaded $JSON_COUNT JSON files, $LOG_COUNT log files${NC}"
done

echo ""

# ====================
# Generate Summary
# ====================
echo -e "${GREEN}Generating summary statistics...${NC}"

SUMMARY_FILE="$OUTPUT_DIR/summary_${EXPERIMENT_ID:-all}.txt"

cat > "$SUMMARY_FILE" <<EOF
Quip Protocol Mining Results Summary
=====================================
Generated: $(date)
S3 Bucket: s3://$S3_BUCKET
Experiment ID: ${EXPERIMENT_ID:-N/A}

EOF

for MINER_TYPE in $MINER_TYPES; do
    MINER_DIR="$OUTPUT_DIR/$MINER_TYPE"

    if [ ! -d "$MINER_DIR" ]; then
        continue
    fi

    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "Miner Type: $MINER_TYPE" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"

    JSON_FILES=$(find "$MINER_DIR" -name "*.json")
    NUM_FILES=$(echo "$JSON_FILES" | wc -l)

    if [ -z "$JSON_FILES" ] || [ "$NUM_FILES" -eq 0 ]; then
        echo "No results found" >> "$SUMMARY_FILE"
        continue
    fi

    echo "Number of experiments: $NUM_FILES" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    # Extract key metrics using jq (if available)
    if command -v jq &> /dev/null; then
        echo "Aggregated Statistics:" >> "$SUMMARY_FILE"
        echo "---------------------" >> "$SUMMARY_FILE"

        # Total attempts
        TOTAL_ATTEMPTS=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.total_attempts // 0) | add')
        echo "  Total Attempts: ${TOTAL_ATTEMPTS:-0}" >> "$SUMMARY_FILE"

        # Total blocks found
        TOTAL_BLOCKS=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.blocks_found // 0) | add')
        echo "  Total Blocks Found: ${TOTAL_BLOCKS:-0}" >> "$SUMMARY_FILE"

        # Average success rate
        AVG_SUCCESS_RATE=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.success_rate // 0) | add / length')
        echo "  Average Success Rate: ${AVG_SUCCESS_RATE:-0}" >> "$SUMMARY_FILE"

        # Average blocks per minute
        AVG_BLOCKS_PER_MIN=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.blocks_per_minute // 0) | add / length')
        echo "  Average Blocks/Minute: ${AVG_BLOCKS_PER_MIN:-0}" >> "$SUMMARY_FILE"

        # Average energy
        AVG_ENERGY=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.avg_energy // 0) | add / length')
        echo "  Average Energy: ${AVG_ENERGY:-0}" >> "$SUMMARY_FILE"

        # Average diversity
        AVG_DIVERSITY=$(echo "$JSON_FILES" | xargs cat | jq -s 'map(.results.avg_diversity // 0) | add / length')
        echo "  Average Diversity: ${AVG_DIVERSITY:-0}" >> "$SUMMARY_FILE"

        echo "" >> "$SUMMARY_FILE"
        echo "Individual Experiments:" >> "$SUMMARY_FILE"
        echo "----------------------" >> "$SUMMARY_FILE"

        # List each experiment with key stats
        for JSON_FILE in $JSON_FILES; do
            FILENAME=$(basename "$JSON_FILE")
            BLOCKS=$(cat "$JSON_FILE" | jq -r '.results.blocks_found // 0')
            RATE=$(cat "$JSON_FILE" | jq -r '.results.blocks_per_minute // 0')
            ENERGY=$(cat "$JSON_FILE" | jq -r '.results.avg_energy // 0')

            echo "  $FILENAME: $BLOCKS blocks, $RATE blocks/min, $ENERGY avg energy" >> "$SUMMARY_FILE"
        done
    else
        echo "Install 'jq' for detailed statistics" >> "$SUMMARY_FILE"
        echo "Files:" >> "$SUMMARY_FILE"
        ls -lh "$MINER_DIR"/*.json >> "$SUMMARY_FILE" || true
    fi
done

echo -e "${GREEN}✓ Summary saved to: $SUMMARY_FILE${NC}"
echo ""

# ====================
# Display Summary
# ====================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"
cat "$SUMMARY_FILE"

# ====================
# Optional: Create CSV
# ====================
if command -v jq &> /dev/null; then
    echo ""
    echo -e "${GREEN}Generating CSV export...${NC}"

    CSV_FILE="$OUTPUT_DIR/results_${EXPERIMENT_ID:-all}.csv"

    # CSV header
    echo "miner_type,instance_id,timestamp,total_attempts,blocks_found,success_rate,blocks_per_minute,avg_energy,avg_diversity,avg_solutions" > "$CSV_FILE"

    for MINER_TYPE in $MINER_TYPES; do
        MINER_DIR="$OUTPUT_DIR/$MINER_TYPE"

        if [ ! -d "$MINER_DIR" ]; then
            continue
        fi

        JSON_FILES=$(find "$MINER_DIR" -name "*.json")

        for JSON_FILE in $JSON_FILES; do
            FILENAME=$(basename "$JSON_FILE" .json)

            # Extract instance ID and timestamp from filename
            # Format: {miner_type}_{instance_id}_{timestamp}.json
            INSTANCE_ID=$(echo "$FILENAME" | cut -d'_' -f2)
            TIMESTAMP=$(echo "$FILENAME" | cut -d'_' -f3-)

            # Extract metrics
            ATTEMPTS=$(cat "$JSON_FILE" | jq -r '.results.total_attempts // 0')
            BLOCKS=$(cat "$JSON_FILE" | jq -r '.results.blocks_found // 0')
            SUCCESS=$(cat "$JSON_FILE" | jq -r '.results.success_rate // 0')
            RATE=$(cat "$JSON_FILE" | jq -r '.results.blocks_per_minute // 0')
            ENERGY=$(cat "$JSON_FILE" | jq -r '.results.avg_energy // 0')
            DIVERSITY=$(cat "$JSON_FILE" | jq -r '.results.avg_diversity // 0')
            SOLUTIONS=$(cat "$JSON_FILE" | jq -r '.results.avg_solutions // 0')

            echo "$MINER_TYPE,$INSTANCE_ID,$TIMESTAMP,$ATTEMPTS,$BLOCKS,$SUCCESS,$RATE,$ENERGY,$DIVERSITY,$SOLUTIONS" >> "$CSV_FILE"
        done
    done

    echo -e "${GREEN}✓ CSV exported to: $CSV_FILE${NC}"
fi

# ====================
# Completion
# ====================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Collection Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Output Files:${NC}"
echo "  Results Directory: $OUTPUT_DIR"
echo "  Summary: $SUMMARY_FILE"
if [ -f "$CSV_FILE" ]; then
    echo "  CSV Export: $CSV_FILE"
fi
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review summary statistics in $SUMMARY_FILE"
echo "2. Analyze individual results in $OUTPUT_DIR/{cpu,cuda,metal}/"
echo "3. Import CSV into spreadsheet or analysis tools"

# ====================
# Generate Visualizations
# ====================
echo ""
echo -e "${GREEN}Generating visualizations...${NC}"

# Find one representative JSON file from each miner type
CPU_JSON=$(find "$OUTPUT_DIR/cpu" -name "*.json" 2>/dev/null | head -n1)
CUDA_JSON=$(find "$OUTPUT_DIR/cuda" -name "*.json" 2>/dev/null | head -n1)
METAL_JSON=$(find "$OUTPUT_DIR/metal" -name "*.json" 2>/dev/null | head -n1)
QPU_JSON=$(find "$OUTPUT_DIR/qpu" -name "*.json" 2>/dev/null | head -n1)

# Build visualization command
VIZ_CMD="python tools/visualize_comparative_performance.py"
VIZ_ARGS=""

[ -n "$CPU_JSON" ] && VIZ_ARGS="$VIZ_ARGS --cpu-mining \"$CPU_JSON\""
[ -n "$CUDA_JSON" ] && VIZ_ARGS="$VIZ_ARGS --cuda-mining \"$CUDA_JSON\""
[ -n "$METAL_JSON" ] && VIZ_ARGS="$VIZ_ARGS --metal-mining \"$METAL_JSON\""
[ -n "$QPU_JSON" ] && VIZ_ARGS="$VIZ_ARGS --qpu-mining \"$QPU_JSON\""

VIZ_OUTPUT="$OUTPUT_DIR/performance_comparison_${EXPERIMENT_ID:-all}.pdf"
VIZ_ARGS="$VIZ_ARGS --output \"$VIZ_OUTPUT\""

if [ -n "$CPU_JSON" ] || [ -n "$CUDA_JSON" ] || [ -n "$METAL_JSON" ] || [ -n "$QPU_JSON" ]; then
    echo -e "${BLUE}Running: $VIZ_CMD $VIZ_ARGS${NC}"
    eval "$VIZ_CMD $VIZ_ARGS" 2>&1 | grep -E "(✅|❌|Error)" || true

    if [ -f "$VIZ_OUTPUT" ]; then
        echo -e "${GREEN}✓ Visualization saved to: $VIZ_OUTPUT${NC}"
        echo ""
        echo -e "${GREEN}Opening visualization...${NC}"
        open "$VIZ_OUTPUT" 2>/dev/null || echo "  (Use: open $VIZ_OUTPUT)"
    else
        echo -e "${YELLOW}⚠️  Visualization generation failed${NC}"
        echo "  Install required dependencies: pip install matplotlib seaborn pandas numpy"
    fi
else
    echo -e "${YELLOW}⚠️  No JSON files found for visualization${NC}"
fi

echo ""
echo "4. View visualization: $VIZ_OUTPUT"
