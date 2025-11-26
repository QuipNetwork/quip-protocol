#!/bin/bash
# Akash-specific entrypoint that exposes logs and results via HTTP
set -e

MINER_TYPE="${MINER_TYPE:-cpu}"
MINING_DURATION="${MINING_DURATION:-90m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"
GPU_DEVICE="${GPU_DEVICE:-0}"

# Akash deployment identifier (from env or hostname)
DEPLOYMENT_ID="${AKASH_DEPLOYMENT_ID:-$(hostname)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output files
OUTPUT_DIR="/output"
mkdir -p "$OUTPUT_DIR"
OUTPUT_JSON="${OUTPUT_DIR}/${MINER_TYPE}_${DEPLOYMENT_ID}_${TIMESTAMP}.json"
OUTPUT_LOG="${OUTPUT_DIR}/${MINER_TYPE}_${DEPLOYMENT_ID}_${TIMESTAMP}.log"

# Create symlinks for easy HTTP access
ln -sf "$OUTPUT_JSON" "${OUTPUT_DIR}/latest.json"
ln -sf "$OUTPUT_LOG" "${OUTPUT_DIR}/latest.log"

echo "========================================" | tee -a "$OUTPUT_LOG"
echo "Quip Protocol - Akash Mining Deployment" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"
echo "Miner Type: $MINER_TYPE" | tee -a "$OUTPUT_LOG"
echo "Deployment ID: $DEPLOYMENT_ID" | tee -a "$OUTPUT_LOG"
echo "Start Time: $(date)" | tee -a "$OUTPUT_LOG"
echo "Duration: $MINING_DURATION" | tee -a "$OUTPUT_LOG"
echo "Difficulty: $DIFFICULTY_ENERGY" | tee -a "$OUTPUT_LOG"
echo "Output JSON: $OUTPUT_JSON" | tee -a "$OUTPUT_LOG"
echo "Output Log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"

# Start simple HTTP server in background to serve results
echo "Starting HTTP server on port 8080..." | tee -a "$OUTPUT_LOG"
(cd "$OUTPUT_DIR" && python3 -m http.server 8080 &)
HTTP_PID=$!

# Construct mining command
CMD="python tools/compare_mining_rates.py \
  --miner-type $MINER_TYPE \
  --difficulty-energy $DIFFICULTY_ENERGY \
  --duration $MINING_DURATION \
  --min-diversity $MIN_DIVERSITY \
  --min-solutions $MIN_SOLUTIONS \
  --topology $TOPOLOGY_FILE"

if [ "$MINER_TYPE" = "cuda" ]; then
  CMD="$CMD --device $GPU_DEVICE"
fi

CMD="$CMD -o $OUTPUT_JSON"

# Run mining experiment
echo "Running: $CMD" | tee -a "$OUTPUT_LOG"
echo "----------------------------------------" | tee -a "$OUTPUT_LOG"

$CMD 2>&1 | tee -a "$OUTPUT_LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------" | tee -a "$OUTPUT_LOG"
echo "Completed at: $(date)" | tee -a "$OUTPUT_LOG"
echo "Exit code: $EXIT_CODE" | tee -a "$OUTPUT_LOG"

# Upload to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    echo "Uploading results to S3: $S3_BUCKET" | tee -a "$OUTPUT_LOG"
    aws s3 cp "$OUTPUT_JSON" "s3://$S3_BUCKET/${MINER_TYPE}/" 2>&1 | tee -a "$OUTPUT_LOG" || echo "S3 upload failed"
    aws s3 cp "$OUTPUT_LOG" "s3://$S3_BUCKET/${MINER_TYPE}/" 2>&1 | tee -a "$OUTPUT_LOG" || echo "S3 upload failed"
fi

# Create completion marker
echo "COMPLETED" > "${OUTPUT_DIR}/status.txt"
date >> "${OUTPUT_DIR}/status.txt"

# Keep HTTP server running for result retrieval
echo "" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"
echo "Mining complete! Results available at:" | tee -a "$OUTPUT_LOG"
echo "  JSON: http://<deployment-url>/latest.json" | tee -a "$OUTPUT_LOG"
echo "  Log:  http://<deployment-url>/latest.log" | tee -a "$OUTPUT_LOG"
echo "  Status: http://<deployment-url>/status.txt" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo "HTTP server will remain running for result retrieval..." | tee -a "$OUTPUT_LOG"
echo "Download results, then close the deployment to stop billing." | tee -a "$OUTPUT_LOG"

# Keep container alive for result retrieval (wait for HTTP server)
wait $HTTP_PID

exit $EXIT_CODE
