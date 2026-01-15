#!/bin/bash
# Entrypoint script for CUDA miner Docker container
set -e

# Get instance ID from AWS metadata service (if running on EC2)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "local_$(hostname)")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set output file path
OUTPUT_FILE="${OUTPUT_FILE:-/output/cuda_${INSTANCE_ID}_${TIMESTAMP}.json}"
LOG_FILE="${OUTPUT_FILE%.json}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Quip Protocol - CUDA Mining Comparison" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Instance ID: $INSTANCE_ID" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Duration: $MINING_DURATION" | tee -a "$LOG_FILE"
echo "Difficulty Energy: $DIFFICULTY_ENERGY" | tee -a "$LOG_FILE"
echo "Min Diversity: $MIN_DIVERSITY" | tee -a "$LOG_FILE"
echo "Min Solutions: $MIN_SOLUTIONS" | tee -a "$LOG_FILE"
echo "Topology: $TOPOLOGY_FILE" | tee -a "$LOG_FILE"
echo "GPU Device: $GPU_DEVICE" | tee -a "$LOG_FILE"
echo "Output File: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Check CUDA availability
echo "CUDA Information:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Construct the command
CMD="python tools/compare_mining_rates.py \
  --miner-type cuda \
  --difficulty-energy $DIFFICULTY_ENERGY \
  --duration $MINING_DURATION \
  --min-diversity $MIN_DIVERSITY \
  --min-solutions $MIN_SOLUTIONS \
  --topology $TOPOLOGY_FILE \
  --device $GPU_DEVICE \
  -o $OUTPUT_FILE"

# Run the mining comparison
echo "Running command: $CMD" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Execute and capture both stdout and stderr
$CMD 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Completed at: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

# If S3 bucket is specified, sync results
if [ -n "$S3_BUCKET" ]; then
    echo "Syncing results to S3: $S3_BUCKET" | tee -a "$LOG_FILE"
    aws s3 cp "$OUTPUT_FILE" "s3://$S3_BUCKET/cuda/" 2>&1 | tee -a "$LOG_FILE"
    aws s3 cp "$LOG_FILE" "s3://$S3_BUCKET/cuda/" 2>&1 | tee -a "$LOG_FILE"
fi

# Cleanup CUDA resources
echo "Cleaning up CUDA resources..." | tee -a "$LOG_FILE"
python -c "import cupy; cupy.cuda.runtime.deviceReset()" 2>&1 | tee -a "$LOG_FILE"

exit $EXIT_CODE
