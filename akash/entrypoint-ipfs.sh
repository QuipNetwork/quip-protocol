#!/bin/bash
# Akash entrypoint with IPFS upload support
set -e

MINER_TYPE="${MINER_TYPE:-cpu}"
MINING_DURATION="${MINING_DURATION:-90m}"
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"
GPU_DEVICE="${GPU_DEVICE:-0}"

# ============================================================================
# GPU DIAGNOSTICS - Run before mining to catch CUDA issues early
# ============================================================================
echo "========================================"
echo "GPU Diagnostics"
echo "========================================"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi || echo "  nvidia-smi FAILED (exit code: $?)"
    echo ""
else
    echo "nvidia-smi: NOT FOUND"
fi

# Check CUDA environment
echo "CUDA Environment:"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "  NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-not set}"
echo ""

# Check if CUDA libraries are accessible
if [ -d "/usr/local/cuda/lib64" ]; then
    echo "CUDA libraries found in /usr/local/cuda/lib64"
    ls /usr/local/cuda/lib64/*.so* 2>/dev/null | head -5
else
    echo "CUDA libraries: NOT FOUND in /usr/local/cuda/lib64"
fi
echo ""

# Python CUDA test
echo "Python CUDA Test:"
python3 -c "
import sys
try:
    import cupy as cp
    print(f'  CuPy version: {cp.__version__}')
    print(f'  CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}')
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f'  CUDA devices detected: {device_count}')
        for i in range(device_count):
            try:
                cp.cuda.Device(i).use()
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f'    Device {i}: {props[\"name\"].decode()} ({props[\"totalGlobalMem\"] // 1024**3} GB)')
                # Try allocation
                _ = cp.zeros(1)
                print(f'    Device {i}: Memory allocation OK')
            except Exception as e:
                print(f'    Device {i}: FAILED - {e}')
    except Exception as e:
        print(f'  Device enumeration failed: {e}')
except ImportError as e:
    print(f'  CuPy import failed: {e}')
except Exception as e:
    print(f'  CUDA test failed: {e}')
" 2>&1
echo ""
echo "========================================"
echo ""

# If CUDA miner requested but CUDA not working, exit early with clear error
if [ "$MINER_TYPE" = "cuda" ]; then
    GPU_OK=$(python3 -c "
import sys
try:
    import cupy as cp
    cp.cuda.Device(0).use()
    _ = cp.zeros(1)
    print('OK')
except:
    print('FAIL')
" 2>/dev/null)

    if [ "$GPU_OK" != "OK" ]; then
        echo "========================================"
        echo "FATAL: CUDA requested but GPU not accessible"
        echo ""
        echo "Possible causes:"
        echo "  1. Host driver too old (need driver 520+ for CUDA 11.8)"
        echo "  2. NVIDIA Container Runtime not configured on this provider"
        echo "  3. No GPU allocated to this container"
        echo "  4. GPU not exposed to container (missing --gpus flag)"
        echo ""
        echo "This image uses CUDA 11.8 for good compatibility."
        echo "If this still fails, try a different Akash provider,"
        echo "or use the CPU image instead."
        echo "========================================"
        exit 1
    fi
fi

# IPFS configuration
IPFS_NODE="${IPFS_NODE:-}"  # e.g., https://carback-ipfs.ngrok.io
IPFS_API_KEY="${IPFS_API_KEY:-}"
IPFS_PIN="${IPFS_PIN:-false}"  # Pin to IPFS (keep permanently)

# S3 configuration (optional fallback)
S3_BUCKET="${S3_BUCKET:-}"

# Akash deployment identifier
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
echo "Quip Protocol - Akash Mining with IPFS" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"
echo "Miner Type: $MINER_TYPE" | tee -a "$OUTPUT_LOG"
echo "Deployment ID: $DEPLOYMENT_ID" | tee -a "$OUTPUT_LOG"
echo "Start Time: $(date)" | tee -a "$OUTPUT_LOG"
echo "Duration: $MINING_DURATION" | tee -a "$OUTPUT_LOG"
echo "Difficulty: $DIFFICULTY_ENERGY" | tee -a "$OUTPUT_LOG"
echo "Output JSON: $OUTPUT_JSON" | tee -a "$OUTPUT_LOG"
echo "Output Log: $OUTPUT_LOG" | tee -a "$OUTPUT_LOG"

if [ -n "$IPFS_NODE" ]; then
    echo "IPFS Node: $IPFS_NODE" | tee -a "$OUTPUT_LOG"
    echo "IPFS Pin: $IPFS_PIN" | tee -a "$OUTPUT_LOG"
fi

echo "========================================" | tee -a "$OUTPUT_LOG"

# Start simple HTTP server in background to serve results
echo "Starting HTTP server on port 8080..." | tee -a "$OUTPUT_LOG"
(cd "$OUTPUT_DIR" && python3 -m http.server 8080 &)
HTTP_PID=$!

# Function to upload to IPFS
upload_to_ipfs() {
    local file=$1
    local filename=$(basename "$file")

    if [ -z "$IPFS_NODE" ]; then
        echo "IPFS upload skipped (IPFS_NODE not configured)" | tee -a "$OUTPUT_LOG"
        return 1
    fi

    echo "Uploading $filename to IPFS..." | tee -a "$OUTPUT_LOG"
    echo "  IPFS Node: $IPFS_NODE" | tee -a "$OUTPUT_LOG"
    echo "  File size: $(ls -lh "$file" | awk '{print $5}')" | tee -a "$OUTPUT_LOG"

    # Upload via IPFS HTTP API (capture both stdout and stderr)
    local CURL_OUTPUT=$(mktemp)
    local HTTP_CODE=$(curl -s -w "%{http_code}" -X POST \
        -H "X-API-Key: $IPFS_API_KEY" \
        -F "file=@$file" \
        "${IPFS_NODE}/api/v0/add?pin=${IPFS_PIN}&wrap-with-directory=false" \
        -o "$CURL_OUTPUT" 2>&1)
    local CURL_EXIT=$?
    local IPFS_RESPONSE=$(cat "$CURL_OUTPUT")
    rm -f "$CURL_OUTPUT"

    echo "  Curl exit code: $CURL_EXIT, HTTP status: $HTTP_CODE" | tee -a "$OUTPUT_LOG"

    if [ $CURL_EXIT -ne 0 ]; then
        echo "  Curl error: $IPFS_RESPONSE" | tee -a "$OUTPUT_LOG"
        echo "✗ IPFS upload failed for $filename (curl error)" | tee -a "$OUTPUT_LOG"
        return 1
    fi

    if [ "$HTTP_CODE" != "200" ]; then
        echo "  HTTP error response: $IPFS_RESPONSE" | tee -a "$OUTPUT_LOG"
        echo "✗ IPFS upload failed for $filename (HTTP $HTTP_CODE)" | tee -a "$OUTPUT_LOG"
        return 1
    fi

    # Extract CID (hash)
    local CID=$(echo "$IPFS_RESPONSE" | grep -o '"Hash":"[^"]*' | cut -d'"' -f4)

    if [ -n "$CID" ]; then
        echo "✓ Uploaded to IPFS: $CID" | tee -a "$OUTPUT_LOG"
        echo "  Gateway URL: https://ipfs.io/ipfs/$CID" | tee -a "$OUTPUT_LOG"
        echo "  Your gateway: ${IPFS_NODE}/ipfs/$CID" | tee -a "$OUTPUT_LOG"

        # Save CID to file for easy retrieval
        echo "$CID" > "${file}.cid"

        # Add to MFS (Mutable File System) so it shows in IPFS Web UI Files view
        local MFS_PATH="/${DEPLOYMENT_ID}/${filename}"
        echo "  Adding to MFS: $MFS_PATH" | tee -a "$OUTPUT_LOG"

        # Create directory if needed
        curl -s -X POST \
            -H "X-API-Key: $IPFS_API_KEY" \
            "${IPFS_NODE}/api/v0/files/mkdir?arg=/${DEPLOYMENT_ID}&parents=true" \
            > /dev/null 2>&1

        # Copy file to MFS
        local MFS_RESPONSE=$(curl -s -X POST \
            -H "X-API-Key: $IPFS_API_KEY" \
            "${IPFS_NODE}/api/v0/files/cp?arg=/ipfs/${CID}&arg=${MFS_PATH}" 2>&1)

        if [ $? -eq 0 ]; then
            echo "  ✓ Added to MFS: $MFS_PATH" | tee -a "$OUTPUT_LOG"
        else
            echo "  ⚠ MFS copy failed (file still available via CID): $MFS_RESPONSE" | tee -a "$OUTPUT_LOG"
        fi

        return 0
    fi

    echo "  Response (no CID found): $IPFS_RESPONSE" | tee -a "$OUTPUT_LOG"
    echo "✗ IPFS upload failed for $filename (no CID in response)" | tee -a "$OUTPUT_LOG"
    return 1
}

# Function to upload to S3
upload_to_s3() {
    local file=$1
    local filename=$(basename "$file")

    if [ -z "$S3_BUCKET" ]; then
        return 1
    fi

    echo "Uploading $filename to S3..." | tee -a "$OUTPUT_LOG"
    aws s3 cp "$file" "s3://$S3_BUCKET/${MINER_TYPE}/" 2>&1 | tee -a "$OUTPUT_LOG"

    return $?
}

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

# Upload results
echo "" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"
echo "Uploading Results" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"

# Try IPFS first (if configured)
IPFS_UPLOAD_SUCCESS=false

if [ -n "$IPFS_NODE" ]; then
    upload_to_ipfs "$OUTPUT_JSON"
    IPFS_JSON_SUCCESS=$?

    upload_to_ipfs "$OUTPUT_LOG"
    IPFS_LOG_SUCCESS=$?

    # Create manifest file with both CIDs
    if [ -f "${OUTPUT_JSON}.cid" ] && [ -f "${OUTPUT_LOG}.cid" ]; then
        MANIFEST_FILE="${OUTPUT_DIR}/manifest_${DEPLOYMENT_ID}_${TIMESTAMP}.json"
        cat > "$MANIFEST_FILE" <<EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "miner_type": "$MINER_TYPE",
  "timestamp": "$TIMESTAMP",
  "duration": "$MINING_DURATION",
  "difficulty": $DIFFICULTY_ENERGY,
  "results": {
    "json_cid": "$(cat ${OUTPUT_JSON}.cid)",
    "log_cid": "$(cat ${OUTPUT_LOG}.cid)",
    "json_url": "https://ipfs.io/ipfs/$(cat ${OUTPUT_JSON}.cid)",
    "log_url": "https://ipfs.io/ipfs/$(cat ${OUTPUT_LOG}.cid)"
  }
}
EOF

        # Upload manifest to IPFS
        upload_to_ipfs "$MANIFEST_FILE"
        IPFS_MANIFEST_SUCCESS=$?

        if [ -f "${MANIFEST_FILE}.cid" ]; then
            MANIFEST_CID=$(cat "${MANIFEST_FILE}.cid")
            echo "" | tee -a "$OUTPUT_LOG"
            echo "========================================" | tee -a "$OUTPUT_LOG"
            echo "📦 IPFS Manifest CID: $MANIFEST_CID" | tee -a "$OUTPUT_LOG"
            echo "🔗 Access at: https://ipfs.io/ipfs/$MANIFEST_CID" | tee -a "$OUTPUT_LOG"
            echo "========================================" | tee -a "$OUTPUT_LOG"

            # Save manifest CID for easy discovery
            echo "$MANIFEST_CID" > "${OUTPUT_DIR}/manifest.cid"
            ln -sf "${MANIFEST_FILE}" "${OUTPUT_DIR}/latest_manifest.json"

            # Check if all uploads succeeded
            if [ $IPFS_JSON_SUCCESS -eq 0 ] && [ $IPFS_LOG_SUCCESS -eq 0 ] && [ $IPFS_MANIFEST_SUCCESS -eq 0 ]; then
                IPFS_UPLOAD_SUCCESS=true
            fi
        fi
    fi
fi

# Fallback to S3 if configured and IPFS failed
if [ -n "$S3_BUCKET" ]; then
    if [ -z "$IPFS_NODE" ] || [ $IPFS_JSON_SUCCESS -ne 0 ]; then
        upload_to_s3 "$OUTPUT_JSON"
        upload_to_s3 "$OUTPUT_LOG"
    fi
fi

# Create completion marker
echo "COMPLETED" > "${OUTPUT_DIR}/status.txt"
date >> "${OUTPUT_DIR}/status.txt"

if [ -f "${OUTPUT_DIR}/manifest.cid" ]; then
    echo "IPFS Manifest: $(cat ${OUTPUT_DIR}/manifest.cid)" >> "${OUTPUT_DIR}/status.txt"
fi

# Decide whether to shut down or keep running
echo "" | tee -a "$OUTPUT_LOG"
echo "========================================" | tee -a "$OUTPUT_LOG"

if [ "$IPFS_UPLOAD_SUCCESS" = true ]; then
    # All IPFS uploads succeeded - shut down automatically
    MANIFEST_CID=$(cat "${OUTPUT_DIR}/manifest.cid")
    echo "✅ All files successfully uploaded to IPFS!" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
    echo "📦 IPFS Manifest CID: $MANIFEST_CID" | tee -a "$OUTPUT_LOG"
    echo "🔗 Access at: https://ipfs.io/ipfs/$MANIFEST_CID" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
    echo "Retrieve your results with:" | tee -a "$OUTPUT_LOG"
    echo "  ./akash/collect_ipfs_results.sh \"$MANIFEST_CID\"" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
    echo "🛑 Auto-shutdown enabled - stopping container..." | tee -a "$OUTPUT_LOG"
    echo "   (IPFS upload successful, no need to keep running)" | tee -a "$OUTPUT_LOG"
    echo "========================================" | tee -a "$OUTPUT_LOG"

    # Kill HTTP server and exit
    kill $HTTP_PID 2>/dev/null || true
    sleep 2
    exit $EXIT_CODE

else
    # IPFS upload failed or not configured - keep HTTP server running
    echo "⚠️  IPFS upload incomplete or failed" | tee -a "$OUTPUT_LOG"
    echo "" | tee -a "$OUTPUT_LOG"
    echo "Mining complete! Results available at:" | tee -a "$OUTPUT_LOG"
    echo "  JSON: http://<deployment-url>/latest.json" | tee -a "$OUTPUT_LOG"
    echo "  Log:  http://<deployment-url>/latest.log" | tee -a "$OUTPUT_LOG"
    echo "  Status: http://<deployment-url>/status.txt" | tee -a "$OUTPUT_LOG"

    if [ -f "${OUTPUT_DIR}/manifest.cid" ]; then
        MANIFEST_CID=$(cat "${OUTPUT_DIR}/manifest.cid")
        echo "" | tee -a "$OUTPUT_LOG"
        echo "Partial IPFS Results:" | tee -a "$OUTPUT_LOG"
        echo "  Manifest: https://ipfs.io/ipfs/$MANIFEST_CID" | tee -a "$OUTPUT_LOG"
        echo "  Manifest (local): http://<deployment-url>/latest_manifest.json" | tee -a "$OUTPUT_LOG"
    fi

    echo "" | tee -a "$OUTPUT_LOG"
    echo "========================================" | tee -a "$OUTPUT_LOG"
    echo "⏳ HTTP server will remain running for manual result retrieval..." | tee -a "$OUTPUT_LOG"
    echo "   Download results, then close the deployment to stop billing." | tee -a "$OUTPUT_LOG"
    echo "========================================" | tee -a "$OUTPUT_LOG"

    # Keep container alive for result retrieval (wait for HTTP server)
    wait $HTTP_PID
    exit $EXIT_CODE
fi
