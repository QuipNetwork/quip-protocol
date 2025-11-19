#!/bin/bash
# User-data script for Metal (macOS) mining EC2 instances
# This script runs on instance startup and performs the mining experiment
# Note: Runs directly on macOS without Docker
set -e

# ====================
# Configuration
# ====================
S3_BUCKET="${S3_BUCKET:-REPLACE_WITH_S3_BUCKET}"
MINING_DURATION="${MINING_DURATION:-24h}"  # Default to 24h (minimum billing)
DIFFICULTY_ENERGY="${DIFFICULTY_ENERGY:--14900}"
MIN_DIVERSITY="${MIN_DIVERSITY:-0.1}"
MIN_SOLUTIONS="${MIN_SOLUTIONS:-5}"
TOPOLOGY_FILE="${TOPOLOGY_FILE:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/your-repo/quip-protocol.git}"
AUTO_TERMINATE="${AUTO_TERMINATE:-false}"  # Never auto-terminate Mac instances (24h minimum)

# ====================
# System Setup
# ====================
echo "Starting Metal (macOS) mining setup..."
date

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Update Homebrew
brew update

# Install required packages
echo "Installing dependencies..."
brew install python@3.13 git awscli

# ====================
# Instance Metadata
# ====================
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

echo "Instance ID: $INSTANCE_ID"
echo "Region: $REGION"
echo "Availability Zone: $AZ"

# Detect GPU cores
if command -v ioreg &> /dev/null; then
    GPU_CORES=$(ioreg -l | grep gpu-core-count | awk '{print $4}' | head -n 1)
    echo "Detected GPU cores: $GPU_CORES"
else
    echo "Warning: Could not detect GPU cores"
    GPU_CORES="unknown"
fi

# ====================
# Clone Repository
# ====================
echo "Cloning Quip Protocol repository..."
cd /tmp
rm -rf quip-protocol
git clone $GITHUB_REPO || { echo "Failed to clone repository"; exit 1; }
cd quip-protocol

# ====================
# Python Environment Setup
# ====================
echo "Setting up Python environment..."

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install \
    "pyobjc-framework-Metal" \
    "pyobjc-framework-MetalPerformanceShaders" \
    "dwave-ocean-sdk>=6.0.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.6.0" \
    "pandas>=2.0.0" \
    "seaborn>=0.12.0" \
    "python-dotenv>=1.0.0" \
    "click>=8.1.7" \
    "hashsigs>=0.0.2" \
    "blake3>=1.0.5"

# ====================
# Run Mining Experiment
# ====================
echo "Starting Metal mining experiment..."
echo "Configuration:"
echo "  Duration: $MINING_DURATION"
echo "  Difficulty Energy: $DIFFICULTY_ENERGY"
echo "  Min Diversity: $MIN_DIVERSITY"
echo "  Min Solutions: $MIN_SOLUTIONS"
echo "  Topology: $TOPOLOGY_FILE"
echo "  GPU Cores: $GPU_CORES"
echo "  S3 Bucket: $S3_BUCKET"

# Create output directory
mkdir -p /tmp/output

# Generate output filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="/tmp/output/metal_${INSTANCE_ID}_${TIMESTAMP}.json"
OUTPUT_LOG="/tmp/output/metal_${INSTANCE_ID}_${TIMESTAMP}.log"

# Run mining comparison
python tools/compare_mining_rates.py \
    --miner-type metal \
    --difficulty-energy $DIFFICULTY_ENERGY \
    --duration $MINING_DURATION \
    --min-diversity $MIN_DIVERSITY \
    --min-solutions $MIN_SOLUTIONS \
    --topology $TOPOLOGY_FILE \
    -o $OUTPUT_JSON \
    2>&1 | tee $OUTPUT_LOG

EXIT_CODE=${PIPESTATUS[0]}

echo "Mining experiment exited with code: $EXIT_CODE"
date

# ====================
# Upload Results to S3
# ====================
echo "Uploading results to S3..."
aws s3 cp $OUTPUT_JSON s3://${S3_BUCKET}/metal/ || echo "Warning: Failed to upload JSON to S3"
aws s3 cp $OUTPUT_LOG s3://${S3_BUCKET}/metal/ || echo "Warning: Failed to upload log to S3"

echo "Results uploaded to s3://${S3_BUCKET}/metal/"

# ====================
# Additional Experiments (Optional)
# ====================
# Since Mac instances have 24-hour minimum billing, you can queue
# additional experiments here to maximize value

# Example: Run multiple difficulty levels
# for DIFF in -4100 -14900 -15450; do
#     python tools/compare_mining_rates.py \
#         --miner-type metal \
#         --difficulty-energy $DIFF \
#         --duration 2h \
#         -o /tmp/output/metal_${INSTANCE_ID}_diff${DIFF}.json
#     aws s3 cp /tmp/output/metal_${INSTANCE_ID}_diff${DIFF}.json s3://${S3_BUCKET}/metal/
# done

# ====================
# Cleanup & Termination
# ====================
echo "Experiment complete!"
echo ""
echo "IMPORTANT: This is a Mac instance with 24-hour minimum billing."
echo "  - Current runtime will be billed for full 24 hours"
echo "  - Consider running additional experiments to maximize value"
echo "  - Manually terminate after 24 hours to avoid extra charges"
echo ""

if [ "$AUTO_TERMINATE" == "true" ]; then
    echo "WARNING: Auto-terminate requested, but NOT recommended for Mac instances"
    echo "Terminating anyway in 60 seconds..."
    sleep 60
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION || true
else
    echo "Auto-termination disabled (recommended for Mac instances)."
    echo "To terminate manually:"
    echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
fi

exit $EXIT_CODE
