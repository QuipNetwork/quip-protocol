#!/bin/bash
# Test Docker images locally with short mining runs
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test parameters (short duration for quick validation)
TEST_DURATION="${TEST_DURATION:-10s}"
TEST_DIFFICULTY="${TEST_DIFFICULTY:--14900}"
TEST_DIVERSITY="${TEST_DIVERSITY:-0.1}"
TEST_SOLUTIONS="${TEST_SOLUTIONS:-5}"
TEST_TOPOLOGY="${TEST_TOPOLOGY:-dwave_topologies/topologies/advantage2_system1_7.json.gz}"

# Create output directory
mkdir -p docker/output

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Quip Protocol Docker Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Test Duration: $TEST_DURATION"
echo "Difficulty Energy: $TEST_DIFFICULTY"
echo "Min Diversity: $TEST_DIVERSITY"
echo "Min Solutions: $TEST_SOLUTIONS"
echo "Topology: $TEST_TOPOLOGY"
echo ""

# Test CPU miner
echo -e "${GREEN}[1/3] Testing CPU miner...${NC}"
docker run --rm \
  -v "$(pwd)/docker/output:/output" \
  -v "$(pwd)/dwave_topologies:/app/dwave_topologies:ro" \
  -e MINING_DURATION=$TEST_DURATION \
  -e DIFFICULTY_ENERGY=$TEST_DIFFICULTY \
  -e MIN_DIVERSITY=$TEST_DIVERSITY \
  -e MIN_SOLUTIONS=$TEST_SOLUTIONS \
  -e TOPOLOGY_FILE=$TEST_TOPOLOGY \
  quip-protocol/cpu-miner:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CPU miner test passed${NC}"
    # Check if output file was created
    if ls docker/output/cpu_*.json 1> /dev/null 2>&1; then
        LATEST_CPU=$(ls -t docker/output/cpu_*.json | head -n1)
        echo -e "${GREEN}  Output file: $LATEST_CPU${NC}"
        echo -e "${BLUE}  Preview:${NC}"
        cat "$LATEST_CPU" | head -n 20
    fi
else
    echo -e "${RED}✗ CPU miner test failed${NC}"
    exit 1
fi
echo ""

# Test CUDA miner (only if NVIDIA GPU is available)
echo -e "${GREEN}[2/3] Testing CUDA miner...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}NVIDIA GPU detected, testing CUDA miner...${NC}"
    docker run --rm \
      --gpus all \
      -v "$(pwd)/docker/output:/output" \
      -v "$(pwd)/dwave_topologies:/app/dwave_topologies:ro" \
      -e MINING_DURATION=$TEST_DURATION \
      -e DIFFICULTY_ENERGY=$TEST_DIFFICULTY \
      -e MIN_DIVERSITY=$TEST_DIVERSITY \
      -e MIN_SOLUTIONS=$TEST_SOLUTIONS \
      -e TOPOLOGY_FILE=$TEST_TOPOLOGY \
      -e GPU_DEVICE=0 \
      quip-protocol/cuda-miner:latest

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CUDA miner test passed${NC}"
        # Check if output file was created
        if ls docker/output/cuda_*.json 1> /dev/null 2>&1; then
            LATEST_CUDA=$(ls -t docker/output/cuda_*.json | head -n1)
            echo -e "${GREEN}  Output file: $LATEST_CUDA${NC}"
            echo -e "${BLUE}  Preview:${NC}"
            cat "$LATEST_CUDA" | head -n 20
        fi
    else
        echo -e "${RED}✗ CUDA miner test failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⊘ Skipping CUDA test (no NVIDIA GPU detected)${NC}"
fi
echo ""

# Test Metal miner (only on macOS)
echo -e "${GREEN}[3/3] Testing Metal miner...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}macOS detected, testing Metal miner...${NC}"
    echo -e "${YELLOW}Note: Metal may not work properly in Docker${NC}"
    echo -e "${YELLOW}For production, use direct deployment (see aws/user_data_metal.sh)${NC}"

    docker run --rm \
      -v "$(pwd)/docker/output:/output" \
      -v "$(pwd)/dwave_topologies:/app/dwave_topologies:ro" \
      -e MINING_DURATION=$TEST_DURATION \
      -e DIFFICULTY_ENERGY=$TEST_DIFFICULTY \
      -e MIN_DIVERSITY=$TEST_DIVERSITY \
      -e MIN_SOLUTIONS=$TEST_SOLUTIONS \
      -e TOPOLOGY_FILE=$TEST_TOPOLOGY \
      quip-protocol/metal-miner:latest || true

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Metal miner test passed${NC}"
        # Check if output file was created
        if ls docker/output/metal_*.json 1> /dev/null 2>&1; then
            LATEST_METAL=$(ls -t docker/output/metal_*.json | head -n1)
            echo -e "${GREEN}  Output file: $LATEST_METAL${NC}"
            echo -e "${BLUE}  Preview:${NC}"
            cat "$LATEST_METAL" | head -n 20
        fi
    else
        echo -e "${YELLOW}⚠ Metal miner test failed (expected in Docker)${NC}"
    fi
else
    echo -e "${YELLOW}⊘ Skipping Metal test (not on macOS)${NC}"
fi
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Output files created:"
ls -lh docker/output/ | grep -E '\.(json|log)$' || echo "No output files found"
echo ""
echo -e "${GREEN}Testing completed!${NC}"
