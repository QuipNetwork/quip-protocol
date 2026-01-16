#!/bin/bash
# Build all Docker images for Quip Protocol mining comparison
set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Version tag (can be overridden with environment variable)
VERSION="${VERSION:-latest}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Quip Protocol Docker Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Version: $VERSION"
echo ""

# Build CPU image
echo -e "${GREEN}Building CPU miner image...${NC}"
docker build \
  -f docker/mining_rates/Dockerfile.cpu \
  -t quip-protocol/cpu-miner:${VERSION} \
  -t quip-protocol/cpu-miner:latest \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CPU miner image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build CPU miner image${NC}"
    exit 1
fi
echo ""

# Build CUDA image
echo -e "${GREEN}Building CUDA miner image...${NC}"
docker build \
  -f docker/mining_rates/Dockerfile.cuda \
  -t quip-protocol/cuda-miner:${VERSION} \
  -t quip-protocol/cuda-miner:latest \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CUDA miner image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build CUDA miner image${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${BLUE}========================================${NC}"
docker images | grep quip-protocol
echo ""

# Optional: Push to registry
if [ -n "$PUSH_TO_REGISTRY" ]; then
    echo -e "${GREEN}Pushing images to registry...${NC}"

    if [ -n "$ECR_REGISTRY" ]; then
        # Tag for ECR
        docker tag quip-protocol/cpu-miner:${VERSION} ${ECR_REGISTRY}/cpu-miner:${VERSION}
        docker tag quip-protocol/cuda-miner:${VERSION} ${ECR_REGISTRY}/cuda-miner:${VERSION}

        # Push to ECR
        docker push ${ECR_REGISTRY}/cpu-miner:${VERSION}
        docker push ${ECR_REGISTRY}/cuda-miner:${VERSION}

        echo -e "${GREEN}✓ Images pushed to ECR: ${ECR_REGISTRY}${NC}"
    else
        echo -e "${RED}Error: ECR_REGISTRY environment variable not set${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}All builds completed!${NC}"
