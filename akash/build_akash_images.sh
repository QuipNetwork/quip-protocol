#!/bin/bash
# Build and push Docker images for Akash deployment
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
REGISTRY="${REGISTRY:-ghcr.io/your-org}"  # Change to your registry
VERSION="${VERSION:-latest}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Akash-Optimized Docker Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo ""

# Build CPU image
echo -e "${GREEN}[1/2] Building CPU miner for Akash...${NC}"
docker build \
  -f akash/Dockerfile.akash-cpu \
  -t ${REGISTRY}/quip-protocol-cpu-miner:${VERSION} \
  -t ${REGISTRY}/quip-protocol-cpu-miner:latest \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CPU image built successfully${NC}"
else
    echo -e "${RED}✗ CPU image build failed${NC}"
    exit 1
fi
echo ""

# Build CUDA image
echo -e "${GREEN}[2/2] Building CUDA miner for Akash...${NC}"
docker build \
  -f akash/Dockerfile.akash-cuda \
  -t ${REGISTRY}/quip-protocol-cuda-miner:${VERSION} \
  -t ${REGISTRY}/quip-protocol-cuda-miner:latest \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CUDA image built successfully${NC}"
else
    echo -e "${RED}✗ CUDA image build failed${NC}"
    exit 1
fi
echo ""

# Push images
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Push Images to Registry?${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Images will be pushed to: $REGISTRY"
echo ""
read -p "Push now? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Pushing images...${NC}"

    docker push ${REGISTRY}/quip-protocol-cpu-miner:${VERSION}
    docker push ${REGISTRY}/quip-protocol-cpu-miner:latest

    docker push ${REGISTRY}/quip-protocol-cuda-miner:${VERSION}
    docker push ${REGISTRY}/quip-protocol-cuda-miner:latest

    echo -e "${GREEN}✓ Images pushed successfully${NC}"
else
    echo -e "${YELLOW}Skipping push. Push manually with:${NC}"
    echo "  docker push ${REGISTRY}/quip-protocol-cpu-miner:${VERSION}"
    echo "  docker push ${REGISTRY}/quip-protocol-cuda-miner:${VERSION}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Update SDL files (deploy-cpu.yaml, deploy-cuda.yaml) with your registry"
echo "2. Deploy to Akash: ./akash/deploy.sh"
echo "3. Monitor deployment: ./akash/monitor.sh <deployment-id>"
echo "4. Retrieve results: ./akash/retrieve_results.sh <deployment-url>"
