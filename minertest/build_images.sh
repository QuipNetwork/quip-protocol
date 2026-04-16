#!/bin/bash
# Build and push Docker images for mining deployment
# Supports multi-platform builds for amd64 and arm64
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
REGISTRY="${REGISTRY:-docker.io/carback1}"  # Docker Hub registry
VERSION="${VERSION:-latest}"
PLATFORM="${PLATFORM:-linux/amd64,linux/arm64}"  # Multi-arch by default

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Docker Mining Images${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Registry: $REGISTRY"
echo "Version: $VERSION"
echo "Platform: $PLATFORM"
echo ""

# Check if buildx is available for cross-platform builds
if ! docker buildx version &> /dev/null; then
    echo -e "${RED}Error: docker buildx not available. Required for multi-platform builds.${NC}"
    echo -e "${YELLOW}Install with: docker buildx install${NC}"
    exit 1
fi

# Create/use buildx builder for cross-platform
BUILDER_NAME="quip-multiarch"
if ! docker buildx inspect ${BUILDER_NAME} &> /dev/null; then
    echo -e "${GREEN}Creating buildx builder for cross-platform builds...${NC}"
    docker buildx create --name ${BUILDER_NAME} --use --bootstrap
else
    docker buildx use ${BUILDER_NAME}
fi

# Detect if multi-platform build
IS_MULTI_PLATFORM=false
if [[ "$PLATFORM" == *","* ]]; then
    IS_MULTI_PLATFORM=true
    echo -e "${YELLOW}Note: Multi-platform builds require --push (cannot use --load)${NC}"
    echo -e "${YELLOW}Images will be built and pushed directly to registry.${NC}"
    echo ""
fi

if [ "$IS_MULTI_PLATFORM" = true ]; then
    # Multi-platform: must push directly (--load doesn't support multi-arch)
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Building & Pushing Multi-Platform Images${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Platforms: $PLATFORM"
    echo "Registry: $REGISTRY"
    echo ""
    read -p "Build and push now? (y/N) " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted. To build single platform locally:${NC}"
        echo "  PLATFORM=linux/amd64 ./minertest/build_images.sh"
        exit 0
    fi

    # Build and push CPU image (multi-platform)
    echo -e "${GREEN}[1/2] Building & pushing CPU miner (${PLATFORM})...${NC}"
    docker buildx build \
      --platform ${PLATFORM} \
      -f minertest/Dockerfile.cpu \
      -t ${REGISTRY}/quip-protocol-cpu-miner:${VERSION} \
      -t ${REGISTRY}/quip-protocol-cpu-miner:latest \
      --push \
      .

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CPU image built and pushed successfully${NC}"
    else
        echo -e "${RED}✗ CPU image build failed${NC}"
        exit 1
    fi
    echo ""

    # Build and push CUDA image (multi-platform)
    echo -e "${GREEN}[2/2] Building & pushing CUDA miner (${PLATFORM})...${NC}"
    docker buildx build \
      --platform ${PLATFORM} \
      -f minertest/Dockerfile.cuda \
      -t ${REGISTRY}/quip-protocol-cuda-miner:${VERSION} \
      -t ${REGISTRY}/quip-protocol-cuda-miner:latest \
      --push \
      .

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CUDA image built and pushed successfully${NC}"
    else
        echo -e "${RED}✗ CUDA image build failed${NC}"
        exit 1
    fi

else
    # Single platform: can use --load for local testing

    # Build CPU image
    echo -e "${GREEN}[1/2] Building CPU miner (${PLATFORM})...${NC}"
    docker buildx build \
      --platform ${PLATFORM} \
      -f minertest/Dockerfile.cpu \
      -t ${REGISTRY}/quip-protocol-cpu-miner:${VERSION} \
      -t ${REGISTRY}/quip-protocol-cpu-miner:latest \
      --load \
      .

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CPU image built successfully${NC}"
    else
        echo -e "${RED}✗ CPU image build failed${NC}"
        exit 1
    fi
    echo ""

    # Build CUDA image
    echo -e "${GREEN}[2/2] Building CUDA miner (${PLATFORM})...${NC}"
    docker buildx build \
      --platform ${PLATFORM} \
      -f minertest/Dockerfile.cuda \
      -t ${REGISTRY}/quip-protocol-cuda-miner:${VERSION} \
      -t ${REGISTRY}/quip-protocol-cuda-miner:latest \
      --load \
      .

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ CUDA image built successfully${NC}"
    else
        echo -e "${RED}✗ CUDA image build failed${NC}"
        exit 1
    fi
    echo ""

    # Push images (single platform)
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
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Deploy to Akash: see minertest/AKASH.md"
echo "2. Deploy to AWS: see minertest/AWS.md"
echo ""
echo -e "${YELLOW}Platform Options:${NC}"
echo "- Default: linux/amd64,linux/arm64 (multi-arch, requires push)"
echo "- Single platform: PLATFORM=linux/amd64 ./minertest/build_images.sh"
echo "- ARM only: PLATFORM=linux/arm64 ./minertest/build_images.sh"
echo ""

# Show image sizes (only works for single-platform local builds)
if [ "$IS_MULTI_PLATFORM" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Image Sizes:${NC}"
    echo -e "${BLUE}========================================${NC}"
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep quip-protocol || true
fi
