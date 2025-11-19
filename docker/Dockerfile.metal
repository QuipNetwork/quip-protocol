# Dockerfile for Metal-based mining comparison
# macOS-compatible image for arm64 architecture (Apple Silicon)
# NOTE: This Dockerfile is provided for reference, but running directly on macOS
# EC2 instances without Docker is recommended due to Metal framework requirements

FROM python:3.13-slim

LABEL maintainer="Quip Protocol"
LABEL description="Metal GPU miner for quantum blockchain mining rate comparison on Apple Silicon"

# Note: This image is designed for macOS arm64 systems
# Xcode Command Line Tools must be installed on the host system
# Docker on macOS may have limitations accessing Metal framework

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./
COPY blockchain_base.py ./
COPY GPU/ ./GPU/
COPY tools/ ./tools/
COPY dwave_topologies/ ./dwave_topologies/

# Install Python dependencies including PyObjC frameworks for Metal
# Note: These will only work properly on macOS hosts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
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

# Create output directory
RUN mkdir -p /output

# Set default environment variables (can be overridden at runtime)
ENV MINING_DURATION=30m
ENV DIFFICULTY_ENERGY=-14900
ENV MIN_DIVERSITY=0.1
ENV MIN_SOLUTIONS=5
ENV TOPOLOGY_FILE=dwave_topologies/topologies/advantage2_system1_7.json.gz
ENV MINER_TYPE=metal

# Copy entrypoint script
COPY docker/entrypoint-metal.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Volume for output files
VOLUME ["/output"]

# Run the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# IMPORTANT NOTES:
# 1. Metal framework access from Docker containers is limited/unsupported
# 2. For AWS EC2 Mac instances, running directly on the host is recommended
# 3. See aws/user_data_metal.sh for direct deployment without Docker
# 4. This Dockerfile is provided for consistency with CPU/CUDA images
