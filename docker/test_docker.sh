#!/bin/bash
# Test script for Docker builds and basic functionality
# This script tests building and running the Docker images

set -e  # Exit on error

# Cleanup function
cleanup() {
    if [ -n "$BUILD_LOG_TDMPC" ] && [ -f "$BUILD_LOG_TDMPC" ]; then
        rm -f "$BUILD_LOG_TDMPC"
    fi
    if [ -n "$BUILD_LOG_DREAMER" ] && [ -f "$BUILD_LOG_DREAMER" ]; then
        rm -f "$BUILD_LOG_DREAMER"
    fi
}

# Set up trap to cleanup on exit
trap cleanup EXIT

echo "========================================="
echo "Docker Build and Test Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[ℹ]${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi
print_status "Docker is available"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_info "docker-compose not found, will use 'docker compose' instead"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Build images
echo ""
echo "Building Docker images..."
echo "========================================="

# Build TDMPC2 image (simpler, no JAX)
print_info "Building TDMPC2 image..."
BUILD_LOG_TDMPC=$(mktemp)
if docker build -f docker/Dockerfile.tdmpc -t hwm-tdmpc:test . > "$BUILD_LOG_TDMPC" 2>&1; then
    print_status "TDMPC2 image built successfully"
else
    print_error "Failed to build TDMPC2 image. Check $BUILD_LOG_TDMPC for details"
    tail -50 "$BUILD_LOG_TDMPC"
    # Don't remove log on failure so user can inspect it
    trap - EXIT  # Disable cleanup trap
    exit 1
fi

# Build DreamerV3 image
print_info "Building DreamerV3 image..."
BUILD_LOG_DREAMER=$(mktemp)
if docker build -f docker/Dockerfile.dreamer -t hwm-dreamer:test . > "$BUILD_LOG_DREAMER" 2>&1; then
    print_status "DreamerV3 image built successfully"
else
    print_error "Failed to build DreamerV3 image. Check $BUILD_LOG_DREAMER for details"
    tail -50 "$BUILD_LOG_DREAMER"
    # Don't remove log on failure so user can inspect it
    trap - EXIT  # Disable cleanup trap
    exit 1
fi

# Test running containers
echo ""
echo "Testing Docker containers..."
echo "========================================="

# Test TDMPC2 container
print_info "Testing TDMPC2 container..."
if docker run --rm hwm-tdmpc:test python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
    print_status "TDMPC2 container runs successfully"
else
    print_error "TDMPC2 container failed to run"
    exit 1
fi

# Test DreamerV3 container
print_info "Testing DreamerV3 container..."
if docker run --rm hwm-dreamer:test python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"; then
    print_status "DreamerV3 container runs successfully"
else
    print_error "DreamerV3 container failed to run"
    exit 1
fi

echo ""
echo "========================================="
print_status "All Docker images built and tested successfully!"
echo "========================================="
echo ""
echo "You can now run:"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace hwm-tdmpc:test"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace hwm-dreamer:test"
echo ""
echo "Or use docker-compose:"
echo "  $COMPOSE_CMD run tdmpc"
echo "  $COMPOSE_CMD run dreamer"
echo ""
