#!/usr/bin/env bash
# setup_backend.sh — Clone, build, and pin llama.cpp with CUDA support
#
# Usage: ./scripts/setup_backend.sh [COMMIT_HASH]
#
# The pinned commit ensures reproducible builds. Update the default
# hash when upgrading llama.cpp.

set -euo pipefail

REPO_URL="https://github.com/ggml-org/llama.cpp.git"
# Pin to a known stable commit — update this when upgrading
DEFAULT_COMMIT="master"
COMMIT="${1:-$DEFAULT_COMMIT}"
BUILD_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/llama.cpp"

echo "=== ADHIE Backend Setup ==="
echo "Repository:  $REPO_URL"
echo "Commit:      $COMMIT"
echo "Build dir:   $BUILD_DIR"
echo ""

# Clone if not present
if [ ! -d "$BUILD_DIR" ]; then
    echo ">>> Cloning llama.cpp..."
    git clone "$REPO_URL" "$BUILD_DIR"
else
    echo ">>> llama.cpp already cloned at $BUILD_DIR"
fi

cd "$BUILD_DIR"

# Checkout pinned commit
echo ">>> Checking out $COMMIT..."
git fetch origin
git checkout "$COMMIT"

ACTUAL_HASH=$(git rev-parse HEAD)
echo ">>> Pinned at commit: $ACTUAL_HASH"

# Build with CUDA support
echo ">>> Building with CUDA support..."
mkdir -p build && cd build

cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

cmake --build . --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo ""
echo "=== Build complete ==="
echo "Server binary: $BUILD_DIR/build/bin/llama-server"
echo "CLI binary:    $BUILD_DIR/build/bin/llama-cli"
echo ""

# Verify the build
if [ -f "$BUILD_DIR/build/bin/llama-server" ]; then
    echo ">>> Verifying server..."
    "$BUILD_DIR/build/bin/llama-server" --help 2>&1 | head -5
    echo "..."
    echo "✓ llama-server built successfully"
else
    echo "✗ llama-server not found — build may have failed"
    exit 1
fi

# Save build info
cat > "$BUILD_DIR/build/BUILD_INFO.txt" << EOF
commit: $ACTUAL_HASH
date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
cuda: $(nvcc --version 2>/dev/null | grep "release" || echo "unknown")
cmake: $(cmake --version | head -1)
os: $(uname -s -r)
EOF

echo ""
echo "Build info saved to $BUILD_DIR/build/BUILD_INFO.txt"
echo "Done."

