#!/bin/sh
# test_wasm_build.sh: Test script to verify WASM build succeeds and produces outputs.

# Fail on any error
set -e

echo "Running WASM build..."
./build_wasm.sh

if [ -f "web/barracuda.js" ] && [ -f "web/barracuda.wasm" ]; then
    echo "WASM artifacts generated successfully."
    exit 0
else
    echo "Error: WASM artifacts missing."
    exit 1
fi
