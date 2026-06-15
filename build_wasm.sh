#!/bin/sh
# build_wasm.sh: Script to compile the BarraCUDA compiler to WebAssembly.
# This script uses Emscripten's emcc compiler.

# Fail on any error
set -e

# Prepend homebrew bin to PATH to ensure python 3.10+ is used by emcc
export PATH="/opt/homebrew/bin:$PATH"

# Define the output directory
OUT_DIR="web"

# Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Collect all C files in the src directory recursively
# We use find to locate all files with a .c extension
SRC_FILES=$(find src -name "*.c")

# Invoke emcc to compile the C files.
# Explanation of flags:
# -O3: Aggressive optimization for performance and size.
# -I...: Include directories needed for compilation.
# -s ALLOW_MEMORY_GROWTH=1: Allows the WASM linear memory to grow at runtime if more is needed.
# -s EXPORTED_RUNTIME_METHODS='["FS","callMain"]': Exports the Emscripten Virtual File System API (FS) 
#    and the main function entry point (callMain) so we can call them from JS.
# -s INVOKE_RUN=0: Prevents the main() function from executing automatically when the WASM module loads.
# -o web/barracuda.js: Specifies the output file. Emscripten will automatically generate both 
#    barracuda.js (the loader/glue code) and barracuda.wasm (the WebAssembly binary).
emcc $SRC_FILES -O3 \
    -Isrc -Isrc/fe -Isrc/ir -Isrc/tdf -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/cpu -Isrc/runtime -Iruntime \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_RUNTIME_METHODS='["FS","callMain"]' \
    -s INVOKE_RUN=0 \
    -s EXPORT_ES6=0 \
    -o "$OUT_DIR/barracuda.js"

echo "WASM compilation successful. Output in $OUT_DIR/"
