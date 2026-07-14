# Usage

[← back to README](../README.md)

## Building

```bash
make
```

### Requirements

- A C99 compiler (gcc, clang, whatever you've got)
- A will to live (optional but recommended)
- LLVM is NOT required. Booth does its own instruction encoding like an adult.

## Command reference

The compiled binary is `kath` (after Kathleen Booth); the project is Booth. They differ on purpose, since a `booth` already lives in the Linux HA stack and you may have both on your PATH.

```bash
# Compile to AMD GPU binary (RDNA 3, default)
./kath --amdgpu-bin kernel.cu -o kernel.hsaco

# Compile for RDNA 2
./kath --amdgpu-bin --gfx1030 kernel.cu -o kernel.hsaco

# Compile for RDNA 4
./kath --amdgpu-bin --gfx1200 kernel.cu -o kernel.hsaco

# Compile to NVIDIA PTX
./kath --nvidia-ptx kernel.cu -o kernel.ptx

# Compile to Tenstorrent Metalium C++
./kath --tensix kernel.cu -o kernel_compute.cpp

# Compile to native RV32IM ELF for Tenstorrent baby cores
./kath --rv-elf kernel.cu -o kernel.elf

# Dump the TDF (Tile DataFlow) layout: regions, channels, NoC arcs
./kath --tdf kernel.cu

# HIP frontend (auto-on for .hip files, predefines __HIPCC__ and platform
# macros). Pair with any backend.
./kath --hip --amdgpu-bin kernel.hip -o kernel.hsaco
./kath --hip --nvidia-ptx kernel.hip -o kernel.ptx

# Triton frontend (parses @triton.jit Python through to BIR). Pair with
# any backend, or use --lex / --parse / --sema for inspection.
./kath --triton --amdgpu-bin kernel.py -o kernel.hsaco
./kath --triton --nvidia-ptx kernel.py -o kernel.ptx
./kath --triton --tensix       kernel.py -o kernel_compute.cpp

# CPU backend: compile a kernel to a host-runnable x86-64 object, no GPU
# needed. Link it with a host driver and run it like any other function.
./kath --cpu kernel.cu -o kernel.o
./kath --triton --cpu tests/tri_vadd.py -o vadd.o
# Calling convention: pass the kernel's own params, then one extra arg on
# the end, nthreads. The body runs once per thread_id up to nthreads, so a
# 1-D launch just hands it the element count. See examples/cpu_launch_vadd.c.

# RISC-V backend: same idea, RV64IMFD object you run under qemu-riscv64.
./kath --rv64 kernel.cu -o kernel_rv.o
# These are System V ELF objects, so link and run them on Linux (or WSL),
# not under MinGW (wrong ABI). The rv64 host has to be freestanding, see
# tests/diff for a worked runner.

# Differential testing: run the same kernel on two backends and diff the
# results. The CPU backend is the oracle, so a disagreement points at the
# other backend's codegen. Genuine cross-backend (x86 vs RISC-V), no GPU.
bash tests/diff/run_diff.sh

# Dump the IR (for debugging or curiosity)
./kath --ir kernel.cu

# Just parse and dump the AST
./kath --ast kernel.cu

# Run semantic analysis
./kath --sema kernel.cu

# Error messages in te reo Maori (or any language with a translation file)
./kath --lang lang/mi.txt --amdgpu-bin kernel.cu -o kernel.hsaco
```

## Runtime Launcher

Booth includes a minimal HSA runtime (`src/runtime/`) for dispatching compiled kernels on real AMD hardware. Zero compile-time dependency on ROCm. It loads `libhsa-runtime64.so` at runtime via `dlopen`.

```bash
# Compile the runtime and example together
gcc -std=c99 -O2 -I src/runtime \
    examples/launch_saxpy.c src/runtime/bc_runtime.c \
    -ldl -lm -o launch_saxpy

# Compile a kernel and run it
./kath --amdgpu-bin -o test.hsaco tests/canonical.cu
./launch_saxpy test.hsaco
```

Requires Linux with ROCm installed. See `examples/launch_saxpy.c` for a complete example.
