# Roadmap

[← back to README](../README.md)

## Near Term: Hardening

Fix the known gaps: integer literal suffixes, `const`, parameter reassignment. These are all small parser/lowerer changes. The goal is to compile real-world `.cu` files without modifications.

## Medium Term: Optimisation

The generated code works but isn't winning any benchmarks. Done so far: instruction scheduling, constant folding, dead code elimination, divergence-aware SSA register allocation. Priorities:

- Loop-invariant code motion
- Occupancy tuning based on register pressure

## Long Term: More Architectures

The IR (BIR) is target-independent. The backend is cleanly separated. Adding a new target means writing a new `isel` + `emit` pair.

- **NVIDIA PTX** - Done. Compiles CUDA to PTX, validated on RTX 4060 Ti. `--nvidia-ptx`
- **Tenstorrent Tensix Metalium** - Done. Compiles CUDA to TT-Metalium C++ for Blackhole. `--tensix`
- **Tenstorrent RV32IM** - Done. Native RV32IM ELF for Wormhole baby cores with TDF layer for L1/NoC orchestration. `--rv-elf`
- **Apple Metal** - Stub backend exists, hardware validation pending. `--metal`
- **Intel Arc** - Xe architecture, SPIR-V emit stub. Would give Booth coverage across all four major GPU vendors. `--intel-spirv`
- **CPU (x86-64)** - Done via `--cpu`. CUDA and Triton kernels compile to a host object and run with no GPU, Triton matmul (`tl.dot` + K-loop) included. The SIMT-to-loop trick wraps the kernel body in a thread loop. Stack-everything codegen for now, no register allocator (correct first, fast later). ARM64 is still on the radar.
- **RISC-V Vector Extension** - For when GPUs are too mainstream and you want to run CUDA on a softcore.
