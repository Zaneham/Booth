# BarraCUDA

An open-source CUDA C++ compiler written from scratch in C99 that takes `.cu` files and compiles them to AMD GPU machine code, NVIDIA PTX, and Tenstorrent Tensix C++, with more architectures planned. No LLVM, no dependencies, and no permission asked.

This is what happens when you look at NVIDIA's walled garden and think "how hard can it be?" The answer is: quite hard, actually, but I did it anyway.

See [CHANGELOG.txt](CHANGELOG.txt) for recent updates.

**update** HIP is now being supported.

## What It Does

Takes CUDA C source code, the same `.cu` files you'd feed to `nvcc`, and compiles them to AMD RDNA 2/3/4 binaries, NVIDIA PTX, or Tenstorrent Tensix Metalium C++.

## Building

```bash
# It's C99. It builds with gcc. There are no dependencies.
make

# That's it. No cmake. No autoconf. No 47-step build process.
# If this doesn't work, your gcc is broken, not the Makefile.
```

### Requirements

- A C99 compiler (gcc, clang, whatever you've got)
- A will to live (optional but recommended)
- LLVM is NOT required. BarraCUDA does its own instruction encoding like an adult.

## Usage

```bash
# Compile to AMD GPU binary (RDNA 3, default)
./barracuda --amdgpu-bin kernel.cu -o kernel.hsaco

# Compile for RDNA 2
./barracuda --amdgpu-bin --gfx1030 kernel.cu -o kernel.hsaco

# Compile for RDNA 4
./barracuda --amdgpu-bin --gfx1200 kernel.cu -o kernel.hsaco

# Compile to NVIDIA PTX
./barracuda --nvidia-ptx kernel.cu -o kernel.ptx

# Compile to Tenstorrent Metalium C++
./barracuda --tensix kernel.cu -o kernel_compute.cpp

# Dump the IR (for debugging or curiosity)
./barracuda --ir kernel.cu

# Just parse and dump the AST
./barracuda --ast kernel.cu

# Run semantic analysis
./barracuda --sema kernel.cu

# Error messages in te reo Maori (or any language with a translation file)
./barracuda --lang lang/mi.txt --amdgpu-bin kernel.cu -o kernel.hsaco
```

## Runtime Launcher

BarraCUDA includes a minimal HSA runtime (`src/runtime/`) for dispatching compiled kernels on real AMD hardware. Zero compile-time dependency on ROCm — loads `libhsa-runtime64.so` at runtime via `dlopen`.

```bash
# Compile the runtime and example together
gcc -std=c99 -O2 -I src/runtime \
    examples/launch_saxpy.c src/runtime/bc_runtime.c \
    -ldl -lm -o launch_saxpy

# Compile a kernel and run it
./barracuda --amdgpu-bin -o test.hsaco tests/canonical.cu
./launch_saxpy test.hsaco
```

Requires Linux with ROCm installed. See `examples/launch_saxpy.c` for a complete example.

## What Works

 The following CUDA features compile to working GFX9/GFX10/GFX11/GFX12 machine code, NVIDIA PTX, and Tensix Metalium C++:

### Core Language
- `__global__`, `__device__`, `__host__` function qualifiers
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` builtins
- Structs (named + anonymous inline), enums, typedefs, namespaces
- Pointers, arrays, pointer arithmetic
- All C control flow: `if`/`else`, `for`, `while`, `do-while`, `switch`/`case`, `goto`/`label`
- Short-circuit `&&` and `||`
- Ternary operator
- Templates (basic instantiation)
- Multiple return paths, `continue`, `break`

### CUDA Features
- `__shared__` memory (allocated from LDS, properly tracked)
- `__syncthreads()` → `s_barrier`
- Atomic operations: `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`
- Warp intrinsics: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`
- Warp votes: `__ballot_sync`, `__any_sync`, `__all_sync`
- Vector types: `float2`, `float3`, `float4`, `int2`, `int3`, `int4` with `.x`/`.y`/`.z`/`.w` access
- Half precision: `__half`, `__float2half()`, `__half2float()`, `__nv_bfloat16`
- `__launch_bounds__` (parsed, propagated, enforces VGPR caps)
- Cooperative groups: `cooperative_groups::this_thread_block()` with `.sync()`, `.thread_rank()`, `.size()`
- Operator overloading
- Math builtins: `sqrtf`, `rsqrtf`, `expf`, `exp2f`, `logf`, `log2f`, `log10f`, `sinf`, `cosf`, `tanf`, `tanhf`, `powf`, `fabsf`, `floorf`, `ceilf`, `truncf`, `roundf`, `rintf`, `fmaxf`, `fminf`, `fmodf`, `copysignf`
- `__constant__` memory, `__device__` globals

### Compiler Features
- Full C preprocessor: `#include`, `#define`/`#undef`, function-like macros, `#ifdef`/`#ifndef`/`#if`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`, `-I`/`-D` flags
- Error recovery (reports multiple errors without hanging)
- Multilingual error messages (`--lang <file>`) with language-neutral E-codes
- Source location tracking in IR dumps
- Struct pass-by-value

## Example

```cuda
__global__ void vector_add(float *c, float *a, float *b, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
```

```
$ ./barracuda --amdgpu-bin vector_add.cu -o vector_add.hsaco
wrote vector_add.hsaco (528 bytes code, 1 kernels)
```

No LLVM required :-) 


## Validated on Hardware

BarraCUDA-compiled kernels have been tested and produce correct results on real silicon:

- **AMD MI300X (CDNA3, GFX942)** — 8/8 test kernels passing. Monte Carlo neutron transport producing correct physics (k_eff = 0.995, matching reference).
- **AMD RDNA3 (GFX1100)** — Full test suite passing via RDNA3 emulator CI.
- **NVIDIA RTX 4060 Ti** — PTX backend, loaded via CUDA Driver API, JIT-compiled by NVIDIA driver. Monte Carlo neutron transport benchmark produces correct results with 3.8x speedup over single-thread CPU. No NVCC involved anywhere in the pipeline.
- **Tenstorrent Blackhole** — Compiles to valid Metalium C++. Hardware validation pending dev kit access.

## What Doesn't Work (Yet)

Being honest about limitations is important. Here's what's missing:

- Parameter reassignment in `__device__` functions (use local variables)
- Textures and surfaces
- Dynamic parallelism (device-side kernel launch)
- Multiple translation units
- Host code generation (only device code is compiled)

None of these are architectural blockers. They're all "haven't got round to it yet" items.

## Test Suite

14 test files, 35+ kernels, ~1,700 BIR instructions, ~27,000 bytes of machine code:

- `vector_add.cu` - The "hello world" of GPU computing
- `cuda_features.cu` - Atomics, warp ops, barriers, gotos, switch, short-circuit
- `test_tier12.cu` - Vectors, shared memory, operator overloading
- `notgpt.cu` - AI-generated CUDA with extremely sarcastic comments (tiled SGEMM, reductions, histograms, prefix scan, stencils, half precision, cooperative groups, and the "kitchen sink" kernel)
- `stress.cu` - N-body simulation, nested control flow, bit manipulation, struct pass-by-value, chained function calls
- `canonical.cu` - Canonical patterns from NVIDIA samples adapted for the parser
- `test_errors.cu` - Deliberate syntax errors to verify error recovery
- `test_launch_bounds.cu` - `__launch_bounds__` parsing and VGPR cap enforcement
- `test_coop_groups.cu` - Cooperative groups lowering
- `mymathhomework.cu` - Trig identities, exponential growth, Newton-Raphson, log laws, hyperbolic functions, floor/ceil/round, power rule, clamping
- Plus preprocessor tests, template tests, unsigned integer tests

## Roadmap

### Near Term: Hardening

Fix the known gaps: integer literal suffixes, `const`, parameter reassignment. These are all small parser/lowerer changes. The goal is to compile real-world `.cu` files without modifications.

### Medium Term: Optimisation

The generated code works but isn't winning any benchmarks. Done so far: instruction scheduling, constant folding, dead code elimination, divergence-aware SSA register allocation. Priorities:

- Loop-invariant code motion
- Occupancy tuning based on register pressure

### Long Term: More Architectures

The IR (BIR) is target-independent. The backend is cleanly separated. Adding a new target means writing a new `isel` + `emit` pair.

- **NVIDIA PTX** - Done. Compiles CUDA to PTX, validated on RTX 4060 Ti. `--nvidia-ptx`
- **Tenstorrent Tensix** - Done. Compiles CUDA to TT-Metalium C++ for Blackhole. `--tensix`
- **Intel Arc** - Xe architecture. Would give BarraCUDA coverage across all four major GPU vendors.
- **RISC-V Vector Extension** - For when GPUs are too mainstream and you want to run CUDA on a softcore.


## Contributing

**Issues and PRs in any language are welcome** — just include an English translation alongside. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide on style, naming, and where to help.

The HLASM-style short identifiers (`ra_gc`, `mk_hash`, `enc_vop3`) are culturally neutral by accident, there's nothing English about a 5-character label. If you've found a bug or have an idea, write it up in whatever language you think in.

## Contact

Found a bug? Want to discuss the finer points of AMDGPU instruction encoding? Need someone to commiserate with about the state of GPU computing?

**zanehambly@gmail.com**

Open an issue if there's anything you want to discuss. Or don't. I'm not your mum.

Based in New Zealand, where it's already tomorrow and the GPUs are just as confused as everywhere else.

## License

Apache 2.0. Do whatever you want. If this compiler somehow ends up in production, I'd love to hear about it, mostly so I can update my LinkedIn with something more interesting than wrote a CUDA compiler for fun.

## Acknowledgements

- **Fernando Magno Quintão Pereira** and the **Compilers Lab at UFMG** (Universidade Federal de Minas Gerais). Fernando reached out after seeing the project, pointed me to the divergence analysis papers, and offered guidance. The SSA register allocator exists because of that conversation.
- **The academic community** — Cooper, Harvey & Kennedy for dominators; Braun & Hack for SSA spilling; Sampaio, Souza, Collange & Pereira for divergence analysis. I'm just a hobbyist who reads papers and writes C. The actual hard work was done by the researchers.
- **Steven Muchnick** for *Advanced Compiler Design and Implementation*. If this compiler does anything right, that book is why.
- **Low Level** for the Zero to Hero C course and the YouTube channel. That's where I learnt C.
- **Abe Kornelis** for being an amazing teacher. His work on the [z390 Portable Mainframe Assembler](https://github.com/z390development/z390) project is well worth your time.
- To the people who've sent messages of kindness and critique, thank you from a forever student and a happy hobbyist.
- My Granny, Grandad, Nana and Baka. Love you x

*He aha te mea nui o te ao. He tāngata, he tāngata, he tāngata.*

What is the most important thing in the world? It is people, it is people, it is people.

---

