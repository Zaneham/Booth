# BarraCUDA

BarraCUDA is a multi architecture, multi language compiler with the intended goal of allowing for cross platform development on GPU's and CPU's. This is also, from what I've seen, a series of 'firsts'. BarraCUDA is the first attempt to apply mainframe operational discipline to GPU computing and it's the first attempt to allow users to run GPU-based-languages on CPU's on the same compiler (for some languages like Triton, this is a first without LLVM). You can see the mainframe influence throughout in the abend dumps, MVS Snap, SYSPRINT and Tile Dataflow for dataflow based gpu's which is based of CICS transactions. If you've ever read about or worked with z/OS you'll feel right at home, if you're asking "whats z/OS?" give yourself an uppercut (jokes!). 

This project originally started because I wanted to run CUDA on my laptop and I found ROCm and all that very confusing. So, nau mai haere mai (welcome) to me looking at Nvidias walled garden and going "oh it couldn't be that hard" and realising that yes, actually it is. But I did it anyway. 

See [CHANGELOG.txt](CHANGELOG.txt) for recent updates.

**update**: HIP is now being supported.

**update 2**: Triton is now being supported as well.

**update 3**: Native RV32IM codegen for Tenstorrent Wormhole baby cores via `--rv-elf`, plus a TDF (Tile DataFlow) IR layer above BIR that models L1 placement and NoC arcs as first-class compiler concepts.

**update 4**: there's a CPU backend now (`--cpu`). CUDA and Triton kernels compile to a normal x86 object and just run, no GPU needed. Triton matmul goes the whole way through, `tl.dot` and a K-loop and the lot, so you can mess about with Triton on a laptop.

## What It Does

Takes CUDA C, HIP, or Triton source (the same files you'd hand to `nvcc`, `ROCm`, or Triton's JIT) and turns them into AMD RDNA 2/3/4 binaries, NVIDIA PTX, Tenstorrent Metalium C++ or native RV32IM, or just plain x86-64 you can run on a laptop with no GPU in it.

That last one still surprises me a bit. You can write a Triton kernel, matmul and all, and run it on a machine that's never seen a GPU. I haven't come across anyone else doing Triton like this (from scratch, no LLVM, straight to native), but I'd happily be proven wrong, so give me a yell if you've seen it somewhere.

## Building

```bash
# It's C99. It builds with gcc. There are no dependencies.
make

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

# Compile to native RV32IM ELF for Tenstorrent baby cores
./barracuda --rv-elf kernel.cu -o kernel.elf

# Dump the TDF (Tile DataFlow) layout: regions, channels, NoC arcs
./barracuda --tdf kernel.cu

# HIP frontend (auto-on for .hip files, predefines __HIPCC__ and platform
# macros). Pair with any backend.
./barracuda --hip --amdgpu-bin kernel.hip -o kernel.hsaco
./barracuda --hip --nvidia-ptx kernel.hip -o kernel.ptx

# Triton frontend (parses @triton.jit Python through to BIR). Pair with
# any backend, or use --lex / --parse / --sema for inspection.
./barracuda --triton --amdgpu-bin kernel.py -o kernel.hsaco
./barracuda --triton --nvidia-ptx kernel.py -o kernel.ptx
./barracuda --triton --tensix       kernel.py -o kernel_compute.cpp

# CPU backend: compile a kernel to a host-runnable x86-64 object, no GPU
# needed. Link it with a host driver and run it like any other function.
./barracuda --cpu kernel.cu -o kernel.o
./barracuda --triton --cpu tests/tri_vadd.py -o vadd.o
# Calling convention: pass the kernel's own params, then one extra arg on
# the end, nthreads. The body runs once per thread_id up to nthreads, so a
# 1-D launch just hands it the element count. See examples/cpu_launch_vadd.c.

# RISC-V backend: same idea, RV64IMFD object you run under qemu-riscv64.
./barracuda --rv64 kernel.cu -o kernel_rv.o
# These are System V ELF objects, so link and run them on Linux (or WSL),
# not under MinGW (wrong ABI). The rv64 host has to be freestanding, see
# tests/diff for a worked runner.

# Differential testing: run the same kernel on two backends and diff the
# results. The CPU backend is the oracle, so a disagreement points at the
# other backend's codegen. Genuine cross-backend (x86 vs RISC-V), no GPU.
bash tests/diff/run_diff.sh

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

BarraCUDA includes a minimal HSA runtime (`src/runtime/`) for dispatching compiled kernels on real AMD hardware. Zero compile-time dependency on ROCm. It loads `libhsa-runtime64.so` at runtime via `dlopen`.

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

## Mainframe Curios

This is the bit where I admit I read a pile of z/OS manuals and got a little obsessed. The mainframe folks sorted out crash diagnostics and structured job output decades ago, and honestly a lot of it is nicer than what I had while squinting at broken GPU kernels. So I borrowed the ideas. I'm sure I'm doing them more clumsily than the people who invented them, and I'm still learning this stuff, but they're not a gimmick to me, they're the things that actually helped me find bugs.

### ABEND dumps

When a kernel faults you get a real dump, not a shrug. `src/runtime/bc_abend.*` gives GPU faults proper IBM-style completion codes (G0Cx, the GPU cousins of S0Cx), correlates the faulting address against tracked allocations, and prints a dispatch snapshot. It's wired into the HSA runtime and fires automatically off the system event callback, so a memory aperture violation tells you which buffer and which dispatch went wrong instead of just dying quietly. Live on the AMD/HSA path.

### SNAP (`--snap`)

A parameter dump, basically. The mainframe crowd had this in the 70s and I kept wishing for it while debugging. With `--snap` the AMD backend writes each kernel parameter's register value into a host-visible buffer on entry, so when things go sideways you can read the evidence instead of staring at disassembly like it owes you money. AMD only for now.

### SYSPRINT

Structured kernel output, routed by class, the way every mainframe job has emitted records to named SYSPRINT classes since 1965. Kernels emit class-tagged records into a host-visible buffer, the host registers sinks by pattern (`STEP1.*`, `*.ERROR`, `*`), and `bc_sp_drain` walks the buffer once the kernel finishes.

```c
/* kernel.cu */
#include "sysprint_device.h"
#include "sysprint_classes.h"  /* your CLS_ constants */

__global__ void k(bc_sp_buf_t *sp, ...) {
    if (threadIdx.x == 0) {
        BC_SYSPRINT(sp, CLS_RESULT, "kernel done", 11);
    }
}
```

```c
/* host.c */
bc_sp_intern("DEMO.RESULT");                       /* gets id 1 */
bc_sp_register_sink("DEMO.RESULT", my_sink, NULL);
/* allocate buffer, dispatch kernel passing buffer pointer */
bc_sp_drain(&buf);                                 /* sinks fire */
```

See `examples/sysprint_kernel.cu` + `examples/launch_sysprint.c` for a full end-to-end demo. Works on the NVIDIA PTX and Tensix backends; the AMD path currently trips a regalloc bug on the byte-copy loop ([open issue](https://github.com/Zaneham/BarraCUDA/issues)), which gets its own follow-up.

### TDF (Tile DataFlow)

For the dataflow GPUs (Tenstorrent), the layer above BIR is modelled on CICS transactions: regions, channels, and NoC arcs as first-class compiler concepts, with L1 placement and a fission pass for multi-core kernels. Dump it with `--tdf`.

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
- Triton tile shape inference: rank-0/1/2 shape annotation on every expression, constexpr default propagation (`BLOCK: tl.constexpr = 256` resolves to `vec[256]`), numpy-style broadcasting, `[:, None]` / `[None, :]` reshape patterns
- Triton matmul on the CPU: `tl.dot` lowers and runs via `--cpu`, with a runtime K-loop so the contraction can be any size. Rank-2 tiles materialise and unroll
- x86-64 CPU backend (`--cpu`): CUDA and Triton kernels compile to a host object and run with no GPU. SIMT becomes a thread loop. Stack-everything codegen, no register allocator yet
- TDF (Tile DataFlow) IR layer above BIR: regions / channels / NoC arcs as first-class compiler concepts, L1 placement, fission pass for multi-core kernels
- SYSPRINT: class-tagged structured kernel output, pattern-routed sinks on the host. See the Mainframe Curios section above for the kernel/host workflow.

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

- **AMD MI300X (CDNA3, GFX942)**: 8/8 test kernels passing. Monte Carlo neutron transport producing correct physics (k_eff = 0.995, matching reference).
- **AMD RDNA3 (GFX1100)**: Full test suite passing via RDNA3 emulator CI.
- **NVIDIA RTX 4060 Ti**: PTX backend, loaded via CUDA Driver API, JIT-compiled by NVIDIA driver. Monte Carlo neutron transport benchmark produces correct results with 3.8x speedup over single-thread CPU. No NVCC involved anywhere in the pipeline.
- **Tenstorrent Blackhole**: Compiles to valid Metalium C++. Hardware validation pending dev kit access.

## What Doesn't Work (Yet)

Being honest about limitations is important. Here's what's missing:

- Parameter reassignment in `__device__` functions (use local variables)
- Textures and surfaces
- Dynamic parallelism (device-side kernel launch)
- Multiple translation units
- Host code generation (only device code is compiled)
- Rank-2 matrix codegen on the GPU backends (MFMA on AMD, mma.sync on NVIDIA). Triton `tl.dot` already runs on the CPU backend, materialised and unrolled with a K-loop, but the GPU matrix-instruction path is a separate job. On GPU targets rank-2 tiles still refuse cleanly with E099, no silent wrong code.
- CPU backend is correct-first: stack-everything codegen, no register allocator yet, single block per call, and `tl.load` masks aren't honoured (so keep the launch's nthreads equal to the element count). It runs; it isn't fast.
- Soft-float for the Tenstorrent native RV32IM path. The runtime exists and validates against host FPU; wiring it into `--rv-elf` is a sitting away.

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
- **Tenstorrent Tensix Metalium** - Done. Compiles CUDA to TT-Metalium C++ for Blackhole. `--tensix`
- **Tenstorrent RV32IM** - Done. Native RV32IM ELF for Wormhole baby cores with TDF layer for L1/NoC orchestration. `--rv-elf`
- **Apple Metal** - Stub backend exists, hardware validation pending. `--metal`
- **Intel Arc** - Xe architecture, SPIR-V emit stub. Would give BarraCUDA coverage across all four major GPU vendors. `--intel-spirv`
- **CPU (x86-64)** - Done via `--cpu`. CUDA and Triton kernels compile to a host object and run with no GPU, Triton matmul (`tl.dot` + K-loop) included. The SIMT-to-loop trick wraps the kernel body in a thread loop. Stack-everything codegen for now, no register allocator (correct first, fast later). ARM64 is still on the radar.
- **RISC-V Vector Extension** - For when GPUs are too mainstream and you want to run CUDA on a softcore.


## Contributing

**Issues and PRs in any language are welcome**, just include an English translation alongside. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide on style, naming, and where to help.

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
- **The academic community**: Cooper, Harvey & Kennedy for dominators; Braun & Hack for SSA spilling; Sampaio, Souza, Collange & Pereira for divergence analysis. I'm just a hobbyist who reads papers and writes C. The actual hard work was done by the researchers.
- **Steven Muchnick** for *Advanced Compiler Design and Implementation*. If this compiler does anything right, that book is why.
- **Low Level** for the Zero to Hero C course and the YouTube channel. That's where I learnt C.
- **Abe Kornelis** for being an amazing teacher. His work on the [z390 Portable Mainframe Assembler](https://github.com/z390development/z390) project is well worth your time.
- To the people who've sent messages of kindness and critique, thank you from a forever student and a happy hobbyist.
- My Granny, Grandad, Nana and Baka. Love you x

*He aha te mea nui o te ao. He tāngata, he tāngata, he tāngata.*

What is the most important thing in the world? It is people, it is people, it is people.

---

