# Feature Status

[← back to README](../README.md)

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
- SYSPRINT: class-tagged structured kernel output, pattern-routed sinks on the host. See [mainframe.md](mainframe.md) for the kernel/host workflow.

## What Doesn't Work (Yet)

Being honest about limitations is important. Here's what's missing:

- Parameter reassignment in `__device__` functions (use local variables)
- Textures and surfaces
- Dynamic parallelism (device-side kernel launch)
- Multiple translation units
- Host code generation (only device code is compiled)
- Rank-2 matrix codegen on the GPU backends (MFMA on AMD, mma.sync on NVIDIA). Triton `tl.dot` already runs on the CPU backend, materialised and unrolled with a K-loop, but the GPU matrix-instruction path is a separate job. On GPU targets rank-2 tiles still refuse cleanly with E099, no silent wrong code.
- CPU backend is correct-first: stack-everything codegen, no register allocator yet, single block per call, and `tl.load` masks aren't honoured (so keep the launch's nthreads equal to the element count). It runs; it isn't fast.
- Soft-float for the Tenstorrent native RV32IM path. The runtime exists and validates against host FPU; wiring it into `--rv-elf` is a sitting's work away.

None of these are architectural blockers. They're all "haven't got round to it yet" items.
