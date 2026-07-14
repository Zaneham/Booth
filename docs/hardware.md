# Validated on Hardware

[← back to README](../README.md)

Booth-compiled kernels have been tested and produce correct results on real silicon:

- **AMD MI300X (CDNA3, GFX942)**: 8/8 test kernels passing. Monte Carlo neutron transport producing correct physics (k_eff = 0.995, matching reference).
- **AMD RDNA3 (GFX1100)**: Full test suite passing via RDNA3 emulator CI.
- **NVIDIA RTX 4060 Ti**: PTX backend, loaded via CUDA Driver API, JIT-compiled by NVIDIA driver. Monte Carlo neutron transport benchmark produces correct results with 3.8x speedup over single-thread CPU. No NVCC involved anywhere in the pipeline.
- **Tenstorrent Blackhole**: Compiles to valid Metalium C++. Hardware validation pending dev kit access.

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
$ ./kath --amdgpu-bin vector_add.cu -o vector_add.hsaco
wrote vector_add.hsaco (528 bytes code, 1 kernels)
```

No LLVM required :-)

## Test Suite

This is the kernel corpus that gets compiled end to end: 14 source files, 35+ kernels, ~1,700 BIR instructions, ~27,000 bytes of machine code. (Separately, the C test harness that exercises the compiler internals runs 274 cases across the frontends, IR, backends and runtime, see [CONTRIBUTING](../CONTRIBUTING.md).)

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
