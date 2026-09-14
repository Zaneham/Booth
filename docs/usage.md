# Usage

[← back to README](../README.md)

## Building

```bash
make
```

### Requirements

- A C99 compiler (gcc, clang, whatever you've got)

Nothing else is needed to build or use Booth. Two frontends take their input
from another compiler, and you only need that compiler if you want that
frontend:

- Fortran kernels need [LFortran](https://lfortran.org/) to produce the CUDA
  source Booth then compiles. See [Fortran](#fortran).
- OCaml kernels need [OCaml](https://ocaml.org/) 5.x and dune to type-check
  them and write the `.cmt` Booth reads. See [OCaml](#ocaml).

## Command reference

The compiled binary is `kath` (after Kathleen Booth); the project is Booth. They differ on purpose, since a `booth` already lives in the Linux HA stack and you may have both on your PATH.

```bash
# Front-door verbs: build, then run on the detected device
./kath run <file>                 # compile and run on whatever device is here
./kath run --cpu <file>           # force the CPU host
./kath build <file> -o out.o      # compile only
./kath doctor                     # report devices and toolchain, then self-test

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

## Several files at once

Hand `kath` more than one `.cu` and each is compiled on its own, then linked into the single module the backend emits.

```bash
./kath --amdgpu-bin softmax.cu rope.cu common_kernels.cu -o kernels.hsaco
```

Each file gets a fresh preprocessor, lexer, parser and sema, so a macro, typedef or struct in one cannot reach the next, and two files may each keep their own `static` helper of the same name without either calling the other's. Anything not marked `static` is one symbol the whole program shares, so a `__device__` function, a `__constant__` global, or an `inline` function arriving from a header several files include, all give you one copy rather than one per file. Two files defining the same non-`inline` symbol is E126 rather than a quiet decision about which one wins.

Symbols become visible in the order you list the files, so a `__device__` function has to be defined in a file earlier on the command line than the one calling it. The other way round comes back as E105, unknown function. The other frontends read one document each, so `--triton`, `--mlir` and `--bir-in` still take a single file and will say so if given more.

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

## Fortran

Booth compiles Fortran `do concurrent` kernels by taking the CUDA source
[LFortran](https://lfortran.org/) emits for its GPU offload and compiling that
the rest of the way down. Elementwise arithmetic works today. See the
limitations at the end before planning around it.

Write a kernel the same way you would for LFortran:

```fortran
subroutine saxpy(x, y, a, n)
  integer, intent(in) :: n
  real, intent(in) :: x(n), a
  real, intent(inout) :: y(n)
  integer :: i
  do concurrent (i = 1:n)
    y(i) = a * x(i) + y(i)
  end do
end subroutine
```

Ask LFortran for the kernel source. No nvcc needed, the sidecar is written
beside the object file:

```bash
lfortran --gpu=cuda -c saxpy.f90 -o saxpy.o
# writes saxpy.o.cuda.cu
```

That file ends with a CUDA-runtime registration block, which is host glue for
`cudaLaunchKernel` and means nothing here. Cut it:

```bash
sed '/Auto-generated kernel registration/,$d' saxpy.o.cuda.cu > saxpy.cu
```

Then compile it for whatever you have:

```bash
./kath --amdgpu-bin saxpy.cu -o saxpy.hsaco   # AMD RDNA3
./kath --nvidia-ptx saxpy.cu -o saxpy.ptx     # NVIDIA
./kath --cpu        saxpy.cu -o saxpy.o       # x86-64, no GPU needed
```

### Calling it

Read the generated signature out of the `.cu` before you write a host driver,
because it will not match the Fortran argument order:

```c
extern "C" __global__ void __lfortran_gpu_kernel_0(
    float *x, float *y, float a, int n, int __loop_end_0)
```

Arrays come first, then scalars in symbol-table order rather than declaration
order, then a trailing `__loop_end_0` carrying the loop bound. The `--cpu`
backend adds one more argument on the end, `nthreads`, as it does for any
kernel.

`src/runtime/lf_gpu.c` implements LFortran's GPU offload ABI on top of the
NVIDIA runtime, so a Fortran program can launch kernels Booth produced.
`lf_gpu_hsa.c` is the AMD sibling.

### What is tested

`tests/numeric/` compiles Fortran kernels through this whole chain on every
push and checks the results against SLATEC known-good values, across x86-64,
RDNA3 and RDNA4. Six BLAS-1 routines so far: saxpy, sscal, scopy, sswap, srot
and srotm.

```bash
KATH=./kath LFORTRAN=lfortran \
python3 tests/numeric/run_numeric.py --backends cpu
```

See `tests/numeric/README.md` for adding a kernel.

### Limitations

Elementwise arithmetic only, and the reasons are all upstream of Booth:

- No reductions, so sdot, sasum and snrm2 are absent. A `do concurrent` with a
  `reduce` clause produces an empty kernel.
- Nested loops inside a `do concurrent` are dropped, which blocks Chebyshev
  recurrences and with them the SLATEC special functions
  ([lfortran#12369](https://github.com/lfortran/lfortran/issues/12369)).
- Intrinsics are emitted as `abs` and `exp` rather than `fabsf` and `expf`, so
  transcendentals do not resolve.
- Named constants used in a kernel body come through as uninitialised locals.

Tenstorrent is not supported from Fortran yet. The baby cores are RV32IM with
no FPU, so float has to go through a soft-float runtime, which is in progress.

If something breaks and you cannot tell whether it is an LFortran gap or a
Booth one, raise it here and it will get sorted out from this side.

## OCaml

Booth compiles GPU kernels written as ordinary OCaml functions. `ocamlc` does
the type checking, `kcomp` reads the `.cmt` it leaves behind and lowers it to
BIR, and from there it is the same pipeline CUDA and Triton use.

The kernel language is a module, `Kernel`, whose types are abstract. Writing an
`int` where an `i32` belongs, or reading a shared array as if it were global, is
a type error before any of Booth's code runs.

```ocaml
open Kernel

let vadd (a : f32 garray) (b : f32 garray) (c : f32 garray) (n : i32) =
  let i = block_id () * block_dim () + thread_id () in
  if i < n then set c i (get a i +. get b i)
```

### Building a kernel

Kernel sources live in `src/ocaml/` and are listed in the `kernels` library in
`src/ocaml/dune`. Nothing links against that library; it exists so `ocamlc`
type-checks each file and writes the `.cmt`.

```bash
cd src/ocaml && opam exec -- dune build
./_build/default/kcomp.exe _build/default/.kernels.objs/byte/vadd_k.cmt -o vadd.bir
cd ../.. && ./kath --bir-in --nvidia-ptx src/ocaml/vadd.bir -o vadd.ptx
```

`kcomp` refuses anything outside the subset and says where:

```
vadd_k.ml:7:15: this statement is not in the kernel subset
```

Nothing is written when it refuses.

### What the subset holds

`i32` and `f32`; `'a garray` in global memory and `'a sarray` per block; the
twelve thread-geometry builtins (`thread_id`, `block_id`, `block_dim`,
`grid_dim`, each with `_y` and `_z` forms); integer arithmetic including `/`
and `mod`; float arithmetic including `/.`; the bitwise and shift operators;
both comparison families, with `<.` and friends for floats; `to_f32` and
`to_i32`; `sqrtf`, `fabsf`, `floorf`, `ceilf`, `exp2f`, `log2f`, `sinf`,
`cosf`, `rsqf`, `rcpf`, `fminf`, `fmaxf`; `if` and `if`/`else`; counted `loop`;
`barrier`; and atomic `add`, `sub`, `and`, `or`, `xor` and `xchg`.

Accumulators are ordinary `ref` cells. They become one stack slot each and
mem2reg promotes them, so the frontend never emits a phi node.

A function marked `let[@device]` is a device function rather than a kernel. Its
body is an expression, it takes up to five arguments, and Booth inlines it for
the backends that have no calling convention.

```ocaml
let[@device] expf (x : f32) = exp2f (x *. float 1.4426950408889634)
```

`float 1.5` and `int 3` are how literals are written, since the types are
abstract. Both take a literal, not an expression.

### Limitations

- No `while` and no early exit. `loop` is counted and steps by one.
- No `f64`. Everything is single precision.
- No atomic `min` or `max`. BIR carries one opcode for each and the NVIDIA and
  AMD backends read it with opposite signedness, so it has no single meaning to
  offer until the IR can say which is meant. (this is something I am working on)
- No warp shuffles or ballots yet.
- A device function may not recurse, and neither may a kernel.

This is not OCaml running on a GPU. It is OCaml as the language you write the
kernel in. There is no GC, no closure at runtime, no exception and no
polymorphism on the device. CUDA C is not C either so it does track (gpu's are weird)..

## MLIR

`--mlir` reads MLIR text. The reader is Ondřej Čertík's, vendored under
`src/mlir/vendor`, and there is no LLVM in the path.

```bash
# Lower MLIR and compile it, same as any other frontend
./kath --mlir kernel.mlir --cpu -o kernel.o
./kath --mlir kernel.mlir --rv64 -o kernel.o

# Dump the BIR the lowering produced
./kath --mlir --ir kernel.mlir

# Reprint what was read instead of lowering it, for telling a misreading
# from a bad file
./kath --mlir --pp kernel.mlir
```

Accepted today: `func.func` with named arguments, `return`, `arith.constant`,
every `arith` integer and float binop, `arith.cmpi` and `arith.cmpf`, and the
`arith` conversions. Anything else is named on stderr and the whole lowering
fails, because an op skipped quietly is a kernel that compiles and computes
something different.

Nothing in MLIR marks a function as a kernel yet, so everything lowers as a
device function. `--cpu` and `--rv64` give you real code; the GPU backends will
take the file and then report zero kernels, which is an honest answer until
the `gpu` dialect arrives.
