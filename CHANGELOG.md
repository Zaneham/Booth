Booth — Changelog
=================

## Unreleased

### Runtime

- `kath run`, `kath build` and `kath doctor`, so one command builds a source
  and runs it on whatever device is there (Zane Hambly, 2026-09-14)

### Backends

- `--nvidia-cubin` writes a cubin the card will load, with no NVCC anywhere in
  the chain. All 67 of ggml-cuda's files now reach the IR
  (Zane Hambly, 2026-09-09)

### Frontend

- `constexpr` and `const` objects fold at every use, and anything the folder
  cannot evaluate refuses with E128 (Zane Hambly, 2026-09-03)

- class templates, specialisations, default template arguments and
  `enum class` parse, so 47 of ggml-cuda's 67 files reach the lowerer
  (Zane Hambly, 2026-09-04)

- the lowerer's tables no longer run out of room on a large translation unit,
  taking ggml-cuda's lowering errors from 774 to 210 (Zane Hambly, 2026-09-04)


## Booth 0.5.3

### Runtime

- #169: the runtime is split by where it runs, the `BC_ERR_*` codes no
  longer collide, and the examples and NVIDIA harness are built
  (Zane Hambly, 2026-08-23)


### Frontend

- variadic template parameter packs, several `.cu` files as separate
  translation units, `mma.sync` and `mfma` lowering, and an i1 that no
  longer strides by zero (Zane Hambly, 2026-09-03)

- `(a) + (b)` adds again; the parser treated any parenthesised identifier as a
  type name without asking whether it named one, so the left operand vanished
  into a cast with no diagnostic (Zane Hambly, 2026-09-03)

- the cast test is now the type name registry, so the registry has to be
  complete. Template type parameters, `using X = T` aliases and the type names
  sema resolves without a typedef (`size_t`, `uint32_t`, `float4` and the rest)
  all reach it. A compound literal through a typedef, `(pair){1, 2}`, parses
  for the first time, and `sizeof(name)` where the name is a type reads as a
  type rather than an expression (Zane Hambly, 2026-09-03)

- llama.cpp's ggml-cuda preprocesses, all 67 files; `#pragma once` is
  honoured, variadic and multi-line macro invocations expand, and an
  expansion too big for the output buffer is E053 rather than an
  unterminated buffer the lexer reads past (Zane Hambly, 2026-09-03)

- `kath --mlir` reads MLIR text, no LLVM in the path. Čertík's pure-C
  reader vendored under `src/mlir/vendor` (mlir 826b69c9, corec a160199d),
  reached only through `src/mlir/mlir_fe.c` (Zane Hambly, 2026-08-11)

- `src/mlir/lower.c` walks the parsed module into BIR: `func.func`, `return`,
  `arith.constant` and every arith binop, compare and conversion the reader
  classifies. From there it is the pipeline CUDA and Triton already use, and
  MLIR reaches all four backends. `--mlir --pp` reprints instead
  (Zane Hambly, 2026-08-11)

- an op outside the subset stops the lowering and names itself. Skipping it
  would leave a function that compiles and computes something else
  (Zane Hambly, 2026-08-11)

- five fixes to the vendored reader, all worth upstreaming, and four of them
  are `func.func` being unfinished where `tt.func` is not: `parser_init`
  renamed off Booth's own, `parser_error`'s `exit(1)` replaced by a
  `mlir_parse_fail()` the linker supplies, `func.func` binding its arguments
  before parsing the body rather than after, `func.func` accepting the
  `attributes` clause where MLIR actually writes it, and `arith.xori`,
  `shli` and `shrsi` added to `op_string_to_type`, which the printer could
  already write but the parser could not read back
  (Zane Hambly, 2026-08-11)

- `ml_parse` resets the reader's process-wide type interning, which upstream
  assumes one context per process. Without it a closed context left the next
  parse in freed memory (Zane Hambly, 2026-08-11)

- the Triton lowering records pool overflow through `bir_pfull`, which the C99
  one already did and it never has. It answered a full block pool with index 0,
  a live block, so `bir_pchk` could not see a Triton arena exhaustion at all
  (Zane Hambly, 2026-08-11)

- Triton blocks are named. String offset 0 is a live string, so a nameless
  block printed as whatever went into the table first, and all four blocks of
  a loop kernel were labelled with the kernel's own name
  (Zane Hambly, 2026-08-11)

- `kath --bir-in` reads BIR text and skips the frontend entirely, so a compiler
  outside this tree can target Booth without linking against it. `src/build/`
  parses and builds modules, `src/ocaml/` emits them from OCaml, and `kcomp`
  lowers an ordinary OCaml function from its .cmt, leaving ocamlc to do the
  type checking and refusing anything outside the kernel subset by source
  location. Immediates parse as well as print, so a module holding a constant
  reads back. The PTX from an OCaml-written vadd runs on an RTX 4060 Ti
  (Zane Hambly, 2026-08-18)

- The kernel language grows device functions, shared memory, loops, division
  and the transcendentals, enough to price an Asian option on the GPU, which
  turned up four bugs now fixed: sin and cos took turns rather than radians,
  float constants printed to six digits, the BIR lexer clamped integers above
  INT32_MAX because long is 32 bits on Windows, and a function's total_insts
  was the module count rather than its own, so mem2reg moved one body over the
  next (Zane Hambly, 2026-08-18)

- The Asian pricer takes its model parameters as arguments and reduces across
  the block on device (Zane Hambly, 2026-08-18)

- `get`, `set`, `sget` and `sset` take the element type from the array rather
  than assuming f32, so an integer array is usable and not merely declarable
  (Zane Hambly, 2026-08-18)

- Atomic add, sub, and, or, xor and xchg reach the kernel language, leaving min
  and max out while BIR has one opcode for each and NVIDIA reads it unsigned
  where AMD reads it signed (Zane Hambly, 2026-08-18)

### Documentation

- How to write and build an OCaml kernel, what the subset holds and what it
  does not, and OCaml and LFortran named as the optional dependencies they are
  (Zane Hambly, 2026-08-18)

### Architecture

- A run-side contract in `src/exec`, a variant flag with no target now errors,
  and `parse_type` no longer recurses off the stack (Zane Hambly, 2026-08-18)

- BIR arena writers record a `pool_full` bit rather than returning index 0,
  which is a live entry and not a sentinel. A full pool emitted wrong
  immediates under exit 0; `bir_pchk` now refuses (Zane Hambly, 2026-08-11)

- #160: DCE and mem2reg move instructions without moving `inst_lines[]`
  with them, so every line number past the first deleted instruction
  pointed at the wrong source. Four sites fixed
  (Zane Hambly, 2026-08-09)

### Backends

- `__popc`, `__clz`, `__ffs` and `__brev`, and the BIR ops behind them (#165).
  Both scans answer the operand's width for a zero input. AMD uses
  `v_bcnt_u32_b32` and the ffbl/ffbh pair, PTX the native popc/clz/brev, and
  x86-64, RV64 and the Tensix baby cores a SWAR through the multiplier rather
  than `popcnt`/`tzcnt`, which are SSE4.2 and BMI1. Widths other than 32 are
  refused by name; `n_errs` was a field nobody read, so both CPU backends were
  printing a refusal and writing the object anyway (Zane Hambly, 2026-08-18)

### Build

- #160: vendor Kauri (MIT) as `src/kauri.h`, included from `barracuda.h`,
  so `KA_GUARD`, `KA_CHK` and `KA_PNEW` are available tree-wide
  (Zane Hambly, 2026-08-09)

### CI and tests

- `make mutate` bends one line of Booth at a time in a scratch copy and checks
  the suite notices, from a table in `tests/mutants.tbl`. Ported from Kahu's
  (Zane Hambly, 2026-08-12)

- six tests that were not testing what they looked like they were. The `cfd`
  family could not tell a reversed subtraction from an addition, because both
  fixtures folded to 7. The only SOP2 encoding test used `s_add_u32`, whose
  opcode is 0x00, so the opcode field could sit anywhere in the word. The GFX9
  SMEM branch had no test at all, on a shipping target. `rss` accepted a clean
  rejection everywhere, so an allocator that rejected everything would have
  passed. Nothing checked a memory wait waits on the memory counter, or that a
  plain `func.func` is not a kernel (Zane Hambly, 2026-08-12)

- #160: `make repro` compiles every test file twice under `--amdgpu`,
  `--nvidia-ptx` and `--ir` and compares the bytes, so the deterministic
  layout `bir.h` claims is checked rather than assumed
  (Zane Hambly, 2026-08-09)

- tests are named for their family and position, `rvi01` and `tdf39` rather
  than `rv_isel_max_frame_slots_in_range`, after z390's `TESTDCB1`. The family
  is the file stem, the old descriptive name became a description the runner
  prints, and `fam_order` in `tests/tmain.c` is the one place a family is
  declared (Zane Hambly, 2026-08-10)

- the runner refuses to start if a test registers an unknown family, a name
  that disagrees with its number, or a number already taken. 278 of the 380
  tests were registering under families `cat_order` did not list, so they ran
  unheaded in link order, `--list` showed 102 of them and `--cat rv_enc` ran
  none of them while exiting 0 (Zane Hambly, 2026-08-10)

- `--fam` replaces `--cat`, which still works, and `--families` lists the
  families with their files and counts (Zane Hambly, 2026-08-10)

- the keyword table `lookup_keyword` binary searches is checked for ordering,
  and every keyword is checked to still lex as a keyword. A misfiled entry lexed
  as an identifier and surfaced as a parse error somewhere else entirely
  (Zane Hambly, 2026-08-10)

- `make repro` reads a sidecar `tests/NAME.opt` per fixture instead of counting
  every refusal as a silent skip. Eight refusals were being hidden, one of them
  a live `v_mfma` verifier failure, and an `xfail` that starts passing is now
  reported too (Zane Hambly, 2026-08-10)

- `tests/trpi.c` collects regressions for bugs that shipped, seeded with #160's
  line-number corruption, which had four sites fixed and no test
  (Zane Hambly, 2026-08-10)

## 2026-08-07

Version 0.5.2.

First, a correction. The last release went out tagged v5.01, which was
meant to be 0.5.1 and wasn't, and it left Booth looking four major
versions further along than it actually is. It isn't. This release puts
the numbering back where it belongs, and sorry to anyone who pinned the
old one or took the version at face value. The tag stays where it is so
nothing breaks underneath you, but the compiler now reports what it is.

The theme this cycle, without meaning to be, was the compiler telling
the truth. Semantic errors used to be printed and then ignored by every
mode except `--sema`, so the backend ran on source that had already been
rejected, wrote an output file and exited zero. Asking for several
backends at once wrote all of them over the same `-o` path and left you
whichever finished last, under the name you chose, again exiting zero.
Metal quietly narrowed a double-precision kernel to float and said
nothing, which is a real problem if you were counting on the precision.
`--amdgpu` ignored `-o` entirely and mixed a diagnostic into the
assembly on stdout. All four are fixed, and all four had been sitting
there being cheerfully wrong for a while.

The structural change is the backend contract. Every target now sits
behind a `be_desc_t` and registers itself in one list, and a backend
owns its own command line rather than reaching into a shared config
struct and the driver's argument loop. Adding a target used to mean
reading 420 KB of AMD backend to work out what was expected; now it
means reading one header and copying the skeleton. `main.c` lost about
a third of its length in the process, and the frontend stopped needing
to know what an AMD target enum is.

Booth also installs now. `make install` puts `kath`, the message
catalogues and a CMake package config into a prefix, so a downstream
project can `find_package(Booth)` and compile kernels as part of its own
build with `booth_add_kernel()`. There is a worked example under
`examples/cmake/` and CI builds it against a staged install on every
push, along with a check that the target list in the package config
hasn't drifted from what the backends actually accept.

Getting Booth no longer requires being able to build it. Every release
now carries prebuilt binaries for Linux, macOS and Windows, statically
linked so they have no runtime dependencies whatsoever, with checksums.
Unpack and run. This mattered more than I realised: the Windows build
had been quietly depending on MinGW's `libssp-0.dll`, so handing someone
the binary would not have worked on a machine without a toolchain, which
is exactly the machine they wanted it for.

There are coverage numbers for the first time, 74.1% of lines and 58.4%
of branches, reported by CI on every PR. That immediately turned up the
SSA register allocator having never been executed by a test at all, and
the six fixtures now pinning its behaviour also pin a real bug in it,
which is at least honest.

Thanks to Jorge Galvez, whose do-concurrent ocean benchmarks found three
genuine frontend bugs in an afternoon, and who let me test against his
code. Thanks to @maou3434 for the bare HIP warp and lane intrinsics,
which is their first contribution here and a very welcome one. And
thanks to @FileDelta for asking a simple question about the runtimes
that turned over considerably more than either of us expected.

### Frontend

- double-precision `fmax`, `fmin` and `fmod`. The ocean kernels in
  [Jorge Galvez](https://github.com/JorgeG94)'s do-concurrent benchmarks call
  them, and only the `f`-suffixed single-precision forms were recognised
  (Zane Hambly, 2026-08-06)

- raise the cap on arguments in one call to 64, and say so when a call goes
  past it. Sema stopped counting at 16 and then reported an arity mismatch
  against the count it had stopped at, so a correct 23-argument call in the
  same ocean benchmarks was rejected and told the wrong number
  (Zane Hambly, 2026-08-06)

- #142: parse function pointer declarators, and constructors and destructors
  (Zane Hambly, 2026-07-27)

- #144: move the Triton errors into the shared error catalogue
  (Zane Hambly, 2026-07-28)

- keep block comment state across lines, so a macro quoted in prose is no
  longer expanded
  (Zane Hambly, 2026-07-22)

- stop dereferencing anonymous-struct name sentinels as source offsets
  (Zane Hambly, 2026-07-22)

### Triton

- let `--cpu` and `--rv64` through the mode gate, and fail with a nonzero
  status rather than a silent zero
  (Zane Hambly, 2026-07-16)

### HIP

- #137: support bare convergent warp and lane intrinsics
  (Maou, 2026-07-27)

### Architecture

- one target per run. Several backends at once all wrote to the same
  `-o` path, so you got whichever came last in the registry under the
  name you asked for, and a zero exit

- backend contract (`be_desc_t`) with static registration; every
  existing backend sits behind the same shape and the driver iterates
  `be_list` instead of the copy-pasted if-chain. A backend also owns its
  own command line now, so adding a target means one file and one line in
  the list rather than editing a shared config struct and the driver's
  argument loop. Skeleton in `src/backend/skeleton/` and
  `docs/backends.md` for anyone adding a target
  (Zane Hambly, 2026-08-02)

### Backends

- seed atomic RMW as divergent in the AMD divergence analysis, so a GEP off
  an `atomicAdd` result no longer takes the scalar path and emits
  `s_add_u32` with a VGPR source; unblocks per-thread SYSPRINT on AMD
  (Zane Hambly, 2026-08-01)

- #138: real high-half multiply on x86-64 and RV64, and an honest refusal
  where it cannot be done
  (Zane Hambly, 2026-07-26)

- metal: refuse a kernel that uses `double` rather than narrowing it to
  `float`. Apple GPUs have no fp64, and quietly halving the precision the
  source asked for is worse than saying so
  (Zane Hambly, 2026-08-06)

- a divergent return masks lanes instead of ending the wave, so AMD kernels
  no longer lose the lanes that did not take the branch
  (Zane Hambly, 2026-07-25)

- `--amdgpu` honours `-o`, and the register-plan line goes to stderr rather
  than into the middle of the assembly on stdout, where it stopped the result
  assembling
  (Zane Hambly, 2026-08-06)

### Tensix

- `--tt-chip` selects wormhole or blackhole, so L1 and text limits follow the
  target part instead of being fixed
  (Zane Hambly, 2026-07-23)

- refuse i64 arithmetic instead of silently truncating it to 32-bit
  (Zane Hambly, 2026-07-22)

- the RV32 backend gains control flow, calls, GEP and shared memory, and emits
  an ELF that tt-metal can load
  (Zane Hambly, 2026-07-22)

- reserve a 64K `__shared__` slab at the top of L1
  (Zane Hambly, 2026-07-22)

### Runtime

- ABEND dump renders the captured kernarg block as a mainframe-style SNAP
  section (offset gutter, four 4-byte hex groups, ASCII on the right), so
  the arg bytes the launcher handed off are visible on every backend that
  calls `ab_snag`, not just AMD
  (Zane Hambly, 2026-08-02)

- ABEND arms the CPU backend too: `ab_arm_cpu` hooks POSIX
  SIGSEGV/SIGILL/SIGFPE/SIGBUS or the Windows unhandled-exception filter and
  maps them onto the G0Cx taxonomy, so a `--cpu` kernel that faults gets
  the same dump an AMD kernel would
  (Zane Hambly, 2026-08-01)

- #141: run LFortran `do concurrent` kernels on Booth, NVIDIA and AMD
  (Zane Hambly, 2026-07-26)

### Driver

- semantic errors fail the compile. Every mode but `--sema` printed them and
  then carried on into codegen, wrote an output file and exited zero, so a
  build system saw a clean compile of source we had already rejected
  (Zane Hambly, 2026-08-06)

- collapse the C99 mode gates into one cascade
  (Zane Hambly, 2026-07-23)

- let `--tdf` and `--tdf-fission` through the mode gates
  (Zane Hambly, 2026-07-23)

### Build

- track header dependencies with `-MMD`, so a header edit no longer leaves
  stale objects linked in
  (Zane Hambly, 2026-07-22)

- compile `nv_rt` and `bc_runtime`, platform-gated
  (Zane Hambly, 2026-07-16)

- #146: give each host its own object directory under `build/`, so a Git Bash
  build and a WSL build in one checkout stop overwriting each other and handing
  the linker a mix of COFF and ELF
  (Zane Hambly, 2026-07-28)

- `make install`, honouring `PREFIX` and `DESTDIR`, and a CMake package config
  alongside it, so a downstream project can `find_package(Booth)` and build
  kernels with `booth_add_kernel()`
  (Zane Hambly, 2026-08-06)

- prebuilt binaries on every release for Linux, macOS and Windows, statically
  linked where the platform allows it so they carry no runtime dependencies,
  with SHA256 checksums. The Windows build previously needed MinGW's
  `libssp-0.dll` present, which made the binary useless to anyone without a
  toolchain
  (Zane Hambly, 2026-08-07)

### CI and tests

- #140: numeric regression against SLATEC known-good values, across cpu,
  rdna3 and rdna4
  (Zane Hambly, 2026-07-26)

- #143: add srot and srotm to the numeric suite, and size the emulator kernarg
  block from the kernel rather than a fixed 64 bytes
  (Zane Hambly, 2026-07-27)

- validate Tensix ELFs against tt-metal's loader and run RV64 under QEMU
  (Zane Hambly, 2026-07-23)

- guard the divergent-return lowering against regressing to `s_endpgm`
  (Zane Hambly, 2026-07-26)

- #154: `make coverage` builds an instrumented tree and reports line coverage
  via gcovr, with a CI job that posts the summary and uploads the HTML
  (Zane Hambly, 2026-08-02)

- #154: cover the SSA register allocator, which had never been run by a test.
  Six fixtures and any `--max-vgprs` below 8 leave virtual registers
  unallocated under `--ssa-ra`; those are pinned in `tests/tra_ssa.c` until
  the allocator is fixed
  (Zane Hambly, 2026-08-02)

### Documentation

- drop the LLVM requirement from the usage documentation
  (Zane Hambly, 2026-07-27)

- clarify the compiler requirements in the README
  (Zane Hambly, 2026-07-23)

- move the changelog to `CHANGELOG.md` and log changes per PR under Unreleased,
  with a CI check that asks for an entry unless the PR carries the
  `no-changelog-needed` label
  (Zane Hambly, 2026-07-28)

- link the changelog from the README
  (Zane Hambly, 2026-07-28)

- document the Fortran path in `docs/usage.md`, from `do concurrent` through to
  a compiled kernel, with the generated signature and the current limitations,
  and link it from the README
  (Zane Hambly, 2026-07-28)

## 2026-07-14

Version 5.01, which should have read 0.5.1. See the 0.5.2 note above.
So long, and thanks for all the fish.
BarraCUDA was a good pun and a bad description, so it swam off: the
compiler is Booth now, the binary is `kath`, and the version jumps
to mark the line. First release under the new name; the rename note
below has the why.

Since 0.5 the compiler grew a spine on the CPU side. x86-64 and RV64
both gained most of their scalar arithmetic, so a CUDA, HIP or Triton
kernel now compiles and runs on a machine with no GPU in it at all.
Tensix learned to emit native machine code for the baby RISC-V cores
instead of leaning on a C++ handoff. `__device__` calls are inlined
away before isel, so device helpers with control flow finally work on
the GPU and vector backends. Diagnostics were rebuilt to read like
Clang's, carets and colour and all, and the frontend picked up a pile
of coverage.

Thanks to the people who sent patches this cycle: @GauthamMK-0 for
`__hip_bfloat16` and the `warpSize` builtin, @nataliakokoromyti for
lowering `tl.where` to select and keeping the PTX header ASCII, and
@kstppd for fixing the Makefile on non-x86 machines. Much appreciated.

## 2026-06-29

Renamed from BarraCUDA to Booth. The fish pun said
"CUDA" while the whole point of the project is not needing CUDA, so
it had to go. Booth honours Kathleen Booth: she wrote the first
assembly language and co-built the computer it ran on, which is the
top and the bottom of this compiler's pipeline in one person. The
CLI binary is now `kath`, after her. Old `Zaneham/BarraCUDA` URLs
redirect, so nothing external breaks. Internal `bc_`-prefixed
symbols keep their names for now; renaming those is tracked as a
good-first-issue rather than risked in one big sweep across the
codegen. Every entry below this line predates the rename and still
says BarraCUDA, because that's what happened.

## 2026-06-13

Diagnostics that read like Clang's. A new renderer
(src/fe/bc_render.c) gives every phase the same shape: error[Ennn]:
message, a --> file:line:col line, the offending source line, and a
caret under the column, coloured when stderr is a terminal and plain
when piped. main.c's eight per-phase error loops collapse to one
bc_diag() call, and the parser now names the token it choked on (got
';') instead of its kind (got IDENT).

## 2026-06-06

The scalar backends finish the job, and a real
workload drags three old bugs into the light. Building on
yesterday's arithmetic, `--cpu` and `--rv64` gain the direct-ISA
maths (sqrt, abs, round, floor, ceil, min and max), the libm
transcendentals (sin, cos, exp2, log2) reached through honest ELF
relocations and the platform call sequence, with sine and cosine
prescaled back from turns to radians so the GPU's hardware habit
does not leak into a CPU answer, and the atomics, barriers and warp
primitives, which on a loop that runs one thread at a time collapse
politely to a single lane: an atomic is a load, an op and a store
that hands back the old value, a barrier waits for nobody, a shuffle
returns the lane its own value. Then Moa's Monte Carlo neutron
transport, compiled through `--cpu` with no GPU in the room, landed
at k_eff 0.999, but only after it shook loose three codegen bugs
that had been sitting in the tree since March: a narrow integer
parameter never sign-extended, so SysV's undefined high half quietly
broke the kernel's `if(tid < n)` guard; a struct-array stride that
strode by four bytes instead of the whole struct, so every element
past the first read the wrong place; and a float constant taken from
the low half of its double union, so 1.0f stored as zero and an
infinity stored as a rounding crumb. All three are closed, caught by
compiling the same kernel with a stock compiler and diffing the two
runs particle by particle until they agreed.

## 2026-06-05

The CPU and RISC-V backends learn most of their
arithmetic. The `--cpu` (x86-64) and `--rv64` emitters carried
only a starter set of opcodes and everything else quietly returned
zero, which is a fine way to compute the wrong answer with great
confidence. This fills out the scalar instruction set on both at
once, designed the once and encoded twice. Integers gain and/or/xor,
the shifts (width-aware, so a 32-bit `>>` stops dragging the sign
bits down and the xorwow RNG stops handing out the same number
forever), and signed and unsigned divide and remainder. Floats gain
divide, min/max, remainder, compare and select, and the existing
add/sub/mul stop reading a constant operand out of slot zero, which
is to say out of whatever was lying there, a real bug now quietly
closed. The full conversion set lands too: truncate, sign- and
zero-extend, integer to float and back in both signednesses, float
widths, bitcast, and pointer to int. Pointers also stop being
sign-truncated to 32 bits on the way out of memory, which had been
turning a perfectly good address into litter. Width is kept honest:
values sit sign-extended in their slots, RV64 reaches for the *W
word ops, and x86 reads a source at its own width for cvtsi2ss
because SysV never promised to sign-extend a narrow argument and a
64-bit read of one comes back a giant unsigned. Every opcode was
checked against a plain-C oracle on both targets, x86 natively and
RV64 under qemu. The structuriser also gains explicit bounds on its
recursion and its block walk, so a pathological CFG bails rather
than walking the C stack off the edge. No GPU goes near any of it;
it is the groundwork for running a whole kernel on a chip that
never heard of one.

## 2026-06-04

Apple Silicon, by way of Metal. A CUDA, HIP or
Triton kernel now lowers to Metal Shading Language and compiles
under Apple's own toolchain. The Metal emitter (`--metal`) walks
BIR and writes MSL that `xcrun metal` accepts; the kernel
signatures were already there, this fills in the body, every BIR
instruction to its MSL form, with the thread-model builtins mapped
to `thread_position_in_threadgroup` and friends and pointers
tagged device, threadgroup or constant. The piece that made it
real is a control-flow structuriser, because Metal flatly refuses
a goto and the first cut emitted labels and jumps that Apple's
compiler threw straight back. `src/ir/bir_struct.{c,h}` is a
backend-agnostic analysis: dominators and reverse postorder
recover the loop headers and merge points, a recursive descent
rebuilds the nested if/loop/break/continue shape, and phi nodes
get paid off as copies on the edges that feed them. It describes
the structure and never touches the BIR, so the AMD, NVIDIA, CPU
and Tensix backends carry on emitting jumps and never notice it
exists. Relooper-shaped, after Ramsey's "Beyond Relooper" (ICFP
2022); irreducible CFGs and switch are refused (ok = 0) rather
than miscompiled, with the switch-dispatch fallback left for a
later sitting. tests/tstruct.c covers if, if/else, for and while
with break and continue and checks the output comes out goto-free.
CI gains a macOS job that builds, runs the suite, then compiles the
emitted MSL with `xcrun metal` on a real Apple runner, so "it
emits valid Metal" is checked on every push instead of asserted.
The Makefile learns to drop GCC-only warning flags and -Werror on
clang so the macOS build goes through. 282 tests pass; AMD,
NVIDIA and CPU output is unchanged.

## 2026-05-29

0.5 release. The headline is that you can write a
Triton kernel, matmul and all, and run it on a CPU with no GPU.
The CPU backend (`--cpu`) lowers BIR straight to x86-64
with the SIMT model collapsed into a thread loop, and the rank-2
tile path materialises and unrolls so `tl.dot` plus a K-loop
sweeps an arbitrary contraction. A RISC-V backend (`--rv64`) lands
alongside, emitting RV64IMFD objects that run under qemu, which
makes cross-backend differential testing a thing on a laptop: same
BIR through two backends, diff the output buffers, CPU is the
oracle, every case runs `--inject` so a green result actually
means something (see tests/diff). Triton picks up scalar math
intrinsics (exp/log/sin/cos/tan/tanh/sqrt/rsqrt/abs/floor/ceil/
maximum/minimum/fdiv) thanks to @shivam2931120, who got the
radians-to-turns convention right first time. Triton constexpr
params with a default value now fold to literals at lower time
and drop out of the runtime signature too, so the matmul example
loses three unused args. Two small CUDA fixes for `--cpu` and
`--rv64` on their own: sema actually runs (the lowerer needs
it), and the parse-dump fallback no longer fires by accident, so
typedef-struct kernels compile through `--cpu` and `--parse` no
longer segfaults on the synthetic anon name. Adds `--version`.

## 2026-05-26

Triton matmul K-loop, so big matrices work. The
Triton lowerer now lowers `for k in range(...)` as a real counted
loop, which lets a matmul sweep an arbitrary contraction dimension
K a BLOCK_K tile at a time instead of unrolling the whole thing.
The loop counter is a phi (not an alloca, because mem2reg folds an
alloca counter to a constant here and spins forever), and the
accumulator is scratch-backed so it survives across iterations
(read-add-write per element). Verified end to end for K up to 512.
One real bug fixed on the way: the lowerer was setting a function's
total_insts to the last block's count rather than the sum over all
blocks, so cfold and dce walked a truncated instruction range and
quietly corrupted any kernel with control flow. Single output tile
still; multi-block grids and tl.load mask= remain the next sitting.

## 2026-05-25

Triton matmul on the CPU. tl.dot now lowers for the
x86-64 backend, so a real Triton matmul kernel (load two tiles,
tl.dot, store) compiles from Python and runs natively. Rank-2
tiles take a new path in the Triton lowerer: because block sizes
are constexpr, the whole tile is materialized and fully unrolled
into scalar BIR (arange becomes literal indices, [:,None] /
[None,:] reshape, elementwise ops fan out with broadcasting, and
tl.dot becomes an unrolled sum of products). A kernel with any
rank-2 tile takes this path for its whole body and launches one
thread; elementwise rank-1 kernels keep the lane-collapse path.
Backed by BIR_ALLOCA scratch buffers in the CPU emitter. Verified
4x4, 8x8 and non-square 2x3@3x4 against a host reference. Still a
single output tile, no K-loop tiling or masking, capped at 32x32;
those are the next sitting.

## 2026-05-25

x86-64 CPU backend. A new --cpu flag turns a
__global__ kernel into a host-runnable x86-64 ELF object, so
CUDA and Triton kernels can be developed on a laptop with no
GPU. Triton's canonical vector_add runs end to end (Python ->
BIR -> x86-64 -> native execution) and computes a full block.
Stack-everything codegen, no register allocator. Correct
first, fast later. SIMT on CPU: the kernel body runs in a loop
over thread_id in [0, nthreads), with nthreads passed as a
hidden trailing arg, so one call covers a whole block; a return
ends one thread. Full op set lands: add/sub/mul, gep, width-
aware load/store (i8/i16/i32/i64, f32), fadd/fsub/fmul, icmp
(all predicates), branches with both edges explicit, phi via
predecessor edge-copies, and SysV stack arguments for the
seventh parameter onward. See examples/cpu_launch_vadd.c for
the calling convention. Multi-block grids and tl.load mask=
are the next sitting; rank-2 tiles (matmul) still refuse with
E099 pending matrix codegen.

I took the x86-64 emit machinery (the code buffer, REX/ModRM
helpers, the RBP slot map and branch fixups) straight from my
other compilers, Skyhawk (JOVIAL J73) and Karearea (Fortran
77). No point rewriting a perfectly good instruction encoder
twice.

## 2026-05-24

Triton tile shape inference, sitting one. The sema
pass now walks every expression bottom-up and annotates a
(rank, dims, dtype) triple onto each AST node. Scalar, vec[N]
and mat[M, N] all flow through; arange, zeros, broadcast and
the canonical [:, None] / [None, :] reshape patterns produce
the right shapes; numpy-style broadcasting picks up the cases
real Triton kernels rely on. Constexpr value propagation is the
next sitting, so block-size dims still print as `?` for now,
which is fine for getting lowering to dispatch on rank. See
issue #82.

## 2026-05-24

Tenstorrent native RV32IM backend. The baby
RISC-V cores on Wormhole now have a real codegen path that
does not go through Metalium SFPI. New --rv-elf flag lands a
.elf for a __global__ kernel; 13 of the small CUDA samples
compile through it cleanly.

Also new: a TDF (Tile DataFlow) layer above BIR that models
regions, L1 placement, and NoC arcs as one shared description,
and a soft-float fp32 runtime under runtime/ aimed at the
forthcoming float lowering pass. The runtime is correct against
the host FPU (28 corner cases + 1000 random pairs); wiring it
into --rv-elf is the next sitting.

Compiler fix: typedef struct { ... } name; now registers
properly, so member access on the typedef name no longer fails
with E110. Two frontend bugs found along the way are filed as
#88 (preprocessor expands macros inside /* */ comments) and #89
(lexer treats apostrophes inside /* */ comments as char-literal
starts). Test suite 256/256.

## 2026-05-21

Triton end-to-end test suite. 16 new tests in
tests/ttriton.c covering the four frontend stages (lex, parse,
sema, IR) and the three backends a Triton kernel can land on
(AMD GFX11, NVIDIA PTX, Tensix Metalium). The kernels live as
.py files in tests/: a minimal arithmetic test, the canonical
vector add, a goblin-themed variant to confirm the parser is
name-agnostic, and a deliberately ChatGPT-flavoured
"high-performance optimized vector addition" with a six-paragraph
docstring that lowers to the same code the eight-line goblin
kernel does. Caught one real bug along the way: docstrings inside
@triton.jit functions were tripping the lowerer; bare-string
ExprStmt is now silently discarded the way it should be. Test
suite stands at 107 cases, 106 pass, 1 skip.

## 2026-05-21

Triton frontend wired through to every backend. The
post-lowering pipeline (mem2reg, constant folding, dead code
elimination, then per-backend codegen) lived inside the C99
frontend block in main.c since the day the project gained its
second backend. Factored out into run_bir_backends so both
frontends can call it. Triton kernels now compile end to end:
--triton --amdgpu-bin produces a working .hsaco, --triton
--nvidia-ptx produces valid PTX text the NVIDIA driver will JIT,
--triton --tensix produces TT-Metalium compute / reader / writer
/ host C++ files, and --triton --metal / --intel-spirv route
through their respective backends as those mature.

Validated on the canonical vector_add Triton kernel:
  AMD GFX11      176 bytes of code, 2808-byte .hsaco
  NVIDIA sm_89   929 bytes of PTX, 20 instructions
  Tensix         2629-byte compute kernel with SFPI encoding,
                 reader / writer / host scaffolding, datamov
                 analysis recognising the three buffers

The mask= keyword arguments on tl.load and tl.store became
proper warnings rather than counted errors, so the lowering
returns BC_OK and the backend pipeline runs through. Test suite
still 90/90, and a side-by-side compile of canonical.cu through
all three backends confirms no regression on the CUDA path.

## 2026-05-21

Triton BIR lowering, sitting two. The vector add
kernel now lowers end to end into a syntactically valid BIR
module: tl.program_id and tl.num_programs as before, plus
tl.arange as BIR_THREAD_ID for the canonical Triton "one thread
per lane" mapping, tl.load and tl.store as BIR_LOAD and BIR_STORE
with the BIR convention that store operands are (value, address),
and tl.cdiv inlined as the obvious (a+b-1)/b ceiling division.

The binop lowerer now dispatches between integer and floating
point variants based on operand types, with the special case
that a pointer plus an integer produces a BIR_GEP rather than
a plain add. Comparisons produce BIR_ICMP or BIR_FCMP with the
right predicate in the subop slot. AugAssign reads the current
local value, applies the op, and re-binds the local to the new
result so subsequent references see the updated state.
Parameters whose name ends in "_ptr" are typed as f32*, params
annotated tl.constexpr become i32, and the rest default to
scalar i32; a proper type inference pass arrives in sitting
three. The mask= keyword arguments to tl.load and tl.store are
honoured by a polite diagnostic that explains the load or store
is being emitted unmasked for now.

Proof of life: the canonical vector_add kernel compiles to a
clean BIR module with the right shape: pid, block_start, arange,
mask, two GEPed loads, an fadd, and a GEPed store. The matmul
kernel still chokes on its tile subscripts and the for loop, all
of which are tracked for sitting three. Test suite 90/90 still
passes.

## 2026-05-21

Triton BIR lowering, sitting one. The --triton --ir
mode now walks the sema-annotated AST and produces a BIR module,
the same intermediate representation the C99 frontend already
feeds the backends. Triton kernels marked @triton.jit become
__global__ functions with the right parameter list, BIR_PARAM
instructions for each parameter, and an entry block terminated
with BIR_RET. Statements covered in this sitting: Assign (lower
the RHS and remember the BIR value by the assign node's index so
later Name references can find it), Return, ExprStmt, Pass /
Break / Continue (no-op for now). Expressions covered: integer
and float literals, Name references to params and locals via the
sema annotations, BinOp arithmetic (add / sub / mul / sdiv /
srem / shifts / bitwise), unary plus / minus / invert, and the
scalar thread-model intrinsics tl.program_id and tl.num_programs
mapped onto BIR_BLOCK_ID and BIR_GRID_DIM with the right axis
subop. A trivial test kernel exercising program_id, num_programs,
and a chain of integer arithmetic lowers to 10 BIR instructions
and dumps as a syntactically valid BIR module.

Known gaps for sitting two: control flow (For, While, If), masked
loads and stores, subscripts and slices, tile shape inference,
the full intrinsic catalogue (anything beyond program_id and
num_programs currently emits a polite "not yet lowered"
diagnostic), and per-parameter type inference (sitting one
assumes every pointer is i32* and every scalar is i32). Test
suite 90/90 still passes, all existing backends still compile.

## 2026-05-20

Triton sema, sitting one. The --triton --sema mode now
runs name resolution and intrinsic dispatch over the AST the parser
produces. Every Name node is bound to one of: a kernel parameter,
an assigned local, a for-loop variable, an imported module alias,
a Python builtin (range, len, min, max, abs and friends), or a
diagnostic if it does not bind at all. Every Attr node whose base
resolved to the tl module is checked against the intrinsic table
and tagged with the corresponding tn_intrinsic_t code, so the
later lowering pass can dispatch by id rather than by string
compare.

The intrinsic table covers the tl.* functions and dtypes that
appear in real Triton kernels: program_id, num_programs, load,
store, arange, zeros and friends, dot, where, reductions, the
math library (exp, log, sin, cos, sqrt and the rest), the
compile-time helpers static_assert / static_print, and the
numeric dtype markers float16 through uint64. Adding a new
intrinsic is a one-line table entry.

Known gap for the next sitting: submodule access (tl.math.exp,
tl.extra.libdevice.*) is not yet resolved, because sema does not
track sub-modules below the top-level alias. The diagnostic
(unknown intrinsic: tl.math) is informative and the rest of the
kernel still resolves around it.

Validated end-to-end on the vector add (0 errors) and the matmul
kernel including the for loop, AugAssign, and tuple subscripts
(0 errors). Tile shape inference, constexpr value propagation,
and arity checking on intrinsic calls are deliberately deferred
to sema sitting two. Test suite 90/90 still passes.

## 2026-05-20

Triton expression parser, sitting two. The opaque
ExprSpan token-range node is mostly retired and replaced by a real
AST shape: Name, Literal (with int / float / string / None / True /
False subkinds), Tuple, BinOp (thirteen arithmetic and bitwise
codes), UnOp, BoolOp (and / or), Compare (single comparison, chains
are a future sitting), Call with positional and keyword arguments,
Keyword, Attr, Subscript, Slice (with a present-component bitmask
because Python lets you leave any of the three pieces empty), List
display, and conditional expressions of the x-if-cond-else-y kind.

The parser is precedence climbing, one function per binding level,
following Python's published precedence table from the language
reference. Power is right-associative; unary plus / minus / invert
sit BETWEEN power and the multiplicative level so that -2**2
correctly parses as -(2**2) the way the spec says. For-loop
targets bypass the comparison level so `for k in range(...)` does
not eat the `in` keyword as a comparison operator. ExprSpan is
kept as a graceful fallback for genuinely unrecognised constructs
(comprehensions, walrus, await, yield) which the parser will note
with a polite diagnostic and continue past.

Validated on a Triton vector add, a matmul kernel with all the
heavy address arithmetic and tuple subscripts, and an expression
torture file exercising precedence, conditional expressions,
boolean operators, slicing with all three components, and calls
with keyword arguments. No regressions: CUDA -> AMD, HIP -> AMD,
and Metal MSL emission all still work, test suite 90/90 still
passes.

## 2026-05-20

Triton parser, sitting one. The --triton --parse mode
now builds a real AST from the token stream the lexer produces.
Every statement Triton kernels actually use has its own node kind:
FuncDef with its parameter and decorator slots, Block for the
indent-delimited suite, Assign and AugAssign (the latter carrying
the operator subcode in its flags field), ExprStmt for a bare
expression used for its side effect, If with elif and else
branches, For with optional else, While with optional else,
Return, Pass, Break, Continue, Import and ImportFrom with their
dotted-name children. Expressions are still opaque token spans
because the real expression parser waits its turn for the next
sitting.

The parser is recursive descent with one token of lookahead and
uses a scratch-pool pattern for child storage: each grammar
function records the scratch top on entry, pushes its children
as it parses them, and commits the contiguous range to either
inline kid slots or to the extra_kids overflow pool when the node
finishes. This was the fix for an interleaving bug an earlier
incremental approach suffered from, in which a child node's
overflow writes would happily scribble inside its parent's reserved
overflow range, leaving the parent dump with the wrong children
indexed at the wrong offsets. The scratch-pool guarantees each
node's children are contiguous in extra_kids and never trampled
by a descendant.

Validated on a Triton vector add and a matmul kernel including a
for-loop body with += AugAssign and nested function calls. Dumps
the full structural tree to stdout when --triton --parse is set.

## 2026-05-20

Triton frontend, first sitting. The --triton flag now
routes the input file through a brand new self contained Python
tokenizer at src/triton/lex.c rather than through the C99
preprocessor and lexer. The tokenizer handles the parts of Python's
lexical grammar that Triton kernels actually use: identifiers and
the reserved word table, decimal and hex and octal and binary
integers, floats with exponents, single line strings, every
operator and augmented assignment, and the indentation algorithm
that turns Python's leading whitespace into INDENT and DEDENT
tokens off a column stack while honouring implicit line joining
inside brackets and explicit line joining via backslash newline.
The parser, sema, and BIR lowering passes live alongside as stubs
and arrive in future sittings. Validated on a real Triton vector
add and matmul kernel: 393 tokens, zero errors, correct indent
nesting through a 43 line file. Self contained as the rest of the
compiler, no Python in the build pipeline anywhere.

## 2026-05-20

Apple Metal backend, first sitting. The --metal flag now
produces a complete and syntactically valid MSL source file rather
than the polite refusal of the earlier stub. metal_compile walks
every BIR function, picks out the __global__ kernels and __device__
helpers, extracts parameters from the entry block's BIR_PARAM
instructions, and scans each function for thread-model opcodes so
the emit stage only synthesises the [[thread_position_in_threadgroup]]
and friends the kernel actually references. metal_emit_msl writes
the metal_stdlib include, a kernel signature with the correct
device, constant, and threadgroup address-space qualifiers,
[[buffer(N)]] binding indices on real parameters, and the MSL-mandated
constant-reference form for scalar kernel arguments. Function bodies
remain TODO comments with a value-initialised return for any
non-void __device__ helper. Hardware validation against xcrun metal
on macOS is tracked as issue #68.

## 2026-05-19

HIP frontend mode (--hip, plus auto-detect on the .hip
file extension). HIP source is syntactically a superset of CUDA, so
no new parser was required; the entire change is one block in main.c
that predefines __HIPCC__, __HIP_DEVICE_COMPILE__, and the appropriate
__HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__ macro depending on
the chosen backend. Downstream of the preprocessor the pipeline is
identical to a CUDA compile. A .hip kernel now compiles through to
a working GFX11 .hsaco end to end. Two clearly marked HIP NOTES
blocks in main.c document exactly where the divergence lives and
why everything else stays put.

## 2026-05-19

Apple Metal and Intel Arc Xe backend stubs (--metal,
--intel-spirv with --xe-lpg / --xe-hpg / --xe-hpc / --xe2 variants).
Just a stub for now: the directories exist, the headers declare
module structs and entry points, the CLI flags parse, the modes
flow through preproc, lex, parse, sema, BIR lowering, and the
optimisation passes, and the backend functions politely return an
error code rather than actual machine code. Plan is MSL text emit
for Apple and SPIR-V emit for Intel, both following the "let the
vendor's driver finish" pattern that PTX already uses for NVIDIA.

## 2026-03-18

NVIDIA PTX backend (--nvidia-ptx). Compiles CUDA to PTX
text, loaded via CUDA Driver API and JIT-compiled by the NVIDIA driver.
Validated on RTX 4060 Ti running a Monte Carlo neutron transport
benchmark with correct physics results. No NVCC required. Also:
anonymous struct/union support in parser, sema, and lowerer
(struct { float f; int i; } cvt; pattern).

## 2026-03-14

Divergence-aware SSA register allocator (--ssa-ra).
Eliminates all 186 VGPR spills on a 654-line Monte Carlo transport
kernel — scratch traffic drops 78%, total instructions drop 28%.
Exploits the 64:1 cost asymmetry between divergent and uniform VGPR
spills on Wave64 hardware: uniform values spill via v_readfirstlane
at 4 bytes each, divergent values stay in registers where they belong.
Based on the divergence analysis of Sampaio et al. (2013). ~1,300
lines of C99, all static memory, no malloc.

## 2026-03-09

Post-isel verification pass (bc_vfy). The encoder used
to trust isel to produce valid machine instructions. It shouldn't
have. bc_vfy runs twice (post-isel, post-RA) and catches 5 classes
of encoding violation before the binary leaves the compiler. Its
first run immediately found 7 isel bugs across GFX10 and GFX942 —
every one a silent miscompile that would fault on hardware with
"Reason: Unknown." Fixed them all. Also: bc_abend runtime crash
diagnostics, because if IBM could do post-mortem dumps in 1964, we
can do it for GPUs in 2026.

## 2026-03-08

Error localisation infrastructure. Every diagnostic
now has a language-neutral ID (E001–E111). External translation
files via --lang <file>. English reference at lang/en.txt, te reo
Maori at lang/mi.txt. Unified error structs. Lowering errors now
displayed.

## 2026-03-05

CDNA 3 additions: GFX942 backend hardening, MFMA,
Wave64 divergence, tinygrad compat. 8/8 tests passing on MI300X
(PR#56: https://github.com/Zaneham/BarraCUDA/pull/56).

## 2026-03-05

Instruction scheduling
(PR#52: https://github.com/Zaneham/BarraCUDA/pull/52).

## 2026-03-03

CDNA 2 support (--gfx90a, MI250). Tinygrad
compatibility.

## 2026-02-28

Tenstorrent Tensix backend (--tensix). Compiles CUDA
to TT-Metalium C++ for Blackhole. Constant folding
(PR#51: https://github.com/Zaneham/BarraCUDA/pull/51). Dead code
elimination (PR#48: https://github.com/Zaneham/BarraCUDA/pull/48).

## 2026-02-25

HSA runtime launcher
(PR#40: https://github.com/Zaneham/BarraCUDA/pull/40). RDNA 2
support (--gfx1030, PR#38: https://github.com/Zaneham/BarraCUDA/pull/38).
Test suite (PR#41: https://github.com/Zaneham/BarraCUDA/pull/41).

## 2026-02-20

RDNA 4 support
(--gfx1200, PR#32: https://github.com/Zaneham/BarraCUDA/pull/32).

## 2026-02-16

Initial release. CUDA compiler targeting AMD RDNA 3
(gfx1100).
