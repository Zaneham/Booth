# Roadmap

[← back to README](../README.md)

Kia ora (hello), welcome to Booth's roadmap. Here is a listed collection of my thoughts and where I would like to take this project.

Please note that I am just one dude doing this mainly by myself and with the help of a few amazing contributors from time to time. This is also my hobby and I enjoy doing it. This isn't a product and it's subjective to whatever I feel like working on in a particular day. The Roadmap is also subject to change and can include everything from common sense additions to the utterly absurd. 

On that note: 

## Near term

Things with a known shape (kinda). Could be a sitting in one pr / a couple of commits:

- **Port `nv_rt` to `rt_desc_t`.** The launcher contract and `booth-run` are in
  `src/exec/` with an empty `rt_list[]`. NVIDIA and AMD already implement the
  same ten operations under different names, so porting one proves the new contract
  holds and gives `booth-run` something to launch.
- **Split `BIR_ATOMIC_MIN` and `BIR_ATOMIC_MAX` into signed and unsigned.** BIR
  has one opcode for each, the NVIDIA backend reads it unsigned and AMD reads it
  signed, so the same kernel gives different answers per vendor for negative
  values. Needs `bir.h`, the CUDA frontend picking by operand type, and both
  backends. Until then the OCaml frontend refuses to expose them.
- **Widen `--bir-in`.** It reads the subset the frontends emit. Round-tripping
  the sample corpus is 45 of 103, the rest being aggregate types and opcodes the
  parser does not know yet. Everything it cannot read is named and refused.
- **Parameter reassignment in `__device__` functions.** The last of the original
  hardening list; the others are done.
- **A `--deterministic` flag.** GPU floating point is not reproducible run to
  run, because the order a reduction happens in depends on the scheduler and
  atomics make it worse. For most people that is a rounding curiosity; for
  anything that has to be signed off it is disqualifying, since you cannot audit
  a number you cannot reproduce. Forbid float atomics, fix the shape of the
  reduction tree rather than letting the hardware pick, pin the accumulation
  order. It will be Slower, and the same bits every time on a given backend. 
- **Soft-float into `--rv-elf`.** The runtime exists and validates against the
  host FPU. Wiring it up is the remaining bit.


## Frontends

Booth takes seven inputs. They all meet at BIR and go down the same pipeline.

| Frontend | Flag | State |
|---|---|---|
| CUDA C | *(default)* | The reference frontend |
| HIP | `--hip`, or a `.hip` file | Macro layer over the CUDA path |
| Triton | `--triton` | Python subset, `tl.*`, `tl.dot` on the CPU backend |
| Fortran | via LFortran | `do concurrent`, elementwise, SLATEC-checked in CI |
| OCaml | via `kcomp` | Kernels as typed OCaml functions |
| BIR text | `--bir-in` | What `--ir` prints, read back in |

What each still owes:

- **Triton** — rank-2 tiles refuse on GPU targets with E099. `tl.load` masks are
  not honoured on the CPU backend.
- **OCaml** — no `while` and no early exit, no `f64`, no warp shuffles, and no
  atomic min or max until the IR question above is settled.
- **Fortran** — This is more LFortran work than on Booth as Fortran already has GPU support through `do concurrent`. I'm working on the LFortran compiler at the moment to shore up some support around legacy f77 Fortran and also send a patch upstream so I can get Fortran compiling to GPUs properly rather than the limited subset. 

## Optimisation

The generated code is correct and is not winning benchmarks. Done so far:
instruction scheduling, constant folding, dead code elimination, SROA, mem2reg,
device-call inlining, and divergence-aware SSA register allocation on AMD.

- Loop-invariant code motion
- Occupancy tuning from register pressure
- A register allocator for the CPU backend, which is stack-everything today
- Honestly if there's academics out there who wants to see their algorithms implemented send them my way I'll happily try them out. 

## Architectures

BIR is target-independent and the backends are cleanly separated, so a new
target is an `isel` and an `emit`.

- **AMD RDNA 2/3/4** — done, native binaries. `--amdgpu-bin`
- **NVIDIA PTX** — done, validated on an RTX 4060 Ti. `--nvidia-ptx`
- **Tenstorrent Metalium** — done, C++ for Blackhole. `--tensix`
- **Tenstorrent RV32IM** — done, native ELF for Wormhole baby cores, with the
  TDF layer for L1 and NoC orchestration. `--rv-elf`
- **CPU x86-64** — done. Kernels run with no GPU, Triton matmul included.
  Correct first, fast later. Also the oracle the other backends are checked
  against. `--cpu`
- **RISC-V RV64** — done. A host object the same way `--cpu` is, run under
  qemu-riscv64 or on real hardware. It also doubles as the other half of the
  differential testing, since x86 and RISC-V disagreeing points at codegen. `--rv64`
- **Apple Metal** — compiles, hardware validation pending. `--metal`
- **Intel Arc / SPIR-V** — stub. Would give coverage across all four vendors.
- **ARM64** — on the radar, nothing written
- **RISC-V Vector** — for when GPUs are too mainstream and you want to run CUDA
  on a softcore

## Longer term

Longer term is a bit misleading but it's the closest title I can come up with. You'll see long term stuff happening constantly as its being worked towards but it clusters around a few key ideas. 

- **Mainframes** - Working with mainframes and on the z390 project as well as a few others is so amazing. The error messages are structured, the diagnostic tools are second to none and there is some genuinely great ideas I would like to try and port over to GPUs. These systems were created during a time when a fix meant putting in actual punch cards and it cost $1 per byte to run (in 1960's dollars by the way). That's alongside the fact that the earliest users of these machines were finance, science (such as NASA) and the government and were used in systems that could not fail no matter what. Although our machines have improved from the days when the government used to run on kilobytes the actual debugging and diagnostic tooling has not. You shouldn't need to be a kernel specialist to know what's gone wrong with the machine 

A list of what will be done:

- Adding Abend dumps for all architectures. This gives a structured error message to the user telling them what happened, where and what the machine was doing when it happened. It will require a fair bit of assembler to properly capture registers.
- Extending SNAP to all architectures as well. This allows the user to see what the machine is doing as the program is being executed even if it does not crash. It's been incredibly useful to debug "oh the code compiles and runs but doesn't look correct" which I used when creating MOA.
- Extending Sysprint for all architectures as well. 

- **GPU-less CI** - Booth has a CPU backend which in theory could be used as a kind of reference for people who wish to add this to their CI. The idea being that if it's valid CUDA or whatever you're using and Booth's BIR accepts it and lowers it correctly to CPUs it will also work on GPUs. This is more plumbing and plugging and will happen over time than anything

- **BIR text as a universal target** `--bir-in` allows for any language that can print text to target Booth. It has no linking, no ABI or anything else and the target is pretty small. 

- **Cobol** - Okay hear me out. I did warn you about the reasonable to the absurd thing. Now, fun fact the reason why COBOL is so ubiquitous and the Mainframe is still being used despite every consultancy selling "migrate awaaaay!" like it's easy is because of decimal arithmetic. Comp-3 packed decimal and IEEE 754-2008 decimal underpin essentially every important financial calculation on earth. 

What I am working on is making some packed decimal primitives and potentially showcasing it by getting COBOL to run through it. I'm unaware of any attempts to bring decimal arithmetic onto GPUs and I kinda live on the intersection between these two spaces anyway.

- **Compiling to the mainframe** - Right, the properly absurd one. IBM z has a vector facility, SIMD-128, and I already know HLASM and have spent a long time in z390. There is no technical reason a `--zarch` backend could not exist, emitting vector instructions the same way the AMD and RISC-V backends emit theirs. CUDA and Triton kernels running on a mainframe.

It sounds like a joke and it half is. The straight-faced version: banks have enormous z boxes and their data lives there, and they have essentially no GPUs anywhere near it. Moving a few hundred gigabytes of account data to a GPU cluster is a compliance conversation. Moving the compute to the data is not. If the decimal work above lands then this is where it would want to run anyway.

Also nobody has ever done it, and that is reason enough.

## Edge of what I understand

The bits where I genuinely do not know if it works and would have to find out.

- **Spatial compilation for Tensix** - Tenstorrent is not SIMT. It is a fabric of cores with a network on chip between them, and a kernel is not a grid of threads so much as a shape you lay out across the fabric. The TDF layer already treats regions, channels and NoC arcs as first-class, which is further than most people get, but mapping an arbitrary kernel onto a spatial fabric is placement and routing and buffer sizing and proving you cannot deadlock the network. That is a research problem and I am aware I am poking at it with a working compiler rather than a proof.

- **Error bounds as compiler output** - Compile a kernel so that alongside the answer it produces a bound on its own floating point error, via interval or affine arithmetic. It has been done in papers. As far as I know it has never been in a production GPU compiler. It pairs with deterministic mode, because knowing the number is always the same is only half of it and knowing how wrong it might be is the other half.

- **Verified soft float** - `runtime/soft_fp.c` is around 470 lines of C implementing IEEE-754 single precision with integer operations. No allocation, bounded loops, small enough to actually reason about. Verifying it properly, in Why3 or Frama-C or Coq, would give a formally verified soft float being used by a real compiler on real accelerator hardware. I am studying maths at the moment and this is the sort of thing that would make the two halves meet.
