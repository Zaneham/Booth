# Mainframe Curios

[← back to README](../README.md)

This is the bit where I admit I read a pile of z/OS manuals and got a little obsessed. The mainframe folks sorted out crash diagnostics and structured job output decades ago, and honestly a lot of it is nicer than what I had while squinting at broken GPU kernels. So I borrowed the ideas. I'm sure I'm doing them more clumsily than the people who invented them, and I'm still learning this stuff, but they're not a gimmick to me, they're the things that actually helped me find bugs.

## ABEND dumps

When a kernel faults you get a real dump, not a shrug. `src/runtime/bc_abend.*` gives GPU faults proper IBM-style completion codes (G0Cx, the GPU cousins of S0Cx), correlates the faulting address against tracked allocations, and prints a dispatch snapshot. It's wired into the HSA runtime and fires automatically off the system event callback, so a memory aperture violation tells you which buffer and which dispatch went wrong instead of just dying quietly. Live on the AMD/HSA path.

## SNAP (`--snap`)

A parameter dump, basically. The mainframe crowd had this in the 70s and I kept wishing for it while debugging. With `--snap` the AMD backend writes each kernel parameter's register value into a host-visible buffer on entry, so when things go sideways you can read the evidence instead of staring at disassembly like it owes you money. AMD only for now.

## SYSPRINT

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

See `examples/sysprint_kernel.cu` + `examples/launch_sysprint.c` for a full end-to-end demo. Works on the NVIDIA PTX and Tensix backends; the AMD path currently trips a regalloc bug on the byte-copy loop ([open issue](https://github.com/Zaneham/Booth/issues)), which gets its own follow-up.

## TDF (Tile DataFlow)

For the dataflow GPUs (Tenstorrent), the layer above BIR is modelled on CICS transactions: regions, channels, and NoC arcs as first-class compiler concepts, with L1 placement and a fission pass for multi-core kernels. Dump it with `--tdf`.
