#ifndef BARRACUDA_METAL_H
#define BARRACUDA_METAL_H

#include "bir.h"

/* Apple Metal backend, which is to say the part of BarraCUDA that lowers
 * your perfectly serviceable CUDA source into the Metal Shading Language
 * so that Apple's own driver can compile it at load time, in exactly the
 * way we already convince the NVIDIA driver to compile PTX while we
 * stand a respectful distance away and pretend we did all the difficult
 * parts. We do not descend any deeper than MSL into AIR or AGX bytecode,
 * because Apple did not invite anyone in there and the Asahi Linux folk
 * have already spent literal years doing the rest of that homework on
 * everyone else's behalf.
 *
 * In the spirit of every empire that ever planted a flag on a beach and
 * declared the territory subdued, this backend takes the position that
 * directly emitting Metal-flavoured C++ text counts as colonising the
 * Apple GPU, whereas any deeper claim would require a paperwork excursion
 * into the AIR specification we do not currently feel up to. */

#define BC_ERR_METAL    -8

/* ---- Limits ---- */

#define MTL_MAX_PARAMS  32                  /* per-kernel parameter cap */
#define MTL_MAX_KERNS   64                  /* total kernels per module */
#define MTL_MAX_OUT     (1 * 1024 * 1024)   /* output MSL text buffer */

/* ---- Builtin Bitmask ----
 * The set of MSL-attribute-driven kernel parameters a given kernel
 * actually needs, recorded during metal_compile so the emit phase
 * only declares the ones the kernel asks for. MSL is more particular
 * than CUDA about not synthesising parameters you never reference. */

#define MTL_BI_TID      0x01    /* threadIdx,  thread_position_in_threadgroup */
#define MTL_BI_BID      0x02    /* blockIdx,   threadgroup_position_in_grid */
#define MTL_BI_BDIM     0x04    /* blockDim,   threads_per_threadgroup */
#define MTL_BI_GDIM     0x08    /* gridDim,    threadgroups_per_grid */

/* ---- Parameter Descriptor ----
 * One per kernel argument, in declaration order. For pointer parameters
 * we carry the address space across so the emitter can write `device`,
 * `constant`, or `threadgroup` in the MSL signature. MSL refuses to
 * accept a bare `float *`, on the entirely reasonable grounds that it
 * does not know where the float is supposed to live. */

typedef struct {
    uint32_t    name;       /* string table offset, or 0 for synthetic */
    uint32_t    type;       /* BIR type index */
    uint8_t     addrspace;  /* bir_addrspace_t, meaningful for pointer types */
    uint8_t     is_const;   /* pointer to const-qualified pointee */
    uint8_t     pad[2];
} mtl_param_t;

/* ---- Kernel Descriptor ----
 * One per __global__ or __device__ function we plan to emit. For
 * __device__ functions the is_kern bit is zero and the emitter
 * produces a plain MSL function without the `kernel` qualifier or
 * any of the [[buffer(N)]] machinery. */

typedef struct {
    uint32_t    name;               /* string table offset (function name) */
    uint32_t    bir_func;           /* index into M->funcs */
    uint16_t    num_params;
    uint16_t    is_kern;            /* 1 = __global__ kernel, 0 = __device__ */
    uint32_t    builtins;           /* MTL_BI_* bitmask of references seen */
    uint32_t    launch_bounds_max;  /* mirrors __launch_bounds__ if present */
    uint32_t    launch_bounds_min;
    mtl_param_t params[MTL_MAX_PARAMS];
} mtl_kern_t;

/* ---- Module ----
 * The whole MSL output state in one struct. Fixed-size pools, no malloc,
 * no surprise allocations on the hot path. The output buffer is sized
 * to fit any reasonable kernel and then some, and overflows return an
 * error rather than scribbling past the end like an over-enthusiastic
 * intern with a permanent marker. */

typedef struct metal_module_t {
    const bir_module_t *bir;

    mtl_kern_t  kerns[MTL_MAX_KERNS];
    uint32_t    num_kerns;

    char        out_buf[MTL_MAX_OUT];
    uint32_t    out_len;
} metal_module_t;

/* ---- Public API ---- */

int  metal_compile(const bir_module_t *bir, metal_module_t *mm);
int  metal_emit_msl(metal_module_t *mm, const char *path);

#endif /* BARRACUDA_METAL_H */
