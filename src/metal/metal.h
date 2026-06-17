#ifndef BARRACUDA_METAL_H
#define BARRACUDA_METAL_H

#include "bir.h"
#include "bir_struct.h"

/* Apple Metal backend. Lowers CUDA source to Metal Shading Language text;
 * the Metal toolchain compiles that to AIR at load time. Does not emit AIR
 * or AGX bytecode directly. */

#define BC_ERR_METAL    -8

/* ---- Limits ---- */

#define MTL_MAX_PARAMS  32                  /* per-kernel parameter cap */
#define MTL_MAX_KERNS   64                  /* total kernels per module */
#define MTL_MAX_OUT     (1 * 1024 * 1024)   /* output MSL text buffer */

/* ---- Builtin bitmask ----
 * MSL-attribute-driven kernel parameters a kernel needs, recorded during
 * metal_compile so emit only declares the ones referenced. MSL will not
 * synthesise unreferenced parameters. */

#define MTL_BI_TID      0x01    /* threadIdx,  thread_position_in_threadgroup */
#define MTL_BI_BID      0x02    /* blockIdx,   threadgroup_position_in_grid */
#define MTL_BI_BDIM     0x04    /* blockDim,   threads_per_threadgroup */
#define MTL_BI_GDIM     0x08    /* gridDim,    threadgroups_per_grid */

/* ---- Parameter descriptor ----
 * One per kernel argument, in declaration order. For pointer parameters the
 * address space is carried across so the emitter can write `device`,
 * `constant`, or `threadgroup` in the MSL signature; MSL requires an address
 * space on every pointer parameter. */

typedef struct {
    uint32_t    name;       /* string table offset, or 0 for synthetic */
    uint32_t    type;       /* BIR type index */
    uint8_t     addrspace;  /* bir_addrspace_t, meaningful for pointer types */
    uint8_t     is_const;   /* pointer to const-qualified pointee */
    uint8_t     pad[2];
} mtl_param_t;

/* ---- Kernel descriptor ----
 * One per __global__ or __device__ function to emit. For __device__
 * functions is_kern is zero and the emitter produces a plain MSL function
 * without the `kernel` qualifier or [[buffer(N)]] attributes. */

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
 * MSL output state in one struct. Fixed-size pools, no malloc. Output buffer
 * overflow returns an error rather than writing past the end. */

typedef struct metal_module_t {
    const bir_module_t *bir;

    mtl_kern_t  kerns[MTL_MAX_KERNS];
    uint32_t    num_kerns;

    char        out_buf[MTL_MAX_OUT];
    uint32_t    out_len;

    /* Transient emit state. The structure tree is rebuilt for each kernel as
     * it is written out; indent is the current nesting depth. Both are scratch,
     * meaningful only mid-emit. */
    bst_tree_t  tree;
    uint32_t    indent;
} metal_module_t;

/* ---- Public API ---- */

int  metal_compile(const bir_module_t *bir, metal_module_t *mm);
int  metal_emit_msl(metal_module_t *mm, const char *path);

#endif /* BARRACUDA_METAL_H */
