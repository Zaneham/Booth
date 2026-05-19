#include "metal.h"

/* This file is a polite placeholder for the Apple Metal backend, by which
 * we mean an act of cartography, a declaration that we have staked a
 * claim on this particular patch of the source tree without yet doing
 * any of the actual work that the claim implies. When the time comes to
 * fill in the rest, the plan is to lower BIR into MSL text and to hand
 * that text to Apple's own Metal compiler at runtime, much as we hand
 * PTX text to the NVIDIA driver and try to look casual about it.
 *
 * The CUDA to MSL mappings are mostly tedious rather than difficult,
 * because both languages were designed by people who had encountered the
 * same hardware and had then independently reached broadly the same
 * conclusions about what programmers ought to be allowed to say about it.
 * Where CUDA writes threadIdx.x, MSL prefers thread_position_in_threadgroup.x,
 * which is more letters but is also more honest about what is actually
 * happening; blockIdx.x becomes threadgroup_position_in_grid.x in the
 * same vein; blockDim.x becomes threads_per_threadgroup.x; the eternal
 * __syncthreads becomes threadgroup_barrier(mem_flags::mem_threadgroup)
 * and politely insists that you tell it which sort of memory you wish
 * to synchronise, since it has principles. The __shared__ qualifier
 * becomes the threadgroup qualifier, global memory becomes the device
 * qualifier, and almost everything else is bookkeeping carried out at
 * the leisure of the implementer.
 *
 * No part of any of this requires reverse engineering anything at all,
 * because the Metal Shading Language Specification is a publicly
 * downloadable PDF that was written by people who, on the whole,
 * remembered that other humans were going to read it. */

int
metal_compile(const bir_module_t *bir, metal_module_t *mm)
{
    (void)bir;
    (void)mm;
    return BC_ERR_METAL; /* not yet implemented */
}

int
metal_emit_msl(const metal_module_t *mm, const char *path)
{
    (void)mm;
    (void)path;
    return BC_ERR_METAL; /* not yet implemented */
}
