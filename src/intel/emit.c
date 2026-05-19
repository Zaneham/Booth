#include "intel.h"

/* This file sits inside the Intel directory in much the same way that
 * an unfinished sculpture sits on its plinth in the back of the studio,
 * meaning that it is mostly a public statement of future intent rather
 * than anything one would care to admit to having produced in its
 * current state. The plan, eventually, is to lower BIR into a proper
 * SPIR-V module, serialise the bytes to disk in whatever endianness the
 * specification has decided on this week, and then let the Intel driver
 * pick the module up via Level Zero or OpenCL the first time the host
 * program actually decides to dispatch the kernel.
 *
 * CUDA threads and blocks translate over to SPIR-V's work item and
 * work group model with very little fuss, since both abstractions are
 * the same abstraction wearing different brand-approved jackets. The
 * threadIdx values arrive via an OpLoad on the LocalInvocationId
 * builtin; blockIdx comes from WorkgroupId in exactly the same fashion;
 * blockDim comes from WorkgroupSize and is no more interesting than
 * any of the others; __syncthreads becomes an OpControlBarrier with
 * Workgroup execution scope, Workgroup memory scope, and the
 * AcquireRelease and WorkgroupMemory semantics flags set, because
 * SPIR-V believes very strongly that there is no synchronisation
 * primitive that cannot be improved by adding three more arguments to
 * it. The __shared__ qualifier maps onto the Workgroup storage class,
 * global memory takes the CrossWorkgroup storage class with the Kernel
 * capability declared at module scope, and all of the remaining
 * arithmetic is the kind of grunt work that compilers exist to perform
 * so that humans do not have to.
 *
 * The various Xe variants will mostly share their isel and their
 * emitter, with differences amounting to SIMD width preferences and to
 * the precise set of optional SPIR-V extensions the driver of the day
 * is prepared to consume, including the bf16 conversion family and the
 * dpas dot-product accelerator that the HPC silicon brings to the
 * party and the integrated parts very deliberately do not. */

int
intel_compile(const bir_module_t *bir, intel_module_t *im,
              intel_target_t target)
{
    (void)bir;
    if (im) im->target = target;
    return BC_ERR_INTEL; /* not yet implemented */
}

int
intel_emit_spirv(const intel_module_t *im, const char *path)
{
    (void)im;
    (void)path;
    return BC_ERR_INTEL; /* not yet implemented */
}
