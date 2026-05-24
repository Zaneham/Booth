/* sysprint_kernel.cu -- a kernel that emits a SYSPRINT record.
 *
 * Each thread computes c[i] = a[i] + b[i]. Thread 0 additionally
 * emits a single SYSPRINT record reporting the value it
 * produced; the host drains that record after the kernel
 * completes.
 *
 * The single-thread emit avoids the concurrent atomicAdd path
 * the per-thread variant would exercise; the per-thread version
 * is fine semantically but currently exposes an AMD backend
 * regalloc bug. The single-thread demo is enough to verify the
 * end-to-end wiring. */

#include "sysprint_device.h"
#include "sysprint_classes.h"

__global__ void sp_demo(bc_sp_buf_t *sp, float *c,
                        const float *a, const float *b, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    c[tid] = a[tid] + b[tid];

    if (tid == 0) {
        const char *msg = "sp_demo done";
        BC_SYSPRINT(sp, CLS_RESULT, msg, 12u);
    }
}
