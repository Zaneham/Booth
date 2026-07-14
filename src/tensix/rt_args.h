#ifndef BARRACUDA_RT_ARGS_H
#define BARRACUDA_RT_ARGS_H

/*
 * Tensix kernel runtime arguments layout.
 *
 * Shared contract between two pieces:
 *
 *   1. The Booth RV32IM isel (rv_isel.c), which emits LW from
 *      these offsets when lowering BIR_PARAM and the four CUDA
 *      coordinate intrinsics (threadIdx, blockIdx, blockDim,
 *      gridDim) into baby-core code.
 *
 *   2. The host launcher (examples/koyeb_tensix_launch.cpp and any
 *      future deploy harness), which writes the matching struct
 *      into L1 at TD_L1_RTARG_BASE before dispatching the kernel.
 *
 * Both sides must agree on these offsets exactly. Change them in
 * one place and the kernel reads garbage; the layout is therefore
 * versioned by being centralised here and consumed by name from
 * everywhere else.
 *
 * Single-thread-per-core dispatch model: each baby core runs one
 * program and reads threadIdx = (0, 0, 0) on entry. CUDA kernels
 * that depend on threadIdx for parallelism still compile and run
 * correctly, they just execute the "thread 0 alone" trajectory.
 * Many-threads-per-core dispatch is a future task that has two
 * options: (a) the launcher dispatches the same ELF N times with
 * different thread_id_x values, which is correct but slow; (b)
 * we lower threadIdx-dependent code paths to SFPU SIMD, which is
 * fast but is a whole new emitter. Both live well outside this
 * layer; the runtime arg block accommodates either.
 *
 * Bytes 0..47   coordinate intrinsics (4 x 3 ints)
 * Bytes 48..111 kernel parameters (16 x uint32_t slots)
 * Bytes 112..   reserved for future expansion
 *
 * Total used: 112 bytes. Reserved slab: TD_L1_RTARG_SIZE = 256.
 */

/* Coordinate-intrinsic offsets. The subop on a BIR_THREAD_ID etc.
 * inst gives the dimension (0 = x, 1 = y, 2 = z); we compute the
 * byte offset as base + subop*4. */
#define RT_ARG_OFF_TID_X       0u    /* threadIdx.x */
#define RT_ARG_OFF_TID_Y       4u
#define RT_ARG_OFF_TID_Z       8u
#define RT_ARG_OFF_BID_X       12u   /* blockIdx.x  */
#define RT_ARG_OFF_BID_Y       16u
#define RT_ARG_OFF_BID_Z       20u
#define RT_ARG_OFF_BDIM_X      24u   /* blockDim.x  */
#define RT_ARG_OFF_BDIM_Y      28u
#define RT_ARG_OFF_BDIM_Z      32u
#define RT_ARG_OFF_GDIM_X      36u   /* gridDim.x   */
#define RT_ARG_OFF_GDIM_Y      40u
#define RT_ARG_OFF_GDIM_Z      44u

/* Kernel parameter slot block. Each kernel param (BIR_PARAM i)
 * occupies one 4-byte slot. The cap was 16 initially (the TPF/CBRW
 * number used elsewhere in TDF), but real CUDA kernels routinely
 * have 17-26 parameters once you count pointers, sizes, and
 * scalar inputs separately. Bumped to 32 to cover the actual
 * working set. Still fits inside the 256-byte L1 slab:
 *   48 (intrinsics) + 32 * 4 (kernel args) = 176 bytes used.
 *
 * Anyone hitting 32 should think hard about whether they want a
 * pointer to a struct in DRAM instead. */
#define RT_ARG_KERNEL_BASE     48u
#define RT_ARG_N_KERNEL_SLOTS  32u
#define RT_ARG_OFF_KARG(i)     (RT_ARG_KERNEL_BASE + (i) * 4u)

#define RT_ARG_TOTAL_BYTES     (RT_ARG_KERNEL_BASE + RT_ARG_N_KERNEL_SLOTS * 4u)

#endif /* BARRACUDA_RT_ARGS_H */
