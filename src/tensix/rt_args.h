#ifndef BARRACUDA_RT_ARGS_H
#define BARRACUDA_RT_ARGS_H

/*
 * Tensix kernel runtime arguments layout. This is the shared contract between
 * the Booth RV32IM isel (rv_isel.c), which emits LW from these offsets when
 * lowering BIR_PARAM and the four CUDA coordinate intrinsics (threadIdx,
 * blockIdx, blockDim, gridDim), and the host launcher
 * (examples/koyeb_tensix_launch.cpp and any future deploy harness), which writes
 * the matching struct into L1 at TD_L1_RTARG_BASE before dispatch. Both sides
 * must agree on the offsets exactly, so the layout is versioned by living here
 * and being consumed by name everywhere else; change it in one place or the
 * kernel reads garbage.
 *
 * The dispatch model is single-thread-per-core: each baby core runs one program
 * and reads threadIdx = (0, 0, 0) on entry, so CUDA kernels that depend on
 * threadIdx for parallelism still compile and run correctly, they just execute
 * the "thread 0 alone" trajectory. Many-threads-per-core is future work with two
 * options, either the launcher dispatches the same ELF N times with different
 * thread_id_x values (correct but slow), or we lower threadIdx-dependent paths
 * to SFPU SIMD (fast but a whole new emitter); both live well outside this layer
 * and the runtime arg block accommodates either.
 *
 * Bytes 0..47 hold the coordinate intrinsics (4 x 3 ints), bytes 48..111 the
 * kernel parameters (16 x uint32_t slots), and bytes 112 onward are reserved.
 * Total used is 112 bytes inside the 256-byte reserved slab TD_L1_RTARG_SIZE.
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

/* Kernel parameter slot block. Each kernel param (BIR_PARAM i) occupies one
 * 4-byte slot. The cap was 16 initially (the TPF/CBRW number used elsewhere in
 * TDF), but real CUDA kernels routinely have 17-26 parameters once you count
 * pointers, sizes, and scalar inputs separately, so it was bumped to 32 to cover
 * the actual working set. That still fits inside the 256-byte L1 slab: 48
 * (intrinsics) + 32 * 4 (kernel args) = 176 bytes used. Anyone hitting 32 should
 * think hard about whether they want a pointer to a struct in DRAM instead. */
#define RT_ARG_KERNEL_BASE     48u
#define RT_ARG_N_KERNEL_SLOTS  32u
#define RT_ARG_OFF_KARG(i)     (RT_ARG_KERNEL_BASE + (i) * 4u)

#define RT_ARG_TOTAL_BYTES     (RT_ARG_KERNEL_BASE + RT_ARG_N_KERNEL_SLOTS * 4u)

#endif /* BARRACUDA_RT_ARGS_H */
