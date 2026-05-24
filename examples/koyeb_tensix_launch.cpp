/*
 * koyeb_tensix_launch.cpp
 *
 * Host launcher template for running a BarraCUDA-produced RV32IM
 * ELF on a Tenstorrent Wormhole n300, intended for Koyeb deploys
 * with a tt-metal SDK installed on the runner. This is a TEMPLATE,
 * not a built artefact in the BarraCUDA tree: compile it on the
 * deploy box where tt-metal headers exist.
 *
 * Build (on the deploy host with tt-metal installed):
 *     export TT_METAL_HOME=/path/to/tt-metal
 *     g++ -std=c++17 \
 *         -I$TT_METAL_HOME/tt_metal -I$TT_METAL_HOME \
 *         koyeb_tensix_launch.cpp \
 *         -L$TT_METAL_HOME/build/lib -ltt_metal \
 *         -o koyeb_tensix_launch
 *
 * Then drop the BarraCUDA-produced ELF alongside and invoke:
 *     ./koyeb_tensix_launch ./kernel.elf
 *
 * What this template does:
 *   1. Opens the n300 device via tt-metal's device API.
 *   2. Allocates one DRAM buffer for the output integer.
 *   3. Loads the BarraCUDA ELF onto the compute baby core (T0).
 *   4. Sets runtime args matching the kernel's parameter shape.
 *      (For a kernel `void k(int *out, int a, int b)`: out pointer,
 *      then the two scalars.)
 *   5. Launches, waits, reads back, prints the result.
 *
 * What the template does NOT yet do:
 *   - Float arithmetic (BarraCUDA's RV32IM path has no soft-float
 *     runtime linked yet; integer kernels only for the first cut).
 *   - Multi-core dispatch (single Tensix, single baby core).
 *   - Three-kernel reader/compute/writer fission (our --rv-elf
 *     emits one body per kernel; once BIR fission lands the
 *     launcher gains a reader and writer to load alongside).
 *   - Error recovery beyond exit-on-failure.
 *
 * The actual tt-metal API names below are from the public docs at
 * https://docs.tenstorrent.com/tt-metal/ and may need adjustment
 * to match the SDK version pinned by the deploy host. Treat the
 * skeleton as a starting point, not a guarantee.
 */

#include <tt_metal/host_api.hpp>
#include <tt_metal/impl/device/device.hpp>
#include <tt_metal/impl/buffers/buffer.hpp>
#include <tt_metal/impl/kernels/kernel.hpp>
#include <tt_metal/impl/program/program.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

/*
 * Runtime args layout shared with the BarraCUDA isel. Both sides
 * must keep this in sync; the canonical version of these offsets
 * lives in src/tensix/rt_args.h in the compiler tree. They are
 * duplicated here because the launcher does not include the
 * compiler's headers directly. If you change one, change both.
 */
struct kernel_rt_args_t {
    /* CUDA coordinate intrinsics, bytes 0..47 */
    uint32_t thread_id[3];      /* 0, 4, 8     */
    uint32_t block_id[3];       /* 12, 16, 20  */
    uint32_t block_dim[3];      /* 24, 28, 32  */
    uint32_t grid_dim[3];       /* 36, 40, 44  */
    /* Kernel parameters, bytes 48..175 */
    uint32_t kernel_args[32];
};
static_assert(sizeof(kernel_rt_args_t) == 176,
              "rt_args layout drift; must match src/tensix/rt_args.h");

/* Base L1 address where the BarraCUDA isel reads from. Lifted
 * from tdf.h's TD_L1_RTARG_BASE. */
constexpr uint32_t BC_RT_ARGS_L1_ADDR = 0x00008000u;

using namespace tt;
using namespace tt::tt_metal;

/* Read the entire ELF file into a byte vector. The contents are
 * what BarraCUDA wrote with --rv-elf and the tt-metal kernel-loader
 * memcpys the PT_LOAD payload onto the baby core's L1. */
static std::vector<uint8_t> slurp_elf(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " kernel.elf\n";
        return 1;
    }

    /* ---- Open the device. ---- */
    constexpr int device_id = 0;
    Device *device = CreateDevice(device_id);

    /* The kernel we will run is a simple integer add. The expected
     * source for the matching CUDA was:
     *
     *     __global__ void add_kernel(int *out, int a, int b) {
     *         *out = a + b;
     *     }
     *
     * BarraCUDA compiles this to a single baby-core function whose
     * RV32IM body lives in the supplied ELF. */
    constexpr CoreCoord core = {0, 0};
    Program program = CreateProgram();

    /* ---- One DRAM buffer for the output integer. ---- */
    constexpr uint32_t out_bytes = sizeof(int32_t);
    InterleavedBufferConfig dram_cfg{
        .device      = device,
        .size        = out_bytes,
        .page_size   = out_bytes,
        .buffer_type = BufferType::DRAM};
    auto out_buf = CreateBuffer(dram_cfg);

    /* ---- Load the BarraCUDA ELF onto the baby compute core. ----
     *
     * tt-metal's standard path is to point CreateKernel at a .cpp
     * source file that SFPI gcc compiles. Loading a pre-built ELF
     * directly is a lower-level interface; the exact API differs by
     * SDK version. Two common shapes:
     *
     *   (a) CreateKernelFromBinary(program, elf_path, core, ...)
     *   (b) program.add_kernel_binary(elf_path, core, ...)
     *
     * If neither is exposed in your SDK build, the fallback is to
     * stage the ELF where SFPI expects compiled artefacts and let
     * tt-metal's compile-or-load path pick it up. The user will
     * need to consult their pinned tt-metal version's docs for the
     * exact spelling. */
    auto elf_bytes = slurp_elf(argv[1]);
    KernelHandle kern = CreateKernelFromBinary(
        program,
        elf_bytes.data(),
        elf_bytes.size(),
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc       = NOC::RISCV_0_default});

    /* ---- Runtime args ----
     *
     * Build the rt_args struct in host memory, then ship it to L1
     * at BC_RT_ARGS_L1_ADDR before launching. Single-thread-per-
     * core dispatch, so thread_id stays at (0,0,0); block_id is
     * (0,0,0) for this one-core test, would vary per core for a
     * multi-core grid; block_dim/grid_dim describe the launch
     * geometry (here a 1x1x1 grid of 1x1x1 threads).
     *
     * Kernel signature for the add_kernel test is
     *   void k(int *out, int a, int b)
     * so kernel_args[0] = out pointer, [1] = a, [2] = b. */
    kernel_rt_args_t rt_args = {};
    rt_args.thread_id[0] = 0;
    rt_args.thread_id[1] = 0;
    rt_args.thread_id[2] = 0;
    rt_args.block_id[0]  = 0;
    rt_args.block_id[1]  = 0;
    rt_args.block_id[2]  = 0;
    rt_args.block_dim[0] = 1;
    rt_args.block_dim[1] = 1;
    rt_args.block_dim[2] = 1;
    rt_args.grid_dim[0]  = 1;
    rt_args.grid_dim[1]  = 1;
    rt_args.grid_dim[2]  = 1;
    rt_args.kernel_args[0] = static_cast<uint32_t>(out_buf->address());
    rt_args.kernel_args[1] = static_cast<uint32_t>(11);    /* a */
    rt_args.kernel_args[2] = static_cast<uint32_t>(31);    /* b */

    /* Write the struct into the compute core's L1. tt-metal's
     * low-level API has WriteToDeviceL1; the exact spelling differs
     * by SDK version, this is the documented entry point in recent
     * releases. The compiler-emitted ELF will LW from
     * BC_RT_ARGS_L1_ADDR + offset for each field. */
    detail::WriteToDeviceL1(device, core, BC_RT_ARGS_L1_ADDR,
                            reinterpret_cast<uint32_t *>(&rt_args),
                            sizeof(rt_args));

    /* ---- Enqueue, wait, read back. ---- */
    CommandQueue &cq = device->command_queue();
    EnqueueProgram(cq, program, /*blocking=*/false);
    Finish(cq);

    std::vector<uint8_t> readback(out_bytes);
    EnqueueReadBuffer(cq, out_buf, readback.data(), /*blocking=*/true);

    int32_t result;
    std::memcpy(&result, readback.data(), sizeof(result));
    std::cout << "kernel result: " << result
              << "  (expected " << (a_val + b_val) << ")\n";

    CloseDevice(device);
    return (result == a_val + b_val) ? 0 : 1;
}
