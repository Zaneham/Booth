#include "tensix.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Data movement: reader kernels, writer kernels, host programs.
 * Three RISC-V cores per tile, connected by circular buffers.
 * The compute core thinks it's special. It isn't. */

/* ---- Helpers ---- */

static void dm_write(FILE *fp, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
}

/* ---- SSA Trace ---- */

/* Follow the SSA def-use chain from an address operand back to the
 * BIR_PARAM that originally defined it. GEPs, bitcasts, and other
 * address computations are transparent — we just want the base pointer.
 * Returns param index (subop) or -1 if the trail goes cold. */

static int trace_to_param(const bir_module_t *bir, uint32_t val)
{
    int guard = 16;
    while (guard-- > 0) {
        if (val == BIR_VAL_NONE || BIR_VAL_IS_CONST(val)) return -1;

        uint32_t idx = BIR_VAL_INDEX(val);
        if (idx >= bir->num_insts) return -1;

        const bir_inst_t *I = &bir->insts[idx];

        if (I->op == BIR_PARAM)
            return (int)I->subop;

        /* Transparent address computations — follow the base pointer */
        if (I->op == BIR_GEP || I->op == BIR_BITCAST ||
            I->op == BIR_INTTOPTR || I->op == BIR_PTRTOINT) {
            val = I->operands[0];
            continue;
        }

        /* Also follow loads from alloca'd param copies (struct params) */
        if (I->op == BIR_LOAD) {
            val = I->operands[0];
            continue;
        }

        return -1;
    }
    return -1;
}

/* ---- BIR Analysis ---- */

void tensix_analyze_datamov(const bir_module_t *bir, const tt_module_t *tt,
                            tt_dmov_t *dmov)
{
    memset(dmov, 0, sizeof(*dmov));

    /* Find first kernel function */
    uint32_t func_idx = 0;
    int found = 0;
    int guard = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && guard > 0; fi++, guard--) {
        if (tt->mfuncs[fi].is_kernel) {
            func_idx = tt->mfuncs[fi].bir_func;
            found = 1;
            break;
        }
    }
    if (!found || func_idx >= bir->num_funcs) return;

    const bir_func_t *F = &bir->funcs[func_idx];
    uint32_t first = F->first_block < bir->num_blocks
                     ? bir->blocks[F->first_block].first_inst : 0;

    /* Step 1: identify pointer parameters */
    uint32_t param_is_ptr[64];      /* 1 if param i is a global pointer */
    uint32_t param_inst[64];        /* BIR inst index of param i */
    uint32_t nparams = F->num_params < 64 ? F->num_params : 64;
    memset(param_is_ptr, 0, sizeof(param_is_ptr));
    memset(param_inst, 0, sizeof(param_inst));

    guard = 8192;
    for (uint32_t i = first; i < bir->num_insts && guard > 0; i++, guard--) {
        const bir_inst_t *I = &bir->insts[i];
        if (I->op != BIR_PARAM) continue;

        uint32_t pi = I->subop;
        if (pi >= nparams) continue;

        param_inst[pi] = i;

        if (I->type < bir->num_types) {
            const bir_type_t *T = &bir->types[I->type];
            if (T->kind == BIR_TYPE_PTR &&
                (T->addrspace == BIR_AS_GLOBAL || T->addrspace == BIR_AS_GENERIC))
                param_is_ptr[pi] = 1;
        }
    }

    /* Step 2: classify each pointer param as input or output by scanning
     * all instructions for loads/stores that trace back to it */
    uint32_t param_loaded[64];
    uint32_t param_stored[64];
    memset(param_loaded, 0, sizeof(param_loaded));
    memset(param_stored, 0, sizeof(param_stored));

    guard = 262144;
    for (uint32_t bi = 0; bi < F->num_blocks && guard > 0; bi++, guard--) {
        uint32_t bk_idx = F->first_block + bi;
        if (bk_idx >= bir->num_blocks) break;

        const bir_block_t *B = &bir->blocks[bk_idx];
        int bguard = 65536;
        for (uint32_t ii = 0; ii < B->num_insts && bguard > 0; ii++, bguard--) {
            const bir_inst_t *I = &bir->insts[B->first_inst + ii];

            if (I->op == BIR_LOAD) {
                /* ops[0] = address */
                int pi = trace_to_param(bir, I->operands[0]);
                if (pi >= 0 && (uint32_t)pi < nparams)
                    param_loaded[(uint32_t)pi] = 1;
            }
            else if (I->op == BIR_STORE) {
                /* ops[0] = value, ops[1] = address */
                int pi = trace_to_param(bir, I->operands[1]);
                if (pi >= 0 && (uint32_t)pi < nparams)
                    param_stored[(uint32_t)pi] = 1;
            }
        }
    }

    /* Step 3: build the buffer table */
    uint32_t in_idx = 0, out_idx = 0;

    for (uint32_t pi = 0; pi < nparams; pi++) {
        if (!param_is_ptr[pi]) continue;
        if (dmov->num_bufs >= TT_MAX_DMOV_BUFS) break;

        tt_dmov_buf_t *buf = &dmov->bufs[dmov->num_bufs];
        buf->bir_param = pi;
        buf->tile_size = 2048;  /* BF16 32x32 = 2048 bytes */
        buf->name = 0;          /* no param names in BIR */

        if (param_stored[pi]) {
            /* Output buffer: stored to (may also be loaded) */
            buf->is_output = 1;
            buf->cb_index = 16 + out_idx;
            buf->dst_row = 0;  /* output doesn't need a specific Dst row */
            out_idx++;
            dmov->num_outputs++;
        } else {
            /* Input buffer: loaded only */
            buf->is_output = 0;
            buf->cb_index = in_idx;
            buf->dst_row = in_idx * 16;  /* 16 Dst rows per tile */
            in_idx++;
            dmov->num_inputs++;
        }

        dmov->num_bufs++;
    }

    printf("  datamov: %u buffers (%u in, %u out)\n",
           dmov->num_bufs, dmov->num_inputs, dmov->num_outputs);
}

/* ---- Reader Kernel ---- */

int tensix_emit_reader(const tt_module_t *tt, const tt_dmov_t *dmov,
                       const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    dm_write(fp,
        "/* Generated by Booth — Tensix reader kernel.\n"
        " * Moves tiles from DRAM to L1 via the NoC.\n"
        " * Do not edit unless you enjoy debugging semaphore deadlocks. */\n"
        "\n"
        "#include \"dataflow_api.h\"\n"
        "\n"
        "void kernel_main() {\n");

    /* Runtime args: one address per input buffer, then num_tiles */
    uint32_t arg_idx = 0;
    int guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) continue;

        dm_write(fp,
            "    uint32_t src%u_addr = get_arg_val<uint32_t>(%u);\n",
            buf->cb_index, arg_idx);
        arg_idx++;
    }

    dm_write(fp,
        "    uint32_t num_tiles = get_arg_val<uint32_t>(%u);\n"
        "\n",
        arg_idx);

    /* CB declarations and address generators */
    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) continue;

        dm_write(fp,
            "    constexpr uint32_t cb%u = tt::CBIndex::c_%u;\n",
            buf->cb_index, buf->cb_index);
    }

    dm_write(fp,
        "    const uint32_t tile_bytes = get_tile_size(cb0);\n"
        "\n");

    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) continue;

        dm_write(fp,
            "    const InterleavedAddrGenFast<true> s%u = {\n"
            "        .bank_base_address = src%u_addr,\n"
            "        .page_size = tile_bytes,\n"
            "        .data_format = DataFormat::Float16_b\n"
            "    };\n",
            buf->cb_index, buf->cb_index);
    }

    /* Tile loop */
    dm_write(fp,
        "\n"
        "    for (uint32_t t = 0; t < num_tiles; t++) {\n");

    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) continue;

        dm_write(fp,
            "        cb_reserve_back(cb%u, 1);\n"
            "        uint32_t l1_wr%u = get_write_ptr(cb%u);\n"
            "        noc_async_read_tile(t, s%u, l1_wr%u);\n"
            "        noc_async_read_barrier();\n"
            "        cb_push_back(cb%u, 1);\n"
            "\n",
            buf->cb_index,
            buf->cb_index, buf->cb_index,
            buf->cb_index, buf->cb_index,
            buf->cb_index);
    }

    dm_write(fp,
        "    }\n"
        "}\n");

    fclose(fp);
    printf("wrote %s (reader, %u inputs)\n", path, dmov->num_inputs);

    (void)tt;
    return BC_OK;
}

/* ---- Writer Kernel ---- */

int tensix_emit_writer(const tt_module_t *tt, const tt_dmov_t *dmov,
                       const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    dm_write(fp,
        "/* Generated by Booth — Tensix writer kernel.\n"
        " * Drains tiles from L1 to DRAM via the NoC.\n"
        " * The thankless end of the pipeline. */\n"
        "\n"
        "#include \"dataflow_api.h\"\n"
        "\n"
        "void kernel_main() {\n");

    /* Runtime args: one address per output buffer, then num_tiles */
    uint32_t arg_idx = 0;
    int guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (!buf->is_output) continue;

        dm_write(fp,
            "    uint32_t dst%u_addr = get_arg_val<uint32_t>(%u);\n",
            buf->cb_index - 16, arg_idx);
        arg_idx++;
    }

    dm_write(fp,
        "    uint32_t num_tiles = get_arg_val<uint32_t>(%u);\n"
        "\n",
        arg_idx);

    /* CB declarations and address generators */
    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (!buf->is_output) continue;

        dm_write(fp,
            "    constexpr uint32_t cb_out%u = tt::CBIndex::c_%u;\n",
            buf->cb_index - 16, buf->cb_index);
    }

    dm_write(fp, "\n");

    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (!buf->is_output) continue;

        uint32_t out_n = buf->cb_index - 16;
        dm_write(fp,
            "    const uint32_t tile_bytes%u = get_tile_size(cb_out%u);\n"
            "    const InterleavedAddrGenFast<true> d%u = {\n"
            "        .bank_base_address = dst%u_addr,\n"
            "        .page_size = tile_bytes%u,\n"
            "        .data_format = DataFormat::Float16_b\n"
            "    };\n",
            out_n, out_n,
            out_n,
            out_n,
            out_n);
    }

    /* Tile loop */
    dm_write(fp,
        "\n"
        "    for (uint32_t t = 0; t < num_tiles; t++) {\n");

    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (!buf->is_output) continue;

        uint32_t out_n = buf->cb_index - 16;
        dm_write(fp,
            "        cb_wait_front(cb_out%u, 1);\n"
            "        uint32_t l1_rd%u = get_read_ptr(cb_out%u);\n"
            "        noc_async_write_tile(t, d%u, l1_rd%u);\n"
            "        noc_async_write_barrier();\n"
            "        cb_pop_front(cb_out%u, 1);\n"
            "\n",
            out_n,
            out_n, out_n,
            out_n, out_n,
            out_n);
    }

    dm_write(fp,
        "    }\n"
        "}\n");

    fclose(fp);
    printf("wrote %s (writer, %u outputs)\n", path, dmov->num_outputs);

    (void)tt;
    return BC_OK;
}

/* ---- Host Program ---- */

int tensix_emit_host_full(const tt_module_t *tt, const tt_dmov_t *dmov,
                          const char *host_path, const char *reader_path,
                          const char *compute_path, const char *writer_path)
{
    FILE *fp = fopen(host_path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", host_path);
        return BC_ERR_IO;
    }

    /* Header */
    dm_write(fp,
        "/* Generated by Booth — Tensix host program.\n"
        " * Three kernels, many cores, one dream. */\n"
        "\n"
        "#include \"tt_metal/host_api.hpp\"\n"
        "#include \"tt_metal/common/constants.hpp\"\n"
        "\n"
        "using namespace tt;\n"
        "using namespace tt::tt_metal;\n"
        "\n"
        "int main() {\n"
        "    Device *device = CreateDevice(0);\n"
        "    CommandQueue &cq = device->command_queue();\n"
        "    Program program = CreateProgram();\n"
        "\n"
        "    /* %dx%d Wormhole B0 compute grid */\n"
        "    CoreRange cores({0, 0}, {%d, %d});\n"
        "    constexpr uint32_t num_cores = %d;\n"
        "\n",
        TT_WH_CORES_X, TT_WH_CORES_Y,
        TT_WH_CORES_X - 1, TT_WH_CORES_Y - 1,
        TT_WH_NUM_CORES);

    /* Tile geometry */
    dm_write(fp,
        "    /* Tile geometry — 32x32 tiles, BF16 */\n"
        "    constexpr uint32_t tile_size = 2048;\n"
        "    constexpr uint32_t elements_per_tile = 1024;  /* 32*32 */\n"
        "    constexpr uint32_t total_elements = 1024 * 1024;  /* TODO: from launch config */\n"
        "    uint32_t num_tiles = (total_elements + elements_per_tile - 1) / elements_per_tile;\n"
        "    uint32_t tiles_per_core = (num_tiles + num_cores - 1) / num_cores;\n"
        "\n");

    /* DRAM buffer allocation */
    dm_write(fp,
        "    /* DRAM buffers — interleaved across banks */\n"
        "    uint32_t dram_buf_size = num_tiles * tile_size;\n"
        "    InterleavedBufferConfig buf_config{\n"
        "        .device = device, .size = dram_buf_size,\n"
        "        .page_size = tile_size, .buffer_type = BufferType::DRAM\n"
        "    };\n");

    /* One buffer per dmov buf */
    int guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) {
            dm_write(fp,
                "    auto buf_out%u = CreateBuffer(buf_config);  /* output %u */\n",
                buf->cb_index - 16, buf->cb_index - 16);
        } else {
            dm_write(fp,
                "    auto buf_in%u = CreateBuffer(buf_config);  /* input %u */\n",
                buf->cb_index, buf->cb_index);
        }
    }

    dm_write(fp, "\n");

    /* Circular buffers */
    dm_write(fp, "    /* Circular buffers — the semaphore highway between cores */\n");

    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        dm_write(fp,
            "    CircularBufferConfig cb%u_config = CircularBufferConfig(\n"
            "        tile_size, {{tt::CBIndex::c_%u, tt::DataFormat::Float16_b}})\n"
            "        .set_page_size(tt::CBIndex::c_%u, tile_size);\n"
            "    CreateCircularBuffer(program, cores, cb%u_config);\n"
            "\n",
            buf->cb_index, buf->cb_index,
            buf->cb_index,
            buf->cb_index);
    }

    /* Three kernels */
    dm_write(fp,
        "    /* Three kernels: reader (RISCV_0), compute (SFPU), writer (RISCV_1) */\n"
        "    auto reader = CreateKernel(program, \"%s\", cores,\n"
        "        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,\n"
        "                           .noc = NOC::RISCV_0_default});\n"
        "\n"
        "    auto compute = CreateKernel(program, \"%s\", cores,\n"
        "        ComputeConfig{});\n"
        "\n"
        "    auto writer = CreateKernel(program, \"%s\", cores,\n"
        "        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,\n"
        "                           .noc = NOC::RISCV_1_default});\n"
        "\n",
        reader_path, compute_path, writer_path);

    /* Per-core runtime args */
    dm_write(fp,
        "    /* Per-core runtime args */\n"
        "    for (uint32_t y = 0; y < %d; y++) {\n"
        "        for (uint32_t x = 0; x < %d; x++) {\n"
        "            CoreCoord core = {x, y};\n"
        "\n",
        TT_WH_CORES_Y, TT_WH_CORES_X);

    /* Reader args: all input addresses + tiles_per_core */
    dm_write(fp, "            SetRuntimeArgs(program, reader, core, {");
    int first = 1;
    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (buf->is_output) continue;
        if (!first) dm_write(fp, ", ");
        dm_write(fp, "buf_in%u->address()", buf->cb_index);
        first = 0;
    }
    if (!first) dm_write(fp, ", ");
    dm_write(fp, "tiles_per_core});\n");

    /* Compute args: tiles_per_core */
    dm_write(fp,
        "            SetRuntimeArgs(program, compute, core, {tiles_per_core});\n");

    /* Writer args: all output addresses + tiles_per_core */
    dm_write(fp, "            SetRuntimeArgs(program, writer, core, {");
    first = 1;
    guard = TT_MAX_DMOV_BUFS;
    for (uint32_t i = 0; i < dmov->num_bufs && guard > 0; i++, guard--) {
        const tt_dmov_buf_t *buf = &dmov->bufs[i];
        if (!buf->is_output) continue;
        if (!first) dm_write(fp, ", ");
        dm_write(fp, "buf_out%u->address()", buf->cb_index - 16);
        first = 0;
    }
    if (!first) dm_write(fp, ", ");
    dm_write(fp, "tiles_per_core});\n");

    dm_write(fp,
        "        }\n"
        "    }\n"
        "\n");

    /* Launch */
    dm_write(fp,
        "    /* TODO: EnqueueWriteBuffer for input data */\n"
        "    EnqueueProgram(cq, program, false);\n"
        "    Finish(cq);\n"
        "    /* TODO: EnqueueReadBuffer for output data */\n"
        "\n"
        "    CloseDevice(device);\n"
        "    return 0;\n"
        "}\n");

    fclose(fp);
    printf("wrote %s (host, %u buffers, 3 kernels)\n",
           host_path, dmov->num_bufs);

    (void)tt;
    return BC_OK;
}
