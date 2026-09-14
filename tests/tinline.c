/* tinline.c -- __device__ call inlining (issue #101).
 * The GPU backends have no calling convention for device functions, so the
 * inliner must splice every call away before isel. The scalar CPU backend
 * emits real calls and must be left alone. */

#include "tharns.h"
#include <stdlib.h>
#include "bir.h"
#include "bir_inline.h"

static char obuf[TH_BUFSZ];

/* On a GPU target every device call is inlined, so no call survives in the IR
 * (the standalone device bodies are inlined into their callers too). */
static void inl01(void)
{
    int rc = th_run(BC_BIN " --amdgpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* And the kernel compiles the whole way to a .hsaco with the calls gone. */
static void inl02(void)
{
    const char *out = "test_inline.hsaco";
    int rc = th_run(BC_BIN " --amdgpu-bin tests/device_calls.cu "
                    "-o test_inline.hsaco", obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(th_exist(out));
    remove(out);
    PASS();
}

/* NVIDIA and Tensix isel cannot emit a call either, so they inline too. */
static void inl03(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);

    rc = th_run(BC_BIN " --tensix --ir tests/device_calls.cu",
                obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* The CPU backend has a real SysV call ABI, so the inliner must not touch it:
 * the device calls stay as calls. */
static void inl04(void)
{
    int rc = th_run(BC_BIN " --cpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") != NULL);
    PASS();
}

/* ---- Hand-built modules: what the splice must not break ---- */

static uint32_t bmod(bir_module_t *M, int recur)
{
    uint32_t i32, vt, ft, kt, ps[1];

    bir_module_init(M);
    i32 = bir_type_int(M, 32);
    vt  = bir_type_void(M);
    ps[0] = i32;
    ft = bir_type_func(M, i32, ps, 1);
    kt = bir_type_func(M, vt, ps, 0);

    M->num_insts = 16;
    M->insts[0] = (bir_inst_t){ .op = BIR_PARAM, .num_operands = 0,
                                .subop = 0, .type = i32 };
    M->insts[1] = (bir_inst_t){ .op = BIR_ICMP, .num_operands = 2,
                                .subop = BIR_ICMP_SGT, .type = i32 };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);
    M->insts[1].operands[1] = BIR_MAKE_VAL(0);
    M->insts[2] = (bir_inst_t){ .op = BIR_BR_COND, .num_operands = 4,
                                .subop = 0, .type = vt };
    M->insts[2].operands[0] = BIR_MAKE_VAL(1);
    M->insts[2].operands[1] = 1;
    M->insts[2].operands[2] = 2;
    M->insts[2].operands[3] = 2;
    M->insts[3] = (bir_inst_t){ .op = BIR_MUL, .num_operands = 2,
                                .subop = 0, .type = i32 };
    M->insts[3].operands[0] = BIR_MAKE_VAL(0);
    M->insts[3].operands[1] = BIR_MAKE_VAL(0);
    M->insts[4] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1,
                                .subop = 0, .type = vt };
    M->insts[4].operands[0] = BIR_MAKE_VAL(3);
    M->insts[5] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2,
                                .subop = 0, .type = i32 };
    M->insts[5].operands[0] = BIR_MAKE_VAL(0);
    M->insts[5].operands[1] = BIR_MAKE_VAL(0);
    M->insts[6] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1,
                                .subop = 0, .type = vt };
    M->insts[6].operands[0] = BIR_MAKE_VAL(5);

    M->insts[7] = (bir_inst_t){ .op = BIR_THREAD_ID, .num_operands = 0,
                                .subop = 0, .type = i32 };
    M->insts[8] = (bir_inst_t){ .op = BIR_ICMP, .num_operands = 2,
                                .subop = BIR_ICMP_SGT, .type = i32 };
    M->insts[8].operands[0] = BIR_MAKE_VAL(7);
    M->insts[8].operands[1] = BIR_MAKE_VAL(7);
    M->insts[9] = (bir_inst_t){ .op = BIR_BR_COND, .num_operands = 4,
                                .subop = 0, .type = vt };
    M->insts[9].operands[0] = BIR_MAKE_VAL(8);
    M->insts[9].operands[1] = 4;
    M->insts[9].operands[2] = 5;
    M->insts[9].operands[3] = 6;
    M->insts[10] = (bir_inst_t){ .op = BIR_CALL, .num_operands = 2,
                                 .subop = 0, .type = i32 };
    M->insts[10].operands[0] = 0;
    M->insts[10].operands[1] = BIR_MAKE_VAL(7);
    M->insts[11] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1,
                                 .subop = 0, .type = vt };
    M->insts[11].operands[0] = 6;
    M->insts[12] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2,
                                 .subop = 0, .type = i32 };
    M->insts[12].operands[0] = BIR_MAKE_VAL(7);
    M->insts[12].operands[1] = BIR_MAKE_VAL(7);
    M->insts[13] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1,
                                 .subop = 0, .type = vt };
    M->insts[13].operands[0] = 6;
    M->insts[14] = (bir_inst_t){ .op = BIR_PHI, .num_operands = 4,
                                 .subop = 0, .type = i32 };
    M->insts[14].operands[0] = 4;
    M->insts[14].operands[1] = BIR_MAKE_VAL(10);
    M->insts[14].operands[2] = 5;
    M->insts[14].operands[3] = BIR_MAKE_VAL(12);
    M->insts[15] = (bir_inst_t){ .op = BIR_RET, .num_operands = 0,
                                 .subop = 0, .type = vt };

    M->num_blocks = 7;
    M->blocks[0] = (bir_block_t){ .name = 0, .first_inst = 0, .num_insts = 3 };
    M->blocks[1] = (bir_block_t){ .name = 0, .first_inst = 3, .num_insts = 2 };
    M->blocks[2] = (bir_block_t){ .name = 0, .first_inst = 5, .num_insts = 2 };
    M->blocks[3] = (bir_block_t){ .name = 0, .first_inst = 7, .num_insts = 3 };
    M->blocks[4] = (bir_block_t){ .name = 0, .first_inst = 10, .num_insts = 2 };
    M->blocks[5] = (bir_block_t){ .name = 0, .first_inst = 12, .num_insts = 2 };
    M->blocks[6] = (bir_block_t){ .name = 0, .first_inst = 14, .num_insts = 2 };

    if (recur) {
        M->insts[5].op = BIR_CALL;
        M->insts[5].num_operands = 2;
        M->insts[5].type = i32;
        M->insts[5].operands[0] = 0;
        M->insts[5].operands[1] = BIR_MAKE_VAL(0);
    }

    M->num_funcs = 2;
    M->funcs[0].name = 0;
    M->funcs[0].type = ft;
    M->funcs[0].first_block = 0;
    M->funcs[0].num_blocks = 3;
    M->funcs[0].total_insts = 7;
    M->funcs[0].num_params = 1;
    M->funcs[0].cuda_flags = CUDA_DEVICE;
    M->funcs[1].name = 0;
    M->funcs[1].type = kt;
    M->funcs[1].first_block = 3;
    M->funcs[1].num_blocks = 4;
    M->funcs[1].total_insts = 9;
    M->funcs[1].num_params = 0;
    M->funcs[1].cuda_flags = CUDA_GLOBAL;
    return i32;
}

static int ncall(const bir_module_t *M, uint32_t f)
{
    const bir_func_t *F = &M->funcs[f];
    uint32_t b, k;
    int n = 0;
    for (b = 0; b < F->num_blocks; b++) {
        const bir_block_t *B = &M->blocks[F->first_block + b];
        for (k = 0; k < B->num_insts; k++)
            if (M->insts[B->first_inst + k].op == BIR_CALL) n++;
    }
    return n;
}

static void inl05(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    CHECK(M != NULL);
    (void)bmod(M, 0);

    { int rc = bir_inline_device(M); printf("RC=%d\n", rc); }
    CHECK(bir_vchk(M) == BC_OK);
    CHECK(ncall(M, 1) == 0);
    free(M);
    PASS();
}

static void inl06(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    uint32_t f, live = 0;
    CHECK(M != NULL);
    (void)bmod(M, 0);

    CHECK(bir_inline_device(M) == BC_OK);
    for (f = 0; f < M->num_funcs; f++) live += M->funcs[f].num_blocks;
    CHECK(live == M->num_blocks);
    CHECK(M->funcs[0].first_block == 0);
    CHECK(M->blocks[M->funcs[1].first_block].first_inst
          == M->funcs[0].total_insts);
    free(M);
    PASS();
}

static void inl07(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    CHECK(M != NULL);
    (void)bmod(M, 1);

    CHECK(bir_inline_device(M) == BC_OK);
    CHECK(bir_vchk(M) == BC_OK);
    CHECK(ncall(M, 0) == 1);
    free(M);
    PASS();
}

TH_REG("inl", 1, "a GPU kernel ends up with no calls", inl01);
TH_REG("inl", 2, "a GPU binary inlines", inl02);
TH_REG("inl", 3, "other GPU targets inline too", inl03);
TH_REG("inl", 4, "CPU keeps its calls", inl04);
TH_REG("inl", 5, "a phi survives a split predecessor", inl05);
TH_REG("inl", 6, "the arenas come back packed and in order", inl06);
TH_REG("inl", 7, "a recursive device body stays a call", inl07);
