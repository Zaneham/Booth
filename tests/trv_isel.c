/* trv_isel.c -- BIR -> RV32IM isel + ELF emission end to end.
 * Build a small BIR module by hand, run the isel, write the ELF,
 * read it back and check the instruction bytes are what they
 * should be for an integer add kernel. */

#include "tharns.h"
#include "rv_enc.h"
#include "rv_buf.h"
#include "rv_isel.h"
#include "rv_elf.h"
#include "rt_args.h"
#include "tdf.h"
#include "bir.h"

/* bir_module_t is huge; keep it in BSS. */
static bir_module_t fake_bir;
static rv_buf_t     code;

/* Scan the emitted buffer for a specific instruction word. Used
 * by every shape-check test, defined up here so each TH_REG can
 * call it without forward-declaring. */
static int buf_contains(uint32_t want)
{
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++)
        if (rv_buf_data(&code)[i] == want) return 1;
    return 0;
}

/*
 * Construct a BIR module representing:
 *   __global__ void add_kernel(int *out, int a, int b) {
 *       *out = a + b;
 *   }
 *
 * One function, three parameters, one basic block, five
 * instructions: PARAM x3, ADD, STORE.
 */
static void build_bir_add(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));

    /* Types: ptr<i32, global> at index 0, i32 at index 1, void at 2. */
    fake_bir.num_types = 3;
    fake_bir.types[0].kind = BIR_TYPE_PTR;
    fake_bir.types[0].addrspace = BIR_AS_GLOBAL;
    fake_bir.types[0].inner = 1;
    fake_bir.types[1].kind = BIR_TYPE_INT;
    fake_bir.types[1].width = 32;
    fake_bir.types[2].kind = BIR_TYPE_VOID;

    /* One function with one block holding five instructions. */
    fake_bir.num_funcs  = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts  = 5;

    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks  = 1;
    fake_bir.funcs[0].num_params  = 3;
    fake_bir.funcs[0].total_insts = 5;
    fake_bir.funcs[0].cuda_flags  = CUDA_GLOBAL;

    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts  = 5;

    /* inst 0: PARAM 0  (the pointer 'out') */
    fake_bir.insts[0].op    = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type  = 0;

    /* inst 1: PARAM 1  (the int 'a') */
    fake_bir.insts[1].op    = BIR_PARAM;
    fake_bir.insts[1].subop = 1;
    fake_bir.insts[1].type  = 1;

    /* inst 2: PARAM 2  (the int 'b') */
    fake_bir.insts[2].op    = BIR_PARAM;
    fake_bir.insts[2].subop = 2;
    fake_bir.insts[2].type  = 1;

    /* inst 3: ADD inst1, inst2 */
    fake_bir.insts[3].op           = BIR_ADD;
    fake_bir.insts[3].type         = 1;
    fake_bir.insts[3].num_operands = 2;
    fake_bir.insts[3].operands[0]  = BIR_MAKE_VAL(1);
    fake_bir.insts[3].operands[1]  = BIR_MAKE_VAL(2);

    /* inst 4: STORE value=inst3, address=inst0 */
    fake_bir.insts[4].op           = BIR_STORE;
    fake_bir.insts[4].type         = 2;
    fake_bir.insts[4].num_operands = 2;
    fake_bir.insts[4].operands[0]  = BIR_MAKE_VAL(3);
    fake_bir.insts[4].operands[1]  = BIR_MAKE_VAL(0);
}

/* ---- isel produces some code without erroring ---- */

static void rv_isel_smoke(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(rv_buf_n_words(&code) > 0u);
    PASS();
}
TH_REG("rv_enc", rv_isel_smoke);

/* ---- prologue is the documented one: drop sp, save ra ---- */

static void rv_isel_prologue(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* First two words must be: addi sp, sp, -frame_sz   and   sw ra, 0(sp). */
    uint32_t w0 = rv_buf_data(&code)[0];
    uint32_t w1 = rv_buf_data(&code)[1];
    /* w0 should be an addi sp,sp,negative; check opcode and rd/rs1. */
    CHEQ(w0 & 0x7Fu, 0x13u);              /* OP-IMM */
    CHEQ((w0 >> 7) & 0x1Fu, (uint32_t)RV_SP);
    CHEQ((w0 >> 15) & 0x1Fu, (uint32_t)RV_SP);
    /* w1 should be sw ra, 0(sp). */
    CHEQ(w1, rv_sw(RV_RA, RV_SP, 0));
    PASS();
}
TH_REG("rv_enc", rv_isel_prologue);

/* ---- ADD instruction shows up in the body ---- */

static void rv_isel_has_add(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Look for an `add t0, t0, t1` somewhere in the buffer. */
    uint32_t add_t0 = rv_add(RV_T0, RV_T0, RV_T1);
    int found = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        if (rv_buf_data(&code)[i] == add_t0) { found = 1; break; }
    }
    CHECK(found);
    PASS();
}
TH_REG("rv_enc", rv_isel_has_add);

/* ---- final SW writes through the pointer parameter ---- */

static void rv_isel_has_store(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* The closing store is `sw t0, 0(t1)`. */
    uint32_t sw = rv_sw(RV_T0, RV_T1, 0);
    int found = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        if (rv_buf_data(&code)[i] == sw) { found = 1; break; }
    }
    CHECK(found);
    PASS();
}
TH_REG("rv_enc", rv_isel_has_store);

/* ---- epilogue restores ra, jalr to return ---- */

static void rv_isel_epilogue(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    uint32_t n = rv_buf_n_words(&code);
    CHECK(n >= 3u);
    /* The last instruction is jalr zero, ra, 0. */
    CHEQ(rv_buf_data(&code)[n - 1], rv_jalr(RV_ZERO, RV_RA, 0));
    /* The instruction before that is addi sp, sp, +frame_sz (positive). */
    uint32_t w_addsp = rv_buf_data(&code)[n - 2];
    CHEQ(w_addsp & 0x7Fu, 0x13u);
    CHEQ((w_addsp >> 7) & 0x1Fu, (uint32_t)RV_SP);
    CHEQ((w_addsp >> 15) & 0x1Fu, (uint32_t)RV_SP);
    /* And before that, the saved ra is reloaded. */
    CHEQ(rv_buf_data(&code)[n - 3], rv_lw(RV_RA, RV_SP, 0));
    PASS();
}
TH_REG("rv_enc", rv_isel_epilogue);

/* ---- end-to-end: isel + ELF write produces a valid file ---- */

static void rv_isel_to_elf(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHEQ(rv_elf_write(&code, "tdf_isel_kernel.elf"), BC_OK);
    /* Read back and check the ELF magic. */
    FILE *fp = fopen("tdf_isel_kernel.elf", "rb");
    CHECK(fp != NULL);
    uint8_t hdr[16];
    size_t n = fread(hdr, 1, 16, fp);
    fclose(fp);
    CHEQ(n, (size_t)16);
    CHEQ(hdr[0], 0x7Fu);
    CHEQ(hdr[1], (uint8_t)'E');
    CHEQ(hdr[2], (uint8_t)'L');
    CHEQ(hdr[3], (uint8_t)'F');
    PASS();
}
TH_REG("rv_enc", rv_isel_to_elf);

/* ---- unsupported op produces a clean refusal ---- */

static void rv_isel_unsupported(void)
{
    /* Replace inst 4 with a SHARED_ALLOC, which still has no
     * lowering (the L1 region management for __shared__ memory
     * is a future sitting). The isel must refuse cleanly rather
     * than silently mis-codegen; this test is the canary for
     * that discipline as new opcodes get added. */
    build_bir_add();
    fake_bir.insts[4].op = BIR_SHARED_ALLOC;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_ERR_TDF);
    PASS();
}
TH_REG("rv_enc", rv_isel_unsupported);

/* ---- THREAD_ID etc. lower to LW from L1 runtime args ---- */
/*
 * Each coordinate intrinsic emits two instructions: LUI to
 * materialise the runtime args base into t1, then LW from the
 * appropriate offset in that region. We check both pieces appear
 * in the output rather than trying to find a particular slot in
 * the buffer, because the surrounding code can shift if other
 * lowerings change. The expected encoding is what the spec says
 * the launcher will write into.
 *
 * No stub: the kernel actually reads from L1; what's IN that L1
 * slot is the launcher's responsibility, not the compiler's. */
static void rv_isel_thread_id_x_via_l1(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_THREAD_ID;
    fake_bir.insts[3].subop = 0;
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* LUI t1, 0x8 (TD_L1_RTARG_BASE >> 12) and LW t0, 0(t1) */
    CHECK(buf_contains(rv_lui(RV_T1, TD_L1_RTARG_BASE >> 12)));
    CHECK(buf_contains(rv_lw (RV_T0, RV_T1,
                              (int16_t)RT_ARG_OFF_TID_X)));
    PASS();
}
TH_REG("rv_enc", rv_isel_thread_id_x_via_l1);

static void rv_isel_block_id_z_via_l1(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_BLOCK_ID;
    fake_bir.insts[3].subop = 2;       /* z dim */
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_lui(RV_T1, TD_L1_RTARG_BASE >> 12)));
    CHECK(buf_contains(rv_lw (RV_T0, RV_T1,
                              (int16_t)RT_ARG_OFF_BID_Z)));
    PASS();
}
TH_REG("rv_enc", rv_isel_block_id_z_via_l1);

static void rv_isel_block_dim_y_via_l1(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_BLOCK_DIM;
    fake_bir.insts[3].subop = 1;       /* y dim */
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_BDIM_Y)));
    PASS();
}
TH_REG("rv_enc", rv_isel_block_dim_y_via_l1);

static void rv_isel_grid_dim_x_via_l1(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_GRID_DIM;
    fake_bir.insts[3].subop = 0;
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_GDIM_X)));
    PASS();
}
TH_REG("rv_enc", rv_isel_grid_dim_x_via_l1);

/* Bad dimension still refuses cleanly. */
static void rv_isel_intrinsic_bad_dim(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_THREAD_ID;
    fake_bir.insts[3].subop = 5;       /* out of range */
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_ERR_TDF);
    PASS();
}
TH_REG("rv_enc", rv_isel_intrinsic_bad_dim);

/* PARAM now also reads from L1 runtime args, not from a0..a7. */
static void rv_isel_param_via_l1(void)
{
    build_bir_add();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Three PARAMs (out, a, b) should each produce a LW from
     * RT_ARG_KERNEL_BASE + i*4. We check the first slot, slot 1,
     * and slot 2 explicitly. */
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_KARG(0))));
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_KARG(1))));
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_KARG(2))));
    PASS();
}
TH_REG("rv_enc", rv_isel_param_via_l1);

static void rv_isel_refuses_barrier(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_BARRIER;
    fake_bir.insts[3].num_operands = 0;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_ERR_TDF);
    PASS();
}
TH_REG("rv_enc", rv_isel_refuses_barrier);

/* ---- Helper: build a minimal "binop kernel" template ----
 *
 * Same shape as build_bir_add but lets the caller override the
 * binop opcode. Used by all the new arithmetic / bitwise / shift
 * / divide tests so each one runs against an identical scaffold
 * and we know any difference in the resulting instructions comes
 * from the opcode under test, not from layout drift. */

static void build_bir_binop(uint16_t op)
{
    build_bir_add();        /* leaves insts[3] = ADD */
    fake_bir.insts[3].op = op;
}

/* For each new binop, scan the buffer for the expected RV32IM
 * instruction. The kernel body is structurally identical to ADD,
 * so the only post-prologue change is the single arithmetic word
 * between the operand loads and the result spill. */

static void rv_isel_binop_and(void)
{
    build_bir_binop(BIR_AND);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_and(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_and);

static void rv_isel_binop_or(void)
{
    build_bir_binop(BIR_OR);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_or(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_or);

static void rv_isel_binop_xor(void)
{
    build_bir_binop(BIR_XOR);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_xor(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_xor);

static void rv_isel_binop_shl(void)
{
    build_bir_binop(BIR_SHL);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_sll(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_shl);

static void rv_isel_binop_lshr(void)
{
    build_bir_binop(BIR_LSHR);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_srl(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_lshr);

static void rv_isel_binop_ashr(void)
{
    build_bir_binop(BIR_ASHR);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_sra(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_ashr);

static void rv_isel_binop_sdiv(void)
{
    build_bir_binop(BIR_SDIV);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_div(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_sdiv);

static void rv_isel_binop_udiv(void)
{
    build_bir_binop(BIR_UDIV);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_divu(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_udiv);

static void rv_isel_binop_srem(void)
{
    build_bir_binop(BIR_SREM);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_rem(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_srem);

static void rv_isel_binop_urem(void)
{
    build_bir_binop(BIR_UREM);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_remu(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_binop_urem);

/* ---- ICMP variants ----
 *
 * Each comparison kind has a canonical 1- or 2-instruction
 * lowering; we check the discriminating instructions appear. */

static void build_bir_icmp(uint8_t kind)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_ICMP;
    fake_bir.insts[3].subop = kind;
}

static void rv_isel_icmp_eq(void)
{
    build_bir_icmp(BIR_ICMP_EQ);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_xor (RV_T0, RV_T0, RV_T1)));
    CHECK(buf_contains(rv_sltiu(RV_T0, RV_T0, 1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_eq);

static void rv_isel_icmp_ne(void)
{
    build_bir_icmp(BIR_ICMP_NE);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_xor (RV_T0, RV_T0, RV_T1)));
    CHECK(buf_contains(rv_sltu(RV_T0, RV_ZERO, RV_T0)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_ne);

static void rv_isel_icmp_slt(void)
{
    build_bir_icmp(BIR_ICMP_SLT);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_slt(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_slt);

static void rv_isel_icmp_sgt(void)
{
    /* a > b implemented as b < a, so the operands are swapped. */
    build_bir_icmp(BIR_ICMP_SGT);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_slt(RV_T0, RV_T1, RV_T0)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_sgt);

static void rv_isel_icmp_sle(void)
{
    /* a <= b == NOT(b < a), so slt with swapped operands plus xori 1. */
    build_bir_icmp(BIR_ICMP_SLE);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_slt (RV_T0, RV_T1, RV_T0)));
    CHECK(buf_contains(rv_xori(RV_T0, RV_T0, 1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_sle);

static void rv_isel_icmp_ult(void)
{
    build_bir_icmp(BIR_ICMP_ULT);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_sltu(RV_T0, RV_T0, RV_T1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_icmp_ult);

/* ---- load_imm32: small constant ---- */
/*
 * Below ADDI's 12-bit signed range, materialisation should be a
 * single ADDI from x0. Just above, it should be LUI followed by a
 * negative ADDI when the low half's high bit is set, and LUI
 * followed by a positive ADDI otherwise.
 *
 * We build a BIR module that produces "return c" for a chosen
 * constant c, then walk the emitted instructions and check the
 * mantissa.
 */
static void build_bir_return_const(int32_t v)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 2;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind = BIR_TYPE_VOID;
    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 1;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 0;
    fake_bir.funcs[0].total_insts = 1;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 1;

    fake_bir.num_consts = 1;
    fake_bir.consts[0].kind = BIR_CONST_INT;
    fake_bir.consts[0].type = 0;
    fake_bir.consts[0].d.ival = v;

    fake_bir.insts[0].op = BIR_RET;
    fake_bir.insts[0].type = 1;
    fake_bir.insts[0].num_operands = 1;
    fake_bir.insts[0].operands[0] = BIR_MAKE_CONST(0);
}

static void rv_isel_const_small_pos(void)
{
    build_bir_return_const(42);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Should be ADDI a0, zero, 42 somewhere in the body. */
    CHECK(buf_contains(rv_addi(RV_A0, RV_ZERO, 42)));
    PASS();
}
TH_REG("rv_enc", rv_isel_const_small_pos);

static void rv_isel_const_small_neg(void)
{
    build_bir_return_const(-1);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_addi(RV_A0, RV_ZERO, -1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_const_small_neg);

static void rv_isel_const_large_positive(void)
{
    /* 0x12345678: low 12 = 0x678 (bit 11 clear), so just LUI+ADDI
     * with hi=0x12345 and lo=0x678 (no carry correction). */
    build_bir_return_const(0x12345678);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_lui (RV_A0, 0x12345u)));
    CHECK(buf_contains(rv_addi(RV_A0, RV_A0, 0x678)));
    PASS();
}
TH_REG("rv_enc", rv_isel_const_large_positive);

static void rv_isel_const_large_signbump(void)
{
    /* 0x12345fff: low 12 = 0xfff (bit 11 set, lo treated as -1),
     * so hi must bump from 0x12345 to 0x12346 and ADDI = -1. */
    build_bir_return_const(0x12345fff);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_lui (RV_A0, 0x12346u)));
    CHECK(buf_contains(rv_addi(RV_A0, RV_A0, -1)));
    PASS();
}
TH_REG("rv_enc", rv_isel_const_large_signbump);

/* ---- UNREACHABLE emits EBREAK ---- */

static void rv_isel_unreachable_ebreak(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 1;
    fake_bir.types[0].kind = BIR_TYPE_VOID;
    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 1;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].total_insts = 1;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 1;
    fake_bir.insts[0].op = BIR_UNREACHABLE;
    fake_bir.insts[0].type = 0;

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_ebreak()));
    PASS();
}
TH_REG("rv_enc", rv_isel_unreachable_ebreak);

/* ---- Identity casts: PTRTOINT / INTTOPTR / BITCAST ---- */
/*
 * Each should produce a load + store cycle for the value pair
 * (input slot -> register -> output slot) and no arithmetic
 * instruction in between. Easiest check: the count of words
 * between the load+store pair should match what we'd get with
 * a no-op transformation. */

static void rv_isel_cast_id_bitcast(void)
{
    build_bir_add();
    fake_bir.insts[3].op = BIR_BITCAST;
    fake_bir.insts[3].num_operands = 1;
    fake_bir.insts[3].operands[0] = BIR_MAKE_VAL(1);
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* The bitcast lowers to lw t0, ...; sw t0, ... */
    CHECK(buf_contains(rv_lw(RV_T0, RV_SP, 12)));   /* load %1 slot */
    PASS();
}
TH_REG("rv_enc", rv_isel_cast_id_bitcast);

/* ---- Width conversions ---- */
/*
 * ZEXT i8 -> i32 should be SLLI 24, SRLI 24 — that's the
 * canonical pattern for clearing the high bits without a mask
 * that doesn't fit in ANDI. SEXT i8 -> i32 differs only in
 * using SRAI instead of SRLI to sign-extend. TRUNC i32 -> i8
 * is the same SLLI/SRLI pair as ZEXT (clear the high bits). */

static void build_bir_zext_i8_to_i32(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 3;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 8;
    fake_bir.types[1].kind = BIR_TYPE_INT;
    fake_bir.types[1].width = 32;
    fake_bir.types[2].kind = BIR_TYPE_VOID;
    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 3;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 1;
    fake_bir.funcs[0].total_insts = 3;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 3;
    /* PARAM 0 : i8 */
    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 0;
    /* ZEXT to i32 */
    fake_bir.insts[1].op = BIR_ZEXT;
    fake_bir.insts[1].type = 1;
    fake_bir.insts[1].num_operands = 1;
    fake_bir.insts[1].operands[0] = BIR_MAKE_VAL(0);
    /* RET i32 */
    fake_bir.insts[2].op = BIR_RET;
    fake_bir.insts[2].type = 2;
    fake_bir.insts[2].num_operands = 1;
    fake_bir.insts[2].operands[0] = BIR_MAKE_VAL(1);
}

static void rv_isel_zext_i8(void)
{
    build_bir_zext_i8_to_i32();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_slli(RV_T0, RV_T0, 24)));
    CHECK(buf_contains(rv_srli(RV_T0, RV_T0, 24)));
    PASS();
}
TH_REG("rv_enc", rv_isel_zext_i8);

static void rv_isel_sext_i8(void)
{
    build_bir_zext_i8_to_i32();
    fake_bir.insts[1].op = BIR_SEXT;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_slli(RV_T0, RV_T0, 24)));
    CHECK(buf_contains(rv_srai(RV_T0, RV_T0, 24)));
    PASS();
}
TH_REG("rv_enc", rv_isel_sext_i8);

/* ---- Multi-function module: kernel calls a helper ----
 *
 * Two functions in one BIR module:
 *
 *   int helper(int a, int b) { return a + b; }
 *   __global__ void kern(int *out, int a, int b) {
 *       *out = helper(a, b);
 *   }
 *
 * helper is function 1, kern is function 0 (emitted first because
 * function 0 is the entry point). BIR_CALL inside kern targets
 * function 1; the JAL offset gets filled in after both functions
 * have been laid out in the code buffer.
 *
 * What we check:
 *   - rv_isel_module returns OK
 *   - Both function prologues appear in the buffer (we look for
 *     two `sw ra, 0(sp)` instructions, one per function)
 *   - At least one JAL with a positive offset exists (the call
 *     from kern to helper, since helper is laid out after kern)
 */
static void build_bir_kern_calls_helper(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    /* Types: 0=i32, 1=ptr<global,i32>, 2=void, 3=func i32(i32,i32),
     *        4=func void(ptr,i32,i32) */
    fake_bir.num_types = 5;
    fake_bir.types[0].kind  = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind       = BIR_TYPE_PTR;
    fake_bir.types[1].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[1].inner      = 0;
    fake_bir.types[2].kind  = BIR_TYPE_VOID;
    fake_bir.types[3].kind = BIR_TYPE_FUNC;
    fake_bir.types[3].inner = 0;     /* return i32 */
    fake_bir.types[3].num_fields = 2;
    fake_bir.types[3].count = 0;
    fake_bir.types[4].kind = BIR_TYPE_FUNC;
    fake_bir.types[4].inner = 2;     /* return void */
    fake_bir.types[4].num_fields = 3;

    fake_bir.num_funcs  = 2;
    fake_bir.num_blocks = 2;
    fake_bir.num_insts  = 9;   /* kern: 0..5 (6 insts), helper: 6..8 (3 insts) */

    /* Function 0: kern. PARAM 0 (out), PARAM 1 (a), PARAM 2 (b),
     * CALL helper(a, b), STORE result through out, RET void. */
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 3;
    fake_bir.funcs[0].total_insts = 6;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.funcs[0].type = 4;

    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 6;

    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 1;
    fake_bir.insts[1].op = BIR_PARAM;
    fake_bir.insts[1].subop = 1;
    fake_bir.insts[1].type = 0;
    fake_bir.insts[2].op = BIR_PARAM;
    fake_bir.insts[2].subop = 2;
    fake_bir.insts[2].type = 0;
    /* CALL helper(a, b): ops[0]=callee func idx, ops[1..]=args */
    fake_bir.insts[3].op = BIR_CALL;
    fake_bir.insts[3].type = 0;
    fake_bir.insts[3].num_operands = 3;
    fake_bir.insts[3].operands[0] = 1;     /* function 1 = helper */
    fake_bir.insts[3].operands[1] = BIR_MAKE_VAL(1);
    fake_bir.insts[3].operands[2] = BIR_MAKE_VAL(2);
    /* STORE call_result through out: ops[0]=value, ops[1]=addr */
    fake_bir.insts[4].op = BIR_STORE;
    fake_bir.insts[4].type = 2;
    fake_bir.insts[4].num_operands = 2;
    fake_bir.insts[4].operands[0] = BIR_MAKE_VAL(3);
    fake_bir.insts[4].operands[1] = BIR_MAKE_VAL(0);
    /* RET void */
    fake_bir.insts[5].op = BIR_RET;
    fake_bir.insts[5].type = 2;
    fake_bir.insts[5].num_operands = 0;

    /* Function 1: helper. PARAM 0, PARAM 1, ADD, RET. */
    fake_bir.funcs[1].first_block = 1;
    fake_bir.funcs[1].num_blocks = 1;
    fake_bir.funcs[1].num_params = 2;
    fake_bir.funcs[1].total_insts = 4;
    fake_bir.funcs[1].cuda_flags = 0;
    fake_bir.funcs[1].type = 3;

    fake_bir.blocks[1].first_inst = 6;
    fake_bir.blocks[1].num_insts = 4;
    /* Need num_insts to include helper too. */
    fake_bir.num_insts = 10;

    fake_bir.insts[6].op = BIR_PARAM;
    fake_bir.insts[6].subop = 0;
    fake_bir.insts[6].type = 0;
    fake_bir.insts[7].op = BIR_PARAM;
    fake_bir.insts[7].subop = 1;
    fake_bir.insts[7].type = 0;
    fake_bir.insts[8].op = BIR_ADD;
    fake_bir.insts[8].type = 0;
    fake_bir.insts[8].num_operands = 2;
    fake_bir.insts[8].operands[0] = BIR_MAKE_VAL(6);
    fake_bir.insts[8].operands[1] = BIR_MAKE_VAL(7);
    fake_bir.insts[9].op = BIR_RET;
    fake_bir.insts[9].type = 0;
    fake_bir.insts[9].num_operands = 1;
    fake_bir.insts[9].operands[0] = BIR_MAKE_VAL(8);
}

static void rv_isel_module_two_funcs(void)
{
    build_bir_kern_calls_helper();
    rv_buf_init(&code);
    CHEQ(rv_isel_module(&fake_bir, &code), BC_OK);

    /* Two prologues' worth of `sw ra, 0(sp)` should appear, one per
     * function. The instruction encoding for `sw ra, 0(sp)` is
     * fixed regardless of frame size since the offset is zero. */
    uint32_t prologue_ra_store = rv_sw(RV_RA, RV_SP, 0);
    int count = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        if (rv_buf_data(&code)[i] == prologue_ra_store) count++;
    }
    CHEQ(count, 2);
    PASS();
}
TH_REG("rv_enc", rv_isel_module_two_funcs);

/* The JAL for the kern->helper call should appear with a positive
 * offset (helper is emitted after kern). The exact value depends
 * on kern's body length, so we just check that some JAL with a
 * positive J-immediate exists. */
static void rv_isel_module_jal_present(void)
{
    build_bir_kern_calls_helper();
    rv_buf_init(&code);
    CHEQ(rv_isel_module(&fake_bir, &code), BC_OK);

    int found_forward_jal = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        uint32_t w = rv_buf_data(&code)[i];
        /* JAL opcode is 0x6F. Sign bit of J-immediate is bit 31. */
        if ((w & 0x7Fu) == 0x6Fu && (w & 0x80000000u) == 0u) {
            found_forward_jal = 1;
            break;
        }
    }
    CHECK(found_forward_jal);
    PASS();
}
TH_REG("rv_enc", rv_isel_module_jal_present);

/* Args land in a0, a1, etc. The call sequence loads the first
 * two arg values from the kern's local slots into a0 and a1 just
 * before the JAL. */
static void rv_isel_module_args_in_regs(void)
{
    build_bir_kern_calls_helper();
    rv_buf_init(&code);
    CHEQ(rv_isel_module(&fake_bir, &code), BC_OK);

    /* The call args are PARAM 1 (slot 4) and PARAM 2 (slot 8).
     * They get loaded into a0 and a1 respectively via LW. The
     * exact instruction is `lw a0, slot_of_inst_1(sp)` and
     * `lw a1, slot_of_inst_2(sp)`. */
    /* slot_offset(1) = ISEL_LOCALS_BASE (8) + 1*4 = 12 */
    CHECK(buf_contains(rv_lw(RV_A0, RV_SP, 12)));
    CHECK(buf_contains(rv_lw(RV_A1, RV_SP, 16)));
    PASS();
}
TH_REG("rv_enc", rv_isel_module_args_in_regs);

/* Direct recursion is refused cleanly. A function calling itself
 * needs more sophisticated prologue/epilogue handling than we have
 * today; the alternative is to silently miscompile, which is the
 * sort of decision that turns up later in a debugger as a value
 * that was perfectly correct an instant ago. */
static void rv_isel_module_refuses_recursion(void)
{
    build_bir_kern_calls_helper();
    /* Edit function 1 (helper) to call itself instead of just
     * adding. Replace the ADD with a CALL to function 1. */
    fake_bir.insts[8].op = BIR_CALL;
    fake_bir.insts[8].type = 0;
    fake_bir.insts[8].num_operands = 3;
    fake_bir.insts[8].operands[0] = 1;     /* recursion! */
    fake_bir.insts[8].operands[1] = BIR_MAKE_VAL(6);
    fake_bir.insts[8].operands[2] = BIR_MAKE_VAL(7);

    rv_buf_init(&code);
    CHEQ(rv_isel_module(&fake_bir, &code), BC_ERR_TDF);
    PASS();
}
TH_REG("rv_enc", rv_isel_module_refuses_recursion);

/* ---- Forward + backward branches ----
 *
 * Construct a function with three blocks that exercises both
 * directions: block 0 has BR to block 2 (forward, patched at end),
 * block 1 has BR to block 0 (backward, resolved inline because
 * block 0's position is known by the time we get to block 1).
 * After isel finishes, all branch placeholders should be filled
 * in with non-zero offsets (placeholders are emitted as 0).
 */
static void build_bir_three_blocks(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 1;
    fake_bir.types[0].kind = BIR_TYPE_VOID;

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 3;
    fake_bir.num_insts = 3;     /* one BR per block */
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 3;
    fake_bir.funcs[0].total_insts = 3;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;

    /* Block 0: BR to block 2 (forward) */
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 1;
    fake_bir.insts[0].op = BIR_BR;
    fake_bir.insts[0].type = 0;
    fake_bir.insts[0].num_operands = 1;
    fake_bir.insts[0].operands[0] = 2;       /* absolute block idx */

    /* Block 1: BR to block 0 (backward) */
    fake_bir.blocks[1].first_inst = 1;
    fake_bir.blocks[1].num_insts = 1;
    fake_bir.insts[1].op = BIR_BR;
    fake_bir.insts[1].type = 0;
    fake_bir.insts[1].num_operands = 1;
    fake_bir.insts[1].operands[0] = 0;

    /* Block 2: RET */
    fake_bir.blocks[2].first_inst = 2;
    fake_bir.blocks[2].num_insts = 1;
    fake_bir.insts[2].op = BIR_RET;
    fake_bir.insts[2].type = 0;
    fake_bir.insts[2].num_operands = 0;
}

static void rv_isel_br_forward_and_backward(void)
{
    build_bir_three_blocks();
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);

    /* No placeholder zeros should survive after patch resolution.
     * The buffer starts with prologue (addi sp,sp,-N; sw ra,0(sp)),
     * which are both non-zero. So if ANY non-prologue word is zero
     * we have an unresolved patch. We skip the first two words
     * (prologue) and check the rest. */
    int found_zero = 0;
    for (uint32_t i = 2u; i < rv_buf_n_words(&code); i++) {
        if (rv_buf_data(&code)[i] == 0u) { found_zero = 1; break; }
    }
    CHECK(!found_zero);

    /* Verify the backward branch's offset is negative. The BR from
     * block 1 jumps to block 0, which is earlier in the code. */
    /* Block 1's BR is the first JAL after the block 0 BR. Find
     * any JAL (opcode 0x6F) in the buffer with a negative offset. */
    int found_back = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        uint32_t w = rv_buf_data(&code)[i];
        if ((w & 0x7Fu) == 0x6Fu) {
            /* Sign-extend the J-immediate bit 20 (which is bit 31
             * of the instruction word). Negative if set. */
            if (w & 0x80000000u) { found_back = 1; break; }
        }
    }
    CHECK(found_back);
    PASS();
}
TH_REG("rv_enc", rv_isel_br_forward_and_backward);

/* ---- BR_COND emits BNE + JAL ---- */
/*
 * `if (cond) goto X; else goto Y;` lowers to:
 *   load cond -> T0
 *   BNE T0, zero, X
 *   JAL zero, Y
 * We construct a two-arm BIR and check both branch flavours
 * appear in the output. */
static void rv_isel_br_cond_shape(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 2;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind = BIR_TYPE_VOID;

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 3;
    fake_bir.num_insts = 4;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 3;
    fake_bir.funcs[0].num_params = 1;
    fake_bir.funcs[0].total_insts = 4;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;

    /* Block 0: PARAM 0 (cond), BR_COND cond, block 1, block 2 */
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 2;
    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 0;
    fake_bir.insts[1].op = BIR_BR_COND;
    fake_bir.insts[1].type = 1;
    fake_bir.insts[1].num_operands = 3;
    fake_bir.insts[1].operands[0] = BIR_MAKE_VAL(0);
    fake_bir.insts[1].operands[1] = 1;
    fake_bir.insts[1].operands[2] = 2;

    /* Block 1: RET */
    fake_bir.blocks[1].first_inst = 2;
    fake_bir.blocks[1].num_insts = 1;
    fake_bir.insts[2].op = BIR_RET;
    fake_bir.insts[2].type = 1;
    fake_bir.insts[2].num_operands = 0;

    /* Block 2: RET */
    fake_bir.blocks[2].first_inst = 3;
    fake_bir.blocks[2].num_insts = 1;
    fake_bir.insts[3].op = BIR_RET;
    fake_bir.insts[3].type = 1;
    fake_bir.insts[3].num_operands = 0;

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);

    /* Scan for a BNE (opcode 0x63, funct3 0x1) and a JAL
     * (opcode 0x6F) after the first BR. */
    int found_bne = 0, found_jal = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        uint32_t w = rv_buf_data(&code)[i];
        if ((w & 0x7Fu) == 0x63u && ((w >> 12) & 0x7u) == 0x1u)
            found_bne = 1;
        if ((w & 0x7Fu) == 0x6Fu)
            found_jal = 1;
    }
    CHECK(found_bne);
    CHECK(found_jal);

    /* No unresolved zero placeholders past the prologue. */
    for (uint32_t i = 2u; i < rv_buf_n_words(&code); i++) {
        CHECK(rv_buf_data(&code)[i] != 0u);
    }
    PASS();
}
TH_REG("rv_enc", rv_isel_br_cond_shape);

/* ---- SELECT: BEQ skip + value loads ---- */
/*
 * cond ? a : b lowers to:
 *   load cond  -> T2
 *   load b     -> T0
 *   BEQ T2, zero, skip past true arm
 *   load a     -> T0
 *   store T0
 * We can't easily verify the exact skip offset without depending
 * on every instruction count downstream, but we can check that a
 * BEQ appears and the buffer parses past it without unresolved
 * placeholders. */
static void rv_isel_select_shape(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 2;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind = BIR_TYPE_VOID;

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 5;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 3;
    fake_bir.funcs[0].total_insts = 5;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 5;

    /* PARAM 0,1,2 — cond, a, b */
    for (int p = 0; p < 3; p++) {
        fake_bir.insts[p].op = BIR_PARAM;
        fake_bir.insts[p].subop = (uint8_t)p;
        fake_bir.insts[p].type = 0;
    }
    /* SELECT cond, a, b */
    fake_bir.insts[3].op = BIR_SELECT;
    fake_bir.insts[3].type = 0;
    fake_bir.insts[3].num_operands = 3;
    fake_bir.insts[3].operands[0] = BIR_MAKE_VAL(0);
    fake_bir.insts[3].operands[1] = BIR_MAKE_VAL(1);
    fake_bir.insts[3].operands[2] = BIR_MAKE_VAL(2);
    /* RET */
    fake_bir.insts[4].op = BIR_RET;
    fake_bir.insts[4].type = 1;
    fake_bir.insts[4].num_operands = 1;
    fake_bir.insts[4].operands[0] = BIR_MAKE_VAL(3);

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);

    /* BEQ has opcode 0x63 with funct3 0x0. */
    int found_beq = 0;
    for (uint32_t i = 0; i < rv_buf_n_words(&code); i++) {
        uint32_t w = rv_buf_data(&code)[i];
        if ((w & 0x7Fu) == 0x63u && ((w >> 12) & 0x7u) == 0x0u) {
            found_beq = 1; break;
        }
    }
    CHECK(found_beq);
    PASS();
}
TH_REG("rv_enc", rv_isel_select_shape);

/* ---- Beyond a0..a7 still works because all params go via L1 ----
 *
 * With the L1 runtime-args model the old psABI distinction between
 * register-passed params and stack-passed params disappears. Every
 * BIR_PARAM(i) for i < RT_ARG_N_KERNEL_SLOTS reads from
 * RT_ARG_KERNEL_BASE + i*4 in L1. We construct a kernel with 10
 * params and check that the 9th (subop=8) and 10th (subop=9)
 * read from slot 8 and slot 9 of the kernel-arg block. */
static void rv_isel_param_9_via_l1(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 2;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind = BIR_TYPE_VOID;
    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 11;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 10;
    fake_bir.funcs[0].total_insts = 11;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 11;
    for (int p = 0; p < 10; p++) {
        fake_bir.insts[p].op = BIR_PARAM;
        fake_bir.insts[p].subop = (uint8_t)p;
        fake_bir.insts[p].type = 0;
    }
    fake_bir.insts[10].op = BIR_RET;
    fake_bir.insts[10].type = 1;
    fake_bir.insts[10].num_operands = 1;
    fake_bir.insts[10].operands[0] = BIR_MAKE_VAL(8);

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Params 8 and 9 should both LW from the L1 args slots. */
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_KARG(8))));
    CHECK(buf_contains(rv_lw(RV_T0, RV_T1,
                             (int16_t)RT_ARG_OFF_KARG(9))));
    PASS();
}
TH_REG("rv_enc", rv_isel_param_9_via_l1);

/* Past RT_ARG_N_KERNEL_SLOTS the isel refuses cleanly. */
static void rv_isel_param_overflow_refused(void)
{
    build_bir_add();
    fake_bir.insts[1].subop = RT_ARG_N_KERNEL_SLOTS;
    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_ERR_TDF);
    PASS();
}
TH_REG("rv_enc", rv_isel_param_overflow_refused);

/* ---- Struct GEP: field index becomes byte offset ----
 *
 * Construct a kernel with a 4-int struct (gp_cell_t shape from
 * test_cell_load.cu) and a GEP that accesses field 1. The isel
 * should emit ADDI t0, t0, 4 (the byte offset of field 1, since
 * each preceding field is 4 bytes).
 *
 * The BIR we build:
 *   types: 0=i32, 1=struct{i32,i32,f32,i32}, 2=ptr<global,struct>,
 *          3=ptr<global,i32>, 4=void, 5=f32 (for the struct field)
 *   consts: 0 = i32 const 1
 *   insts: 0 = PARAM 0 (ptr<global,struct>)
 *          1 = GEP base=%0, idx=0 (just the struct pointer)
 *          2 = GEP base=%1, idx=1 (field 1 of struct, i32 mat)
 *          3 = RET void
 */
static void rv_isel_struct_gep_field1(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 6;
    fake_bir.types[0].kind  = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[5].kind  = BIR_TYPE_FLOAT;
    fake_bir.types[5].width = 32;
    /* struct: 4 fields, types[count..count+3] = [i32, i32, f32, i32] */
    fake_bir.num_type_fields = 4;
    fake_bir.type_fields[0] = 0;
    fake_bir.type_fields[1] = 0;
    fake_bir.type_fields[2] = 5;
    fake_bir.type_fields[3] = 0;
    fake_bir.types[1].kind       = BIR_TYPE_STRUCT;
    fake_bir.types[1].count      = 0;       /* start of fields */
    fake_bir.types[1].num_fields = 4;
    fake_bir.types[2].kind       = BIR_TYPE_PTR;
    fake_bir.types[2].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[2].inner      = 1;       /* ptr to struct */
    fake_bir.types[3].kind       = BIR_TYPE_PTR;
    fake_bir.types[3].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[3].inner      = 0;       /* ptr to i32 */
    fake_bir.types[4].kind       = BIR_TYPE_VOID;

    fake_bir.num_consts = 2;
    fake_bir.consts[0].kind  = BIR_CONST_ZERO;
    fake_bir.consts[0].type  = 0;
    fake_bir.consts[1].kind  = BIR_CONST_INT;
    fake_bir.consts[1].type  = 0;
    fake_bir.consts[1].d.ival = 1;

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 4;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 1;
    fake_bir.funcs[0].total_insts = 4;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 4;

    /* PARAM 0: ptr<global,struct> */
    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 2;
    /* GEP %0, 0: yields ptr<struct> (no-op) */
    fake_bir.insts[1].op = BIR_GEP;
    fake_bir.insts[1].type = 2;
    fake_bir.insts[1].num_operands = 2;
    fake_bir.insts[1].operands[0] = BIR_MAKE_VAL(0);
    fake_bir.insts[1].operands[1] = BIR_MAKE_CONST(0);
    /* GEP %1, 1: yields ptr<i32> pointing at field 1 of struct */
    fake_bir.insts[2].op = BIR_GEP;
    fake_bir.insts[2].type = 3;
    fake_bir.insts[2].num_operands = 2;
    fake_bir.insts[2].operands[0] = BIR_MAKE_VAL(1);
    fake_bir.insts[2].operands[1] = BIR_MAKE_CONST(1);
    /* RET void */
    fake_bir.insts[3].op = BIR_RET;
    fake_bir.insts[3].type = 4;
    fake_bir.insts[3].num_operands = 0;

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Field 1 of {i32,i32,f32,i32} is at byte offset 4. We expect
     * an ADDI t0, t0, 4 for the struct GEP. */
    CHECK(buf_contains(rv_addi(RV_T0, RV_T0, 4)));
    PASS();
}
TH_REG("rv_enc", rv_isel_struct_gep_field1);

/* ---- type_bytes for nested types ----
 *
 * The size walker has to recurse into ARRAY, VECTOR and STRUCT.
 * We don't expose type_bytes directly, so the test exercises it
 * via a struct GEP that picks a field whose offset depends on
 * a preceding ARRAY field's full size being known.
 *
 * Struct: { i8, [4 x i32], i32 }
 *   field 0: i8 at offset 0, size 1
 *   field 1: [4 x i32] starts at 4 (aligned), size 16
 *   field 2: i32 at offset 20, size 4
 * GEP-to-field-2 should produce ADDI 20.
 */
static void rv_isel_struct_gep_with_array(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 6;
    fake_bir.types[0].kind  = BIR_TYPE_INT;
    fake_bir.types[0].width = 8;
    fake_bir.types[1].kind  = BIR_TYPE_INT;
    fake_bir.types[1].width = 32;
    fake_bir.types[2].kind  = BIR_TYPE_ARRAY;
    fake_bir.types[2].inner = 1;
    fake_bir.types[2].count = 4;
    /* struct fields: [i8, [4 x i32], i32] */
    fake_bir.num_type_fields = 3;
    fake_bir.type_fields[0] = 0;
    fake_bir.type_fields[1] = 2;
    fake_bir.type_fields[2] = 1;
    fake_bir.types[3].kind       = BIR_TYPE_STRUCT;
    fake_bir.types[3].count      = 0;
    fake_bir.types[3].num_fields = 3;
    fake_bir.types[4].kind       = BIR_TYPE_PTR;
    fake_bir.types[4].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[4].inner      = 3;
    fake_bir.types[5].kind       = BIR_TYPE_PTR;
    fake_bir.types[5].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[5].inner      = 1;

    fake_bir.num_consts = 1;
    fake_bir.consts[0].kind  = BIR_CONST_INT;
    fake_bir.consts[0].type  = 1;
    fake_bir.consts[0].d.ival = 2;

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 3;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 1;
    fake_bir.funcs[0].total_insts = 3;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 3;

    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 4;
    fake_bir.insts[1].op = BIR_GEP;
    fake_bir.insts[1].type = 5;
    fake_bir.insts[1].num_operands = 2;
    fake_bir.insts[1].operands[0] = BIR_MAKE_VAL(0);
    fake_bir.insts[1].operands[1] = BIR_MAKE_CONST(0);  /* field index 2 */
    /* Need num_types > 5 + a void type for ret */
    fake_bir.num_types = 7;
    fake_bir.types[6].kind = BIR_TYPE_VOID;
    fake_bir.insts[2].op = BIR_RET;
    fake_bir.insts[2].type = 6;
    fake_bir.insts[2].num_operands = 0;

    /* Patch the index const to 2 (field number for the trailing
     * i32), already done above via consts[0].d.ival = 2. */

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Expected: ADDI t0, t0, 20 (offset of field 2 in the struct). */
    CHECK(buf_contains(rv_addi(RV_T0, RV_T0, 20)));
    PASS();
}
TH_REG("rv_enc", rv_isel_struct_gep_with_array);

/* ---- Array-of-struct GEP: stride access (the bug regression) ----
 *
 * Distinct from struct-field GEP: when base AND result pointees
 * are the SAME struct type, the index is a stride multiplier
 * (typically a runtime threadIdx), not a field number. The isel
 * must recognise this and emit a MUL by sizeof(struct), not a
 * struct-field offset lookup.
 *
 * Struct: { i32, i32 } -> size 8 bytes
 * BIR:
 *   %0 = PARAM 0 (ptr<global, struct>)
 *   %1 = PARAM 1 (i32 idx, runtime)
 *   %2 = GEP base=%0, idx=%1   (yields ptr<global, struct>, same pointee)
 *   %3 = RET void
 *
 * Expected codegen: load base, load idx, multiply idx by 8, add
 * to base, store. The defining instruction is MUL with elem_size
 * loaded into a register; we look for the LI for the size of 8
 * and the MUL. */
static void rv_isel_array_of_struct_stride(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 5;
    fake_bir.types[0].kind  = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    /* struct {i32, i32} */
    fake_bir.num_type_fields = 2;
    fake_bir.type_fields[0] = 0;
    fake_bir.type_fields[1] = 0;
    fake_bir.types[1].kind       = BIR_TYPE_STRUCT;
    fake_bir.types[1].count      = 0;
    fake_bir.types[1].num_fields = 2;
    fake_bir.types[2].kind       = BIR_TYPE_PTR;
    fake_bir.types[2].addrspace  = BIR_AS_GLOBAL;
    fake_bir.types[2].inner      = 1;
    fake_bir.types[3].kind       = BIR_TYPE_VOID;
    /* (unused fourth type slot) */

    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 4;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].num_params = 2;
    fake_bir.funcs[0].total_insts = 4;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 4;

    fake_bir.insts[0].op = BIR_PARAM;
    fake_bir.insts[0].subop = 0;
    fake_bir.insts[0].type = 2;     /* ptr<struct> */
    fake_bir.insts[1].op = BIR_PARAM;
    fake_bir.insts[1].subop = 1;
    fake_bir.insts[1].type = 0;     /* i32 */
    fake_bir.insts[2].op = BIR_GEP;
    fake_bir.insts[2].type = 2;     /* same ptr<struct>, stride GEP */
    fake_bir.insts[2].num_operands = 2;
    fake_bir.insts[2].operands[0] = BIR_MAKE_VAL(0);
    fake_bir.insts[2].operands[1] = BIR_MAKE_VAL(1);
    fake_bir.insts[3].op = BIR_RET;
    fake_bir.insts[3].type = 3;
    fake_bir.insts[3].num_operands = 0;

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    /* Struct size is 8 bytes (i32+i32). The GEP must multiply
     * the runtime index by 8, so we expect an ADDI t2, zero, 8
     * and a MUL t1, t1, t2. */
    CHECK(buf_contains(rv_addi(RV_T2, RV_ZERO, 8)));
    CHECK(buf_contains(rv_mul (RV_T1, RV_T1, RV_T2)));
    PASS();
}
TH_REG("rv_enc", rv_isel_array_of_struct_stride);

/* ---- ALLOCA: pointer into the frame's alloca region ---- */
/*
 * One alloca of i32. After prologue, the pointer it produces
 * should be sp + (ISEL_LOCALS_BASE + total_insts*4 + 0).
 * total_insts here is 2 (the ALLOCA itself and the RET). So
 * the absolute offset is 8 + 8 = 16. We expect ADDI t0, sp, 16. */
static void rv_isel_alloca_offset(void)
{
    memset(&fake_bir, 0, sizeof(fake_bir));
    fake_bir.num_types = 3;
    fake_bir.types[0].kind = BIR_TYPE_INT;
    fake_bir.types[0].width = 32;
    fake_bir.types[1].kind = BIR_TYPE_PTR;
    fake_bir.types[1].addrspace = BIR_AS_PRIVATE;
    fake_bir.types[1].inner = 0;
    fake_bir.types[2].kind = BIR_TYPE_VOID;
    fake_bir.num_funcs = 1;
    fake_bir.num_blocks = 1;
    fake_bir.num_insts = 2;
    fake_bir.funcs[0].first_block = 0;
    fake_bir.funcs[0].num_blocks = 1;
    fake_bir.funcs[0].total_insts = 2;
    fake_bir.funcs[0].cuda_flags = CUDA_GLOBAL;
    fake_bir.blocks[0].first_inst = 0;
    fake_bir.blocks[0].num_insts = 2;
    fake_bir.insts[0].op = BIR_ALLOCA;
    fake_bir.insts[0].type = 1;
    fake_bir.insts[1].op = BIR_RET;
    fake_bir.insts[1].type = 2;
    fake_bir.insts[1].num_operands = 0;

    rv_buf_init(&code);
    CHEQ(rv_isel_func(&fake_bir, 0, &code), BC_OK);
    CHECK(buf_contains(rv_addi(RV_T0, RV_SP, 16)));
    PASS();
}
TH_REG("rv_enc", rv_isel_alloca_offset);
