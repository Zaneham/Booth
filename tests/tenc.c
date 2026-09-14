/* tenc.c -- encoding unit tests
 * Constructs machine instructions by hand, encodes them, stares at the bits.
 * The most important file in this suite by a country mile.
 * If these fail, you've invented new instructions. Congratulations. */

#include "tharns.h"
#include "amdgpu.h"
#include "encode.h"

/* ---- Shared State ---- */

static amd_module_t *A;

static void enc_init(void)
{
    if (!A) A = (amd_module_t *)malloc(sizeof(amd_module_t));
}

static void enc_setup(amd_target_t tgt)
{
    enc_init();
    memset(A, 0, sizeof(*A));
    A->target = tgt;
    A->mfuncs[0].first_block = 0;
    A->mfuncs[0].num_blocks  = 1;
    A->num_mfuncs  = 1;
    A->mblocks[0].first_inst = 0;
    A->mblocks[0].num_insts  = 1;
    A->num_mblocks = 1;
    A->num_minsts  = 1;
}

/* Read dword from code buffer. The moment of truth. */
static uint32_t dw(int i)
{
    uint32_t v;
    memcpy(&v, A->code + i * 4, 4);
    return v;
}

/* ---- Operand Helpers ---- */

static moperand_t sgpr(uint16_t r)
{
    moperand_t op = {0};
    op.kind = MOP_SGPR;
    op.reg_num = r;
    return op;
}

static moperand_t vgpr(uint16_t r)
{
    moperand_t op = {0};
    op.kind = MOP_VGPR;
    op.reg_num = r;
    return op;
}

static moperand_t imm(int32_t val)
{
    moperand_t op = {0};
    op.kind = MOP_IMM;
    op.imm  = val;
    return op;
}

/* ---- encode: SOP1 ---- */
/* s_mov_b32 s7, s2 -- GFX11
 * [31:23]=0xBE8 prefix [22:16]=SDST(7) [15:8]=OP(0x00) [7:0]=SSRC0(2)
 * Expected: 0xBE870002 */

static void enc01(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_MOV_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);

    encode_function(A, 0);
    CHEQX(dw(0), 0xBE870002u);
    PASS();
}
TH_REG("enc", 1, "SOP1 encodes s_mov_b32", enc01)

/* ---- encode: SOP2 ---- */
/* s_add_u32 s7, s2, s3 -- GFX11
 * [31:30]=10 [29:23]=OP(0x00) [22:16]=SDST(7) [15:8]=SSRC1(3) [7:0]=SSRC0(2)
 * Expected: 0x80070302 */

static void enc02(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_ADD_U32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);
    mi->operands[2] = sgpr(3);

    encode_function(A, 0);
    CHEQX(dw(0), 0x80070302u);
    PASS();
}
TH_REG("enc", 2, "SOP2 encodes s_add_u32", enc02)

/* ---- encode: VOP2 ---- */
/* v_add_f32 v1, v2, v3 -- GFX11
 * [31]=0 [30:25]=OP(0x03) [24:17]=VDST(1) [16:9]=VSRC1(3) [8:0]=SRC0(v2=258)
 * Expected: 0x06020702 */

static void enc03(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_ADD_F32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = vgpr(1);   /* VDST */
    mi->operands[1] = vgpr(2);   /* SRC0 */
    mi->operands[2] = vgpr(3);   /* VSRC1 */

    encode_function(A, 0);
    CHEQX(dw(0), 0x06020702u);
    PASS();
}
TH_REG("enc", 3, "VOP2 encodes v_add_f32", enc03)

/* ---- encode: VOP1 + literal ---- */
/* v_mov_b32 v5, 0xDEAD -- GFX11
 * [31:25]=0x3F [24:17]=VDST(5) [16:9]=OP(0x01) [8:0]=SRC0(255=literal)
 * DW0: 0x7E0A02FF  DW1: 0x0000DEAD */

static void enc04(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_MOV_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = vgpr(5);
    mi->operands[1] = imm(0xDEAD);

    encode_function(A, 0);
    CHEQX(dw(0), 0x7E0A02FFu);
    CHEQX(dw(1), 0x0000DEADu);
    PASS();
}
TH_REG("enc", 4, "VOP1 takes a literal operand", enc04)

/* ---- encode: SMEM GFX11 vs GFX10 ---- */
/* s_load_dword s7, s[2:3], 0
 * Both targets: DW0 = 0xF40001C1
 * GFX11: DW1 SOFFSET null=0x7C -> 0xF8000000
 * GFX10: DW1 SOFFSET null=0x7D -> 0xFA000000 */

static void enc05(void)
{
    /* GFX11 */
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_LOAD_DWORD;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);   /* SDATA */
    mi->operands[1] = sgpr(2);   /* SBASE (pair s[2:3], /2 = 1) */
    mi->operands[2] = imm(0);    /* offset */

    encode_function(A, 0);
    CHEQX(dw(0), 0xF40001C1u);
    CHEQX(dw(1), 0xF8000000u);

    /* GFX10 -- same DW0, different SOFFSET null */
    enc_setup(AMD_TARGET_GFX1030);
    mi              = &A->minsts[0];
    mi->op          = AMD_S_LOAD_DWORD;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);
    mi->operands[2] = imm(0);

    encode_function(A, 0);
    CHEQX(dw(0), 0xF40001C1u);
    CHEQX(dw(1), 0xFA000000u);
    PASS();
}
TH_REG("enc", 5, "SMEM differs between GFX11 and GFX10", enc05)

/* ---- encode: global_load_dword ---- */
/* global_load_dword v5, v2 -- GFX11
 * DW0: [31:26]=0x37 [25:18]=OP(0x14) [17:16]=SEG(2=global) [12:0]=offset(0)
 *      = 0xDC520000
 * DW1: [31:24]=VDST(5) [23:16]=SADDR(0x7C=null) [15:8]=DATA(0) [7:0]=ADDR(2)
 *      = 0x057C0002 */

static void enc06(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_GLOBAL_LOAD_DWORD;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = vgpr(5);   /* VDST */
    mi->operands[1] = vgpr(2);   /* ADDR */

    encode_function(A, 0);
    CHEQX(dw(0), 0xDC520000u);
    CHEQX(dw(1), 0x057C0002u);
    PASS();
}
TH_REG("enc", 6, "global_load_dword encodes", enc06)

/* ---- encode: DS ---- */
/* ds_load_b32 v5, v2 -- GFX11
 * DW0: [31:26]=0x36 [25:18]=OP(0x36) [15:0]=offset(0)
 *      = 0xD8D80000
 * DW1: [31:24]=VDST(5) [15:8]=DATA0(0) [7:0]=ADDR(2)
 *      = 0x05000002 */

static void enc07(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_DS_READ_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = vgpr(5);   /* VDST */
    mi->operands[1] = vgpr(2);   /* ADDR */

    encode_function(A, 0);
    CHEQX(dw(0), 0xD8D80000u);
    CHEQX(dw(1), 0x05000002u);
    PASS();
}
TH_REG("enc", 7, "DS encodes ds_read_b32", enc07)

/* ---- encode: s_endpgm ---- */
/* GFX10: SOPP hw_op=0x01, SIMM16=0
 * Expected: 0xBF810000 */

static void enc08(void)
{
    enc_setup(AMD_TARGET_GFX1030);
    minst_t *mi  = &A->minsts[0];
    mi->op       = AMD_S_ENDPGM;
    mi->num_defs = 0;
    mi->num_uses = 0;

    encode_function(A, 0);
    CHEQX(dw(0), 0xBF810000u);
    PASS();
}
TH_REG("enc", 8, "s_endpgm encodes", enc08)

/* ---- encode: s_waitcnt vmcnt(0) ---- */
/* GFX10: hw_op=0x0C, vmcnt=0, lgkmcnt=63, expcnt=7
 * SIMM16 = [15:14]=vmhi(0) [13:8]=lgkm(63) [6:4]=exp(7) [3:0]=vmlo(0)
 *         = 0x3F70
 * DW = 0xBF8C3F70 */

static void enc09(void)
{
    enc_setup(AMD_TARGET_GFX1030);
    minst_t *mi  = &A->minsts[0];
    mi->op       = AMD_S_WAITCNT;
    mi->num_defs = 0;
    mi->num_uses = 0;
    mi->flags    = AMD_WAIT_VMCNT0;

    encode_function(A, 0);
    CHEQX(dw(0), 0xBF8C3F70u);
    PASS();
}
TH_REG("enc", 9, "s_waitcnt vmcnt(0) encodes", enc09)

/* ---- encode: SOP2 with a non-zero opcode ---- */
/* enc02 above uses s_add_u32, whose opcode is 0x00, so the OP field can sit
 * anywhere in the word and the bits still come out right. s_andn2_b32 is 0x22
 * on GFX11 and 0x12 on GFX10, which pins both the field position and the
 * per-generation table.
 *
 * s_andn2_b32 s7, s2, s3
 * [31:30]=10 [29:23]=OP [22:16]=SDST(7) [15:8]=SSRC1(3) [7:0]=SSRC0(2) */

static void enc10(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_ANDN2_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);
    mi->operands[2] = sgpr(3);

    encode_function(A, 0);
    CHEQX(dw(0), 0x91070302u);
    PASS();
}
TH_REG("enc", 10, "SOP2 opcode field is where it should be", enc10)

static void enc11(void)
{
    enc_setup(AMD_TARGET_GFX1030);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_ANDN2_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);
    mi->operands[2] = sgpr(3);

    encode_function(A, 0);
    CHEQX(dw(0), 0x8A070302u);
    PASS();
}
TH_REG("enc", 11, "GFX10 renumbers the same mnemonic", enc11)

/* ---- encode: SMEM on CDNA ---- */
/* enc05 covers GFX11 and GFX10. The GFX9 branch has its own dword layout and
 * had no test at all, which matters because MI300X is a shipping target.
 *
 * s_load_dword s7, s[2:3], 0 -- GFX942
 * DW0: [31:26]=110000 [25:18]=OP(0x00) [17]=IMM(1) [12:6]=SDATA(7) [5:0]=SBASE(1)
 * DW1: [20:0]=OFFSET */

static void enc12(void)
{
    enc_setup(AMD_TARGET_GFX942);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_S_LOAD_DWORD;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = sgpr(7);
    mi->operands[1] = sgpr(2);
    mi->operands[2] = imm(0);

    encode_function(A, 0);
    /* Bit 17 is IMM. Clear it and the hardware reads the offset out of a
     * register that was never set up. */
    CHEQX(dw(0), 0xC00201C1u);
    CHEQX(dw(1), 0x00000000u);
    PASS();
}
TH_REG("enc", 12, "SMEM encodes on CDNA", enc12)

/* ---- encode: bit counting ---- */
/* The opcodes below were read back out of llvm-mc for gfx1100, gfx1030 and
 * gfx900, so if one of these fails either the table drifted or AMD renamed
 * something again. GFX11 calls ffbl "v_ctz_i32_b32"; same bits either way. */

static void enc13(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_FFBL_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = vgpr(10);
    mi->operands[1] = vgpr(11);

    encode_function(A, 0);
    CHEQX(dw(0), 0x7E14750Bu);
    PASS();
}
TH_REG("enc", 13, "VOP1 encodes ffbl/ctz", enc13)

/* v_bcnt_u32_b32 is VOP3 and the opcode moves with every generation:
 * 0x31E on GFX11, 0x364 on GFX10, 0x28B back on GFX9. */
static void enc14(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_BCNT_U32_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = vgpr(1);
    mi->operands[1] = vgpr(11);
    mi->operands[2] = imm(0);

    encode_function(A, 0);
    CHEQX(dw(0), 0xD71E0001u);
    CHEQX(dw(1), 0x0001010Bu);
    PASS();
}
TH_REG("enc", 14, "VOP3 encodes v_bcnt_u32_b32", enc14)

static void enc15(void)
{
    enc_setup(AMD_TARGET_GFX942);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_BFREV_B32;
    mi->num_defs    = 1;
    mi->num_uses    = 1;
    mi->operands[0] = vgpr(1);
    mi->operands[1] = vgpr(11);

    encode_function(A, 0);
    CHEQX(dw(0), 0x7E02590Bu);
    PASS();
}
TH_REG("enc", 15, "CDNA renumbers v_bfrev_b32", enc15)

/* src0 here is the inline constant 32, which lives at 160 in the operand
 * map, not a literal dword trailing the instruction. */
static void enc16(void)
{
    enc_setup(AMD_TARGET_GFX1100);
    minst_t *mi     = &A->minsts[0];
    mi->op          = AMD_V_MIN_U32;
    mi->num_defs    = 1;
    mi->num_uses    = 2;
    mi->operands[0] = vgpr(0);
    mi->operands[1] = imm(32);
    mi->operands[2] = vgpr(10);

    encode_function(A, 0);
    CHEQX(dw(0), 0x260014A0u);
    PASS();
}
TH_REG("enc", 16, "VOP2 encodes v_min_u32 with inline 32", enc16)
