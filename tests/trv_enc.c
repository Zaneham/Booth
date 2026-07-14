/* trv_enc.c -- RV32IM instruction encoder tests.
 * Verify bit patterns against hand-computed values from the spec
 * layouts. Each test cites the encoding that should appear. */

#include "tharns.h"
#include "rv_enc.h"

/* ---- R-type ALU ---- */
/* add x0, x0, x0 is 0x00000033: opcode 0x33 with every field zero.
 * add x5, x6, x7 = 0x00730333... wait, let's compute properly:
 * funct7=0, rs2=7, rs1=6, funct3=0, rd=5, opcode=0x33 packs to
 * (7<<20) | (6<<15) | (5<<7) | 0x33 = 0x007302B3. */

static void rv_add_zero(void)
{
    CHEQ(rv_add(RV_X0, RV_X0, RV_X0), 0x00000033u);
    PASS();
}
TH_REG("rv_enc", rv_add_zero);

static void rv_add_x5_x6_x7(void)
{
    CHEQ(rv_add(RV_X5, RV_X6, RV_X7), 0x007302B3u);
    PASS();
}
TH_REG("rv_enc", rv_add_x5_x6_x7);

/* sub differs from add only in funct7 bit 5 (funct7 = 0x20), so
 * sub x5, x6, x7 = 0x40000000 | 0x007302B3 = 0x407302B3. */

static void rv_sub_x5_x6_x7(void)
{
    CHEQ(rv_sub(RV_X5, RV_X6, RV_X7), 0x407302B3u);
    PASS();
}
TH_REG("rv_enc", rv_sub_x5_x6_x7);

/* sra differs from srl only in funct7 bit 5. */

static void rv_sra_vs_srl(void)
{
    uint32_t srl = rv_srl(RV_X1, RV_X2, RV_X3);
    uint32_t sra = rv_sra(RV_X1, RV_X2, RV_X3);
    CHEQ(sra ^ srl, 0x40000000u);   /* differ only in bit 30 */
    PASS();
}
TH_REG("rv_enc", rv_sra_vs_srl);

/* ---- I-type ALU ---- */
/* addi x1, x0, 5: imm=5 in bits [31:20], rd=1, opcode 0x13, giving
 * (5<<20) | (1<<7) | 0x13 = 0x00500093. */

static void rv_addi_pos(void)
{
    CHEQ(rv_addi(RV_X1, RV_X0, 5), 0x00500093u);
    PASS();
}
TH_REG("rv_enc", rv_addi_pos);

/* Negative immediates sign-extend through the 12-bit field. For
 * addi x1, x0, -1 the imm masks to 0xFFF, so (0xFFF<<20) | (1<<7) | 0x13
 * = 0xFFF00093. */

static void rv_addi_neg(void)
{
    CHEQ(rv_addi(RV_X1, RV_X0, -1), 0xFFF00093u);
    PASS();
}
TH_REG("rv_enc", rv_addi_neg);

/* SLLI uses a 5-bit shamt in bits [24:20] with funct7=0x00, so
 * slli x1, x2, 4 = (4<<20) | (2<<15) | (1<<12) | (1<<7) | 0x13 = 0x00411093. */

static void rv_slli_basic(void)
{
    CHEQ(rv_slli(RV_X1, RV_X2, 4), 0x00411093u);
    PASS();
}
TH_REG("rv_enc", rv_slli_basic);

/* SRAI funct7=0x20. */

static void rv_srai_basic(void)
{
    uint32_t srli = rv_srli(RV_X1, RV_X2, 4);
    uint32_t srai = rv_srai(RV_X1, RV_X2, 4);
    CHEQ(srai ^ srli, 0x40000000u);
    PASS();
}
TH_REG("rv_enc", rv_srai_basic);

/* ---- Loads ---- */
/* lw x1, 0(x2): rs1=2, funct3=2 (LW), rd=1, opcode 0x03, giving
 * (2<<15) | (2<<12) | (1<<7) | 0x03 = 0x00012083. */

static void rv_lw_basic(void)
{
    CHEQ(rv_lw(RV_X1, RV_X2, 0), 0x00012083u);
    PASS();
}
TH_REG("rv_enc", rv_lw_basic);

/* lw with a non-zero, sign-extended offset: for lw x1, -4(x2) the
 * imm masks to 0xFFC, giving 0xFFC00000 | 0x00012083 = 0xFFC12083. */

static void rv_lw_neg_off(void)
{
    CHEQ(rv_lw(RV_X1, RV_X2, -4), 0xFFC12083u);
    PASS();
}
TH_REG("rv_enc", rv_lw_neg_off);

/* ---- Stores ---- */
/* sw x2, 0(x1): rs2=2, rs1=1, funct3=2, opcode 0x23, and no S-type
 * bit-scrambling matters since imm=0, giving (2<<20) | (1<<15) | (2<<12)
 * | 0x23 = 0x0020A023. */

static void rv_sw_basic(void)
{
    CHEQ(rv_sw(RV_X2, RV_X1, 0), 0x0020A023u);
    PASS();
}
TH_REG("rv_enc", rv_sw_basic);

/* sw with a scrambled immediate verifies the S-type bit split. For
 * sw x2, 24(x1), 24 splits to imm[11:5]=0 and imm[4:0]=0x18, so
 * (2<<20) | (1<<15) | (2<<12) | (0x18<<7) | 0x23 = 0x0020AC23. */

static void rv_sw_off24(void)
{
    CHEQ(rv_sw(RV_X2, RV_X1, 24), 0x0020AC23u);
    PASS();
}
TH_REG("rv_enc", rv_sw_off24);

/* ---- Branches ---- */
/* beq x0, x0, 0: opcode 0x63, funct3 0, all registers and immediate
 * bits zero, giving 0x00000063. */

static void rv_beq_zero(void)
{
    CHEQ(rv_beq(RV_X0, RV_X0, 0), 0x00000063u);
    PASS();
}
TH_REG("rv_enc", rv_beq_zero);

/* beq x5, x6, 8: offset 8 puts imm[4:1]=0b0100 into bits [11:8], so
 * (6<<20) | (5<<15) | (4<<8) | 0x63 = 0x00628463. */

static void rv_beq_off8(void)
{
    CHEQ(rv_beq(RV_X5, RV_X6, 8), 0x00628463u);
    PASS();
}
TH_REG("rv_enc", rv_beq_off8);

/* A negative branch tests the sign bit landing in inst[31]. Offset -4
 * is 0x1FFC in the 13-bit field, scattering to 0x80000000 | 0x7E000000
 * | 0xE00 | 0x80 | 0x63 = 0xFE000EE3. */

static void rv_beq_back(void)
{
    CHEQ(rv_beq(RV_X0, RV_X0, -4), 0xFE000EE3u);
    PASS();
}
TH_REG("rv_enc", rv_beq_back);

/* ---- Jumps and uppers ---- */
/* jal x1, 0: opcode 0x6F, rd=1, offset 0, giving (1<<7) | 0x6F = 0x000000EF. */

static void rv_jal_zero(void)
{
    CHEQ(rv_jal(RV_X1, 0), 0x000000EFu);
    PASS();
}
TH_REG("rv_enc", rv_jal_zero);

/* lui x1, 0x12345: imm[31:12]=0x12345, rd=1, opcode 0x37, giving
 * 0x12345000 | 0xB7 = 0x123450B7. */

static void rv_lui_basic(void)
{
    CHEQ(rv_lui(RV_X1, 0x12345u), 0x123450B7u);
    PASS();
}
TH_REG("rv_enc", rv_lui_basic);

/* auipc differs from lui only in opcode (0x17 vs 0x37). */

static void rv_auipc_vs_lui(void)
{
    uint32_t lui   = rv_lui(RV_X1, 0x12345u);
    uint32_t auipc = rv_auipc(RV_X1, 0x12345u);
    CHEQ(lui ^ auipc, 0x20u);  /* opcode bits 6:0 differ by 0x20 */
    PASS();
}
TH_REG("rv_enc", rv_auipc_vs_lui);

/* ---- M extension ---- */
/* mul x5, x6, x7: funct7=0x01, rs2=7, rs1=6, rd=5, opcode 0x33, giving
 * 0x02000000 | (7<<20) | (6<<15) | (5<<7) | 0x33 = 0x027302B3. */

static void rv_mul_basic(void)
{
    CHEQ(rv_mul(RV_X5, RV_X6, RV_X7), 0x027302B3u);
    PASS();
}
TH_REG("rv_enc", rv_mul_basic);

/* div and divu share the funct7=0x01 prefix with mul; differ only
 * in funct3. div: funct3=4, mul: funct3=0. */

static void rv_div_vs_mul(void)
{
    uint32_t mul = rv_mul(RV_X5, RV_X6, RV_X7);
    uint32_t div = rv_div(RV_X5, RV_X6, RV_X7);
    CHEQ(div ^ mul, 0x4000u);  /* funct3 bit 2 differs */
    PASS();
}
TH_REG("rv_enc", rv_div_vs_mul);

/* All 8 M-extension instructions must have funct7 bit 25 set. */

static void rv_m_ext_funct7(void)
{
    CHEQ((rv_mul   (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_mulh  (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_mulhsu(RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_mulhu (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_div   (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_divu  (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_rem   (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    CHEQ((rv_remu  (RV_X1, RV_X2, RV_X3) >> 25) & 0x7Fu, 0x01u);
    PASS();
}
TH_REG("rv_enc", rv_m_ext_funct7);

/* ---- System ---- */

static void rv_ecall_const(void)
{
    CHEQ(rv_ecall(),  0x00000073u);
    PASS();
}
TH_REG("rv_enc", rv_ecall_const);

static void rv_ebreak_const(void)
{
    CHEQ(rv_ebreak(), 0x00100073u);
    PASS();
}
TH_REG("rv_enc", rv_ebreak_const);

/* nop is addi x0, x0, 0 = 0x00000013 */

static void rv_nop_canonical(void)
{
    CHEQ(rv_nop(), 0x00000013u);
    PASS();
}
TH_REG("rv_enc", rv_nop_canonical);

/* fence rw, rw with pred=0xF and succ=0xF gives (0xF<<24) | (0xF<<20)
 * | 0x0F = 0x0FF0000F. */

static void rv_fence_rwrw(void)
{
    CHEQ(rv_fence(0xF, 0xF), 0x0FF0000Fu);
    PASS();
}
TH_REG("rv_enc", rv_fence_rwrw);
