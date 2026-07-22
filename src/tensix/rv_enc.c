#include "rv_enc.h"

/*
 * RV32IM bit-packing. One inline format helper per encoding shape,
 * then a one-liner per instruction that fills in opcode, funct3,
 * funct7 from the spec tables. Spec page references are in the
 * header next to the instruction class headings.
 *
 * Style choice: every immediate gets masked to its declared width
 * before being shifted in, so a caller that passes a value out of
 * range produces deterministic wrap rather than UB. The isel above
 * is responsible for range-checking before we get here; the encoder
 * just packs bits the spec says to pack.
 */

/* ---- Format helpers ---- */

static uint32_t enc_r(uint32_t f7, uint8_t rs2, uint8_t rs1,
                      uint32_t f3, uint8_t rd, uint32_t op)
{
    return ((f7 & 0x7Fu)         << 25)
         | ((uint32_t)(rs2 & 0x1Fu) << 20)
         | ((uint32_t)(rs1 & 0x1Fu) << 15)
         | ((f3 & 0x7u)          << 12)
         | ((uint32_t)(rd  & 0x1Fu) << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_i(int16_t imm, uint8_t rs1, uint32_t f3,
                      uint8_t rd, uint32_t op)
{
    return (((uint32_t)imm & 0xFFFu) << 20)
         | ((uint32_t)(rs1 & 0x1Fu)  << 15)
         | ((f3 & 0x7u)              << 12)
         | ((uint32_t)(rd  & 0x1Fu)  << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_ish(uint32_t f7, uint8_t shamt, uint8_t rs1,
                            uint32_t f3, uint8_t rd, uint32_t op)
{
    /* I-type immediate split for shifts: bits[24:20] are the 5-bit
     * shift amount, bits[31:25] become a funct7 field that
     * distinguishes arithmetic SRAI from logical SRLI. Spec p.26. */
    return ((f7 & 0x7Fu)            << 25)
         | ((uint32_t)(shamt & 0x1Fu) << 20)
         | ((uint32_t)(rs1   & 0x1Fu) << 15)
         | ((f3 & 0x7u)              << 12)
         | ((uint32_t)(rd    & 0x1Fu) << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_s(int16_t imm, uint8_t rs2, uint8_t rs1,
                      uint32_t f3, uint32_t op)
{
    uint32_t u = (uint32_t)imm & 0xFFFu;
    return ((u >> 5)                     << 25)
         | ((uint32_t)(rs2 & 0x1Fu)      << 20)
         | ((uint32_t)(rs1 & 0x1Fu)      << 15)
         | ((f3 & 0x7u)                  << 12)
         | ((u & 0x1Fu)                  << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_b(int16_t offset, uint8_t rs1, uint8_t rs2,
                      uint32_t f3, uint32_t op)
{
    /* B-type immediate scrambling, spec p.24 and p.30: bit 12 goes to
     * inst[31], bits 10:5 to inst[30:25], bits 4:1 to inst[11:8], and bit
     * 11 to inst[7]. Bit 0 of the offset is always zero because branch
     * targets are 2-byte aligned and the encoding has no room for it. */
    uint32_t u  = (uint32_t)offset & 0x1FFFu;     /* 13-bit field */
    uint32_t b12 = (u >> 12) & 0x1u;
    uint32_t b11 = (u >> 11) & 0x1u;
    uint32_t b10_5 = (u >> 5) & 0x3Fu;
    uint32_t b4_1  = (u >> 1) & 0xFu;
    return (b12   << 31)
         | (b10_5 << 25)
         | ((uint32_t)(rs2 & 0x1Fu) << 20)
         | ((uint32_t)(rs1 & 0x1Fu) << 15)
         | ((f3 & 0x7u) << 12)
         | (b4_1  << 8)
         | (b11   << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_u(uint32_t imm_hi20, uint8_t rd, uint32_t op)
{
    /* U-type: the 20-bit immediate occupies bits[31:12]; the lower
     * 12 bits of the produced register are zero. Callers pass the
     * upper 20 bits already shifted-down so we just place them. */
    return ((imm_hi20 & 0xFFFFFu) << 12)
         | ((uint32_t)(rd & 0x1Fu) << 7)
         |  (op & 0x7Fu);
}

static uint32_t enc_j(int32_t offset, uint8_t rd, uint32_t op)
{
    /* J-type immediate scrambling, spec p.24: bit 20 goes to inst[31],
     * bits 10:1 to inst[30:21], bit 11 to inst[20], and bits 19:12 stay
     * at inst[19:12]. Same trick as B-type, the middle bits stay put, the
     * sign bit sits at top, and the 11/12-bit boundary slots in where
     * there is room. */
    uint32_t u  = (uint32_t)offset & 0x1FFFFFu;       /* 21-bit field */
    uint32_t b20    = (u >> 20) & 0x1u;
    uint32_t b19_12 = (u >> 12) & 0xFFu;
    uint32_t b11    = (u >> 11) & 0x1u;
    uint32_t b10_1  = (u >> 1)  & 0x3FFu;
    return (b20     << 31)
         | (b10_1   << 21)
         | (b11     << 20)
         | (b19_12  << 12)
         | ((uint32_t)(rd & 0x1Fu) << 7)
         |  (op & 0x7Fu);
}

/* ---- R-type ALU (OP, opcode=0x33) ---- */

uint32_t rv_add (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x0, rd, 0x33); }
uint32_t rv_sub (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x20, rs2, rs1, 0x0, rd, 0x33); }
uint32_t rv_and (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x7, rd, 0x33); }
uint32_t rv_or  (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x6, rd, 0x33); }
uint32_t rv_xor (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x4, rd, 0x33); }
uint32_t rv_sll (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x1, rd, 0x33); }
uint32_t rv_srl (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x5, rd, 0x33); }
uint32_t rv_sra (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x20, rs2, rs1, 0x5, rd, 0x33); }
uint32_t rv_slt (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x2, rd, 0x33); }
uint32_t rv_sltu(uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x00, rs2, rs1, 0x3, rd, 0x33); }

/* ---- I-type ALU (OP-IMM, opcode=0x13) ---- */

uint32_t rv_addi (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x0, rd, 0x13); }
uint32_t rv_andi (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x7, rd, 0x13); }
uint32_t rv_ori  (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x6, rd, 0x13); }
uint32_t rv_xori (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x4, rd, 0x13); }
uint32_t rv_slti (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x2, rd, 0x13); }
uint32_t rv_sltiu(uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x3, rd, 0x13); }

uint32_t rv_slli(uint8_t rd, uint8_t rs1, uint8_t shamt) { return enc_ish(0x00, shamt, rs1, 0x1, rd, 0x13); }
uint32_t rv_srli(uint8_t rd, uint8_t rs1, uint8_t shamt) { return enc_ish(0x00, shamt, rs1, 0x5, rd, 0x13); }
uint32_t rv_srai(uint8_t rd, uint8_t rs1, uint8_t shamt) { return enc_ish(0x20, shamt, rs1, 0x5, rd, 0x13); }

/* ---- Loads (LOAD, opcode=0x03) ---- */

uint32_t rv_lw (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x2, rd, 0x03); }
uint32_t rv_lh (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x1, rd, 0x03); }
uint32_t rv_lhu(uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x5, rd, 0x03); }
uint32_t rv_lb (uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x0, rd, 0x03); }
uint32_t rv_lbu(uint8_t rd, uint8_t rs1, int16_t imm) { return enc_i(imm, rs1, 0x4, rd, 0x03); }

/* ---- Stores (STORE, opcode=0x23) ---- */

uint32_t rv_sw(uint8_t rs2, uint8_t rs1, int16_t imm) { return enc_s(imm, rs2, rs1, 0x2, 0x23); }
uint32_t rv_sh(uint8_t rs2, uint8_t rs1, int16_t imm) { return enc_s(imm, rs2, rs1, 0x1, 0x23); }
uint32_t rv_sb(uint8_t rs2, uint8_t rs1, int16_t imm) { return enc_s(imm, rs2, rs1, 0x0, 0x23); }

/* ---- Branches (BRANCH, opcode=0x63) ---- */

uint32_t rv_beq (uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x0, 0x63); }
uint32_t rv_bne (uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x1, 0x63); }
uint32_t rv_blt (uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x4, 0x63); }
uint32_t rv_bge (uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x5, 0x63); }
uint32_t rv_bltu(uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x6, 0x63); }
uint32_t rv_bgeu(uint8_t rs1, uint8_t rs2, int16_t off) { return enc_b(off, rs1, rs2, 0x7, 0x63); }

/* ---- Jumps and upper immediates ---- */

uint32_t rv_jal  (uint8_t rd, int32_t off)               { return enc_j(off, rd, 0x6F); }
uint32_t rv_jalr (uint8_t rd, uint8_t rs1, int16_t imm)  { return enc_i(imm, rs1, 0x0, rd, 0x67); }
uint32_t rv_lui  (uint8_t rd, uint32_t hi20)             { return enc_u(hi20, rd, 0x37); }
uint32_t rv_auipc(uint8_t rd, uint32_t hi20)             { return enc_u(hi20, rd, 0x17); }

/* ---- M extension ---- */

uint32_t rv_mul   (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x0, rd, 0x33); }
uint32_t rv_mulh  (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x1, rd, 0x33); }
uint32_t rv_mulhsu(uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x2, rd, 0x33); }
uint32_t rv_mulhu (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x3, rd, 0x33); }
uint32_t rv_div   (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x4, rd, 0x33); }
uint32_t rv_divu  (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x5, rd, 0x33); }
uint32_t rv_rem   (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x6, rd, 0x33); }
uint32_t rv_remu  (uint8_t rd, uint8_t rs1, uint8_t rs2) { return enc_r(0x01, rs2, rs1, 0x7, rd, 0x33); }

/* ---- System / synchronisation ---- */

uint32_t rv_fence(uint8_t pred, uint8_t succ)
{
    /* FENCE encoding, spec p.32: fm in bits[31:28] is zero for a normal
     * fence, pred (operations before) in bits[27:24], succ (operations
     * after) in bits[23:20], with rs1, funct3 and rd all zero on opcode
     * 0x0F. Baby cores treat the whole thing as a nop but we emit the
     * canonical encoding so future hardware revisions and any downstream
     * disassembler see the right bits. */
    return ((uint32_t)(pred & 0xFu) << 24)
         | ((uint32_t)(succ & 0xFu) << 20)
         | 0x0Fu;
}

uint32_t rv_ecall (void) { return 0x00000073u; }
uint32_t rv_ebreak(void) { return 0x00100073u; }

uint32_t rv_nop(void) { return rv_addi(RV_X0, RV_X0, 0); }
