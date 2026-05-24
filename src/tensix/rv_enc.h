#ifndef BARRACUDA_RV_ENC_H
#define BARRACUDA_RV_ENC_H

#include "barracuda.h"

/*
 * RV32IM instruction encoder for the Tenstorrent Wormhole baby
 * RISC-V cores. Pure bit-packing, no assembler, no parser. Each
 * function returns a 32-bit little-endian instruction word.
 *
 * Spec sources cited inline next to each opcode:
 *   docs/RISC-V_Unprivileged_ISA_2024.pdf
 *   tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/README.md
 *
 * RV32I base + M extension only. No F/D/A/C/V/CSR. The baby cores
 * do not implement those and the .ttinsn coprocessor extension
 * lives in a separate emitter once we get there.
 *
 * Calling convention reminder (from spec page 21):
 *   x0       hard zero
 *   x1       ra, return address
 *   x2       sp, stack pointer
 *   x10-x17  a0-a7, argument and return registers
 *   x5-x7    t0-t2, temporaries (caller-saved)
 *   x28-x31  t3-t6, temporaries (caller-saved)
 *   x8-x9    s0-s1, callee-saved
 *   x18-x27  s2-s11, callee-saved
 *
 * Soft-float libcalls pass FP values in a0/a1 and return there too,
 * which is what the soft-float runtime expects when we link it in.
 */

/* ---- Register names ---- */

enum {
    RV_X0 = 0,  RV_X1,  RV_X2,  RV_X3,  RV_X4,  RV_X5,  RV_X6,  RV_X7,
    RV_X8,      RV_X9,  RV_X10, RV_X11, RV_X12, RV_X13, RV_X14, RV_X15,
    RV_X16,     RV_X17, RV_X18, RV_X19, RV_X20, RV_X21, RV_X22, RV_X23,
    RV_X24,     RV_X25, RV_X26, RV_X27, RV_X28, RV_X29, RV_X30, RV_X31,
    RV_REG_COUNT
};

/* ABI aliases. The compiler emits the same bits either way but the
 * isel reads better if it can say a0 instead of x10. */
enum {
    RV_ZERO = 0,  RV_RA = 1,  RV_SP = 2,  RV_GP = 3,
    RV_TP   = 4,  RV_T0 = 5,  RV_T1 = 6,  RV_T2 = 7,
    RV_S0   = 8,  RV_S1 = 9,  RV_A0 = 10, RV_A1 = 11,
    RV_A2   = 12, RV_A3 = 13, RV_A4 = 14, RV_A5 = 15,
    RV_A6   = 16, RV_A7 = 17, RV_S2 = 18, RV_S3 = 19,
    RV_S4   = 20, RV_S5 = 21, RV_S6 = 22, RV_S7 = 23,
    RV_S8   = 24, RV_S9 = 25, RV_S10= 26, RV_S11= 27,
    RV_T3   = 28, RV_T4 = 29, RV_T5 = 30, RV_T6 = 31
};

/* ---- R-type ALU (OP, 0x33) ---- */

uint32_t rv_add (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_sub (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_and (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_or  (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_xor (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_sll (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_srl (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_sra (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_slt (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_sltu(uint8_t rd, uint8_t rs1, uint8_t rs2);

/* ---- I-type ALU (OP-IMM, 0x13) ---- */

uint32_t rv_addi (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_andi (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_ori  (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_xori (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_slti (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_sltiu(uint8_t rd, uint8_t rs1, int16_t imm);

/* I-type shifts use a 5-bit shamt in bits[24:20], funct7 in [31:25]
 * distinguishes SRAI (funct7 bit 5 = 1) from SRLI. */
uint32_t rv_slli(uint8_t rd, uint8_t rs1, uint8_t shamt);
uint32_t rv_srli(uint8_t rd, uint8_t rs1, uint8_t shamt);
uint32_t rv_srai(uint8_t rd, uint8_t rs1, uint8_t shamt);

/* ---- Loads (LOAD, 0x03) ---- */

uint32_t rv_lw (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_lh (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_lhu(uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_lb (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_lbu(uint8_t rd, uint8_t rs1, int16_t imm);

/* ---- Stores (STORE, 0x23) ----
 * Assembler order: sw rs2, offset(rs1). Value first, then address. */

uint32_t rv_sw(uint8_t rs2, uint8_t rs1, int16_t imm);
uint32_t rv_sh(uint8_t rs2, uint8_t rs1, int16_t imm);
uint32_t rv_sb(uint8_t rs2, uint8_t rs1, int16_t imm);

/* ---- Branches (BRANCH, 0x63) ----
 * Offset must be a multiple of 2; range is signed 13-bit which
 * gives 4 KiB forward or back. */

uint32_t rv_beq (uint8_t rs1, uint8_t rs2, int16_t offset);
uint32_t rv_bne (uint8_t rs1, uint8_t rs2, int16_t offset);
uint32_t rv_blt (uint8_t rs1, uint8_t rs2, int16_t offset);
uint32_t rv_bge (uint8_t rs1, uint8_t rs2, int16_t offset);
uint32_t rv_bltu(uint8_t rs1, uint8_t rs2, int16_t offset);
uint32_t rv_bgeu(uint8_t rs1, uint8_t rs2, int16_t offset);

/* ---- Jumps and upper immediates ----
 * JAL offset is signed 21-bit, multiple of 2: +/- 1 MiB.
 * LUI/AUIPC immediate occupies bits[31:12] of the result, the
 * lower 12 are zero. */

uint32_t rv_jal  (uint8_t rd, int32_t offset);
uint32_t rv_jalr (uint8_t rd, uint8_t rs1, int16_t imm);
uint32_t rv_lui  (uint8_t rd, uint32_t imm_hi20);
uint32_t rv_auipc(uint8_t rd, uint32_t imm_hi20);

/* ---- M extension (OP=0x33 with funct7=0x01) ---- */

uint32_t rv_mul   (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_mulh  (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_mulhsu(uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_mulhu (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_div   (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_divu  (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_rem   (uint8_t rd, uint8_t rs1, uint8_t rs2);
uint32_t rv_remu  (uint8_t rd, uint8_t rs1, uint8_t rs2);

/* ---- System / synchronisation (SYSTEM, 0x73) ----
 * FENCE is encoded faithfully even though baby cores execute it
 * as nop; ECALL/EBREAK trigger debug pause on baby cores rather
 * than a real trap, per the BabyRISCV README. */

uint32_t rv_fence (uint8_t pred, uint8_t succ);
uint32_t rv_ecall (void);
uint32_t rv_ebreak(void);

/* ---- Convenience ----
 * Canonical NOP per spec: addi x0, x0, 0. */

uint32_t rv_nop(void);

#endif /* BARRACUDA_RV_ENC_H */
