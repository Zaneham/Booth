/*
 * noc.c -- NoC transfer codegen for the baby RISC-V data-movement cores
 *
 * Lowers a tile read/write into the RISC-V store sequence that programs an NIU
 * request initiator and triggers it. Register addresses from the Wormhole B0
 * ISA docs (NoC/MemoryMap.md). The encoding is checkable with a RISC-V
 * disassembler; whether the transfer actually moves the tile is a question
 * only real hardware can answer.
 */

#include "barracuda.h"
#include "rv_enc.h"
#include "rv_buf.h"
#include "noc.h"

/* NIU request initiator 0 (NoC #0). */
#define NIU_BASE        0xFFB20000u
#define NOC_TARG_LO     0x00
#define NOC_TARG_MID    0x04
#define NOC_RET_LO      0x0C
#define NOC_RET_MID     0x10
#define NOC_CTRL        0x1C
#define NOC_AT_LEN_BE   0x20
#define NOC_CMD_CTRL    0x28

/* NOC_CTRL request type (bits [1:0]). */
#define NOC_CMD_RD      0u
#define NOC_CMD_WR      2u

/* RISC-V registers (ABI names): x5 = t0 base, x6 = t1 value, x0 = zero. */
#define RV_X0   0
#define RV_T0   5
#define RV_T1   6

/* Materialise a 32-bit value into reg via lui + addi, compensating for addi's
 * sign extension of the low 12 bits. */
static void
li32(rv_buf_t *c, uint8_t reg, uint32_t v)
{
    uint32_t hi = (v >> 12) & 0xFFFFFu;
    int32_t  lo = (int32_t)(v & 0xFFFu);
    if (lo & 0x800) { lo -= 0x1000; hi = (hi + 1u) & 0xFFFFFu; }
    if (hi) {
        rv_buf_emit(c, rv_lui(reg, hi));
        if (lo) rv_buf_emit(c, rv_addi(reg, reg, (int16_t)lo));
    } else {
        rv_buf_emit(c, rv_addi(reg, RV_X0, (int16_t)lo));
    }
}

/* Store value v to NIU register at offset off from the base in t0. */
static void
store_reg(rv_buf_t *c, uint32_t v, int16_t off)
{
    li32(c, RV_T1, v);
    rv_buf_emit(c, rv_sw(RV_T1, RV_T0, off));
}

static int
noc_xfer(rv_buf_t *c, uint32_t ctrl, uint32_t tlo, uint32_t tmid,
         uint32_t rlo, uint32_t rmid, uint32_t len)
{
    li32(c, RV_T0, NIU_BASE);
    store_reg(c, tlo,  NOC_TARG_LO);
    store_reg(c, tmid, NOC_TARG_MID);
    store_reg(c, rlo,  NOC_RET_LO);
    store_reg(c, rmid, NOC_RET_MID);
    store_reg(c, len,  NOC_AT_LEN_BE);
    store_reg(c, ctrl, NOC_CTRL);
    store_reg(c, 1u,   NOC_CMD_CTRL);   /* write 1 to trigger the request */
    return BC_OK;
}

int
tt_noc_read(rv_buf_t *c, uint32_t tlo, uint32_t tmid,
            uint32_t rlo, uint32_t rmid, uint32_t len)
{
    return noc_xfer(c, NOC_CMD_RD, tlo, tmid, rlo, rmid, len);
}

int
tt_noc_write(rv_buf_t *c, uint32_t tlo, uint32_t tmid,
             uint32_t rlo, uint32_t rmid, uint32_t len)
{
    return noc_xfer(c, NOC_CMD_WR, tlo, tmid, rlo, rmid, len);
}
