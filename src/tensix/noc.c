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
#define NOC_AT_DATA     0x24
#define NOC_CMD_CTRL    0x28

/* NOC_CTRL request type (bits [1:0]). */
#define NOC_CMD_RD      0u
#define NOC_CMD_AT      1u
#define NOC_CMD_WR      2u

/* NOC_AT_LEN_BE for an atomic increment (NoC/Atomics.md): op selector in
 * [15:12] (1 = increment), IntWidth in [6:2], Ofs in [1:0]. IntWidth=31 is a
 * full 32-bit add; Ofs=0 makes the target address the operand address. */
#define NOC_AT_INC_FULL ((1u << 12) | (31u << 2))

/* RISC-V registers (ABI names): x5 = t0 base, x6 = t1 value, x0 = zero. */
#define RV_X0   0
#define RV_T0   5
#define RV_T1   6
#define RV_T2   7

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

/* ---- Circular-buffer synchronisation ----
 *
 * A CB is a producer/consumer pipe between two baby cores backed by a counter
 * in L1. The producer signals "n pages ready" by atomically bumping a counter
 * the consumer polls; the consumer signals "n pages free" the same way in
 * reverse. The signal is a NoC atomic increment to the *other* core's L1; the
 * wait is a local spin on this core's L1. Together they are the whole handshake.
 */

/* Atomically add `incr` to the 32-bit counter at the remote L1 address
 * (sem_lo + sem_mid X/Y coords). Posted: no response is requested. */
int
tt_sem_inc(rv_buf_t *c, uint32_t sem_lo, uint32_t sem_mid, uint32_t incr)
{
    li32(c, RV_T0, NIU_BASE);
    store_reg(c, sem_lo,         NOC_TARG_LO);
    store_reg(c, sem_mid,        NOC_TARG_MID);
    store_reg(c, NOC_AT_INC_FULL, NOC_AT_LEN_BE);
    store_reg(c, incr,           NOC_AT_DATA);
    store_reg(c, NOC_CMD_AT,     NOC_CTRL);
    store_reg(c, 1u,             NOC_CMD_CTRL);   /* fire */
    return BC_OK;
}

/* Spin until the 32-bit counter at local L1 byte address `sem_addr` is at
 * least `threshold`. Two-instruction loop: reload, branch back if short. */
int
tt_sem_wait_ge(rv_buf_t *c, uint32_t sem_addr, uint32_t threshold)
{
    uint32_t loop;
    int32_t  off;

    li32(c, RV_T0, sem_addr);              /* t0 = &counter */
    li32(c, RV_T2, threshold);             /* t2 = threshold */
    loop = rv_buf_n_words(c);
    rv_buf_emit(c, rv_lw(RV_T1, RV_T0, 0));        /* t1 = *counter */
    off = rv_buf_offset(c, rv_buf_n_words(c), loop);
    rv_buf_emit(c, rv_bltu(RV_T1, RV_T2, (int16_t)off));  /* t1 < t2 ? loop */
    return BC_OK;
}

/* The four CB ops are just the wait/signal pair under their pipeline names.
 * `recv` counts pages the producer has supplied; `free` counts pages the
 * consumer has released. Each side polls the counter it owns and bumps the
 * one its partner owns. */

int
tt_cb_reserve_back(rv_buf_t *c, uint32_t free_addr, uint32_t credits)
{
    return tt_sem_wait_ge(c, free_addr, credits);
}

int
tt_cb_push_back(rv_buf_t *c, uint32_t recv_lo, uint32_t recv_mid, uint32_t n)
{
    return tt_sem_inc(c, recv_lo, recv_mid, n);
}

int
tt_cb_wait_front(rv_buf_t *c, uint32_t recv_addr, uint32_t n)
{
    return tt_sem_wait_ge(c, recv_addr, n);
}

int
tt_cb_pop_front(rv_buf_t *c, uint32_t free_lo, uint32_t free_mid, uint32_t n)
{
    return tt_sem_inc(c, free_lo, free_mid, n);
}
