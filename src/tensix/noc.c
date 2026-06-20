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

/* NOC_CTRL request type (bits [1:0]) and the marked-response flag (bit 4). */
#define NOC_CMD_RD      0u
#define NOC_CMD_AT      1u
#define NOC_CMD_WR      2u
#define NOC_CMD_RESP_MARKED (1u << 4)

/* NIU counters (read-only) at NIU_BASE + 0x200, one 32-bit word each. The few
 * we need to barrier on: reads issued vs completed, writes issued vs acked. */
#define NIU_CNT_BASE    0xFFB20200u
#define CNT_WR_ACK      (1u * 4u)    /* NIU_MST_WR_ACK_RECEIVED      */
#define CNT_RD_RESP     (2u * 4u)    /* NIU_MST_RD_RESP_RECEIVED     */
#define CNT_RD_REQ      (5u * 4u)    /* NIU_MST_RD_REQ_SENT          */
#define CNT_WR_SENT     (10u * 4u)   /* NIU_MST_NONPOSTED_WR_REQ_SENT */

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

/* Seed a local counter to `val` (one store; done once before the loop). */
int
tt_sem_init(rv_buf_t *c, uint32_t sem_addr, uint32_t val)
{
    li32(c, RV_T0, sem_addr);
    li32(c, RV_T1, val);
    rv_buf_emit(c, rv_sw(RV_T1, RV_T0, 0));
    return BC_OK;
}

/* Acquire `n` credits from the local counter: spin until it holds at least n,
 * then atomically subtract n. The subtract goes through the NoC atomic unit
 * (incrementing by -n) rather than a plain load-store, so it cannot lose a
 * concurrent remote increment from the partner core returning credits. */
int
tt_sem_acquire(rv_buf_t *c, uint32_t sem_addr, uint32_t n)
{
    tt_sem_wait_ge(c, sem_addr, n);
    tt_sem_inc(c, sem_addr, 0u, (uint32_t)(0u - n));   /* atomic -= n (local) */
    return BC_OK;
}

/* The four CB ops over the counting-semaphore pair. `free` counts empty slots
 * (seeded to the ring depth), `recv` counts filled slots (seeded to zero). The
 * acquiring ops (reserve_back, wait_front) wait then consume a credit; the
 * signalling ops (push_back, pop_front) hand a credit to the partner over the
 * NoC. Constant counts are correct because the counters track what is available
 * right now, not a running total. */

int
tt_cb_reserve_back(rv_buf_t *c, uint32_t free_addr, uint32_t credits)
{
    return tt_sem_acquire(c, free_addr, credits);
}

int
tt_cb_push_back(rv_buf_t *c, uint32_t recv_lo, uint32_t recv_mid, uint32_t n)
{
    return tt_sem_inc(c, recv_lo, recv_mid, n);
}

int
tt_cb_wait_front(rv_buf_t *c, uint32_t recv_addr, uint32_t n)
{
    return tt_sem_acquire(c, recv_addr, n);
}

int
tt_cb_pop_front(rv_buf_t *c, uint32_t free_lo, uint32_t free_mid, uint32_t n)
{
    return tt_sem_inc(c, free_lo, free_mid, n);
}

/* Expose the 32-bit materialiser; the tile-loop emitter needs it too. */
void
tt_li32(rv_buf_t *c, uint8_t reg, uint32_t v)
{
    li32(c, reg, v);
}

/* ---- Register-sourced transfer + completion barrier ----
 *
 * tt_noc_read/write take immediate addresses, which is all a fixed transfer
 * needs. A tile loop instead keeps the source/destination addresses in
 * registers and bumps them each iteration, so it needs a transfer whose low
 * addresses come from registers. mid (the high address bits + X/Y coords) and
 * the length stay immediate. Clobbers t0/t1; treg and rreg must be neither.
 */
int
tt_noc_xfer_reg(rv_buf_t *c, int is_write, uint8_t treg, uint8_t rreg,
                uint32_t tmid, uint32_t rmid, uint32_t len)
{
    uint32_t ctrl = is_write ? (NOC_CMD_WR | NOC_CMD_RESP_MARKED) : NOC_CMD_RD;
    li32(c, RV_T0, NIU_BASE);
    rv_buf_emit(c, rv_sw(treg, RV_T0, NOC_TARG_LO));   /* target low from reg */
    store_reg(c, tmid, NOC_TARG_MID);
    rv_buf_emit(c, rv_sw(rreg, RV_T0, NOC_RET_LO));    /* return low from reg */
    store_reg(c, rmid, NOC_RET_MID);
    store_reg(c, len,  NOC_AT_LEN_BE);
    store_reg(c, ctrl, NOC_CTRL);
    store_reg(c, 1u,   NOC_CMD_CTRL);                  /* fire */
    return BC_OK;
}

/* Spin until every transfer issued so far has completed: for reads, until the
 * responses-received counter catches the requests-sent counter; for writes,
 * until acks catch the (marked) requests sent. The docs require reading
 * NOC_CMD_CTRL back before the first counter read so the fire and the poll are
 * not reordered, so we do that first. Clobbers t0/t1/t2. */
int
tt_noc_barrier(rv_buf_t *c, int is_write)
{
    uint32_t sent = is_write ? CNT_WR_SENT : CNT_RD_REQ;
    uint32_t got  = is_write ? CNT_WR_ACK  : CNT_RD_RESP;
    uint32_t loop;
    int32_t  off;

    li32(c, RV_T0, NIU_BASE);
    rv_buf_emit(c, rv_lw(RV_T1, RV_T0, NOC_CMD_CTRL));   /* ordering read-back */
    li32(c, RV_T0, NIU_CNT_BASE);
    loop = rv_buf_n_words(c);
    rv_buf_emit(c, rv_lw(RV_T1, RV_T0, (int16_t)sent));
    rv_buf_emit(c, rv_lw(RV_T2, RV_T0, (int16_t)got));
    off = rv_buf_offset(c, rv_buf_n_words(c), loop);
    rv_buf_emit(c, rv_bne(RV_T1, RV_T2, (int16_t)off));  /* sent != got ? loop */
    return BC_OK;
}
