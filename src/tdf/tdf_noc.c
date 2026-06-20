#include "tdf.h"
#include "rv_buf.h"
#include "rv_enc.h"
#include "rt_args.h"
#include "noc.h"
#include <stdio.h>

/*
 * NoC orchestration pass.
 *
 * Two jobs. First, the encoder: given an X/Y coordinate pair and a
 * local memory address, build the 64-bit NoC address that any
 * later `noc_async_read` or `noc_async_write` instruction expects.
 * The bit layout is in tdf.h next to the TD_NOC_* constants and is
 * cited against WormholeB0/NoC/MemoryMap.md.
 *
 * Second, the per-arc orchestration: for every RD and WR arc in
 * the module, decide which of the two physical NoCs to use and
 * what length per fire. The choice today is mechanical: RD on
 * NoC 0 because that fabric flows top-left to bottom-right and
 * matches the Tensix-to-DRAM read direction for the column-16
 * DRAM banks; WR on NoC 1 for the reverse-flow analogue, which
 * Tenstorrent's own RoutingPaths.md plus the DRAMTile congestion
 * table both recommend. Smarter routing is a future pass once the
 * placer knows about multiple cores and DRAM bank affinity.
 *
 * Transfers longer than TD_NOC_MAX_XFER (8192 bytes per NoC
 * request, from Alignment.md) get refused rather than silently
 * truncated. Splitting a too-big transfer into multiple ops is
 * a worthwhile pass but the moment it lands the RV32IM emitter
 * has to know how to issue a sequence of NoC ops with bumped
 * source/destination pointers, and that needs the emitter to
 * exist in the first place. Until then, loud failure beats quiet
 * corruption.
 */

uint64_t td_noc_addr(uint8_t x, uint8_t y, uint64_t local)
{
    /* MemoryMap.md unicast layout:
     *   [35:0]   local address
     *   [41:36]  X coordinate (6 bits)
     *   [47:42]  Y coordinate (6 bits)
     *   [63:48]  reserved, ignored.
     * The broadcast format reuses bits [47:36] for an end-corner
     * and packs the start-corner into [59:48], but we do not
     * emit multicasts yet so the unicast encoder is enough. */
    return (local & TD_NOC_LOCAL_MASK)
         | ((uint64_t)(x & TD_NOC_COORD_MASK) << TD_NOC_LOCAL_BITS)
         | ((uint64_t)(y & TD_NOC_COORD_MASK) << (TD_NOC_LOCAL_BITS + 6u));
}

int td_noc_orchestrate(td_mod_t *M)
{
    for (uint16_t i = 0; i < M->narc; i++) {
        td_arc_t *a = &M->arcs[i];

        if (a->kind != TD_AR_RD && a->kind != TD_AR_WR) {
            /* CB sync arcs do not touch the NoC. Leave noc_id and
             * length at whatever they happen to be; the dump
             * filters those fields out for CB kinds. */
            continue;
        }

        if (a->chan >= M->ncha) {
            fprintf(stderr,
                    "tdf: NoC arc %u references unknown channel %u\n",
                    i, a->chan);
            return BC_ERR_TDF;
        }
        const td_chan_t *c = &M->chans[a->chan];

        /* RD pulls from outside into the channel slot; WR drains
         * the channel slot to outside. NoC choice falls out of the
         * direction, per the doc comment up top. */
        a->noc_id = (a->kind == TD_AR_RD) ? (uint8_t)TD_NOC0
                                          : (uint8_t)TD_NOC1;

        uint32_t len = td_tile_bytes(c->tag);
        if (len == 0u) {
            fprintf(stderr,
                    "tdf: NoC arc %u on channel %u has zero-byte tile\n",
                    i, a->chan);
            return BC_ERR_TDF;
        }
        if (len > TD_NOC_MAX_XFER) {
            fprintf(stderr,
                    "tdf: NoC arc %u length %u exceeds max %u "
                    "(needs splitting pass)\n",
                    i, len, TD_NOC_MAX_XFER);
            return BC_ERR_TDF;
        }
        a->length = len;
    }
    return BC_OK;
}

/*
 * CB-arc emitter: lower one circular-buffer synchronisation arc to baby-core
 * RISC-V, using the primitives in tensix/noc.c. This is the RV32IM emitter the
 * comments above kept deferring, for the four CB arc kinds. RD/WR arcs need
 * runtime DRAM addresses and a per-tile loop and are emitted elsewhere.
 *
 * The two counters live in the channel's FIFO header (the 8 bytes placement
 * reserves right after the tile-data buffer): recv at +0 counts pages the
 * producer has supplied, free at +4 counts pages the consumer has released.
 * Each region polls the counter it owns (local spin) and bumps the one its
 * partner owns (NoC atomic to the partner's coordinates).
 *
 * This targets the single-Tensix model the placer produces today: producer and
 * consumer are different baby cores sharing one L1, so both counters sit in
 * that L1 and the atomic targets the tile's own coordinates (permitted per
 * NoC/Atomics.md). Cross-tile counter placement lands with multi-core placement.
 */
int td_emit_cb_arc(td_mod_t *M, const td_arc_t *a, rv_buf_t *code)
{
    const td_chan_t *c;
    uint32_t data_b, recv, freec;

    if (a->chan >= M->ncha) {
        fprintf(stderr, "tdf: CB arc references unknown channel %u\n", a->chan);
        return BC_ERR_TDF;
    }
    c = &M->chans[a->chan];
    data_b = td_tile_bytes(c->tag) * (uint32_t)c->depth;
    recv   = c->l1_off + data_b;          /* pages produced */
    freec  = c->l1_off + data_b + 4u;     /* pages free      */

    switch (a->kind) {
    case TD_AR_RSV:     /* producer blocks until the consumer has freed slots */
        return tt_cb_reserve_back(code, freec, a->cnt);
    case TD_AR_WAIT:    /* consumer blocks until the producer has supplied tiles */
        return tt_cb_wait_front(code, recv, a->cnt);
    case TD_AR_PUSH: {  /* producer bumps the consumer's recv counter */
        const td_rgn_t *cons = &M->rgns[c->cons];
        uint64_t na = td_noc_addr((uint8_t)cons->x, (uint8_t)cons->y, recv);
        return tt_cb_push_back(code, (uint32_t)na, (uint32_t)(na >> 32), a->cnt);
    }
    case TD_AR_POP: {   /* consumer bumps the producer's free counter */
        const td_rgn_t *prod = &M->rgns[c->prod];
        uint64_t na = td_noc_addr((uint8_t)prod->x, (uint8_t)prod->y, freec);
        return tt_cb_pop_front(code, (uint32_t)na, (uint32_t)(na >> 32), a->cnt);
    }
    default:
        fprintf(stderr, "tdf: td_emit_cb_arc given non-CB arc kind %u\n",
                a->kind);
        return BC_ERR_TDF;
    }
}

/*
 * Tile-transfer loop: the machine-code form of a Metalium reader or writer
 * kernel. Reads the DRAM base address and the tile count from the runtime-args
 * slab the host populates, then loops once per tile:
 *
 *   reader: reserve a CB slot, NoC-read one tile DRAM -> L1, wait for it to
 *           land, push the slot to the consumer.
 *   writer: wait for a produced tile, NoC-write it L1 -> DRAM, wait for the
 *           ack, pop the slot back to the producer.
 *
 * Loop state lives in a-registers the called primitives never touch (they only
 * use t0/t1/t2): a3 = DRAM pointer, a4 = L1 ring pointer, a5 = tile counter,
 * a6 = tile_bytes, a7 = L1 ring end. The L1 pointer wraps to the buffer base
 * after `depth` tiles, which is the circular buffer doing its job.
 *
 * Contiguous DRAM layout: the DRAM pointer advances by one tile each step and
 * `dram_mid` is fixed. Interleaved bank addressing (where consecutive tiles
 * live in different DRAM banks with different coordinates) is a later pass.
 */
int td_emit_dma_loop(rv_buf_t *code, int is_write,
                     uint32_t dram_arg_slot, uint32_t ntiles_arg_slot,
                     uint32_t l1_buf, uint32_t depth,
                     uint32_t dram_mid, uint32_t l1_mid,
                     uint32_t tile_bytes,
                     uint32_t recv_addr, uint32_t free_addr)
{
    uint32_t loop, br_exit, br_wrap, jback, nowrap, done;
    uint32_t ring_end = l1_buf + depth * tile_bytes;

    /* seed the counter this core acquires: a reader waits on free slots (start
     * full = depth), a writer waits on produced tiles (start empty = 0). */
    if (is_write) tt_sem_init(code, recv_addr, 0u);
    else          tt_sem_init(code, free_addr, depth);

    /* prologue: pull args from L1, set up the loop registers. */
    tt_li32(code, RV_T0, TD_L1_RTARG_BASE);
    rv_buf_emit(code, rv_lw(RV_A3, RV_T0,
                            (int16_t)RT_ARG_OFF_KARG(dram_arg_slot)));
    rv_buf_emit(code, rv_lw(RV_A5, RV_T0,
                            (int16_t)RT_ARG_OFF_KARG(ntiles_arg_slot)));
    tt_li32(code, RV_A4, l1_buf);
    tt_li32(code, RV_A6, tile_bytes);
    tt_li32(code, RV_A7, ring_end);

    /* loop: exit once the tile counter hits zero (offset patched below). */
    loop    = rv_buf_n_words(code);
    br_exit = (uint32_t)rv_buf_emit(code, 0u);

    /* pre-transfer CB gate. */
    if (is_write) tt_cb_wait_front(code, recv_addr, 1u);
    else          tt_cb_reserve_back(code, free_addr, 1u);

    /* the transfer. reader: DRAM(a3) -> L1(a4); writer: L1(a4) -> DRAM(a3). */
    if (is_write)
        tt_noc_xfer_reg(code, 1, RV_A4, RV_A3, l1_mid, dram_mid, tile_bytes);
    else
        tt_noc_xfer_reg(code, 0, RV_A3, RV_A4, dram_mid, l1_mid, tile_bytes);
    tt_noc_barrier(code, is_write);

    /* post-transfer CB signal. */
    if (is_write) tt_cb_pop_front(code, free_addr, 0u, 1u);
    else          tt_cb_push_back(code, recv_addr, 0u, 1u);

    /* advance both pointers; wrap the L1 ring pointer at its end. */
    rv_buf_emit(code, rv_add(RV_A3, RV_A3, RV_A6));
    rv_buf_emit(code, rv_add(RV_A4, RV_A4, RV_A6));
    br_wrap = (uint32_t)rv_buf_emit(code, 0u);  /* bltu a4,a7,nowrap (patched) */
    tt_li32(code, RV_A4, l1_buf);               /* wrap: reset to ring base    */
    nowrap = rv_buf_n_words(code);
    rv_buf_patch(code, br_wrap,
                 rv_bltu(RV_A4, RV_A7,
                         (int16_t)rv_buf_offset(code, br_wrap, nowrap)));

    /* decrement and loop. */
    rv_buf_emit(code, rv_addi(RV_A5, RV_A5, -1));
    jback = rv_buf_n_words(code);
    rv_buf_emit(code, rv_jal(RV_ZERO, rv_buf_offset(code, jback, loop)));

    /* done: patch the exit branch now that we know where here is. */
    done = rv_buf_n_words(code);
    rv_buf_patch(code, br_exit,
                 rv_beq(RV_A5, RV_ZERO,
                        (int16_t)rv_buf_offset(code, br_exit, done)));
    return BC_OK;
}
