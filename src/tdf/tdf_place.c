#include "tdf.h"
#include <stdio.h>

/*
 * L1 placement pass.
 *
 * For each channel, decide where its tile-data buffer and FIFO header live in the
 * home core's L1. Today L1 is a single pool because we only target one Tensix at
 * a time; when multi-core lowering lands, each core gets its own running offset
 * and channels go to whichever core hosts the producer. Spec citations sit in
 * tdf.h next to the TD_L1_* constants: Wormhole L1 is 1,464 KiB at 0x00000000,
 * NoC reads/writes to L1 need 16-byte alignment, the first 32 KiB is reserved for
 * code, and channels pack first-fit upward from TD_L1_CB_BASE.
 */

static uint32_t round_up(uint32_t x, uint32_t a)
{
    return (x + (a - 1u)) & ~(a - 1u);
}

/*
 * Tile byte size from a channel tag. Width assumptions follow what the fission
 * pass currently writes: FLOAT is fp32, BFLOAT is the 16-bit brain float, INT is
 * int32. Once td_tag_t carries an explicit width field (when issue #82 tile-shape
 * inference lands and starts producing fp16 or fp8 channels) this collapses to a
 * simple width lookup.
 */
static uint32_t dtype_bytes(uint8_t dt)
{
    switch (dt) {
    case BIR_TYPE_FLOAT:  return 4u;
    case BIR_TYPE_BFLOAT: return 2u;
    case BIR_TYPE_INT:    return 4u;
    default:              return 4u;
    }
}

uint32_t td_tile_bytes(td_tag_t tag)
{
    return (uint32_t)tag.rows * (uint32_t)tag.cols * dtype_bytes(tag.dtype);
}

int td_place_l1(td_mod_t *M)
{
    uint32_t cur = TD_L1_CB_BASE;

    for (uint16_t i = 0; i < M->ncha; i++) {
        td_chan_t *c = &M->chans[i];

        /* Data buffer first, then FIFO header immediately after.
         * Both round up to TD_L1_ALIGN so the next channel starts
         * on a NoC-legal boundary. The FIFO header is 4-byte
         * aligned naturally because the data buffer is 16-byte
         * aligned and the header sits flush against it. */
        cur = round_up(cur, TD_L1_ALIGN);
        c->l1_off = cur;

        uint32_t tile_b = td_tile_bytes(c->tag);
        uint32_t data_b = tile_b * (uint32_t)c->depth;
        uint32_t total  = round_up(data_b + TD_FIFO_BYTES, TD_L1_ALIGN);

        /* Ceiling is the shared slab, not the end of L1: __shared__ is carved
         * off the top, and packing a channel past this would overlap it. */
        if (cur > TD_L1_SHARED_BASE || total > TD_L1_SHARED_BASE - cur) {
            fprintf(stderr,
                    "tdf: L1 budget exceeded at channel %u "
                    "(want %u bytes at 0x%x, CB area ends at 0x%x)\n",
                    c->id, total, cur, TD_L1_SHARED_BASE);
            return BC_ERR_TDF;
        }
        cur += total;
    }
    return BC_OK;
}
