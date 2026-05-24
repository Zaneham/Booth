#ifndef BARRACUDA_RV_BUF_H
#define BARRACUDA_RV_BUF_H

#include "barracuda.h"

/*
 * RV32IM code buffer.
 *
 * Append-only ring of 32-bit instruction words plus a small set of
 * helpers for back-patching forward references. Used by the isel to
 * accumulate a function body before handing it off to the ELF emitter.
 *
 * Fixed-size arena, no malloc; if a kernel ever needs more than
 * RV_BUF_MAX_WORDS the emit calls return -1 and the caller bails.
 * 16 KiB of code is enough for any single baby-core kernel we have
 * a realistic plan to compile in the foreseeable future, and the
 * Tensix L1 only reserves about 32 KiB for code anyway, so spending
 * more memory here would just hide the overflow rather than fix it.
 */

#define RV_BUF_MAX_WORDS  4096    /* 16 KiB of instruction storage */

typedef struct {
    uint32_t words[RV_BUF_MAX_WORDS];
    uint32_t n;                     /* count of words written */
} rv_buf_t;

void           rv_buf_init(rv_buf_t *b);

/* Append one word, return its index (0..n-1) or -1 on overflow.
 * Most callers don't need the index but the isel reaches for it
 * when it wants to back-patch a forward branch later. */
int            rv_buf_emit(rv_buf_t *b, uint32_t word);

/* Overwrite a previously emitted word. Used by the isel once a
 * forward-reference label is resolved and the original placeholder
 * branch can be filled in with the right offset. */
int            rv_buf_patch(rv_buf_t *b, uint32_t idx, uint32_t word);

uint32_t       rv_buf_n_words(const rv_buf_t *b);
uint32_t       rv_buf_pos_bytes(const rv_buf_t *b);   /* n * 4 */
const uint32_t *rv_buf_data(const rv_buf_t *b);

/*
 * Compute the byte offset from word index `from` to word index `to`.
 * Negative for backward references, positive for forward. Used by
 * the isel when filling in B-type and J-type immediate fields:
 *
 *   uint32_t br_idx = rv_buf_emit(b, 0);             // placeholder
 *   ... emit more code ...
 *   uint32_t tgt_idx = rv_buf_n_words(b);            // branch target
 *   int32_t  off = rv_buf_offset(b, br_idx, tgt_idx);
 *   rv_buf_patch(b, br_idx, rv_beq(rs1, rs2, off));
 */
int32_t        rv_buf_offset(const rv_buf_t *b,
                             uint32_t from, uint32_t to);

#endif /* BARRACUDA_RV_BUF_H */
