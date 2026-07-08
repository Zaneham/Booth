#ifndef BARRACUDA_RV_BUF_H
#define BARRACUDA_RV_BUF_H

#include "barracuda.h"

/*
 * RV32IM code buffer: an append-only ring of 32-bit instruction words plus a
 * few helpers for back-patching forward references, used by the isel to build
 * a function body before handing it to the ELF emitter. It's a fixed-size
 * arena with no malloc, so past RV_BUF_MAX_WORDS the emit calls return -1 and
 * the caller bails. 16 KiB is plenty for any baby-core kernel we plan to
 * compile, and the Tensix L1 only reserves about 32 KiB for code, so a bigger
 * buffer would just hide the overflow rather than fix it.
 */

#define RV_BUF_MAX_WORDS  4096    /* 16 KiB of instruction storage */

typedef struct {
    uint32_t words[RV_BUF_MAX_WORDS];
    uint32_t n;                     /* count of words written */
} rv_buf_t;

void           rv_buf_init(rv_buf_t *b);

/* Append one word, return its index (0..n-1) or -1 on overflow. Most callers
 * ignore the index, but the isel reaches for it to back-patch a forward
 * branch later. */
int            rv_buf_emit(rv_buf_t *b, uint32_t word);

/* Overwrite a previously emitted word, used by the isel once a forward-reference
 * label resolves and the placeholder branch can be filled in with the right
 * offset. */
int            rv_buf_patch(rv_buf_t *b, uint32_t idx, uint32_t word);

uint32_t       rv_buf_n_words(const rv_buf_t *b);
uint32_t       rv_buf_pos_bytes(const rv_buf_t *b);   /* n * 4 */
const uint32_t *rv_buf_data(const rv_buf_t *b);

/*
 * Compute the byte offset from word index `from` to word index `to`, negative
 * for backward references and positive for forward. The isel uses it to fill in
 * B-type and J-type immediate fields: emit a placeholder word, keep emitting,
 * then patch the placeholder with the offset from its index to the branch target.
 */
int32_t        rv_buf_offset(const rv_buf_t *b,
                             uint32_t from, uint32_t to);

#endif /* BARRACUDA_RV_BUF_H */
