#include "rv_buf.h"
#include <string.h>
#include <stdio.h>

void rv_buf_init(rv_buf_t *b)
{
    memset(b, 0, sizeof(*b));
}

int rv_buf_emit(rv_buf_t *b, uint32_t word)
{
    if (b->n >= RV_BUF_MAX_WORDS) {
        fprintf(stderr, "rv_buf: code buffer overflow at %u words\n",
                RV_BUF_MAX_WORDS);
        return -1;
    }
    uint32_t idx = b->n++;
    b->words[idx] = word;
    return (int)idx;
}

int rv_buf_patch(rv_buf_t *b, uint32_t idx, uint32_t word)
{
    if (idx >= b->n) {
        fprintf(stderr, "rv_buf: patch index %u past end %u\n", idx, b->n);
        return -1;
    }
    b->words[idx] = word;
    return 0;
}

uint32_t rv_buf_n_words(const rv_buf_t *b)      { return b->n; }
uint32_t rv_buf_nbytes(const rv_buf_t *b)    { return b->n * 4u; }
const uint32_t *rv_buf_data(const rv_buf_t *b)  { return b->words; }

int32_t rv_buf_offset(const rv_buf_t *b, uint32_t from, uint32_t to)
{
    /* Word indices come in unsigned; convert to a signed byte offset. The
     * buffer pointer isn't strictly needed since this is pure index arithmetic,
     * hence the (void)b, but taking it keeps the API uniform and leaves room to
     * add bounds checks later. */
    (void)b;
    return ((int32_t)to - (int32_t)from) * 4;
}
