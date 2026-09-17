#ifndef BOOTH_HGEN_H
#define BOOTH_HGEN_H

#include "barracuda.h"
#include "token.h"


typedef struct {
    const char    *file;
    const char    *src;
    uint32_t       slen;
    const token_t *toks;
    uint32_t       ntok;
    const char    *fat;
} hg_in_t;

int hg_emit(const hg_in_t *in, const char *out);

int hg_fat(const char *cubin, const char *ptx, uint32_t sm, const char *out);

int hg_binc(char *buf, size_t n);

int hg_cc(const char *src, const char *out, const char *const *inc, int ninc,
          const char *const *def, int ndef);

#endif /* BOOTH_HGEN_H */
