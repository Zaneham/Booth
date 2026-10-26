#ifndef BARRACUDA_BIR_LOWER_H
#define BARRACUDA_BIR_LOWER_H

#include "bir.h"

struct parser_s;
struct sema_ctx_s;


/*
 * Lower AST to BIR. Device-side functions only.
 * Host functions are scanned for kernel launches to trigger template instantiation.
 * Returns 0 on success, negative on error.
 * sema may be NULL (fallback: everything treated as signed).
 * out_errs/out_nerrs: if non-NULL, receives lowering errors for display.
 */
int bir_lower(const struct parser_s *P, uint32_t ast_root, bir_module_t *M,
              const struct sema_ctx_s *sema,
              bc_error_t *out_errs, int *out_nerrs);

int bir_ltu(const struct parser_s *P, uint32_t ast_root, bir_module_t *M,
            const struct sema_ctx_s *sema, uint16_t tu,
            bc_error_t *out_errs, int *out_nerrs);

uint32_t bir_nlnch(void);
int      bir_lnch(uint32_t i, const char **base, const char **sym,
                  const char **targ);

#endif /* BARRACUDA_BIR_LOWER_H */
