#include "bir_softfp.h"
#include <stdio.h>
#include <string.h>

/*
 * bir_softfp: fp32 to soft-float calls. RV32IM has no F extension.
 *
 * Every op here rewrites in place, so nothing renumbers. FCMP is absent
 * because it needs a call plus a compare, which is an insertion.
 *
 * I was listening to Let It Happen by Tame Impala when writing this file.
 */

#define SF_NO_FUNC 0xFFFFFFFFu

/* Hello, I see we're both floating on by. */
static const struct { uint16_t op; const char *fn; uint8_t nargs; } SFMAP[] = {
    { BIR_FADD,   "__addsf3",      2 },
    { BIR_FSUB,   "__subsf3",      2 },
    { BIR_FMUL,   "__mulsf3",      2 },
    { BIR_FDIV,   "__divsf3",      2 },
    { BIR_SITOFP, "__floatsisf",   1 },
    { BIR_UITOFP, "__floatunsisf", 1 },
    { BIR_FPTOSI, "__fixsfsi",     1 },
    { BIR_FPTOUI, "__fixunssfsi",  1 },
};
#define SFMAP_N (sizeof(SFMAP) / sizeof(SFMAP[0]))

static uint32_t sf_find(const bir_module_t *M, const char *name)
{
    for (uint32_t i = 0; i < M->num_funcs; i++) {
        if (strcmp(&M->strings[M->funcs[i].name], name) == 0) return i;
    }
    return SF_NO_FUNC;
}

/* A call planted inside __addsf3 recurses until the stack runs out, and a baby
   core has very little to run out of. */
static int sf_is_runtime(const bir_module_t *M, uint32_t fi)
{
    const char *n = &M->strings[M->funcs[fi].name];
    for (uint32_t i = 0; i < SFMAP_N; i++) {
        if (strcmp(n, SFMAP[i].fn) == 0) return 1;
    }
    return strncmp(n, "sfp_", 4) == 0;
}

int bir_softfp(bir_module_t *M)
{
    uint32_t callee[SFMAP_N];
    for (uint32_t i = 0; i < SFMAP_N; i++) callee[i] = sf_find(M, SFMAP[i].fn);

    for (uint32_t fi = 0; fi < M->num_funcs; fi++) {
        if (sf_is_runtime(M, fi)) continue;
        const bir_func_t *F = &M->funcs[fi];

        for (uint32_t b = 0; b < F->num_blocks; b++) {
            const bir_block_t *B = &M->blocks[F->first_block + b];

            for (uint32_t k = 0; k < B->num_insts; k++) {
                bir_inst_t *I = &M->insts[B->first_inst + k];

                for (uint32_t m = 0; m < SFMAP_N; m++) {
                    if (I->op != SFMAP[m].op) continue;

                    if (callee[m] == SF_NO_FUNC) {
                        fprintf(stderr,
                                "bir_softfp: %s needs %s, which is not in the "
                                "module (soft-float runtime not linked in?)\n",
                                bir_op_name(I->op), SFMAP[m].fn);
                        return BC_ERR_VERIFY;
                    }
                    if (I->num_operands != SFMAP[m].nargs) {
                        fprintf(stderr,
                                "bir_softfp: %s has %u operands, expected %u\n",
                                bir_op_name(I->op), I->num_operands,
                                SFMAP[m].nargs);
                        return BC_ERR_VERIFY;
                    }

                    /* BIR_CALL wants the callee in operand 0. */
                    for (uint32_t a = SFMAP[m].nargs; a > 0u; a--) {
                        I->operands[a] = I->operands[a - 1u];
                    }
                    I->operands[0]   = callee[m];
                    I->num_operands  = (uint8_t)(SFMAP[m].nargs + 1u);
                    I->subop         = 0;
                    I->op            = BIR_CALL;
                    break;
                }
            }
        }
    }
    return BC_OK;
}
