/* bir_sroa.c -- Scalar Replacement of Aggregates.
 *
 * mem2reg won't go near a struct alloca: a GEP sits between it and every
 * load/store, so the whole thing stays in memory, sulking. All right, but
 * apart from prising that GEP layer off, handing every accessed field its
 * own scalar alloca, repointing the loads and stores, and letting mem2reg
 * promote the lot... what has SROA ever done for us? The dead struct alloca
 * and its GEPs are carried off by DCE, and good riddance.
 *
 * We only bless the righteous case: a struct alloca whose every use is a
 * constant-index GEP to a scalar field, each GEP used only as a load or
 * store address. Anything stranger (variable index, nested aggregate, a
 * pointer that wanders off) is not the Messiah, it's a very naughty alloca,
 * and we leave it well alone.
 *
 * Runs before mem2reg, and only moves where a field lives, never what it
 * holds, so output is bit-identical with SROA on or off. If a single bit so
 * much as twitches, SROA has sinned, and the diff test will not be looking
 * on the bright side of life. */

#include "bir_sroa.h"
#include "bir_insert.h"
#include <string.h>

#define SROA_MAX_GEP    2048   /* const-index GEPs on one struct alloca */
#define SROA_MAX_USE    2048   /* load/store users across those GEPs    */
#define SROA_MAX_FIELD  64     /* distinct fields we will split out     */

/* Working state, kept static so the arrays stay off the stack. */
typedef struct {
    uint32_t   gep[SROA_MAX_GEP];        /* GEP instruction index        */
    uint32_t   gep_field[SROA_MAX_GEP];  /* its field number             */
    uint32_t   n_gep;

    uint32_t   use_inst[SROA_MAX_USE];   /* the load/store               */
    uint8_t    use_slot[SROA_MAX_USE];   /* 0 = load addr, 1 = store addr*/
    uint32_t   use_field[SROA_MAX_USE];  /* field it accesses            */
    uint32_t   n_use;

    uint32_t   field[SROA_MAX_FIELD];        /* distinct accessed fields */
    uint32_t   field_ptype[SROA_MAX_FIELD];  /* ptr-to-field type each   */
    uint32_t   n_field;

    bir_inst_t newa[SROA_MAX_FIELD];     /* scalar allocas to insert     */
} sroa_t;

static sroa_t S;

/* ---- Small helpers ---- */

static int is_scalar(const bir_module_t *M, uint32_t t)
{
    uint8_t k;
    if (t >= M->num_types) return 0;
    k = M->types[t].kind;
    return k == BIR_TYPE_INT || k == BIR_TYPE_FLOAT
        || k == BIR_TYPE_BFLOAT || k == BIR_TYPE_PTR;
}

static uint32_t pointee(const bir_module_t *M, uint32_t t)
{
    if (t < M->num_types && M->types[t].kind == BIR_TYPE_PTR)
        return M->types[t].inner;
    return BIR_VAL_NONE;
}

/* Does instruction I reference value `v` as a value operand? Skips block
 * indices, the CALL callee, constants and BIR_VAL_NONE. Same per-opcode
 * classifier as bir_insert and the printer. */
static int refs_value(const bir_module_t *M, const bir_inst_t *I, uint32_t v)
{
    int      ovf   = (I->num_operands == BIR_OPERANDS_OVERFLOW);
    uint32_t start = I->operands[0];
    uint32_t count = I->operands[1];
    uint32_t i;
    uint8_t  k, no;

    switch (I->op) {
    case BIR_BR:
        return 0;
    case BIR_BR_COND:
        return I->operands[0] == v;
    case BIR_SWITCH:
        if (ovf) return count >= 1 && M->extra_operands[start] == v;
        return I->operands[0] == v;
    case BIR_PHI:
        if (ovf) {
            for (i = 0; i + 1 < count; i += 2)
                if (M->extra_operands[start + i + 1] == v) return 1;
        } else {
            for (k = 0; (uint32_t)k + 1 < I->num_operands; k += 2)
                if (I->operands[k + 1] == v) return 1;
        }
        return 0;
    case BIR_CALL:
        if (ovf) {
            for (i = 1; i < count; i++)
                if (M->extra_operands[start + i] == v) return 1;
        } else {
            for (k = 1; k < I->num_operands; k++)
                if (I->operands[k] == v) return 1;
        }
        return 0;
    default:
        if (!ovf) {
            no = I->num_operands;
            if (no > BIR_OPERANDS_INLINE) no = BIR_OPERANDS_INLINE;
            for (k = 0; k < no; k++)
                if (I->operands[k] == v) return 1;
        } else {
            for (i = 0; i < count; i++)
                if (M->extra_operands[start + i] == v) return 1;
        }
        return 0;
    }
}

/* Find the scalar-alloca slot (0..n_field-1) assigned to a field. */
static uint32_t slot_of(uint32_t field)
{
    uint32_t j;
    for (j = 0; j < S.n_field; j++)
        if (S.field[j] == field) return j;
    return 0;   /* unreachable; reach it and the invariants have been crucified */
}

/* ---- Eligibility + transform for one struct alloca ---- */

/* Returns 1 (and rewrites the module) if the struct alloca at absolute
 * index `sa` was eligible and split, else 0 (module unchanged). */
static int try_alloca(bir_module_t *M, const bir_func_t *F, uint32_t sa)
{
    uint32_t st, av, b, j, blk = 0, pos = 0, base, K;

    st = pointee(M, M->insts[sa].type);
    if (st == BIR_VAL_NONE || st >= M->num_types ||
        M->types[st].kind != BIR_TYPE_STRUCT)
        return 0;

    av       = BIR_MAKE_VAL(sa);
    S.n_gep  = 0;
    S.n_use  = 0;
    S.n_field = 0;

    /* Pass 1: every use of the struct pointer must be a constant-index GEP
     * to a scalar field; anything else is an escape. */
    for (b = F->first_block; b < F->first_block + F->num_blocks; b++) {
        const bir_block_t *B = &M->blocks[b];
        uint32_t jj;
        for (jj = 0; jj < B->num_insts; jj++) {
            uint32_t ii = B->first_inst + jj;
            bir_inst_t *I = &M->insts[ii];

            if (I->op == BIR_GEP && I->operands[0] == av) {
                uint32_t idxop = I->operands[1];
                int64_t  fi;
                uint32_t ft;
                if (!BIR_VAL_IS_CONST(idxop))                  return 0;
                fi = M->consts[BIR_VAL_INDEX(idxop)].d.ival;
                if (fi < 0 || (uint32_t)fi >= M->types[st].num_fields)
                    return 0;
                ft = M->type_fields[M->types[st].count + (uint32_t)fi];
                if (!is_scalar(M, ft))                         return 0;
                if (S.n_gep >= SROA_MAX_GEP)                   return 0;
                S.gep[S.n_gep]       = ii;
                S.gep_field[S.n_gep] = (uint32_t)fi;
                S.n_gep++;
            } else if (ii != sa && refs_value(M, I, av)) {
                return 0;   /* struct pointer used outside a GEP base */
            }
        }
    }
    if (S.n_gep == 0) return 0;

    /* Pass 2: every GEP result must be used only as a load or store
     * address. Collect those users; any other use is an escape. */
    for (j = 0; j < S.n_gep; j++) {
        uint32_t gv = BIR_MAKE_VAL(S.gep[j]);
        for (b = F->first_block; b < F->first_block + F->num_blocks; b++) {
            const bir_block_t *B = &M->blocks[b];
            uint32_t jj;
            for (jj = 0; jj < B->num_insts; jj++) {
                uint32_t ii = B->first_inst + jj;
                bir_inst_t *I = &M->insts[ii];
                if (ii == S.gep[j]) continue;

                if (I->op == BIR_LOAD && I->operands[0] == gv) {
                    if (S.n_use >= SROA_MAX_USE) return 0;
                    S.use_inst[S.n_use]  = ii;
                    S.use_slot[S.n_use]  = 0;
                    S.use_field[S.n_use] = S.gep_field[j];
                    S.n_use++;
                } else if (I->op == BIR_STORE && I->operands[1] == gv) {
                    if (S.n_use >= SROA_MAX_USE) return 0;
                    S.use_inst[S.n_use]  = ii;
                    S.use_slot[S.n_use]  = 1;
                    S.use_field[S.n_use] = S.gep_field[j];
                    S.n_use++;
                } else if (refs_value(M, I, gv)) {
                    return 0;   /* field pointer escaped */
                }
            }
        }
    }
    /* Orphan guard: nothing left to read. We already split this one to
     * bits, its GEPs point nowhere; out the door, line on the left, and the
     * driver's rescan can stop. */
    if (S.n_use == 0) return 0;

    /* The distinct accessed fields, each splitting off with its own
     * ptr-to-field type (the GEP's result type). Splitters. */
    for (j = 0; j < S.n_gep; j++) {
        uint32_t f = S.gep_field[j];
        uint32_t q;
        int seen = 0;
        for (q = 0; q < S.n_field; q++)
            if (S.field[q] == f) { seen = 1; break; }
        if (!seen) {
            if (S.n_field >= SROA_MAX_FIELD) return 0;
            S.field[S.n_field]       = f;
            S.field_ptype[S.n_field] = M->insts[S.gep[j]].type;
            S.n_field++;
        }
    }

    /* Locate the struct alloca's block + local position. */
    for (b = F->first_block; b < F->first_block + F->num_blocks; b++) {
        const bir_block_t *B = &M->blocks[b];
        if (sa >= B->first_inst && sa < B->first_inst + B->num_insts) {
            blk = b;
            pos = sa - B->first_inst;
            break;
        }
    }

    /* Build one scalar alloca per distinct field, same alignment as the
     * struct, and insert them at the struct alloca's spot. */
    K = S.n_field;
    for (j = 0; j < K; j++) {
        memset(&S.newa[j], 0, sizeof(S.newa[j]));
        S.newa[j].op           = BIR_ALLOCA;
        S.newa[j].num_operands = 0;
        S.newa[j].subop        = M->insts[sa].subop;
        S.newa[j].type         = S.field_ptype[j];   /* ptr-to-field */
    }
    base = bir_insert(M, blk, pos, S.newa, K);
    if (base == BIR_VAL_NONE) return 0;   /* overflow: leave as-is */

    /* Insertion shifted every instruction at or past the struct alloca
     * up by K. All recorded users were after it, so adjust and repoint
     * each load/store from its GEP to the field's scalar alloca. */
    for (j = 0; j < S.n_use; j++)
        S.use_inst[j] += K;

    for (j = 0; j < S.n_use; j++) {
        uint32_t sv = BIR_MAKE_VAL(base + slot_of(S.use_field[j]));
        M->insts[S.use_inst[j]].operands[S.use_slot[j]] = sv;
    }

    return 1;
}

/* ---- Driver ---- */

void bir_sroa(bir_module_t *M)
{
    uint32_t f;
    if (!M) return;

    for (f = 0; f < M->num_funcs; f++) {
        const bir_func_t *F = &M->funcs[f];
        int progress = 1;

        /* One struct alloca per pass, then rescan, since each transform
         * shifts every index out from under us. It terminates: each pass
         * leaves its struct alloca an orphan, the guard waves it off to the
         * left with the other crosses, and we never pick it up again. */
        while (progress) {
            uint32_t b;
            progress = 0;
            for (b = F->first_block;
                 b < F->first_block + F->num_blocks && !progress; b++) {
                const bir_block_t *B = &M->blocks[b];
                uint32_t jj;
                for (jj = 0; jj < B->num_insts; jj++) {
                    uint32_t ii = B->first_inst + jj;
                    uint32_t st;
                    if (M->insts[ii].op != BIR_ALLOCA) continue;
                    st = pointee(M, M->insts[ii].type);
                    if (st == BIR_VAL_NONE || st >= M->num_types ||
                        M->types[st].kind != BIR_TYPE_STRUCT)
                        continue;
                    if (try_alloca(M, F, ii)) { progress = 1; break; }
                }
            }
        }
    }
}
