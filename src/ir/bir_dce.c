#include "bir_dce.h"
#include <string.h>

/*
 * bir_dce: dead code elimination.
 *
 * Remove instructions whose results are never referenced and that
 * have no side effects.  Iterates to fixpoint since removing one
 * dead instruction may make its operands' producers dead too.
 * Then compact surviving instructions and close inter-function gaps.
 */

#define OPT_UNDEF 0xFFFFFFFFu

/* ---- Working State ---- */

typedef struct {
    bir_module_t *M;
    uint32_t func_idx;
    uint32_t base_block, num_blocks;
    uint32_t base_inst;
    uint32_t num_insts;
    uint32_t dead[BIR_MAX_INSTS / 32];
    uint32_t use_count[BIR_MAX_INSTS];
    uint32_t inst_renum[BIR_MAX_INSTS];
    uint8_t  rch[BIR_MAX_BLOCKS];
} opt_t;

static opt_t G;

/* ---- Helpers ---- */

static int is_inline_block_ref(uint16_t op, uint8_t j)
{
    switch (op) {
    case BIR_BR:      return j == 0;
    case BIR_BR_COND: return j >= 1 && j <= 3;
    case BIR_SWITCH:  return j == 1;
    case BIR_PHI:     return j % 2 == 0;
    case BIR_CALL:    return j == 0;
    case BIR_FNREF:   return j == 0;
    case BIR_GLOBAL_REF: return j == 0;
    default:          return 0;
    }
}

/* Is extra operand j a block reference? */
static int is_extra_block_ref(uint16_t op, uint32_t j)
{
    if (op == BIR_PHI)    return j % 2 == 0;
    if (op == BIR_SWITCH) return j == 1 || (j >= 3 && j % 2 == 1);
    if (op == BIR_CALL)   return j == 0;
    return 0;
}

/* Check if an opcode is pure (no side effects, safe to eliminate) */
static int is_pure_op(uint16_t op)
{
    switch (op) {
    /* Arithmetic */
    case BIR_ADD: case BIR_SUB: case BIR_MUL:
    case BIR_SDIV: case BIR_UDIV: case BIR_SREM: case BIR_UREM:
    case BIR_FADD: case BIR_FSUB: case BIR_FMUL: case BIR_FDIV:
    case BIR_FREM:
    /* Bitwise */
    case BIR_AND: case BIR_OR: case BIR_XOR:
    case BIR_SHL: case BIR_LSHR: case BIR_ASHR:
    /* Comparison */
    case BIR_ICMP: case BIR_FCMP:
    /* Conversion */
    case BIR_TRUNC: case BIR_ZEXT: case BIR_SEXT:
    case BIR_FPTRUNC: case BIR_FPEXT:
    case BIR_FPTOSI: case BIR_FPTOUI: case BIR_SITOFP: case BIR_UITOFP:
    case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST:
    /* Math */
    case BIR_SQRT: case BIR_RSQ: case BIR_RCP:
    case BIR_EXP2: case BIR_LOG2:
    case BIR_SIN: case BIR_COS:
    case BIR_FABS: case BIR_FLOOR: case BIR_CEIL:
    case BIR_FTRUNC: case BIR_RNDNE:
    case BIR_FMAX: case BIR_FMIN:
    /* SSA */
    case BIR_PHI:
    /* Memory (no observable effect if unused) */
    case BIR_ALLOCA: case BIR_SHARED_ALLOC: case BIR_GLOBAL_REF: case BIR_GEP:
    /* Thread model (read-only intrinsics) */
    case BIR_THREAD_ID: case BIR_BLOCK_ID:
    case BIR_BLOCK_DIM: case BIR_GRID_DIM:
    /* Misc */
    case BIR_SELECT:
    case BIR_FNREF:
        return 1;
    default:
        return 0;
    }
}

/* ---- Dead Code Elimination ---- */

static int dce_pass(opt_t *S)
{
    bir_module_t *M = S->M;
    uint32_t base = S->base_inst;
    uint32_t end = base + S->num_insts;
    int changes = 0;
    int changed = 1;
    int guard = 100;

    while (changed && --guard) {
        changed = 0;

        /* Build use counts for instructions in this function */
        memset(S->use_count + base, 0, S->num_insts * sizeof(uint32_t));

        for (uint32_t i = base; i < end; i++) {
            if ((S->dead[i / 32] >> (i % 32)) & 1) continue;
            const bir_inst_t *I = &M->insts[i];

            if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
                uint32_t start = I->operands[0];
                uint32_t count = I->operands[1];
                for (uint32_t j = 0; j < count
                     && (start + j) < M->num_extra_ops; j++) {
                    if (is_extra_block_ref(I->op, j)) continue;
                    uint32_t ref = M->extra_operands[start + j];
                    if (ref != BIR_VAL_NONE && !BIR_VAL_IS_CONST(ref)) {
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= base && idx < end)
                            S->use_count[idx]++;
                    }
                }
            } else {
                for (uint8_t j = 0; j < I->num_operands
                     && j < BIR_OPERANDS_INLINE; j++) {
                    if (is_inline_block_ref(I->op, j)) continue;
                    uint32_t ref = I->operands[j];
                    if (ref != BIR_VAL_NONE && !BIR_VAL_IS_CONST(ref)) {
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= base && idx < end)
                            S->use_count[idx]++;
                    }
                }
            }
        }

        /* Remove dead pure instructions */
        for (uint32_t i = base; i < end; i++) {
            if ((S->dead[i / 32] >> (i % 32)) & 1) continue;
            const bir_inst_t *inst = &M->insts[i];
            int pure = is_pure_op(inst->op);
            if (!pure && inst->op == BIR_LOAD && inst->subop == 0)
                pure = 1;  /* non-volatile load, unused result = dead */
            if (S->use_count[i] == 0 && pure) {
                S->dead[i / 32] |= 1u << (i % 32);
                changed = 1;
                changes++;
            }
        }
    }

    return changes;
}

/* ---- Compact ---- */

/* Renumber a single operand after dead instruction removal */
static uint32_t remap_operand(uint32_t ref, const uint32_t *inst_renum,
                              uint32_t max_inst)
{
    if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref))
        return ref;

    uint32_t idx = BIR_VAL_INDEX(ref);
    if (idx < max_inst && inst_renum[idx] != OPT_UNDEF)
        return BIR_MAKE_VAL(inst_renum[idx]);

    return ref;
}

static void compact(opt_t *S)
{
    bir_module_t *M = S->M;
    uint32_t old_num_insts = M->num_insts;
    bir_func_t *F = &M->funcs[S->func_idx];

    /* Build inst_renum: old absolute index -> new absolute index */
    for (uint32_t i = 0; i < old_num_insts; i++)
        S->inst_renum[i] = OPT_UNDEF;

    uint32_t new_idx = S->base_inst;
    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        uint32_t abs_b = S->base_block + bi;
        const bir_block_t *B = &M->blocks[abs_b];
        for (uint32_t j = 0; j < B->num_insts; j++) {
            uint32_t ii = B->first_inst + j;
            if ((S->dead[ii / 32] >> (ii % 32)) & 1) continue;
            S->inst_renum[ii] = new_idx++;
        }
    }

    uint32_t new_total = new_idx - S->base_inst;

    /* Compact in-place: write cursor <= read cursor since we only
     * remove instructions, so no scratch buffer needed. */
    uint32_t wr = S->base_inst;
    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        uint32_t abs_b = S->base_block + bi;
        const bir_block_t *B = &M->blocks[abs_b];
        uint32_t block_start = wr;
        for (uint32_t j = 0; j < B->num_insts; j++) {
            uint32_t ii = B->first_inst + j;
            if ((S->dead[ii / 32] >> (ii % 32)) & 1) continue;
            if (wr != ii) {
                M->insts[wr] = M->insts[ii];
                /* The line table is indexed by instruction position, so it
                 * has to move with them or every line past the first dead
                 * instruction points at the wrong source. */
                M->inst_lines[wr] = M->inst_lines[ii];
            }
            wr++;
        }
        M->blocks[abs_b].first_inst = block_start;
        M->blocks[abs_b].num_insts = wr - block_start;
    }

    F->total_insts = new_total;

    /* Remap all operands in this function's instructions */
    for (uint32_t i = S->base_inst; i < S->base_inst + new_total; i++) {
        bir_inst_t *I = &M->insts[i];

        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            for (uint32_t j = 0; j < count
                 && (start + j) < M->num_extra_ops; j++) {
                if (is_extra_block_ref(I->op, j)) continue;
                M->extra_operands[start + j] =
                    remap_operand(M->extra_operands[start + j],
                                  S->inst_renum, old_num_insts);
            }
        } else {
            for (uint8_t j = 0; j < I->num_operands
                 && j < BIR_OPERANDS_INLINE; j++) {
                if (is_inline_block_ref(I->op, j)) continue;
                I->operands[j] =
                    remap_operand(I->operands[j], S->inst_renum,
                                  old_num_insts);
            }
        }
    }
}

static uint32_t nsuc(const bir_module_t *M, const bir_block_t *B)
{
    const bir_inst_t *I;

    if (B->num_insts == 0u || B->first_inst >= M->num_insts) return 0u;
    I = &M->insts[B->first_inst + B->num_insts - 1u];
    if (I->op == BIR_BR)      return 1u;
    if (I->op == BIR_BR_COND) return 2u;
    if (I->op != BIR_SWITCH)  return 0u;
    if (I->num_operands != BIR_OPERANDS_OVERFLOW) return 1u;
    if (I->operands[1] < 2u) return 0u;
    return 1u + (I->operands[1] - 2u) / 2u;
}

static uint32_t suc(const bir_module_t *M, const bir_block_t *B, uint32_t k)
{
    const bir_inst_t *I = &M->insts[B->first_inst + B->num_insts - 1u];

    if (I->op == BIR_BR)      return bir_oper(M, I, 0u);
    if (I->op == BIR_BR_COND) return bir_oper(M, I, 1u + k);
    if (I->num_operands != BIR_OPERANDS_OVERFLOW) return bir_oper(M, I, 1u);
    return (k == 0u) ? bir_oper(M, I, 1u) : bir_oper(M, I, 2u * k + 1u);
}

static int plive(const opt_t *S, uint32_t p, uint32_t b)
{
    const bir_module_t *M = S->M;
    uint32_t lp, ns, k;

    if (p < S->base_block) return 1;
    lp = p - S->base_block;
    if (lp >= S->num_blocks) return 1;
    if (!S->rch[lp]) return 0;
    ns = nsuc(M, &M->blocks[p]);
    for (k = 0u; k < ns; k++)
        if (suc(M, &M->blocks[p], k) == b) return 1;
    return 0;
}

static int phix(opt_t *S, uint32_t b, bir_inst_t *I)
{
    bir_module_t *M = S->M;
    uint32_t w = 0u, k, n;
    int drop = 0;

    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t st = I->operands[0];
        n = I->operands[1];
        for (k = 0u; k + 1u < n; k += 2u) {
            if (st + k + 1u >= M->num_extra_ops) break;
            if (!plive(S, M->extra_operands[st + k], b)) { drop = 1; continue; }
            M->extra_operands[st + w]      = M->extra_operands[st + k];
            M->extra_operands[st + w + 1u] = M->extra_operands[st + k + 1u];
            w += 2u;
        }
        I->operands[1] = w;
        return drop;
    }

    n = I->num_operands;
    for (k = 0u; k + 1u < n; k += 2u) {
        if (!plive(S, I->operands[k], b)) { drop = 1; continue; }
        I->operands[w]      = I->operands[k];
        I->operands[w + 1u] = I->operands[k + 1u];
        w += 2u;
    }
    I->num_operands = (uint8_t)w;
    return drop;
}

static void rmark(opt_t *S)
{
    bir_module_t *M = S->M;
    uint32_t nb = S->num_blocks, i, k;
    int chg = 1, g = (int)nb + 1;

    for (i = 0u; i < nb; i++) S->rch[i] = 0u;
    S->rch[0] = 1u;

    while (chg && g-- > 0) {
        chg = 0;
        for (i = 0u; i < nb; i++) {
            const bir_block_t *B = &M->blocks[S->base_block + i];
            uint32_t ns;
            if (!S->rch[i]) continue;
            ns = nsuc(M, B);
            for (k = 0u; k < ns; k++) {
                uint32_t t = suc(M, B, k);
                if (t < S->base_block) continue;
                t -= S->base_block;
                if (t >= nb || S->rch[t]) continue;
                S->rch[t] = 1u;
                chg = 1;
            }
        }
    }
}

static int rchk(opt_t *S)
{
    bir_module_t *M = S->M;
    uint32_t nb = S->num_blocks, i, k;
    int n = 0;

    rmark(S);

    for (i = 1u; i < nb; i++) {
        bir_block_t *B = &M->blocks[S->base_block + i];
        bir_inst_t *I;
        if (S->rch[i] || B->num_insts == 0u) continue;
        for (k = 1u; k < B->num_insts; k++) {
            uint32_t ii = B->first_inst + k;
            S->dead[ii / 32] |= 1u << (ii % 32);
            n++;
        }
        I = &M->insts[B->first_inst];
        if (I->op == BIR_UNREACHABLE) continue;
        I->op = BIR_UNREACHABLE;
        I->subop = 0;
        I->num_operands = 0;
        I->type = bir_type_void(M);
        n++;
    }

    for (i = 0u; i < nb; i++) {
        const bir_block_t *B = &M->blocks[S->base_block + i];
        if (!S->rch[i]) continue;
        for (k = 0u; k < B->num_insts; k++) {
            uint32_t ii = B->first_inst + k;
            if (M->insts[ii].op != BIR_PHI) continue;
            if ((S->dead[ii / 32] >> (ii % 32)) & 1u) continue;
            n += phix(S, S->base_block + i, &M->insts[ii]);
        }
    }

    return n;
}

/* ---- Per-Function Driver ---- */

static int opt_run_func(opt_t *S, uint32_t fi)
{
    const bir_func_t *F = &S->M->funcs[fi];
    if (F->num_blocks == 0 || F->total_insts == 0) return 0;

    S->func_idx   = fi;
    S->base_block = F->first_block;
    S->num_blocks = F->num_blocks;
    S->base_inst  = S->M->blocks[F->first_block].first_inst;
    S->num_insts  = F->total_insts;

    {
        uint32_t lo = S->base_inst / 32;
        uint32_t hi = (S->base_inst + S->num_insts + 31) / 32;
        memset(S->dead + lo, 0, (hi - lo) * sizeof(uint32_t));
    }

    int changes = rchk(S);

    changes += dce_pass(S);

    if (changes > 0)
        compact(S);

    return changes;
}

/* ---- Public API ---- */

int bir_dce(bir_module_t *M)
{
    opt_t *S = &G;
    memset(S, 0, sizeof(*S));
    S->M = M;

    int total = 0;
    for (uint32_t fi = 0; fi < M->num_funcs; fi++)
        total += opt_run_func(S, fi);

    /* Close inter-function gaps (same pattern as bir_mem2reg) */
    if (total > 0) {
        uint32_t dst = 0;
        for (uint32_t fi = 0; fi < M->num_funcs; fi++) {
            bir_func_t *F = &M->funcs[fi];
            if (F->num_blocks == 0) continue;

            uint32_t src = M->blocks[F->first_block].first_inst;
            uint32_t count = F->total_insts;

            if (src == dst) {
                dst += count;
                continue;
            }

            int32_t shift = (int32_t)dst - (int32_t)src;

            memmove(&M->insts[dst], &M->insts[src],
                    count * sizeof(bir_inst_t));
            memmove(&M->inst_lines[dst], &M->inst_lines[src],
                    count * sizeof(M->inst_lines[0]));

            for (uint16_t bi = 0; bi < F->num_blocks; bi++) {
                uint32_t abs_b = F->first_block + bi;
                M->blocks[abs_b].first_inst =
                    (uint32_t)((int32_t)M->blocks[abs_b].first_inst + shift);
            }

            for (uint32_t i = dst; i < dst + count; i++) {
                bir_inst_t *I = &M->insts[i];
                if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
                    uint32_t start = I->operands[0];
                    uint32_t cnt = I->operands[1];
                    for (uint32_t j = 0; j < cnt
                         && (start + j) < M->num_extra_ops; j++) {
                        if (is_extra_block_ref(I->op, j)) continue;
                        uint32_t ref = M->extra_operands[start + j];
                        if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE)
                            continue;
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= src && idx < src + count)
                            M->extra_operands[start + j] =
                                BIR_MAKE_VAL(
                                    (uint32_t)((int32_t)idx + shift));
                    }
                } else {
                    for (uint8_t j = 0; j < I->num_operands
                         && j < BIR_OPERANDS_INLINE; j++) {
                        if (is_inline_block_ref(I->op, j)) continue;
                        uint32_t ref = I->operands[j];
                        if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE)
                            continue;
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= src && idx < src + count)
                            I->operands[j] =
                                BIR_MAKE_VAL(
                                    (uint32_t)((int32_t)idx + shift));
                    }
                }
            }

            dst += count;
        }
        M->num_insts = dst;
    }

    return total;
}
