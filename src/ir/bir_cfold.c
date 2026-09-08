#include "bir_cfold.h"
#include <math.h>

/*
 * bir_cfold: constant folding.
 *
 * Single forward pass per function.  SSA guarantees defs dominate
 * uses, so when we reach an instruction all its operands have
 * already been resolved through val_rewrite[].  After folding,
 * DCE cleans up the dead instructions.
 */

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

/* Rewrite a single operand through val_rewrite[]. */
static uint32_t rewrite_val(uint32_t ref, const uint32_t *val_rewrite)
{
    if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref))
        return ref;
    uint32_t rw = val_rewrite[BIR_VAL_INDEX(ref)];
    return rw != BIR_VAL_NONE ? rw : ref;
}

/* Get the integer bit width from a type index. Returns 0 if not integer. */
static int int_width(const bir_module_t *M, uint32_t tidx)
{
    if (tidx >= M->num_types) return 0;
    if (M->types[tidx].kind != BIR_TYPE_INT) return 0;
    return (int)M->types[tidx].width;
}

/* Get the float bit width from a type index. Returns 0 if not float. */
static int float_width(const bir_module_t *M, uint32_t tidx)
{
    if (tidx >= M->num_types) return 0;
    if (M->types[tidx].kind != BIR_TYPE_FLOAT) return 0;
    return (int)M->types[tidx].width;
}

/* Mask an integer to w bits (unsigned). */
static int64_t mask_to_width(int64_t val, int w)
{
    if (w >= 64) return val;
    return val & (int64_t)((1ULL << w) - 1);
}

/* Sign-extend an integer from w bits to int64_t. */
static int64_t sign_extend(int64_t val, int w)
{
    if (w >= 64) return val;
    val = mask_to_width(val, w);
    int64_t sign_bit = (int64_t)(1ULL << (w - 1));
    return (val ^ sign_bit) - sign_bit;
}

/* ---- Working State ---- */

typedef struct {
    bir_module_t *M;
    uint32_t val_rewrite[BIR_MAX_INSTS];
} cf_t;

static cf_t G;

/* ---- Fold One Instruction ---- */

/*
 * Try to fold instruction at absolute index ii.
 * If foldable, sets S->val_rewrite[ii] and returns 1.
 * Otherwise returns 0.
 */
static int try_fold(cf_t *S, uint32_t ii)
{
    bir_module_t *M = S->M;
    bir_inst_t *I = &M->insts[ii];
    uint16_t op = I->op;

    /* SELECT with constant condition — propagate chosen operand */
    if (op == BIR_SELECT && I->num_operands >= 3) {
        uint32_t cond = I->operands[0];
        if (!BIR_VAL_IS_CONST(cond)) return 0;
        uint32_t ci = BIR_VAL_INDEX(cond);
        if (ci >= M->num_consts) return 0;
        const bir_const_t *c = &M->consts[ci];
        if (c->kind != BIR_CONST_INT) return 0;
        S->val_rewrite[ii] = c->d.ival ? I->operands[1] : I->operands[2];
        return 1;
    }

    /* From here, require all value operands to be constants */
    if (I->num_operands == 0 || I->num_operands == BIR_OPERANDS_OVERFLOW)
        return 0;

    for (uint8_t j = 0; j < I->num_operands && j < BIR_OPERANDS_INLINE; j++) {
        if (is_inline_block_ref(op, j)) continue;
        uint32_t v = I->operands[j];
        if (v == BIR_VAL_NONE) continue;
        if (!BIR_VAL_IS_CONST(v)) return 0;
    }

    /* Collect constant operands */
    const bir_const_t *c0 = NULL, *c1 = NULL;

    if (I->num_operands >= 1 && !is_inline_block_ref(op, 0)) {
        uint32_t idx = BIR_VAL_INDEX(I->operands[0]);
        if (idx < M->num_consts) c0 = &M->consts[idx];
    }
    if (I->num_operands >= 2 && !is_inline_block_ref(op, 1)) {
        uint32_t idx = BIR_VAL_INDEX(I->operands[1]);
        if (idx < M->num_consts) c1 = &M->consts[idx];
    }

    int rw = int_width(M, I->type);
    int fw = float_width(M, I->type);

    /* Integer comparison (before binary ops — icmp result is i1,
     * which would match the rw > 0 gate and hit default: return 0) */
    if (op == BIR_ICMP && c0 && c1
        && c0->kind == BIR_CONST_INT && c1->kind == BIR_CONST_INT) {
        int64_t a = c0->d.ival, b = c1->d.ival;
        int sw = int_width(M, c0->type);
        if (sw == 0) sw = 32;
        int64_t sa = sign_extend(a, sw), sb = sign_extend(b, sw);
        uint64_t ua = (uint64_t)mask_to_width(a, sw);
        uint64_t ub = (uint64_t)mask_to_width(b, sw);
        int result;

        switch (I->subop) {
        case BIR_ICMP_EQ:  result = (ua == ub); break;
        case BIR_ICMP_NE:  result = (ua != ub); break;
        case BIR_ICMP_SLT: result = (sa <  sb); break;
        case BIR_ICMP_SLE: result = (sa <= sb); break;
        case BIR_ICMP_SGT: result = (sa >  sb); break;
        case BIR_ICMP_SGE: result = (sa >= sb); break;
        case BIR_ICMP_ULT: result = (ua <  ub); break;
        case BIR_ICMP_ULE: result = (ua <= ub); break;
        case BIR_ICMP_UGT: result = (ua >  ub); break;
        case BIR_ICMP_UGE: result = (ua >= ub); break;
        default: return 0;
        }

        uint32_t ci = bir_const_int(M, I->type, result);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Integer binary ops */
    if (c0 && c1 && c0->kind == BIR_CONST_INT && c1->kind == BIR_CONST_INT
        && rw > 0) {
        int64_t a = c0->d.ival, b = c1->d.ival;
        int64_t r;
        int sw = int_width(M, c0->type);
        if (sw == 0) sw = rw;

        switch (op) {
        case BIR_ADD:  r = a + b; break;
        case BIR_SUB:  r = a - b; break;
        case BIR_MUL:  r = a * b; break;
        case BIR_SDIV:
            if (b == 0) return 0;
            if (sign_extend(b, sw) == -1 && sign_extend(a, sw) == INT64_MIN)
                return 0;
            r = sign_extend(a, sw) / sign_extend(b, sw);
            break;
        case BIR_UDIV:
            if (b == 0) return 0;
            r = (int64_t)((uint64_t)mask_to_width(a, sw)
                / (uint64_t)mask_to_width(b, sw));
            break;
        case BIR_SREM:
            if (b == 0) return 0;
            if (sign_extend(b, sw) == -1 && sign_extend(a, sw) == INT64_MIN)
                return 0;
            r = sign_extend(a, sw) % sign_extend(b, sw);
            break;
        case BIR_UREM:
            if (b == 0) return 0;
            r = (int64_t)((uint64_t)mask_to_width(a, sw)
                % (uint64_t)mask_to_width(b, sw));
            break;
        case BIR_AND:  r = a & b; break;
        case BIR_OR:   r = a | b; break;
        case BIR_XOR:  r = a ^ b; break;
        case BIR_SHL:  r = (int64_t)((uint64_t)a << (b & (sw - 1))); break;
        case BIR_LSHR:
            r = (int64_t)((uint64_t)mask_to_width(a, sw)
                >> (b & (sw - 1)));
            break;
        case BIR_ASHR:
            r = sign_extend(a, sw) >> (b & (sw - 1));
            break;
        default: return 0;
        }

        r = mask_to_width(r, rw);
        uint32_t ci = bir_const_int(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Float comparison */
    if (op == BIR_FCMP && c0 && c1
        && c0->kind == BIR_CONST_FLOAT && c1->kind == BIR_CONST_FLOAT) {
        double a = c0->d.fval, b = c1->d.fval;
        int ord = (a == a) && (b == b);  /* neither is NaN */
        int result;

        switch (I->subop) {
        case BIR_FCMP_OEQ: result = ord && (a == b); break;
        case BIR_FCMP_ONE: result = ord && (a != b); break;
        case BIR_FCMP_OLT: result = ord && (a <  b); break;
        case BIR_FCMP_OLE: result = ord && (a <= b); break;
        case BIR_FCMP_OGT: result = ord && (a >  b); break;
        case BIR_FCMP_OGE: result = ord && (a >= b); break;
        case BIR_FCMP_UEQ: result = !ord || (a == b); break;
        case BIR_FCMP_UNE: result = !ord || (a != b); break;
        case BIR_FCMP_ULT: result = !ord || (a <  b); break;
        case BIR_FCMP_ULE: result = !ord || (a <= b); break;
        case BIR_FCMP_UGT: result = !ord || (a >  b); break;
        case BIR_FCMP_UGE: result = !ord || (a >= b); break;
        case BIR_FCMP_ORD: result = ord; break;
        case BIR_FCMP_UNO: result = !ord; break;
        default: return 0;
        }

        uint32_t ci = bir_const_int(M, I->type, result);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Float binary ops */
    if (c0 && c1 && c0->kind == BIR_CONST_FLOAT
        && c1->kind == BIR_CONST_FLOAT && fw > 0) {
        double a = c0->d.fval, b = c1->d.fval;
        double r;

        switch (op) {
        case BIR_FADD: r = a + b; break;
        case BIR_FSUB: r = a - b; break;
        case BIR_FMUL: r = a * b; break;
        case BIR_FDIV: r = a / b; break;
        case BIR_FREM: r = fmod(a, b); break;
        case BIR_FMAX: r = a > b ? a : b; break;
        case BIR_FMIN: r = a < b ? a : b; break;
        default: return 0;
        }

        if (fw == 32) r = (double)(float)r;
        uint32_t ci = bir_const_float(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Integer conversions (unary, one constant operand) */
    if (c0 && c0->kind == BIR_CONST_INT && rw > 0) {
        int64_t a = c0->d.ival;
        int64_t r;
        int sw = int_width(M, c0->type);
        if (sw == 0) sw = 32;

        switch (op) {
        case BIR_TRUNC: r = mask_to_width(a, rw); break;
        case BIR_ZEXT:  r = mask_to_width(a, sw); break;
        case BIR_SEXT:  r = sign_extend(a, sw); break;
        default: return 0;
        }

        r = mask_to_width(r, rw);
        uint32_t ci = bir_const_int(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Int-to-float conversions */
    if (c0 && c0->kind == BIR_CONST_INT && fw > 0) {
        int64_t a = c0->d.ival;
        int sw = int_width(M, c0->type);
        if (sw == 0) sw = 32;
        double r;

        switch (op) {
        case BIR_SITOFP: r = (double)sign_extend(a, sw); break;
        case BIR_UITOFP: r = (double)(uint64_t)mask_to_width(a, sw); break;
        default: return 0;
        }

        if (fw == 32) r = (double)(float)r;
        uint32_t ci = bir_const_float(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Float-to-int conversions */
    if (c0 && c0->kind == BIR_CONST_FLOAT && rw > 0) {
        double a = c0->d.fval;
        int64_t r;

        switch (op) {
        case BIR_FPTOSI: r = (int64_t)a; break;
        case BIR_FPTOUI: r = (int64_t)(uint64_t)a; break;
        default: return 0;
        }

        r = mask_to_width(r, rw);
        uint32_t ci = bir_const_int(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    /* Float width conversions */
    if (c0 && c0->kind == BIR_CONST_FLOAT && fw > 0) {
        double r = c0->d.fval;

        switch (op) {
        case BIR_FPTRUNC:
            if (fw == 32) r = (double)(float)r;
            else return 0; /* skip folding for f16 */
            break;
        case BIR_FPEXT:   break; /* already double internally */
        default: return 0;
        }

        uint32_t ci = bir_const_float(M, I->type, r);
        S->val_rewrite[ii] = BIR_MAKE_CONST(ci);
        return 1;
    }

    return 0;
}

/* ---- Per-Function Pass ---- */

static int cf_run_func(cf_t *S, uint32_t fi)
{
    bir_module_t *M = S->M;
    const bir_func_t *F = &M->funcs[fi];
    if (F->num_blocks == 0 || F->total_insts == 0) return 0;

    uint32_t base = M->blocks[F->first_block].first_inst;
    uint32_t end = base + F->total_insts;
    int changes = 0;

    for (uint32_t i = base; i < end; i++)
        S->val_rewrite[i] = BIR_VAL_NONE;

    for (uint32_t i = base; i < end; i++) {
        bir_inst_t *I = &M->insts[i];

        /* Rewrite operands through val_rewrite[] */
        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            for (uint32_t j = 0; j < count
                 && (start + j) < M->num_extra_ops; j++) {
                if (is_extra_block_ref(I->op, j)) continue;
                M->extra_operands[start + j] =
                    rewrite_val(M->extra_operands[start + j],
                                S->val_rewrite);
            }
        } else {
            for (uint8_t j = 0; j < I->num_operands
                 && j < BIR_OPERANDS_INLINE; j++) {
                if (is_inline_block_ref(I->op, j)) continue;
                I->operands[j] = rewrite_val(I->operands[j],
                                             S->val_rewrite);
            }
        }

        /* Try to fold */
        changes += try_fold(S, i);
    }

    return changes;
}

/* ---- Public API ---- */

int bir_cfold(bir_module_t *M)
{
    G.M = M;
    int total = 0;
    for (uint32_t fi = 0; fi < M->num_funcs; fi++)
        total += cf_run_func(&G, fi);
    return total;
}
