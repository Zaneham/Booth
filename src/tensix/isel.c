#include "tensix.h"
#include <string.h>

/* BIR SSA -> SFPU machine IR. 32 lanes pretending to be a million threads. */
static struct {
    tt_module_t         *tt;
    const bir_module_t  *bir;

    uint32_t    dst_offset;
    uint32_t    block_map[BIR_MAX_BLOCKS];

    #define TT_MAX_PRED_DEPTH 32
    struct {
        uint32_t false_bir;     /* BIR block for else path */
        uint32_t merge_bir;     /* BIR block for merge */
        int      has_else;
    } pred_stack[TT_MAX_PRED_DEPTH];
    uint32_t    pred_depth;

    int         had_error;

} S;

/* ---- Operands ---- */

static tt_operand_t mop_none(void)
{
    tt_operand_t o = {0};
    o.kind = TT_MOP_NONE;
    return o;
}

static tt_operand_t mop_vreg(uint32_t v)
{
    tt_operand_t o = {0};
    o.kind = TT_MOP_VREG;
    o.reg_num = (uint16_t)v;
    return o;
}

static tt_operand_t mop_lreg(uint16_t r)
{
    tt_operand_t o = {0};
    o.kind = TT_MOP_LREG;
    o.reg_num = r;
    return o;
}

static tt_operand_t mop_imm(int32_t val)
{
    tt_operand_t o = {0};
    o.kind = TT_MOP_IMM;
    o.imm = val;
    return o;
}

static tt_operand_t mop_dst_row(uint16_t row)
{
    tt_operand_t o = {0};
    o.kind = TT_MOP_DST_ROW;
    o.reg_num = row;
    return o;
}

/* ---- Helpers ---- */

static uint32_t get_num_ops(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW)
        return I->operands[1];
    return I->num_operands;
}

static uint32_t bir_block_successor(uint32_t bir_bi)
{
    if (bir_bi >= S.bir->num_blocks) return 0xFFFFFFFF;
    const bir_block_t *B = &S.bir->blocks[bir_bi];
    if (B->num_insts == 0) return 0xFFFFFFFF;
    const bir_inst_t *last = &S.bir->insts[B->first_inst + B->num_insts - 1];
    if (last->op == BIR_BR)
        return last->operands[0];
    return 0xFFFFFFFF;
}

static uint32_t find_merge_block(uint32_t true_bir, uint32_t false_bir)
{
    uint32_t true_succ = bir_block_successor(true_bir);
    if (true_succ == false_bir) return false_bir;
    uint32_t false_succ = bir_block_successor(false_bir);
    if (true_succ != 0xFFFFFFFF && true_succ == false_succ)
        return true_succ;
    return false_bir;
}

static uint32_t new_vreg(void)
{
    if (S.tt->vreg_count >= TT_MAX_VREGS) return 0;
    return S.tt->vreg_count++;
}

static uint32_t map_bir_val(uint32_t bir_inst)
{
    if (bir_inst < BIR_MAX_INSTS && S.tt->val_vreg[bir_inst] != 0)
        return S.tt->val_vreg[bir_inst];
    uint32_t v = new_vreg();
    if (bir_inst < BIR_MAX_INSTS)
        S.tt->val_vreg[bir_inst] = v;
    return v;
}

/* ---- Emission ---- */

static uint32_t emit(uint16_t op, uint8_t fmt,
                     uint8_t num_defs, uint8_t num_uses,
                     const tt_operand_t *ops, uint16_t flags)
{
    if (S.tt->num_minsts >= TT_MAX_MINSTS) {
        if (!S.had_error) tterr(BC_E537, (unsigned)TT_MAX_MINSTS);
        S.had_error = 1;
        return 0;
    }
    uint32_t idx = S.tt->num_minsts++;
    tt_minst_t *I = &S.tt->minsts[idx];
    I->op = op;
    I->fmt = fmt;
    I->num_defs = num_defs;
    I->num_uses = num_uses;
    I->flags = flags;
    I->pad = 0;
    int total = num_defs + num_uses;
    if (total > TT_MINST_MAX_OPS) total = TT_MINST_MAX_OPS;
    for (int i = 0; i < total; i++)
        I->operands[i] = ops[i];
    for (int i = total; i < TT_MINST_MAX_OPS; i++)
        I->operands[i] = mop_none();
    return idx;
}

static void emit_fmtA(uint16_t op, tt_operand_t dst,
                       tt_operand_t src_a, tt_operand_t src_b,
                       tt_operand_t src_c, uint16_t mod1)
{
    tt_operand_t ops[5] = { dst, src_a, src_b, src_c, mop_imm(mod1) };
    emit(op, TT_FMT_A, 1, 4, ops, mod1);
}

static void emit_fmtB(uint16_t op, tt_operand_t dst,
                       tt_operand_t src, int32_t imm12, uint16_t mod1)
{
    tt_operand_t ops[4] = { dst, src, mop_imm(imm12), mop_imm(mod1) };
    emit(op, TT_FMT_B, 1, 3, ops, mod1);
}

static void emit_fmtC(uint16_t op, int32_t imm16, uint16_t mod1)
{
    tt_operand_t ops[2] = { mop_imm(imm16), mop_imm(mod1) };
    emit(op, TT_FMT_C, 0, 2, ops, mod1);
}

static void emit_fmtD(uint16_t op, tt_operand_t lreg,
                       uint16_t mod0, uint16_t addr_mode, uint16_t dst_addr)
{
    tt_operand_t ops[4] = { lreg, mop_imm(mod0), mop_imm(addr_mode),
                            mop_dst_row(dst_addr) };
    if (op == TT_SFPLOAD)
        emit(op, TT_FMT_D, 1, 3, ops, mod0);
    else
        emit(op, TT_FMT_D, 0, 4, ops, mod0);
}

static void emit_fmtE(uint16_t op, tt_operand_t lreg,
                       uint16_t mod0, uint16_t imm16)
{
    tt_operand_t ops[3] = { lreg, mop_imm(mod0), mop_imm(imm16) };
    emit(op, TT_FMT_E, 1, 2, ops, mod0);
}

/* ---- Constants ---- */

/* Zero and one are free. Everything else is two loads. */
static tt_operand_t materialise_const(uint32_t const_idx)
{
    if (const_idx >= S.bir->num_consts)
        return mop_lreg(TT_LREG_ZERO);

    const bir_const_t *C = &S.bir->consts[const_idx];

    if (C->kind == BIR_CONST_ZERO ||
        (C->kind == BIR_CONST_INT && C->d.ival == 0) ||
        (C->kind == BIR_CONST_FLOAT && C->d.fval == 0.0) ||
        C->kind == BIR_CONST_NULL) {
        return mop_lreg(TT_LREG_ZERO);
    }

    if (C->kind == BIR_CONST_FLOAT && C->d.fval == 1.0)
        return mop_lreg(TT_LREG_ONE);

    uint32_t vr = new_vreg();
    tt_operand_t dst = mop_vreg(vr);

    uint32_t val;
    if (C->kind == BIR_CONST_FLOAT) {
        union { float f; uint32_t u; } pun;
        pun.f = (float)C->d.fval;
        val = pun.u;
    } else {
        val = (uint32_t)C->d.ival;
    }
    emit_fmtE(TT_SFPLOADI, dst, TT_LOADI_MOD0_UPPER,
              (uint16_t)(val >> 16));
    emit_fmtE(TT_SFPLOADI, dst, TT_LOADI_MOD0_LOWER,
              (uint16_t)(val & 0xFFFF));

    return dst;
}

static tt_operand_t resolve_val(uint32_t val)
{
    if (val == BIR_VAL_NONE) return mop_lreg(TT_LREG_ZERO);

    if (BIR_VAL_IS_CONST(val))
        return materialise_const(BIR_VAL_INDEX(val));

    uint32_t inst_idx = BIR_VAL_INDEX(val);
    uint32_t vr = map_bir_val(inst_idx);
    return mop_vreg(vr);
}

/* ---- FP Arithmetic ---- */

static void isel_fp_arith(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t lhs = resolve_val(I->operands[0]);
    tt_operand_t rhs = resolve_val(I->operands[1]);

    switch (I->op) {
    case BIR_FADD:
        emit_fmtA(TT_SFPADD, dst, lhs, mop_lreg(TT_LREG_ONE), rhs, 0);
        break;
    case BIR_FSUB:
        emit_fmtA(TT_SFPADD, dst, lhs, mop_lreg(TT_LREG_ONE), rhs, 1);
        break;
    case BIR_FMUL:
        emit_fmtA(TT_SFPMUL, dst, lhs, rhs, mop_lreg(TT_LREG_ZERO), 0);
        break;
    default:
        break;
    }
}

static int cimm(uint32_t val, int32_t *out)
{
    if (!BIR_VAL_IS_CONST(val)) return 0;

    uint32_t ci = BIR_VAL_INDEX(val);
    if (ci >= S.bir->num_consts) return 0;

    const bir_const_t *C = &S.bir->consts[ci];
    if (C->kind == BIR_CONST_ZERO) { *out = 0; return 1; }
    if (C->kind == BIR_CONST_INT)  { *out = (int32_t)C->d.ival; return 1; }
    return 0;
}

static void mulk(tt_operand_t dst, tt_operand_t x, uint32_t mag, int neg)
{
    int first = 1;

    if (mag == 0u) {
        emit_fmtB(TT_SFPMOV, dst, mop_lreg(TT_LREG_ZERO), 0, 0);
        return;
    }

    for (int b = 0; b < 32; b++) {
        if (((mag >> b) & 1u) == 0u) continue;

        if (first) {
            emit_fmtB(TT_SFPMOV, dst, x, 0, 0);
            if (b > 0)
                emit_fmtB(TT_SFPSHFT, dst, dst, b, TT_SHFT_MOD1_IMM);
            first = 0;
        } else {
            tt_operand_t t = mop_vreg(new_vreg());
            emit_fmtB(TT_SFPMOV, t, x, 0, 0);
            if (b > 0)
                emit_fmtB(TT_SFPSHFT, t, t, b, TT_SHFT_MOD1_IMM);
            emit_fmtB(TT_SFPIADD, dst, t, 0, TT_IADD_MOD1_NOCC);
        }
    }

    if (neg)
        emit_fmtB(TT_SFPIADD, dst, mop_lreg(TT_LREG_ZERO), 0,
                  TT_IADD_MOD1_SUB | TT_IADD_MOD1_NOCC);
}

static void mulrt(tt_operand_t dst, tt_operand_t a, tt_operand_t b)
{
    tt_operand_t aa  = mop_vreg(new_vreg());
    tt_operand_t bb  = mop_vreg(new_vreg());
    tt_operand_t one = mop_vreg(new_vreg());
    tt_operand_t t   = mop_vreg(new_vreg());

    emit_fmtB(TT_SFPMOV, aa, a, 0, 0);
    emit_fmtB(TT_SFPMOV, bb, b, 0, 0);
    emit_fmtE(TT_SFPLOADI, one, TT_LOADI_MOD0_U16, 1);
    emit_fmtB(TT_SFPMOV, dst, mop_lreg(TT_LREG_ZERO), 0, 0);

    for (int i = 0; i < 32; i++) {
        emit_fmtB(TT_SFPMOV, t, bb, 0, 0);
        emit_fmtB(TT_SFPAND, t, one, 0, 0);
        emit_fmtB(TT_SFPIADD, t, mop_lreg(TT_LREG_ZERO), 0,
                  TT_IADD_MOD1_SUB | TT_IADD_MOD1_NOCC);
        emit_fmtB(TT_SFPAND, t, aa, 0, 0);
        emit_fmtB(TT_SFPIADD, dst, t, 0, TT_IADD_MOD1_NOCC);
        if (i < 31) {
            emit_fmtB(TT_SFPSHFT, aa, aa, 1, TT_SHFT_MOD1_IMM);
            emit_fmtB(TT_SFPSHFT, bb, bb, -1, TT_SHFT_MOD1_IMM);
        }
    }
}

static void isel_mul(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    int32_t k;

    if (cimm(I->operands[1], &k)) {
        uint32_t mag = k < 0 ? 0u - (uint32_t)k : (uint32_t)k;
        mulk(dst, resolve_val(I->operands[0]), mag, k < 0);
        return;
    }
    if (cimm(I->operands[0], &k)) {
        uint32_t mag = k < 0 ? 0u - (uint32_t)k : (uint32_t)k;
        mulk(dst, resolve_val(I->operands[1]), mag, k < 0);
        return;
    }

    mulrt(dst, resolve_val(I->operands[0]), resolve_val(I->operands[1]));
}

/* ---- Integer Arithmetic ---- */

/* SFPIADD is the only reg+reg integer op. Two instructions per op. */
static void isel_int_arith(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t lhs = resolve_val(I->operands[0]);
    tt_operand_t rhs = resolve_val(I->operands[1]);

    switch (I->op) {
    case BIR_ADD:
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPIADD, dst, rhs, 0, TT_IADD_MOD1_NOCC);
        break;
    case BIR_SUB:
        emit_fmtB(TT_SFPMOV, dst, rhs, 0, 0);
        emit_fmtB(TT_SFPIADD, dst, lhs, 0,
                  TT_IADD_MOD1_SUB | TT_IADD_MOD1_NOCC);
        break;
    case BIR_AND:
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPAND, dst, rhs, 0, 0);
        break;
    case BIR_OR:
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPOR, dst, rhs, 0, 0);
        break;
    case BIR_XOR:
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPXOR, dst, rhs, 0, 0);
        break;
    case BIR_SHL:
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPSHFT, dst, rhs, 0, TT_SHFT_MOD1_REG);
        break;
    case BIR_LSHR: {
        tt_operand_t neg = mop_vreg(new_vreg());
        emit_fmtB(TT_SFPMOV, neg, rhs, 0, 0);
        emit_fmtB(TT_SFPIADD, neg, mop_lreg(TT_LREG_ZERO), 0,
                  TT_IADD_MOD1_SUB | TT_IADD_MOD1_NOCC);
        emit_fmtB(TT_SFPMOV, dst, lhs, 0, 0);
        emit_fmtB(TT_SFPSHFT, dst, neg, 0, TT_SHFT_MOD1_REG);
        break;
    }
    case BIR_ASHR:
        tterr(BC_E527);
        S.had_error = 1;
        break;
    default:
        break;
    }
}

/* ---- Comparison ---- */

static int uicmp(uint32_t sub)
{
    return sub == BIR_ICMP_ULT || sub == BIR_ICMP_UGE ||
           sub == BIR_ICMP_UGT || sub == BIR_ICMP_ULE;
}

static tt_operand_t cmpv(tt_operand_t x, tt_operand_t y, int uns)
{
    tt_operand_t d = mop_vreg(new_vreg());
    tt_operand_t s = mop_vreg(new_vreg());
    tt_operand_t n = mop_vreg(new_vreg());
    tt_operand_t m = mop_vreg(new_vreg());

    emit_fmtB(TT_SFPMOV, d, y, 0, 0);
    emit_fmtB(TT_SFPIADD, d, x, 0, TT_IADD_MOD1_SUB | TT_IADD_MOD1_NOCC);

    emit_fmtB(TT_SFPMOV, s, x, 0, 0);
    emit_fmtB(TT_SFPXOR, s, y, 0, 0);

    emit_fmtB(TT_SFPNOT, n, s, 0, 0);
    emit_fmtB(TT_SFPAND, n, d, 0, 0);

    if (uns) {
        emit_fmtB(TT_SFPNOT, m, x, 0, 0);
        emit_fmtB(TT_SFPAND, m, y, 0, 0);
    } else {
        emit_fmtB(TT_SFPNOT, m, y, 0, 0);
        emit_fmtB(TT_SFPAND, m, x, 0, 0);
    }
    emit_fmtB(TT_SFPOR, m, n, 0, 0);

    return m;
}

static tt_operand_t xorv(tt_operand_t x, tt_operand_t y)
{
    tt_operand_t t = mop_vreg(new_vreg());
    emit_fmtB(TT_SFPMOV, t, x, 0, 0);
    emit_fmtB(TT_SFPXOR, t, y, 0, 0);
    return t;
}

static void isel_icmp(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t lhs = resolve_val(I->operands[0]);
    tt_operand_t rhs = resolve_val(I->operands[1]);
    int uns = uicmp(I->subop);
    tt_operand_t t;
    uint16_t cc;

    switch (I->subop) {
    case BIR_ICMP_EQ:
        t = xorv(lhs, rhs); cc = TT_CC_EQ; break;
    case BIR_ICMP_SLT: case BIR_ICMP_ULT:
        t = cmpv(lhs, rhs, uns); cc = TT_CC_LT; break;
    case BIR_ICMP_SGE: case BIR_ICMP_UGE:
        t = cmpv(lhs, rhs, uns); cc = TT_CC_GE; break;
    case BIR_ICMP_SGT: case BIR_ICMP_UGT:
        t = cmpv(rhs, lhs, uns); cc = TT_CC_LT; break;
    case BIR_ICMP_SLE: case BIR_ICMP_ULE:
        t = cmpv(rhs, lhs, uns); cc = TT_CC_GE; break;
    default:
        t = xorv(lhs, rhs); cc = TT_CC_NE; break;
    }

    emit_fmtB(TT_SFPSETCC, dst, t, 0, cc);
}

static void isel_fcmp(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t lhs = resolve_val(I->operands[0]);
    tt_operand_t rhs = resolve_val(I->operands[1]);

    uint32_t tmp_vr = new_vreg();
    tt_operand_t tmp = mop_vreg(tmp_vr);
    emit_fmtA(TT_SFPADD, tmp, lhs, mop_lreg(TT_LREG_ONE), rhs, 1);  /* tmp = lhs - rhs */

    uint16_t cc;
    switch (I->subop) {
    case BIR_FCMP_OEQ: case BIR_FCMP_UEQ:      cc = TT_CC_EQ; break;
    case BIR_FCMP_ONE: case BIR_FCMP_UNE:      cc = TT_CC_NE; break;
    case BIR_FCMP_OLT: case BIR_FCMP_ULT:      cc = TT_CC_LT; break;
    case BIR_FCMP_OGE: case BIR_FCMP_UGE:      cc = TT_CC_GE; break;
    case BIR_FCMP_OGT: case BIR_FCMP_UGT:
        /* GT(a,b) = LT(b,a) */
        emit_fmtA(TT_SFPADD, tmp, rhs, mop_lreg(TT_LREG_ONE), lhs, 1);
        cc = TT_CC_LT;
        break;
    case BIR_FCMP_OLE: case BIR_FCMP_ULE:
        emit_fmtA(TT_SFPADD, tmp, rhs, mop_lreg(TT_LREG_ONE), lhs, 1);
        cc = TT_CC_GE;
        break;
    default: cc = TT_CC_NE; break;
    }

    emit_fmtB(TT_SFPSETCC, dst, tmp, 0, cc);
}

/* ---- Control Flow -> Predication ---- */

static void isel_br_cond(const bir_inst_t *I)
{
    tt_operand_t cond = resolve_val(I->operands[0]);
    uint32_t true_bir  = I->operands[1];
    uint32_t false_bir = I->operands[2];

    emit_fmtC(TT_SFPPUSHC, 0, 0);
    emit_fmtB(TT_SFPSETCC, mop_vreg(new_vreg()), cond, 0, TT_CC_NE);
    emit_fmtC(TT_SFPENCC, 0, TT_ENCC_INIT);

    if (S.pred_depth < TT_MAX_PRED_DEPTH) {
        uint32_t merge = find_merge_block(true_bir, false_bir);
        int has_else = (merge != false_bir);
        S.pred_stack[S.pred_depth].false_bir = false_bir;
        S.pred_stack[S.pred_depth].merge_bir = merge;
        S.pred_stack[S.pred_depth].has_else = has_else;
        S.pred_depth++;
    }
}

static void isel_br(const bir_inst_t *I)
{
    (void)I; /* unconditional branches are implicit in predicated execution */
}

/* ---- Data Movement ---- */

static void isel_alloca(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    uint16_t row = (uint16_t)S.dst_offset;
    if (S.dst_offset < TT_DST_ROWS) S.dst_offset++;
    emit_fmtE(TT_SFPLOADI, mop_vreg(vr), TT_LOADI_MOD0_INTA, row);
    (void)I;
}

static uint32_t ptras(uint32_t ptr)
{
    if (ptr == BIR_VAL_NONE || BIR_VAL_IS_CONST(ptr))
        return BIR_AS_GLOBAL;

    uint32_t si = BIR_VAL_INDEX(ptr);
    if (si >= S.bir->num_insts) return BIR_AS_GLOBAL;

    uint32_t pt = S.bir->insts[si].type;
    if (pt < S.bir->num_types && S.bir->types[pt].kind == BIR_TYPE_PTR)
        return S.bir->types[pt].addrspace;

    return BIR_AS_GLOBAL;
}

static int tilas(uint32_t as, const char *what)
{
    if (as == BIR_AS_GLOBAL || as == BIR_AS_GENERIC ||
        as == BIR_AS_CONSTANT)
        return 1;
    tterr(BC_E526, what, (unsigned)as);
    S.had_error = 1;
    return 0;
}

static void isel_load(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);

    if (!tilas(ptras(I->operands[0]), "load")) return;

    emit_fmtD(TT_SFPLOAD, dst, TT_LDST_MOD0_FP32, 0, 0);
}

static void isel_store(const bir_inst_t *I)
{
    tt_operand_t val = resolve_val(I->operands[0]);

    if (!tilas(ptras(I->operands[1]), "store")) return;

    emit_fmtD(TT_SFPSTORE, val, TT_LDST_MOD0_FP32, 0, 0);
}

static void isel_gep(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t base = resolve_val(I->operands[0]);
    uint32_t foff = 0;
    tt_operand_t offset = bir_fgep(S.bir, I, 4, &foff)
        ? mop_imm((int32_t)foff)
        : (get_num_ops(I) > 1)
        ? resolve_val(I->operands[1])
        : mop_lreg(TT_LREG_ZERO);

    emit_fmtB(TT_SFPMOV, dst, base, 0, 0);
    emit_fmtB(TT_SFPIADD, dst, offset, 0, TT_IADD_MOD1_NOCC);
}

/* ---- Conversion ---- */

static void isel_conversion(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t src = resolve_val(I->operands[0]);

    switch (I->op) {
    case BIR_FPTOSI:
    case BIR_FPTOUI:
        emit_fmtB(TT_SFPCAST, dst, src, 0, 1);
        break;
    case BIR_SITOFP:
    case BIR_UITOFP:
        emit_fmtB(TT_SFPCAST, dst, src, 0, 0);
        break;
    case BIR_FPTRUNC:
        emit_fmtB(TT_SFPCAST, dst, src, 0, 2);
        break;
    case BIR_FPEXT:
        emit_fmtB(TT_SFPCAST, dst, src, 0, 3);
        break;
    case BIR_TRUNC:
    case BIR_BITCAST:
    case BIR_ZEXT:
    case BIR_SEXT:
    case BIR_PTRTOINT:
    case BIR_INTTOPTR:
        emit_fmtB(TT_SFPMOV, dst, src, 0, 0);
        break;
    default:
        emit_fmtB(TT_SFPMOV, dst, src, 0, 0);
        break;
    }
}

/* ---- Thread Model ---- */

static void isel_thread_model(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);

    switch (I->op) {
    case BIR_THREAD_ID:
        if (I->subop == 0) {
            /* L15 = lane_id (0,2,4,...,62), shift right 1 for threadIdx.x */
            emit_fmtB(TT_SFPMOV, dst, mop_lreg(TT_LREG_LANE_ID), 0, 0);
            emit_fmtB(TT_SFPSHFT, dst, dst, -1, TT_SHFT_MOD1_IMM);
        } else {
            emit_fmtE(TT_SFPLOADI, dst, TT_LOADI_MOD0_INTA, 0);
        }
        break;

    case BIR_BLOCK_ID:
        emit_fmtD(TT_SFPLOAD, dst, TT_LDST_MOD0_FP32, 0, 0);
        break;

    case BIR_BLOCK_DIM:
        emit_fmtE(TT_SFPLOADI, dst, TT_LOADI_MOD0_INTA, TT_SFPU_LANES);
        break;

    case BIR_GRID_DIM:
        emit_fmtD(TT_SFPLOAD, dst, TT_LDST_MOD0_FP32, 0, 1);
        break;

    default:
        break;
    }
}

/* ---- Parameters ---- */

static void isel_param(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    uint32_t param_idx = I->subop;
    uint16_t row = (uint16_t)(param_idx + 2);
    emit_fmtD(TT_SFPLOAD, dst, TT_LDST_MOD0_FP32, 0, row);
}

/* ---- PHI Nodes ---- */

static void isel_phi(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t ops[2] = { mop_vreg(vr), mop_imm(0) };
    emit(TT_PSEUDO_PHI, TT_FMT_PSEUDO, 1, 1, ops, 0);
    (void)I;
}

/* ---- Select ---- */

static void isel_select(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    tt_operand_t dst = mop_vreg(vr);
    tt_operand_t cond     = resolve_val(I->operands[0]);
    tt_operand_t true_val = resolve_val(I->operands[1]);
    tt_operand_t false_val = resolve_val(I->operands[2]);

    emit_fmtB(TT_SFPMOV, dst, false_val, 0, 0);
    emit_fmtC(TT_SFPPUSHC, 0, 0);
    emit_fmtB(TT_SFPSETCC, mop_vreg(new_vreg()), cond, 0, TT_CC_NE);
    emit_fmtC(TT_SFPENCC, 0, TT_ENCC_INIT);
    emit_fmtB(TT_SFPMOV, dst, true_val, 0, 0);
    emit_fmtC(TT_SFPPOPC, 0, 0);
}

/* ---- Barrier ---- */

static void isel_barrier(void)
{
    emit_fmtC(TT_SFPNOP, 0, 0); /* single core: nop */
}

/* per TT-ISA BH/TensixCoprocessor/VectorUnit.md the SFPU computes on LReg */
/* per TT-ISA BH/TensixCoprocessor/SFPLOAD.md a load moves Dst to LReg */
/* per TT-ISA BH/TensixCoprocessor/SFPSTORE.md a store moves LReg to Dst */
/* per TT-ISA BH/TensixCoprocessor/STALLWAIT.md C0 drains ThCon requests for the issuing thread only */
static const char *fscop(uint8_t sc)
{
    if (sc == 0u) return "__threadfence_block";
    if (sc == 2u) return "__threadfence_system";
    return "__threadfence";
}

static void isel_fence(const bir_inst_t *I)
{
    tterr(BC_E866, fscop(I->subop));
    S.had_error = 1;
}

/* ---- Call ---- */

static void isel_call(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx);
    (void)I;
    emit_fmtB(TT_SFPMOV, mop_vreg(vr), mop_lreg(TT_LREG_ZERO), 0, 0);
}

/* ---- Return ---- */

static void isel_ret(const bir_inst_t *I)
{
    if (I->num_operands > 0 && I->operands[0] != BIR_VAL_NONE) {
        tt_operand_t val = resolve_val(I->operands[0]);
        emit_fmtD(TT_SFPSTORE, val, TT_LDST_MOD0_FP32, 0, 0);
    }
}

/* ---- Per-Block Instruction Selection ---- */

static void isel_block(uint32_t bir_bi)
{
    const bir_block_t *B = &S.bir->blocks[bir_bi];

    for (uint32_t di = 0; di < S.pred_depth; di++) {
        if (S.pred_stack[di].has_else && S.pred_stack[di].false_bir == bir_bi) {
            emit_fmtC(TT_SFPCOMPC, 0, 0);  /* else: flip mask */
        }
    }
    int popped = 0;
    for (uint32_t di = S.pred_depth; di > 0 && !popped; di--) {
        if (S.pred_stack[di - 1].merge_bir == bir_bi) {
            emit_fmtC(TT_SFPPOPC, 0, 0);  /* merge: restore mask */
            S.pred_depth--;
            popped = 1;
        }
    }
    int guard = 65536;
    for (uint32_t ii = 0; ii < B->num_insts && guard > 0; ii++, guard--) {
        uint32_t idx = B->first_inst + ii;
        const bir_inst_t *I = &S.bir->insts[idx];

        switch (I->op) {
        case BIR_FADD: case BIR_FSUB: case BIR_FMUL:
            isel_fp_arith(idx, I);
            break;

        case BIR_FDIV:
            tterr(BC_E522, "floating-point division");
            S.had_error = 1;
            break;
        case BIR_FREM:
            tterr(BC_E522, "floating-point remainder");
            S.had_error = 1;
            break;
        case BIR_ADD: case BIR_SUB:
        case BIR_AND: case BIR_OR: case BIR_XOR:
        case BIR_SHL: case BIR_LSHR: case BIR_ASHR:
            isel_int_arith(idx, I);
            break;

        case BIR_MUL:
            isel_mul(idx, I);
            break;

        case BIR_SDIV: case BIR_UDIV:
            tterr(BC_E523, "integer division");
            S.had_error = 1;
            break;
        case BIR_SREM: case BIR_UREM:
            tterr(BC_E523, "integer remainder");
            S.had_error = 1;
            break;
        case BIR_ICMP:
            isel_icmp(idx, I);
            break;
        case BIR_FCMP:
            isel_fcmp(idx, I);
            break;
        case BIR_TRUNC: case BIR_ZEXT: case BIR_SEXT:
        case BIR_FPTRUNC: case BIR_FPEXT:
        case BIR_FPTOSI: case BIR_FPTOUI:
        case BIR_SITOFP: case BIR_UITOFP:
        case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST:
            isel_conversion(idx, I);
            break;
        case BIR_LOAD:
            isel_load(idx, I);
            break;
        case BIR_STORE:
            isel_store(I);
            break;
        case BIR_GEP:
            isel_gep(idx, I);
            break;
        case BIR_ALLOCA:
            isel_alloca(idx, I);
            break;
        case BIR_SHARED_ALLOC:
            tterr(BC_E524);
            S.had_error = 1;
            break;
        case BIR_BR:
            isel_br(I);
            break;
        case BIR_BR_COND:
            isel_br_cond(I);
            break;
        case BIR_SWITCH:
            tterr(BC_E525);
            S.had_error = 1;
            break;
        case BIR_RET:
            isel_ret(I);
            break;
        case BIR_UNREACHABLE:
            break;
        case BIR_PHI:
            isel_phi(idx, I);
            break;
        case BIR_PARAM:
            isel_param(idx, I);
            break;
        case BIR_THREAD_ID:
        case BIR_BLOCK_ID:
        case BIR_BLOCK_DIM:
        case BIR_GRID_DIM:
            isel_thread_model(idx, I);
            break;
        case BIR_BARRIER:
        case BIR_BARRIER_GROUP:
            isel_barrier();
            break;
        case BIR_FENCE:
            isel_fence(I);
            break;

        case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
        case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
        case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
        case BIR_ATOMIC_XCHG: case BIR_ATOMIC_CAS:
        case BIR_ATOMIC_LOAD: case BIR_ATOMIC_STORE:
            tterr(BC_E520, "an atomic read-modify-write");
            S.had_error = 1;
            break;

        case BIR_SHFL: case BIR_SHFL_UP:
        case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            tterr(BC_E521, "a warp shuffle");
            S.had_error = 1;
            break;

        case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
            tterr(BC_E521, "a warp vote");
            S.had_error = 1;
            break;

        case BIR_POPCOUNT: case BIR_CTZ:
        case BIR_CLZ: case BIR_BREV:
            tterr(BC_E528);
            S.had_error = 1;
            break;

        case BIR_TRAP:
            tterr(BC_E529);
            S.had_error = 1;
            break;

        case BIR_FNREF:
            tterr(BC_E530);
            S.had_error = 1;
            break;

        case BIR_PRINTF:
            tterr(BC_E531);
            S.had_error = 1;
            break;

        case BIR_MMA: case BIR_MFRG:
            tterr(BC_E532);
            S.had_error = 1;
            break;
        case BIR_SELECT:
            isel_select(idx, I);
            break;
        case BIR_CALL:
            isel_call(idx, I);
            break;
        case BIR_INLINE_ASM:
            tterr(BC_E533);
            S.had_error = 1;
            break;

        default:
            tterr(BC_E867, bir_op_name((int)I->op));
            S.had_error = 1;
            break;
        }
    }
}

/* ---- Per-Function Setup ---- */

static int isel_function(uint32_t func_idx)
{
    const bir_func_t *F = &S.bir->funcs[func_idx];
    if (!(F->cuda_flags & (CUDA_GLOBAL | CUDA_DEVICE))) return BC_OK;

    if (S.tt->num_mfuncs >= TT_MAX_MFUNCS) return BC_ERR_TENSIX;

    uint32_t mf_idx = S.tt->num_mfuncs++;
    tt_mfunc_t *MF = &S.tt->mfuncs[mf_idx];

    MF->name = F->name;
    MF->first_block = S.tt->num_mblocks;
    MF->num_blocks = 0;
    MF->num_lregs_used = 0;
    MF->dst_rows_used = 0;
    MF->is_kernel = (F->cuda_flags & CUDA_GLOBAL) ? 1 : 0;
    MF->launch_bounds_max = F->launch_bounds_max;
    MF->launch_bounds_min = F->launch_bounds_min;
    MF->tiles_per_core = 0;
    MF->bir_func = func_idx;
    MF->coarsen_pattern = TT_PATTERN_GENERIC;

    S.dst_offset = 16;  /* reserve rows 0-15 for runtime args */
    S.pred_depth = 0;

    /* Pre-create block map */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        S.block_map[bir_bi] = S.tt->num_mblocks + bi;
    }

    /* Select instructions block by block */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        if (S.tt->num_mblocks >= TT_MAX_MBLOCKS) break;

        uint32_t mb_idx = S.tt->num_mblocks;
        tt_mblock_t *MB = &S.tt->mblocks[mb_idx];
        MB->first_inst = S.tt->num_minsts;
        MB->bir_block = bir_bi;

        isel_block(bir_bi);

        MB->num_insts = S.tt->num_minsts - MB->first_inst;
        S.tt->num_mblocks++;
    }

    MF->num_blocks = (uint16_t)(S.tt->num_mblocks - MF->first_block);
    MF->dst_rows_used = (uint16_t)S.dst_offset;
    return BC_OK;
}

/* ---- Public API ---- */

int tensix_compile(const bir_module_t *bir, tt_module_t *tt)
{
    memset(&S, 0, sizeof(S));
    S.tt = tt;
    S.bir = bir;

    tt->bir = bir;
    tt->num_minsts = 0;
    tt->num_mblocks = 0;
    tt->num_mfuncs = 0;
    tt->vreg_count = 1;  /* vreg 0 reserved as sentinel */
    tt->out_len = 0;

    memset(tt->val_vreg, 0, sizeof(tt->val_vreg));
    memset(tt->reg_map, 0, sizeof(tt->reg_map));

    int guard = 8192;
    for (uint32_t fi = 0; fi < bir->num_funcs && guard > 0; fi++, guard--) {
        int rc = isel_function(fi);
        if (rc != BC_OK) return rc;
    }
    if (S.had_error) return BC_ERR_TENSIX;

    return BC_OK;
}
