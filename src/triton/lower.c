/* Triton frontend BIR lowering.
 *
 * Walks the sema-annotated AST and produces BIR, the same target the
 * CUDA frontend lowers into. Once a Triton kernel reaches BIR, every
 * backend (AMD, NVIDIA, Tensix, Metal, Intel, x86 CPU) consumes it.
 *
 * Handled:
 *   - Module skeleton: each @triton.jit FuncDef becomes a __global__
 *     function with one BIR_PARAM per parameter and a BIR_RET.
 *   - Statements: Assign, ExprStmt, Return, AugAssign, and
 *     `for k in range(...)` as a counted loop.
 *   - Scalar expressions: int/float literals, params and locals, BinOp
 *     arithmetic, unary negation, comparisons, and the scalar
 *     intrinsics (program_id, num_programs, arange, cdiv).
 *   - Tiles: 2-D shapes from [:,None]/[None,:] broadcasts, tl.load,
 *     tl.store, tl.zeros, and tl.dot. A kernel with rank-2 tiles
 *     materialises and unrolls them; a K-loop sweeps the contraction.
 *
 * Not yet: multi-block grids (one block per call), tl.load mask=, while
 * loops, if/else, rank-2 reductions.
 *
 * Type policy is crude: a param whose name ends in _ptr is f32*, a
 * tl.constexpr param is i32, everything else is i32, float literals are
 * f32. Use-site type inference comes later. */

#include "triton.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- Helpers: diagnostic ---- */

static void l_err(tn_lower_t *L, uint16_t eid,
                  const tn_tok_t *t, const char *msg)
{
    if (L->num_errors >= BC_MAX_ERRORS) return;
    bc_error_t *e = &L->errors[L->num_errors++];
    e->eid = eid;
    e->loc.line = t ? t->line : 0;
    e->loc.col  = t ? t->col  : 0;
    snprintf(e->msg, sizeof(e->msg), "%s", msg);
}

static const tn_tok_t *l_tok(tn_lower_t *L, uint32_t node_idx)
{
    const tn_parse_t *P = L->parser;
    if (node_idx >= P->num_nodes) return NULL;
    uint32_t t = P->nodes[node_idx].tok_off;
    if (t >= P->lex->num_tokens) return NULL;
    return &P->lex->tokens[t];
}

/* ---- Helpers: BIR construction ---- */

static uint32_t l_kid(const tn_lower_t *L, uint32_t node_idx, uint32_t i)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    if (n->num_kids == TN_NODE_KIDS_OVERFLOW) {
        return L->parser->extra_kids[n->kids[0] + i];
    }
    return n->kids[i];
}

static uint32_t l_nkids(const tn_node_t *n)
{
    return (n->num_kids == TN_NODE_KIDS_OVERFLOW)
           ? n->kids[1]
           : n->num_kids;
}

/* Emit a BIR instruction with the given opcode, subop, and result
 * type. Operands are appended via l_op afterwards. Returns the BIR
 * value reference for the instruction's result, or BIR_VAL_NONE on
 * overflow. The block's instruction count is bumped so block scanners
 * pick up the new instruction immediately. */

static uint32_t l_emit(tn_lower_t *L, int op, uint32_t type, int subop)
{
    bir_module_t *M = L->bir;
    if (M->num_insts >= BIR_MAX_INSTS) return BIR_VAL_NONE;
    uint32_t idx = M->num_insts++;
    bir_inst_t *I = &M->insts[idx];
    I->op = (uint16_t)op;
    I->num_operands = 0;
    I->subop = (uint8_t)subop;
    I->type  = type;
    for (int k = 0; k < BIR_OPERANDS_INLINE; k++) I->operands[k] = BIR_VAL_NONE;
    /* Maintain the host block's instruction count so subsequent
     * scanners count the new instruction. */
    M->blocks[L->cur_block].num_insts++;
    return BIR_MAKE_VAL(idx);
}

/* Append an operand to the last-emitted instruction. Inline up to
 * BIR_OPERANDS_INLINE slots; overflow is not handled because operand
 * counts here stay well below six. */

static void l_op(tn_lower_t *L, uint32_t inst_val, uint32_t operand)
{
    if (inst_val == BIR_VAL_NONE) return;
    if (BIR_VAL_IS_CONST(inst_val)) return;
    uint32_t idx = BIR_VAL_INDEX(inst_val);
    if (idx >= L->bir->num_insts) return;
    bir_inst_t *I = &L->bir->insts[idx];
    if (I->num_operands < BIR_OPERANDS_INLINE) {
        I->operands[I->num_operands++] = operand;
    }
}

/* Create a new BIR block as a child of the current function. */

static uint32_t l_new_block(tn_lower_t *L)
{
    bir_module_t *M = L->bir;
    if (M->num_blocks >= BIR_MAX_BLOCKS) return 0;
    uint32_t idx = M->num_blocks++;
    M->blocks[idx].name = 0;
    M->blocks[idx].first_inst = M->num_insts;
    M->blocks[idx].num_insts = 0;
    /* Attach to the current function. */
    bir_func_t *F = &M->funcs[L->cur_func];
    if (F->num_blocks == 0) F->first_block = idx;
    F->num_blocks++;
    return idx;
}

/* ---- Type bootstrap ---- */

static void l_types_init(tn_lower_t *L)
{
    bir_module_t *M = L->bir;
    L->t_void    = bir_type_void(M);
    L->t_i32     = bir_type_int(M, 32);
    L->t_f32     = bir_type_float(M, 32);
    L->t_ptr_i32 = bir_type_ptr(M, L->t_i32, BIR_AS_GLOBAL);
    L->t_ptr_f32 = bir_type_ptr(M, L->t_f32, BIR_AS_GLOBAL);
}

/* ---- Literal parsing ----
 * The token text for a literal is whatever the source spelled. Accepts
 * decimal integers with optional underscores, hex/octal/binary integers
 * with the corresponding prefix, and floats with optional exponent.
 * Anything fancier (imaginary suffix, literals that overflow int64)
 * falls back to zero. */

static int64_t l_parse_int(const char *s, uint32_t len)
{
    int64_t v = 0;
    if (len == 0) return 0;

    uint32_t p = 0;
    int base = 10;

    if (len > 2 && s[0] == '0') {
        if (s[1] == 'x' || s[1] == 'X') { base = 16; p = 2; }
        else if (s[1] == 'o' || s[1] == 'O') { base = 8;  p = 2; }
        else if (s[1] == 'b' || s[1] == 'B') { base = 2;  p = 2; }
    }
    for (; p < len; p++) {
        char c = s[p];
        if (c == '_') continue;
        int d = -1;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        if (d < 0 || d >= base) break;
        v = v * base + d;
    }
    return v;
}

static double l_parse_float(const char *s, uint32_t len)
{
    /* Small stack buffer to feed strtod a null-terminated string
     * without malloc. Anything longer than the buffer gets zero. */
    char buf[64];
    uint32_t n = len < 63 ? len : 63;
    memcpy(buf, s, n);
    buf[n] = '\0';
    /* Strip underscores in place; strtod does not know about them. */
    uint32_t w = 0;
    for (uint32_t r = 0; r < n; r++) {
        if (buf[r] != '_') buf[w++] = buf[r];
    }
    buf[w] = '\0';
    return strtod(buf, NULL);
}

/* ---- Expression lowering ----
 * Each expression node either yields a BIR value or, for constructs not
 * yet handled, emits a diagnostic and returns BIR_VAL_NONE. The walker
 * recurses over child nodes; recursion stays shallow because Triton
 * kernels are shallow trees. */

static uint32_t l_expr(tn_lower_t *L, uint32_t node_idx);

/* Refuse to lower expressions whose inferred shape is rank-2 or higher.
 * The shape annotation comes from sema; matrix-instruction codegen
 * (MFMA on AMD, mma.sync on NVIDIA) does not yet exist, so lowering a
 * rank-2 tile would produce silent wrong code. */

static int l_shape_supported(tn_lower_t *L, uint32_t node_idx)
{
    if (node_idx >= TN_MAX_NODES) return 1;
    tn_shape_t sh = L->sema->node_shape[node_idx];
    if (sh.rank < 2) return 1;
    char sbuf[64];
    tn_shape_format(sh, sbuf, sizeof(sbuf));
    char msg[256];
    snprintf(msg, sizeof(msg),
             "rank-2 tile (%s) needs matrix codegen "
             "(MFMA / mma.sync), not yet lowered", sbuf);
    l_err(L, 99, l_tok(L, node_idx), msg);
    return 0;
}

static uint32_t l_lit(tn_lower_t *L, uint32_t node_idx)
{
    const tn_parse_t *P = L->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    if (n->tok_off >= P->lex->num_tokens) return BIR_VAL_NONE;
    const tn_tok_t *t = &P->lex->tokens[n->tok_off];
    const char *s = P->lex->src + t->off;
    switch (n->flags) {
    case TN_LIT_INT: {
        int64_t v = l_parse_int(s, t->len);
        return BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, v));
    }
    case TN_LIT_FLOAT: {
        double v = l_parse_float(s, t->len);
        return BIR_MAKE_CONST(bir_const_float(L->bir, L->t_f32, v));
    }
    case TN_LIT_TRUE:
        return BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, 1));
    case TN_LIT_FALSE:
    case TN_LIT_NONE:
        return BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, 0));
    case TN_LIT_STRING:
        l_err(L, 90, t, "string literals are not lowerable in sitting one");
        return BIR_VAL_NONE;
    default:
        return BIR_VAL_NONE;
    }
}

static uint32_t l_name(tn_lower_t *L, uint32_t node_idx)
{
    int kind = L->sema->node_sym_kind[node_idx];
    uint32_t aux = L->sema->node_sym_aux[node_idx];

    switch (kind) {
    case TN_SYM_PARAM: {
        /* The parameter's BIR value was emitted when the function
         * header was lowered. Look it up by its position relative to
         * cur_param_base. Constexpr params with a default folded into a
         * literal during the header (no BIR_PARAM emitted), so the
         * constant is handed back directly. */
        if (aux < 32 && L->param_remap[aux] == 0xFF) {
            return BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32,
                                                L->param_const[aux]));
        }
        uint32_t bir_idx = (aux < 32) ? L->param_remap[aux] : (uint8_t)aux;
        return BIR_MAKE_VAL(L->cur_param_base + bir_idx);
    }
    case TN_SYM_LOCAL:
    case TN_SYM_LOOPVAR: {
        /* aux holds the declaring node (Assign or For). The value
         * the RHS produced was stashed in node_val under that
         * declaring node's index. */
        if (aux < TN_MAX_NODES) {
            uint32_t v = L->node_val[aux];
            if (v != BIR_VAL_NONE) return v;
        }
        l_err(L, 91, l_tok(L, node_idx),
              "local referenced before it produced a BIR value");
        return BIR_VAL_NONE;
    }
    case TN_SYM_MODULE:
    case TN_SYM_INTRINSIC:
    case TN_SYM_TYPE:
        /* Bare module / intrinsic / type names should not appear as
         * value expressions on their own; usually means the kernel
         * uses something not yet lowered. Report and continue. */
        l_err(L, 92, l_tok(L, node_idx),
              "bare module or intrinsic name in expression position");
        return BIR_VAL_NONE;
    case TN_SYM_UNBOUND:
    default:
        return BIR_VAL_NONE;
    }
}

/* Look up the BIR type of a value reference. Constants read the type
 * from the consts pool; instructions read it from the result type slot.
 * Lets BinOp dispatch between integer and floating-point operations and
 * lets pointer-plus-integer lower as a GEP. */

static uint32_t l_val_type(const tn_lower_t *L, uint32_t v)
{
    if (v == BIR_VAL_NONE) return L->t_i32;
    if (BIR_VAL_IS_CONST(v)) {
        uint32_t ci = BIR_VAL_INDEX(v);
        if (ci >= L->bir->num_consts) return L->t_i32;
        return L->bir->consts[ci].type;
    }
    uint32_t ii = BIR_VAL_INDEX(v);
    if (ii >= L->bir->num_insts) return L->t_i32;
    return L->bir->insts[ii].type;
}

static int l_type_kind(const tn_lower_t *L, uint32_t type_idx)
{
    if (type_idx >= L->bir->num_types) return -1;
    return L->bir->types[type_idx].kind;
}

/* Map a Triton BinOp subcode onto the matching BIR opcode, choosing
 * integer or floating-point flavour based on the operand types. The
 * pointer-plus-int case is handled separately because it lowers to
 * BIR_GEP rather than to BIR_ADD. */

static int l_bop_int(int bop)
{
    switch (bop) {
    case TN_BOP_ADD:    return BIR_ADD;
    case TN_BOP_SUB:    return BIR_SUB;
    case TN_BOP_MUL:    return BIR_MUL;
    case TN_BOP_DIV:    return BIR_SDIV;
    case TN_BOP_FDIV:   return BIR_SDIV;
    case TN_BOP_MOD:    return BIR_SREM;
    case TN_BOP_AND:    return BIR_AND;
    case TN_BOP_OR:     return BIR_OR;
    case TN_BOP_XOR:    return BIR_XOR;
    case TN_BOP_SHL:    return BIR_SHL;
    case TN_BOP_SHR:    return BIR_ASHR;
    default:            return -1;
    }
}

static int l_bop_float(int bop)
{
    switch (bop) {
    case TN_BOP_ADD:    return BIR_FADD;
    case TN_BOP_SUB:    return BIR_FSUB;
    case TN_BOP_MUL:    return BIR_FMUL;
    case TN_BOP_DIV:    return BIR_FDIV;
    case TN_BOP_FDIV:   return BIR_FDIV;
    case TN_BOP_MOD:    return BIR_FREM;
    default:            return -1;
    }
}

/* Value-level binop: dispatch ptr+int -> GEP, float -> FADD..., int ->
 * ADD... Shared by the scalar path (l_binop) and the per-element tile
 * path so both encode arithmetic identically. */
static uint32_t l_emit_binop(tn_lower_t *L, int bop, uint32_t lhs, uint32_t rhs)
{
    if (lhs == BIR_VAL_NONE || rhs == BIR_VAL_NONE) return BIR_VAL_NONE;
    uint32_t lt = l_val_type(L, lhs), rt = l_val_type(L, rhs);
    int lk = l_type_kind(L, lt), rk = l_type_kind(L, rt);

    if (bop == TN_BOP_ADD && (lk == BIR_TYPE_PTR || rk == BIR_TYPE_PTR)) {
        uint32_t base = (lk == BIR_TYPE_PTR) ? lhs : rhs;
        uint32_t idx  = (lk == BIR_TYPE_PTR) ? rhs : lhs;
        uint32_t bt   = (lk == BIR_TYPE_PTR) ? lt  : rt;
        uint32_t inst = l_emit(L, BIR_GEP, bt, 0);
        l_op(L, inst, base); l_op(L, inst, idx);
        return inst;
    }
    if (lk == BIR_TYPE_FLOAT || rk == BIR_TYPE_FLOAT) {
        int op = l_bop_float(bop);
        if (op < 0) return BIR_VAL_NONE;
        uint32_t inst = l_emit(L, op, L->t_f32, 0);
        l_op(L, inst, lhs); l_op(L, inst, rhs);
        return inst;
    }
    int op = l_bop_int(bop);
    if (op < 0) return BIR_VAL_NONE;
    uint32_t inst = l_emit(L, op, L->t_i32, 0);
    l_op(L, inst, lhs); l_op(L, inst, rhs);
    return inst;
}

static uint32_t l_binop(tn_lower_t *L, uint32_t node_idx)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    uint32_t lhs = l_expr(L, l_kid(L, node_idx, 0));
    uint32_t rhs = l_expr(L, l_kid(L, node_idx, 1));
    uint32_t r = l_emit_binop(L, (int)n->flags, lhs, rhs);
    if (r == BIR_VAL_NONE && lhs != BIR_VAL_NONE && rhs != BIR_VAL_NONE)
        l_err(L, 93, l_tok(L, node_idx), "binop not yet lowered");
    return r;
}

/* Comparison nodes lower to BIR_ICMP or BIR_FCMP with the predicate
 * encoded in the instruction's subop slot. */

static uint32_t l_compare(tn_lower_t *L, uint32_t node_idx)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    if (l_nkids(n) < 2) return BIR_VAL_NONE;
    uint32_t lhs = l_expr(L, l_kid(L, node_idx, 0));
    uint32_t rhs = l_expr(L, l_kid(L, node_idx, 1));
    if (lhs == BIR_VAL_NONE || rhs == BIR_VAL_NONE) return BIR_VAL_NONE;

    int lhs_kind = l_type_kind(L, l_val_type(L, lhs));
    int rhs_kind = l_type_kind(L, l_val_type(L, rhs));
    int is_float = (lhs_kind == BIR_TYPE_FLOAT || rhs_kind == BIR_TYPE_FLOAT);
    int pred;

    if (is_float) {
        switch (n->flags) {
        case TN_CMP_LT: pred = BIR_FCMP_OLT; break;
        case TN_CMP_LE: pred = BIR_FCMP_OLE; break;
        case TN_CMP_GT: pred = BIR_FCMP_OGT; break;
        case TN_CMP_GE: pred = BIR_FCMP_OGE; break;
        case TN_CMP_EQ: pred = BIR_FCMP_OEQ; break;
        case TN_CMP_NE: pred = BIR_FCMP_ONE; break;
        default:
            l_err(L, 96, l_tok(L, node_idx),
                  "comparison predicate not yet lowered for floats");
            return BIR_VAL_NONE;
        }
        uint32_t inst = l_emit(L, BIR_FCMP, L->t_i32, pred);
        l_op(L, inst, lhs);
        l_op(L, inst, rhs);
        return inst;
    }

    switch (n->flags) {
    case TN_CMP_LT: pred = BIR_ICMP_SLT; break;
    case TN_CMP_LE: pred = BIR_ICMP_SLE; break;
    case TN_CMP_GT: pred = BIR_ICMP_SGT; break;
    case TN_CMP_GE: pred = BIR_ICMP_SGE; break;
    case TN_CMP_EQ: pred = BIR_ICMP_EQ;  break;
    case TN_CMP_NE: pred = BIR_ICMP_NE;  break;
    default:
        l_err(L, 96, l_tok(L, node_idx),
              "comparison predicate not yet lowered");
        return BIR_VAL_NONE;
    }
    uint32_t inst = l_emit(L, BIR_ICMP, L->t_i32, pred);
    l_op(L, inst, lhs);
    l_op(L, inst, rhs);
    return inst;
}

static uint32_t l_unop(tn_lower_t *L, uint32_t node_idx)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    uint32_t operand = l_expr(L, l_kid(L, node_idx, 0));
    if (operand == BIR_VAL_NONE) return BIR_VAL_NONE;

    switch (n->flags) {
    case TN_UOP_POS:
        return operand;
    case TN_UOP_NEG: {
        /* Lower -x as 0 - x; BIR does not have a dedicated negate. */
        uint32_t zero = BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, 0));
        uint32_t inst = l_emit(L, BIR_SUB, L->t_i32, 0);
        l_op(L, inst, zero);
        l_op(L, inst, operand);
        return inst;
    }
    case TN_UOP_INV: {
        /* ~x as x XOR -1. */
        uint32_t mone = BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, -1));
        uint32_t inst = l_emit(L, BIR_XOR, L->t_i32, 0);
        l_op(L, inst, operand);
        l_op(L, inst, mone);
        return inst;
    }
    default:
        l_err(L, 94, l_tok(L, node_idx),
              "unary operator not yet lowered");
        return BIR_VAL_NONE;
    }
}

static uint32_t l_pos_arg(tn_lower_t *L, uint32_t call_idx, int want)
{
    const tn_node_t *n = &L->parser->nodes[call_idx];
    uint32_t nk = l_nkids(n);
    int pos = 0;

    for (uint32_t i = 1; i < nk; i++) {
        uint32_t kid = l_kid(L, call_idx, i);
        if (L->parser->nodes[kid].kind == TN_NK_KEYWORD)
            continue;
        if (pos == want)
            return kid;
        pos++;
    }
    return 0;
}

static uint32_t l_math_type(tn_lower_t *L, uint32_t v)
{
    uint32_t ty = l_val_type(L, v);
    return l_type_kind(L, ty) == BIR_TYPE_FLOAT ? ty : L->t_f32;
}

static uint32_t l_math1(tn_lower_t *L, uint32_t call_idx, int op)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    uint32_t r = l_emit(L, op, l_math_type(L, v), 0);
    l_op(L, r, v);
    return r;
}

static uint32_t l_math2(tn_lower_t *L, uint32_t call_idx, int op)
{
    uint32_t a0n = l_pos_arg(L, call_idx, 0);
    uint32_t a1n = l_pos_arg(L, call_idx, 1);
    if (a0n == 0 || a1n == 0) return BIR_VAL_NONE;
    uint32_t a0 = l_expr(L, a0n);
    uint32_t a1 = l_expr(L, a1n);
    if (a0 == BIR_VAL_NONE || a1 == BIR_VAL_NONE) return BIR_VAL_NONE;
    uint32_t r = l_emit(L, op, l_math_type(L, a0), 0);
    l_op(L, r, a0);
    l_op(L, r, a1);
    return r;
}

static uint32_t l_fmul_k(tn_lower_t *L, uint32_t v, double k)
{
    uint32_t c = BIR_MAKE_CONST(bir_const_float(L->bir, L->t_f32, k));
    uint32_t r = l_emit(L, BIR_FMUL, L->t_f32, 0);
    l_op(L, r, v);
    l_op(L, r, c);
    return r;
}

static uint32_t l_exp_nat(tn_lower_t *L, uint32_t call_idx)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    /* log2(e): implement exp(x) through BIR_EXP2. */
    uint32_t x = l_fmul_k(L, v, 1.4426950408889634);
    uint32_t r = l_emit(L, BIR_EXP2, L->t_f32, 0);
    l_op(L, r, x);
    return r;
}

static uint32_t l_log_nat(tn_lower_t *L, uint32_t call_idx)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    uint32_t lg = l_emit(L, BIR_LOG2, L->t_f32, 0);
    l_op(L, lg, v);
    /* ln(2): convert BIR_LOG2 output back to natural log. */
    return l_fmul_k(L, lg, 0.6931471805599453);
}

static uint32_t l_sincos(tn_lower_t *L, uint32_t call_idx, int op)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    /* 1/(2*pi): convert radians to turns for BIR_SIN/BIR_COS. */
    uint32_t t = l_fmul_k(L, v, 0.15915494309189535);
    uint32_t r = l_emit(L, op, L->t_f32, 0);
    l_op(L, r, t);
    return r;
}

static uint32_t l_tan(tn_lower_t *L, uint32_t call_idx)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    /* 1/(2*pi): convert radians to turns for BIR_SIN/BIR_COS. */
    uint32_t t = l_fmul_k(L, v, 0.15915494309189535);
    uint32_t s = l_emit(L, BIR_SIN, L->t_f32, 0);
    l_op(L, s, t);
    uint32_t c = l_emit(L, BIR_COS, L->t_f32, 0);
    l_op(L, c, t);
    uint32_t r = l_emit(L, BIR_FDIV, L->t_f32, 0);
    l_op(L, r, s);
    l_op(L, r, c);
    return r;
}

static uint32_t l_tanh(tn_lower_t *L, uint32_t call_idx)
{
    uint32_t an = l_pos_arg(L, call_idx, 0);
    if (an == 0) return BIR_VAL_NONE;
    uint32_t v = l_expr(L, an);
    if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
    /* 2*log2(e): scale for tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). */
    uint32_t x = l_fmul_k(L, v, 2.8853900817779268);
    uint32_t e2 = l_emit(L, BIR_EXP2, L->t_f32, 0);
    l_op(L, e2, x);
    uint32_t one = BIR_MAKE_CONST(bir_const_float(L->bir, L->t_f32, 1.0));
    uint32_t nm = l_emit(L, BIR_FSUB, L->t_f32, 0);
    l_op(L, nm, e2);
    l_op(L, nm, one);
    uint32_t dn = l_emit(L, BIR_FADD, L->t_f32, 0);
    l_op(L, dn, e2);
    l_op(L, dn, one);
    uint32_t r = l_emit(L, BIR_FDIV, L->t_f32, 0);
    l_op(L, r, nm);
    l_op(L, r, dn);
    return r;
}

/* Lower a Call where the callee is an Attr that sema resolved to a
 * tl.* intrinsic. Scalar math lands here; rank-2 tile calls take the
 * materialised tile path below. */

static uint32_t l_intrinsic_call(tn_lower_t *L, uint32_t call_idx,
                                 int intrinsic_id)
{
    uint32_t nk = l_nkids(&L->parser->nodes[call_idx]);
    /* kids[0] is the callee (the Attr). Positional args start at index
     * 1. Keyword args appear as TN_NK_KEYWORD nodes. */
    switch (intrinsic_id) {
    case TN_TLI_PROGRAM_ID: {
        /* tl.program_id(axis=N) -> BIR_BLOCK_ID with subop=N. The axis
         * is positional or keyword; read the value from an integer
         * literal child either way. */
        int axis = 0;
        for (uint32_t i = 1; i < nk; i++) {
            uint32_t kid = l_kid(L, call_idx, i);
            const tn_node_t *kn = &L->parser->nodes[kid];
            uint32_t value = kid;
            if (kn->kind == TN_NK_KEYWORD && l_nkids(kn) > 0) {
                value = l_kid(L, kid, 0);
                kn = &L->parser->nodes[value];
            }
            if (kn->kind == TN_NK_LITERAL && kn->flags == TN_LIT_INT) {
                const tn_tok_t *t = &L->parser->lex->tokens[kn->tok_off];
                axis = (int)l_parse_int(L->parser->lex->src + t->off,
                                         t->len);
            }
        }
        if (axis < 0) axis = 0;
        if (axis > 2) axis = 2;
        return l_emit(L, BIR_BLOCK_ID, L->t_i32, axis);
    }
    case TN_TLI_NUM_PROGRAMS: {
        int axis = 0;
        for (uint32_t i = 1; i < nk; i++) {
            uint32_t kid = l_kid(L, call_idx, i);
            const tn_node_t *kn = &L->parser->nodes[kid];
            uint32_t value = kid;
            if (kn->kind == TN_NK_KEYWORD && l_nkids(kn) > 0) {
                value = l_kid(L, kid, 0);
                kn = &L->parser->nodes[value];
            }
            if (kn->kind == TN_NK_LITERAL && kn->flags == TN_LIT_INT) {
                const tn_tok_t *t = &L->parser->lex->tokens[kn->tok_off];
                axis = (int)l_parse_int(L->parser->lex->src + t->off,
                                         t->len);
            }
        }
        if (axis < 0) axis = 0;
        if (axis > 2) axis = 2;
        return l_emit(L, BIR_GRID_DIM, L->t_i32, axis);
    }

    case TN_TLI_ARANGE: {
        /* tl.arange(start, stop) produces the lane-relative tile of
         * indices. The Triton-to-CUDA mapping is one thread per lane,
         * so each thread sees its own lane id; lower as
         * BIR_THREAD_ID(0). start/stop are ignored here and become
         * tile-shape metadata later. */
        return l_emit(L, BIR_THREAD_ID, L->t_i32, 0);
    }

    case TN_TLI_LOAD: {
        /* tl.load(ptr, mask=..., other=...). The mask is not honoured;
         * a diagnostic records that the lowering is conservative. */
        uint32_t ptr_val = BIR_VAL_NONE;
        int saw_mask = 0;
        for (uint32_t i = 1; i < nk; i++) {
            uint32_t kid = l_kid(L, call_idx, i);
            const tn_node_t *kn = &L->parser->nodes[kid];
            if (kn->kind == TN_NK_KEYWORD) { saw_mask = 1; continue; }
            if (ptr_val == BIR_VAL_NONE) ptr_val = l_expr(L, kid);
        }
        if (ptr_val == BIR_VAL_NONE) return BIR_VAL_NONE;
        if (saw_mask) {
            const tn_tok_t *t = l_tok(L, call_idx);
            fprintf(stderr, "%u:%u: warning: tl.load mask= ignored "
                            "in sitting two (load emitted unmasked)\n",
                    t ? t->line : 0, t ? t->col : 0);
        }
        uint32_t ptr_type = l_val_type(L, ptr_val);
        uint32_t pointee = (l_type_kind(L, ptr_type) == BIR_TYPE_PTR)
                           ? L->bir->types[ptr_type].inner
                           : L->t_i32;
        uint32_t inst = l_emit(L, BIR_LOAD, pointee, 0);
        l_op(L, inst, ptr_val);
        return inst;
    }

    case TN_TLI_STORE: {
        /* tl.store(ptr, value, mask=...). Same mask treatment as
         * tl.load above. BIR store wants ops[0]=value, ops[1]=addr. */
        uint32_t ptr_val = BIR_VAL_NONE;
        uint32_t val_val = BIR_VAL_NONE;
        int saw_mask = 0;
        int pos = 0;
        for (uint32_t i = 1; i < nk; i++) {
            uint32_t kid = l_kid(L, call_idx, i);
            const tn_node_t *kn = &L->parser->nodes[kid];
            if (kn->kind == TN_NK_KEYWORD) { saw_mask = 1; continue; }
            uint32_t v = l_expr(L, kid);
            if (pos == 0)      ptr_val = v;
            else if (pos == 1) val_val = v;
            pos++;
        }
        if (ptr_val == BIR_VAL_NONE || val_val == BIR_VAL_NONE)
            return BIR_VAL_NONE;
        if (saw_mask) {
            const tn_tok_t *t = l_tok(L, call_idx);
            fprintf(stderr, "%u:%u: warning: tl.store mask= ignored "
                            "in sitting two (store emitted unmasked)\n",
                    t ? t->line : 0, t ? t->col : 0);
        }
        uint32_t inst = l_emit(L, BIR_STORE, L->t_void, 0);
        l_op(L, inst, val_val);
        l_op(L, inst, ptr_val);
        return inst;
    }

    case TN_TLI_CDIV: {
        /* cdiv(a, b) is ceiling division (a + b - 1) / b. Inlined
         * rather than given a dedicated BIR op; the existing
         * optimisation passes fold it. */
        if (nk < 3) return BIR_VAL_NONE;
        uint32_t a = l_expr(L, l_kid(L, call_idx, 1));
        uint32_t b = l_expr(L, l_kid(L, call_idx, 2));
        if (a == BIR_VAL_NONE || b == BIR_VAL_NONE) return BIR_VAL_NONE;
        uint32_t one = BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, 1));
        uint32_t bm1 = l_emit(L, BIR_SUB, L->t_i32, 0);
        l_op(L, bm1, b); l_op(L, bm1, one);
        uint32_t sum = l_emit(L, BIR_ADD, L->t_i32, 0);
        l_op(L, sum, a); l_op(L, sum, bm1);
        uint32_t div = l_emit(L, BIR_SDIV, L->t_i32, 0);
        l_op(L, div, sum); l_op(L, div, b);
        return div;
    }

    case TN_TLI_EXP:     return l_exp_nat(L, call_idx);
    case TN_TLI_EXP2:    return l_math1(L, call_idx, BIR_EXP2);
    case TN_TLI_LOG:     return l_log_nat(L, call_idx);
    case TN_TLI_LOG2:    return l_math1(L, call_idx, BIR_LOG2);
    case TN_TLI_SIN:     return l_sincos(L, call_idx, BIR_SIN);
    case TN_TLI_COS:     return l_sincos(L, call_idx, BIR_COS);
    case TN_TLI_TAN:     return l_tan(L, call_idx);
    case TN_TLI_TANH:    return l_tanh(L, call_idx);
    case TN_TLI_SQRT:    return l_math1(L, call_idx, BIR_SQRT);
    case TN_TLI_RSQRT:   return l_math1(L, call_idx, BIR_RSQ);
    case TN_TLI_ABS:     return l_math1(L, call_idx, BIR_FABS);
    case TN_TLI_FLOOR:   return l_math1(L, call_idx, BIR_FLOOR);
    case TN_TLI_CEIL:    return l_math1(L, call_idx, BIR_CEIL);
    case TN_TLI_MAXIMUM: return l_math2(L, call_idx, BIR_FMAX);
    case TN_TLI_MINIMUM: return l_math2(L, call_idx, BIR_FMIN);
    case TN_TLI_FDIV:    return l_math2(L, call_idx, BIR_FDIV);

    default: {
        /* Everything else is not yet lowered. The diagnostic includes
         * the intrinsic name so the missing one is clear. */
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "intrinsic tl.%s not yet lowered in sitting one",
                 tn_intrinsic_name(intrinsic_id));
        l_err(L, 95, l_tok(L, call_idx), msg);
        return BIR_VAL_NONE;
    }
    }
}

/* ---- Rank-2 / tile materialisation (the tl.dot path) ----
 *
 * Once a kernel touches a 2-D tile, the lane-as-thread model is dropped
 * and the tile is unrolled. Block sizes are compile-time constants, so a
 * tile is an array of per-element BIR values: arange yields literal
 * indices, [:,None]/[None,:] reshape, ops fan out with broadcasting, and
 * tl.dot is a flat sum of products. It runs on one thread, since nothing
 * here reads thread_id. Tiles are capped at TN_TILE_MAX elements; past
 * that, lowering bails rather than emit a wall of instructions. */

static int l_tile(tn_lower_t *L, uint32_t node, int *out);

static uint32_t l_const_i32(tn_lower_t *L, int v){
    return BIR_MAKE_CONST(bir_const_int(L->bir, L->t_i32, (int64_t)v));
}
static int l_tile_is_float(tn_lower_t *L, uint32_t node){
    int dt = L->sema->node_shape[node].dtype;
    return dt==TN_TLI_FLOAT16 || dt==TN_TLI_FLOAT32 ||
           dt==TN_TLI_FLOAT64 || dt==TN_TLI_BFLOAT16;
}
static int l_tile_elems(const tn_tile_t *t){
    return (t->rank>=2) ? t->d0*t->d1 : t->d0;
}
/* element (r,c) with size-1 axes broadcasting */
static uint32_t l_tile_get(const tn_tile_t *t, int r, int c){
    int cols = (t->rank>=2) ? t->d1 : 1;
    int rr = (t->d0==1) ? 0 : r;
    int cc = (cols==1)  ? 0 : c;
    int idx = rr*cols + cc;
    if (idx<0 || idx>=TN_TILE_MAX) return BIR_VAL_NONE;
    return t->elem[idx];
}
static int l_tile_alloc(tn_lower_t *L){
    return (L->n_tiles>=TN_TILE_POOL) ? -1 : L->n_tiles++;
}
/* rank-0 operand becomes a 1x1 broadcast tile of the scalar value */
static int l_tile_operand(tn_lower_t *L, uint32_t node, int *out){
    if (L->sema->node_shape[node].rank == 0){
        int ti=l_tile_alloc(L); if(ti<0) return -1;
        tn_tile_t *t=&L->tile_pool[ti]; t->rank=2; t->d0=1; t->d1=1;
        t->elem[0]=l_expr(L,node);
        if (t->elem[0]==BIR_VAL_NONE) return -1;
        *out=ti; return 0;
    }
    return l_tile(L, node, out);
}

static int l_tile(tn_lower_t *L, uint32_t node, int *out){
    const tn_node_t *n = &L->parser->nodes[node];
    tn_shape_t sh = L->sema->node_shape[node];
    int rows = (sh.rank>=1) ? sh.dims[0] : 1;
    int cols = (sh.rank>=2) ? sh.dims[1] : 1;
    if (rows<=0 || cols<=0 || rows*cols>TN_TILE_MAX){
        l_err(L, 99, l_tok(L,node), "tile shape unsupported or too large");
        return -1;
    }
    switch (n->kind){
    case TN_NK_NAME: {
        int kind = L->sema->node_sym_kind[node];
        if (kind==TN_SYM_LOCAL || kind==TN_SYM_LOOPVAR){
            uint32_t aux = L->sema->node_sym_aux[node];
            if (aux<TN_MAX_NODES && L->node_tile[aux]>=0){ *out=L->node_tile[aux]; return 0; }
        }
        l_err(L, 91, l_tok(L,node), "tile local referenced before defined");
        return -1;
    }
    case TN_NK_SUBSCRIPT: {
        /* offs[:,None] / offs[None,:]: same elements as the child vec,
         * reshaped to the column/row shape sema inferred. */
        int ci; if (l_tile(L, l_kid(L,node,0), &ci)) return -1;
        int ne = l_tile_elems(&L->tile_pool[ci]);
        int ti = l_tile_alloc(L); if (ti<0) return -1;
        tn_tile_t *t=&L->tile_pool[ti], *child=&L->tile_pool[ci];
        t->rank=2; t->d0=rows; t->d1=cols;
        for (int i=0;i<ne && i<TN_TILE_MAX;i++) t->elem[i]=child->elem[i];
        *out=ti; return 0;
    }
    case TN_NK_BINOP: {
        int la, ra;
        if (l_tile_operand(L, l_kid(L,node,0), &la)) return -1;
        if (l_tile_operand(L, l_kid(L,node,1), &ra)) return -1;
        int ti=l_tile_alloc(L); if(ti<0) return -1;
        tn_tile_t *t=&L->tile_pool[ti], *A=&L->tile_pool[la], *B=&L->tile_pool[ra];
        t->rank=2; t->d0=rows; t->d1=cols;
        int bop=(int)n->flags;
        for (int r=0;r<rows;r++) for (int c=0;c<cols;c++)
            t->elem[r*cols+c]=l_emit_binop(L, bop, l_tile_get(A,r,c), l_tile_get(B,r,c));
        *out=ti; return 0;
    }
    case TN_NK_CALL: {
        uint32_t callee=l_kid(L,node,0);
        if (L->sema->node_sym_kind[callee]!=TN_SYM_INTRINSIC){
            l_err(L,96,l_tok(L,node),"non-intrinsic tile call"); return -1; }
        int id=(int)L->sema->node_sym_aux[callee];
        uint32_t nk=l_nkids(n);
        if (id==TN_TLI_ARANGE){
            int ti=l_tile_alloc(L); if(ti<0)return -1; tn_tile_t*t=&L->tile_pool[ti];
            t->rank=1; t->d0=rows; t->d1=1;
            for (int i=0;i<rows;i++) t->elem[i]=l_const_i32(L,i);
            *out=ti; return 0;
        }
        if (id==TN_TLI_ZEROS || id==TN_TLI_ZEROS_LIKE){
            /* Accumulators are scratch-backed so they survive a K-loop:
             * alloca the buffer, store the element addresses, init to 0. */
            int ti=l_tile_alloc(L); if(ti<0)return -1; tn_tile_t*t=&L->tile_pool[ti];
            t->rank=2; t->d0=rows; t->d1=cols; t->mem=1;
            int isf=l_tile_is_float(L,node);
            uint32_t et=isf?L->t_f32:L->t_i32, ept=isf?L->t_ptr_f32:L->t_ptr_i32;
            uint32_t arr=bir_type_array(L->bir, et, (uint32_t)(rows*cols));
            uint32_t apt=bir_type_ptr(L->bir, arr, 0);
            uint32_t base=l_emit(L,BIR_ALLOCA,apt,0);
            uint32_t z = isf ? BIR_MAKE_CONST(bir_const_float(L->bir,L->t_f32,0.0))
                             : l_const_i32(L,0);
            for (int i=0;i<rows*cols;i++){
                uint32_t gep=l_emit(L,BIR_GEP,ept,0); l_op(L,gep,base); l_op(L,gep,l_const_i32(L,i));
                uint32_t st=l_emit(L,BIR_STORE,L->t_void,0); l_op(L,st,z); l_op(L,st,gep);
                t->elem[i]=gep;
            }
            *out=ti; return 0;
        }
        if (id==TN_TLI_LOAD){
            uint32_t addr=0;
            for (uint32_t i=1;i<nk;i++){ uint32_t k=l_kid(L,node,i);
                if (L->parser->nodes[k].kind!=TN_NK_KEYWORD){ addr=k; break; } }
            int ai; if(l_tile(L,addr,&ai)) return -1;
            int ti=l_tile_alloc(L); if(ti<0)return -1;
            tn_tile_t *t=&L->tile_pool[ti], *A=&L->tile_pool[ai];
            t->rank=2; t->d0=rows; t->d1=cols;
            int ne=l_tile_elems(A);
            for (int i=0;i<rows*cols;i++){
                uint32_t a=(i<ne)?A->elem[i]:BIR_VAL_NONE;
                uint32_t pty=l_val_type(L,a);  /* element type from the pointer's pointee */
                uint32_t dty=(l_type_kind(L,pty)==BIR_TYPE_PTR)? L->bir->types[pty].inner : L->t_i32;
                uint32_t ld=l_emit(L,BIR_LOAD,dty,0); l_op(L,ld,a);
                t->elem[i]=ld;
            }
            *out=ti; return 0;
        }
        if (id==TN_TLI_DOT){
            int ai,bi;
            if (l_tile(L,l_kid(L,node,1),&ai)) return -1;
            if (l_tile(L,l_kid(L,node,2),&bi)) return -1;
            tn_tile_t *A=&L->tile_pool[ai], *B=&L->tile_pool[bi];
            int Md=A->d0, Kd=A->d1, Nd=B->d1;
            int ti=l_tile_alloc(L); if(ti<0)return -1; tn_tile_t*t=&L->tile_pool[ti];
            t->rank=2; t->d0=Md; t->d1=Nd;
            /* float vs int dot follows the loaded operand element type */
            int isf = (l_type_kind(L, l_val_type(L, A->elem[0]))==BIR_TYPE_FLOAT);
            uint32_t mty=isf?L->t_f32:L->t_i32;
            for (int m=0;m<Md;m++) for (int nn=0;nn<Nd;nn++){
                uint32_t acc=BIR_VAL_NONE;
                for (int k=0;k<Kd;k++){
                    uint32_t pr=l_emit(L, isf?BIR_FMUL:BIR_MUL, mty, 0);
                    l_op(L,pr,A->elem[m*Kd+k]); l_op(L,pr,B->elem[k*Nd+nn]);
                    if (acc==BIR_VAL_NONE) acc=pr;
                    else { uint32_t s=l_emit(L, isf?BIR_FADD:BIR_ADD, mty, 0);
                           l_op(L,s,acc); l_op(L,s,pr); acc=s; }
                }
                t->elem[m*Nd+nn]=acc;
            }
            *out=ti; return 0;
        }
        l_err(L,95,l_tok(L,node),"tile intrinsic not yet lowered");
        return -1;
    }
    default:
        l_err(L,97,l_tok(L,node),"tile expression kind not lowered");
        return -1;
    }
}

/* tl.store(addr_tile, value_tile): one BIR_STORE per element. */
static void l_store_tile(tn_lower_t *L, uint32_t call_node){
    uint32_t nk=l_nkids(&L->parser->nodes[call_node]);
    uint32_t addr=0, val=0; int pos=0;
    for (uint32_t i=1;i<nk;i++){ uint32_t k=l_kid(L,call_node,i);
        if (L->parser->nodes[k].kind==TN_NK_KEYWORD) continue;
        if (pos==0) addr=k; else if (pos==1) val=k; pos++; }
    int ai,vi; if(l_tile(L,addr,&ai)) return; if(l_tile(L,val,&vi)) return;
    tn_tile_t *A=&L->tile_pool[ai], *V=&L->tile_pool[vi];
    int ne=l_tile_elems(A);
    for (int i=0;i<ne;i++){
        uint32_t vval=V->elem[i];
        if (V->mem){ /* scratch accumulator: read the element first */
            uint32_t pty=l_val_type(L,vval);
            uint32_t dty=(l_type_kind(L,pty)==BIR_TYPE_PTR)?L->bir->types[pty].inner:L->t_f32;
            uint32_t ld=l_emit(L,BIR_LOAD,dty,0); l_op(L,ld,vval); vval=ld;
        }
        uint32_t st=l_emit(L,BIR_STORE,L->t_void,0);
        l_op(L,st,vval); l_op(L,st,A->elem[i]);
    }
}

/* Does any node in this subtree carry a rank-2 shape? Decides whether
 * the kernel uses the tile (materialise/unroll) lowering path. */
static int l_subtree_has_rank2(tn_lower_t *L, uint32_t node){
    if (node==0 || node>=L->parser->num_nodes) return 0;
    if (L->sema->node_shape[node].rank >= 2) return 1;
    uint32_t nk=l_nkids(&L->parser->nodes[node]);
    for (uint32_t i=0;i<nk;i++)
        if (l_subtree_has_rank2(L, l_kid(L,node,i))) return 1;
    return 0;
}

static uint32_t l_call(tn_lower_t *L, uint32_t node_idx)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    if (l_nkids(n) == 0) return BIR_VAL_NONE;

    uint32_t callee = l_kid(L, node_idx, 0);
    int kind = L->sema->node_sym_kind[callee];
    if (kind == TN_SYM_INTRINSIC) {
        int id = (int)L->sema->node_sym_aux[callee];
        return l_intrinsic_call(L, node_idx, id);
    }

    l_err(L, 96, l_tok(L, node_idx),
          "non-intrinsic call not yet lowered");
    return BIR_VAL_NONE;
}

static uint32_t l_expr(tn_lower_t *L, uint32_t node_idx)
{
    if (node_idx == 0 || node_idx >= L->parser->num_nodes) return BIR_VAL_NONE;
    if (!l_shape_supported(L, node_idx)) return BIR_VAL_NONE;
    const tn_node_t *n = &L->parser->nodes[node_idx];

    switch (n->kind) {
    case TN_NK_LITERAL:  return l_lit(L, node_idx);
    case TN_NK_NAME:     return l_name(L, node_idx);
    case TN_NK_BINOP:    return l_binop(L, node_idx);
    case TN_NK_UNOP:     return l_unop(L, node_idx);
    case TN_NK_CALL:     return l_call(L, node_idx);
    case TN_NK_COMPARE:  return l_compare(L, node_idx);
    case TN_NK_TUPLE:
    case TN_NK_LIST:
    case TN_NK_SUBSCRIPT:
    case TN_NK_SLICE:
    case TN_NK_ATTR:
    case TN_NK_IFEXPR:
    case TN_NK_BOOLOP:
        l_err(L, 97, l_tok(L, node_idx),
              "expression kind not yet lowered in sitting two");
        return BIR_VAL_NONE;
    case TN_NK_EXPR_SPAN:
        l_err(L, 98, l_tok(L, node_idx),
              "opaque expression span left over from parser");
        return BIR_VAL_NONE;
    default:
        return BIR_VAL_NONE;
    }
}

/* ---- Statement lowering ---- */

static void l_stmt(tn_lower_t *L, uint32_t node_idx);

static void l_assign(tn_lower_t *L, uint32_t node_idx)
{
    uint32_t value_node = l_kid(L, node_idx, 1);
    /* In a tile kernel, a rank>=1 RHS materialises into a tile bound to
     * this Assign node; Name references resolve through node_tile. */
    if (L->tile_mode && L->sema->node_shape[value_node].rank >= 1){
        int ti;
        if (l_tile(L, value_node, &ti)==0) L->node_tile[node_idx]=ti;
        return;
    }
    uint32_t v = l_expr(L, value_node);
    L->node_val[node_idx] = v;
}

static void l_return(tn_lower_t *L, uint32_t node_idx)
{
    uint32_t inst = l_emit(L, BIR_RET, L->t_void, 0);
    uint32_t nk = l_nkids(&L->parser->nodes[node_idx]);
    if (nk > 0) {
        uint32_t v = l_expr(L, l_kid(L, node_idx, 0));
        if (v != BIR_VAL_NONE) l_op(L, inst, v);
    }
}

static void l_block(tn_lower_t *L, uint32_t node_idx)
{
    uint32_t nk = l_nkids(&L->parser->nodes[node_idx]);
    for (uint32_t i = 0; i < nk; i++) {
        l_stmt(L, l_kid(L, node_idx, i));
    }
}

/* Lower `for k in range(start, stop, step)` as a counted loop.
 *
 * The counter is a phi built by hand, not an alloca. Stashing k in an
 * alloca and letting mem2reg lift it to a phi (as the CUDA loops do)
 * fails here: mem2reg decides the counter never changes, folds it to its
 * start value, and the loop runs forever. Emitting the phi directly
 * avoids that, and the backends already copy a phi's value in along each
 * edge. Bodies are straight-line only, which is all the matmul K-loop
 * needs. */
static void l_for(tn_lower_t *L, uint32_t node_idx)
{
    uint32_t nk = l_nkids(&L->parser->nodes[node_idx]);
    if (nk < 3) { l_err(L,99,l_tok(L,node_idx),"malformed for"); return; }
    uint32_t iter = l_kid(L, node_idx, 1);
    uint32_t body = l_kid(L, node_idx, 2);
    const tn_node_t *it = &L->parser->nodes[iter];
    if (it->kind != TN_NK_CALL){ l_err(L,99,l_tok(L,iter),"for iterable must be range()"); return; }
    int nargs = (int)l_nkids(it) - 1;   /* kid 0 is the 'range' callee */
    uint32_t start, stop, step;
    if (nargs <= 1){ start=l_const_i32(L,0); stop=(nargs==1)?l_expr(L,l_kid(L,iter,1)):l_const_i32(L,0); step=l_const_i32(L,1); }
    else if (nargs == 2){ start=l_expr(L,l_kid(L,iter,1)); stop=l_expr(L,l_kid(L,iter,2)); step=l_const_i32(L,1); }
    else { start=l_expr(L,l_kid(L,iter,1)); stop=l_expr(L,l_kid(L,iter,2)); step=l_expr(L,l_kid(L,iter,3)); }
    if (start==BIR_VAL_NONE||stop==BIR_VAL_NONE||step==BIR_VAL_NONE) return;

    /* Counter is a real phi (no alloca), so mem2reg cannot fold it; the
     * backend's phi edge-copies carry start in on the preheader edge and
     * the increment in on the back-edge. */
    uint32_t pre = L->cur_block;
    uint32_t br0 = l_emit(L, BIR_BR, L->t_void, 0);                 /* preheader -> head */

    uint32_t head = l_new_block(L); l_op(L, br0, head);
    L->cur_block = head;
    uint32_t kphi = l_emit(L, BIR_PHI, L->t_i32, 0);
    l_op(L, kphi, pre); l_op(L, kphi, start);                       /* [preheader: start] */
    uint32_t cond = l_emit(L, BIR_ICMP, L->t_i32, BIR_ICMP_SLT); l_op(L,cond,kphi); l_op(L,cond,stop);
    uint32_t brc = l_emit(L, BIR_BR_COND, L->t_void, 0); l_op(L,brc,cond);  /* [0]=cond */

    uint32_t bodyb = l_new_block(L); l_op(L, brc, bodyb);                   /* [1]=true */
    L->cur_block = bodyb;
    L->node_val[node_idx] = kphi;          /* bind the loop variable k */
    l_block(L, body);
    uint32_t kn = l_emit(L, BIR_ADD, L->t_i32, 0); l_op(L,kn,kphi); l_op(L,kn,step);
    l_op(L, kphi, bodyb); l_op(L, kphi, kn);   /* phi back-edge pair [body: k+step] */
    uint32_t brh = l_emit(L, BIR_BR, L->t_void, 0); l_op(L,brh,head);       /* back-edge */

    uint32_t exitb = l_new_block(L); l_op(L, brc, exitb);                   /* [2]=false */
    L->cur_block = exitb;
}

static void l_stmt(tn_lower_t *L, uint32_t node_idx)
{
    if (node_idx == 0 || node_idx >= L->parser->num_nodes) return;
    const tn_node_t *n = &L->parser->nodes[node_idx];

    switch (n->kind) {
    case TN_NK_ASSIGN:    l_assign(L, node_idx);  break;
    case TN_NK_RETURN:    l_return(L, node_idx);  break;
    case TN_NK_EXPR_STMT: {
        if (l_nkids(n) > 0) {
            uint32_t expr = l_kid(L, node_idx, 0);
            const tn_node_t *en = &L->parser->nodes[expr];
            /* Bare string expression statement is a Python docstring.
             * Triton kernels routinely open with one; they have no
             * runtime meaning, so skip them silently. */
            if (en->kind == TN_NK_LITERAL && en->flags == TN_LIT_STRING) {
                break;
            }
            /* In a tile kernel, tl.store(addr_tile, value_tile) fans out
             * to one BIR_STORE per element. */
            if (L->tile_mode && en->kind == TN_NK_CALL && l_nkids(en) > 0){
                uint32_t callee = l_kid(L, expr, 0);
                if (L->sema->node_sym_kind[callee]==TN_SYM_INTRINSIC &&
                    (int)L->sema->node_sym_aux[callee]==TN_TLI_STORE){
                    l_store_tile(L, expr);
                    break;
                }
            }
            (void)l_expr(L, expr);
        }
        break;
    }
    case TN_NK_PASS:
    case TN_NK_BREAK:
    case TN_NK_CONTINUE:
        /* No-ops at the BIR level until SSA-flavoured loops exist. */
        break;
    case TN_NK_BLOCK:    l_block(L, node_idx);    break;
    case TN_NK_AUG_ASSIGN: {
        /* Lower `x op= rhs` as `x = x op rhs`. Find the declaring node
         * of x so the resulting BIR value replaces the old one in the
         * node_val map. Only single-Name targets are handled; attribute
         * and subscript targets need separate machinery to address the
         * LHS slot rather than read it. */
        if (l_nkids(n) < 2) break;
        uint32_t target_idx = l_kid(L, node_idx, 0);
        uint32_t rhs_idx    = l_kid(L, node_idx, 1);
        const tn_node_t *tn = &L->parser->nodes[target_idx];
        if (tn->kind != TN_NK_NAME) {
            l_err(L, 102, l_tok(L, target_idx),
                  "AugAssign target kind not yet lowered");
            break;
        }

        /* Tile accumulator: acc += <tile>. acc is scratch-backed, so
         * this is read-add-write per element and persists across the
         * loop. */
        if (L->tile_mode && n->flags == TN_AUG_ADD){
            int tk=L->sema->node_sym_kind[target_idx];
            uint32_t aux=L->sema->node_sym_aux[target_idx];
            if ((tk==TN_SYM_LOCAL||tk==TN_SYM_LOOPVAR) && aux<TN_MAX_NODES &&
                L->node_tile[aux]>=0 && L->tile_pool[L->node_tile[aux]].mem){
                int ri; if (l_tile(L, rhs_idx, &ri)==0){
                    tn_tile_t *ACC=&L->tile_pool[L->node_tile[aux]], *R=&L->tile_pool[ri];
                    int ne=l_tile_elems(ACC);
                    for (int i=0;i<ne;i++){
                        uint32_t pty=l_val_type(L,ACC->elem[i]);
                        uint32_t dty=(l_type_kind(L,pty)==BIR_TYPE_PTR)?L->bir->types[pty].inner:L->t_f32;
                        int isf=(l_type_kind(L,dty)==BIR_TYPE_FLOAT);
                        uint32_t cur=l_emit(L,BIR_LOAD,dty,0); l_op(L,cur,ACC->elem[i]);
                        uint32_t add=l_emit(L, isf?BIR_FADD:BIR_ADD, dty,0); l_op(L,add,cur); l_op(L,add,R->elem[i]);
                        uint32_t st=l_emit(L,BIR_STORE,L->t_void,0); l_op(L,st,add); l_op(L,st,ACC->elem[i]);
                    }
                }
                break;
            }
        }

        uint32_t old_val = l_expr(L, target_idx);
        uint32_t rhs_val = l_expr(L, rhs_idx);
        if (old_val == BIR_VAL_NONE || rhs_val == BIR_VAL_NONE) break;

        /* Map TN_AUG_* onto TN_BOP_* to reuse l_bop_int / l_bop_float.
         * The codes line up for the common arithmetic ops because the
         * enums share the same order. */
        int bop = -1;
        switch (n->flags) {
        case TN_AUG_ADD:  bop = TN_BOP_ADD;  break;
        case TN_AUG_SUB:  bop = TN_BOP_SUB;  break;
        case TN_AUG_MUL:  bop = TN_BOP_MUL;  break;
        case TN_AUG_DIV:  bop = TN_BOP_DIV;  break;
        case TN_AUG_FDIV: bop = TN_BOP_FDIV; break;
        case TN_AUG_MOD:  bop = TN_BOP_MOD;  break;
        case TN_AUG_AND:  bop = TN_BOP_AND;  break;
        case TN_AUG_OR:   bop = TN_BOP_OR;   break;
        case TN_AUG_XOR:  bop = TN_BOP_XOR;  break;
        case TN_AUG_SHL:  bop = TN_BOP_SHL;  break;
        case TN_AUG_SHR:  bop = TN_BOP_SHR;  break;
        default:
            l_err(L, 103, l_tok(L, node_idx),
                  "AugAssign operator not yet lowered");
            break;
        }
        if (bop < 0) break;

        int lkind = l_type_kind(L, l_val_type(L, old_val));
        int rkind = l_type_kind(L, l_val_type(L, rhs_val));
        int op;
        uint32_t result_type;
        if (lkind == BIR_TYPE_FLOAT || rkind == BIR_TYPE_FLOAT) {
            op = l_bop_float(bop);
            result_type = L->t_f32;
        } else {
            op = l_bop_int(bop);
            result_type = L->t_i32;
        }
        if (op < 0) break;
        uint32_t inst = l_emit(L, op, result_type, 0);
        l_op(L, inst, old_val);
        l_op(L, inst, rhs_val);

        /* Re-bind the local to the new value so subsequent
         * references read the updated state. */
        int kind = L->sema->node_sym_kind[target_idx];
        uint32_t aux = L->sema->node_sym_aux[target_idx];
        if ((kind == TN_SYM_LOCAL || kind == TN_SYM_LOOPVAR) &&
            aux < TN_MAX_NODES) {
            L->node_val[aux] = inst;
        }
        break;
    }
    case TN_NK_IF:
    case TN_NK_FOR:
        l_for(L, node_idx);
        break;
    case TN_NK_WHILE:
        l_err(L, 99, l_tok(L, node_idx),
              "while loops not yet lowered");
        break;
    default:
        break;
    }
}

/* ---- Function lowering ---- */

static void l_funcdef(tn_lower_t *L, uint32_t node_idx, int is_kernel)
{
    const tn_parse_t *P = L->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    bir_module_t *M = L->bir;
    if (M->num_funcs >= BIR_MAX_FUNCS) return;

    uint32_t fi = M->num_funcs++;
    bir_func_t *F = &M->funcs[fi];
    memset(F, 0, sizeof(*F));

    /* Function name from the IDENT token immediately after `def`. */
    uint32_t name_tok = n->tok_off;
    while (name_tok < P->lex->num_tokens &&
           P->lex->tokens[name_tok].kind != TN_TOK_IDENT) name_tok++;
    if (name_tok < P->lex->num_tokens) {
        const tn_tok_t *t = &P->lex->tokens[name_tok];
        F->name = bir_add_string(M, P->lex->src + t->off, t->len);
    }

    /* Collect parameters: each PARAM child becomes a BIR_PARAM unless
     * it is a constexpr-with-default, which is folded to a literal at
     * lower time and dropped from the runtime signature. param_remap
     * maps source-position to BIR-position; source-position matches the
     * kid index because params come before the body, so sema's
     * TN_SYM_PARAM aux lands in the same slot. */
    uint32_t nk = l_nkids(n);
    uint32_t param_types[32];
    int num_params = 0;
    for (int j = 0; j < 32; j++) {
        L->param_remap[j] = 0xFF;
        L->param_const[j] = -1;
    }
    for (uint32_t i = 0; i < nk && i < 32 && num_params < 32; i++) {
        uint32_t kid = l_kid(L, node_idx, i);
        if (P->nodes[kid].kind == TN_NK_PARAM) {
            /* Default type policy: a parameter whose name ends in
             * "_ptr" gets f32*, a tl.constexpr-annotated scalar stays
             * i32, everything else is treated as i32. Real type
             * inference comes later. */
            uint32_t pname_tok = P->nodes[kid].tok_off;
            while (pname_tok < P->lex->num_tokens &&
                   P->lex->tokens[pname_tok].kind != TN_TOK_IDENT)
                pname_tok++;
            int is_ptr = 0;
            if (pname_tok < P->lex->num_tokens) {
                const tn_tok_t *t = &P->lex->tokens[pname_tok];
                if (t->len >= 4 &&
                    memcmp(P->lex->src + t->off + t->len - 4,
                           "_ptr", 4) == 0) {
                    is_ptr = 1;
                }
            }
            /* If the param has an annotation child that resolves to
             * the constexpr type marker, treat the param as a plain
             * scalar i32 rather than a pointer. */
            int is_constexpr = 0;
            uint32_t pkn = l_nkids(&P->nodes[kid]);
            for (uint32_t j = 0; j < pkn; j++) {
                uint32_t anno = l_kid(L, kid, j);
                int akind = L->sema->node_sym_kind[anno];
                uint32_t aaux = L->sema->node_sym_aux[anno];
                if (akind == TN_SYM_TYPE && aaux == TN_TLI_CONSTEXPR) {
                    is_constexpr = 1;
                    break;
                }
            }
            int has_default = (L->sema->node_const_val[kid] >= 0);
            if (is_constexpr && has_default) {
                /* Fold the value, drop the param from the BIR signature.
                 * 0xFF in the remap tells l_name to return a literal. */
                L->param_remap[i] = 0xFF;
                L->param_const[i] = L->sema->node_const_val[kid];
            } else if (is_constexpr) {
                /* Constexpr without a default keeps a runtime slot, so
                 * existing fixtures (vadd's BLOCK_SIZE) still work. */
                L->param_remap[i] = (uint8_t)num_params;
                param_types[num_params++] = L->t_i32;
            } else if (is_ptr) {
                L->param_remap[i] = (uint8_t)num_params;
                param_types[num_params++] = L->t_ptr_f32;
            } else {
                /* Default: assume i32 scalar for things like
                 * n_elements that look like sizes. */
                L->param_remap[i] = (uint8_t)num_params;
                param_types[num_params++] = L->t_i32;
            }
        }
    }
    F->type = bir_type_func(M, L->t_void, param_types, num_params);
    F->num_params = (uint16_t)num_params;
    F->cuda_flags = is_kernel ? CUDA_GLOBAL : CUDA_DEVICE;

    L->cur_func = fi;
    L->cur_block = l_new_block(L);
    L->cur_param_base = M->num_insts;

    /* Emit one BIR_PARAM instruction per parameter, in order. The
     * subop carries the parameter index; later references to the
     * Name resolve to BIR_MAKE_VAL(cur_param_base + index). */
    for (int i = 0; i < num_params; i++) {
        uint32_t inst = l_emit(L, BIR_PARAM, param_types[i], i);
        (void)inst;
    }

    /* Walk the body. The first child that is a Block is the body;
     * earlier children are params and the optional return annotation. */
    for (uint32_t i = 0; i < nk; i++) {
        uint32_t kid = l_kid(L, node_idx, i);
        if (P->nodes[kid].kind == TN_NK_BLOCK) {
            /* A kernel with any rank-2 tile uses the materialise/unroll
             * path for its whole body; otherwise the scalar lane path. */
            L->tile_mode = l_subtree_has_rank2(L, kid);
            L->n_tiles = 0;
            l_block(L, kid);
            break;
        }
    }

    /* Always terminate with a RET so the BIR block is well-formed,
     * even if the kernel author forgot to write one. */
    bir_block_t *B = &M->blocks[L->cur_block];
    if (B->num_insts == 0 ||
        M->insts[B->first_inst + B->num_insts - 1].op != BIR_RET) {
        (void)l_emit(L, BIR_RET, L->t_void, 0);
    }

    /* Sum every block, not just the last one: a kernel with control flow
     * (e.g. a matmul K-loop) spans several blocks, and cfold/dce walk
     * [first_inst, first_inst + total_insts), so an undercount truncates
     * the range and corrupts the loop. */
    F->total_insts = 0;
    for (uint16_t bi = 0; bi < F->num_blocks; bi++)
        F->total_insts += M->blocks[F->first_block + bi].num_insts;
}

/* Inspect the decorators preceding a FuncDef and decide whether the
 * function is annotated @triton.jit. Only JIT-decorated functions are
 * lowered; everything else is skipped. */

static int l_is_jit(const tn_lower_t *L, uint32_t module_idx, uint32_t func_pos)
{
    /* Look one or two siblings back for a Decorator node. */
    for (uint32_t back = 1; back <= 3 && back <= func_pos; back++) {
        uint32_t sib = l_kid(L, module_idx, func_pos - back);
        const tn_node_t *sn = &L->parser->nodes[sib];
        if (sn->kind == TN_NK_DECORATOR) {
            /* The first child is a DottedName. Containing `triton.jit`
             * marks a kernel; the bare `jit` form is accepted too in
             * case of an alias. */
            if (l_nkids(sn) > 0) {
                uint32_t dn = l_kid(L, sib, 0);
                const tn_node_t *dnn = &L->parser->nodes[dn];
                uint32_t tlen = dnn->tok_len;
                uint32_t off  = dnn->tok_off;
                /* Scan tokens for IDENT `jit`. */
                for (uint32_t t = off; t < off + tlen &&
                                       t < L->parser->lex->num_tokens; t++) {
                    const tn_tok_t *tk = &L->parser->lex->tokens[t];
                    if (tk->kind == TN_TOK_IDENT && tk->len == 3 &&
                        memcmp(L->parser->lex->src + tk->off, "jit", 3) == 0)
                        return 1;
                }
            }
            continue;
        }
        if (sn->kind == TN_NK_FUNCDEF) break;
    }
    return 0;
}

/* ---- Public API ---- */

void tn_lower_init(tn_lower_t *L, const tn_parse_t *P,
                   const tn_sema_t *S, bir_module_t *M)
{
    memset(L, 0, sizeof(*L));
    L->parser = P;
    L->sema   = S;
    L->bir    = M;
    for (uint32_t i = 0; i < TN_MAX_NODES; i++) L->node_val[i] = BIR_VAL_NONE;
    for (uint32_t i = 0; i < TN_MAX_NODES; i++) L->node_tile[i] = -1;
}

int tn_lower(tn_lower_t *L)
{
    bir_module_init(L->bir);
    l_types_init(L);

    /* Walk the module's top-level statements and lower each FuncDef.
     * Anything else (imports, bare decorators, expression statements)
     * is ignored: it produces no machine code. */
    uint32_t root = L->parser->root;
    if (root == 0 || root >= L->parser->num_nodes) return BC_OK;
    uint32_t nk = l_nkids(&L->parser->nodes[root]);
    for (uint32_t i = 0; i < nk; i++) {
        uint32_t kid = l_kid(L, root, i);
        const tn_node_t *kn = &L->parser->nodes[kid];
        if (kn->kind == TN_NK_FUNCDEF) {
            int is_kernel = l_is_jit(L, root, i);
            l_funcdef(L, kid, is_kernel);
        }
    }

    return (L->num_errors > 0) ? BC_ERR_TRITON : BC_OK;
}
