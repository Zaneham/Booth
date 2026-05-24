/* Triton frontend BIR lowering, sitting one.
 *
 * Walks the sema-annotated AST and produces BIR, the same target
 * the CUDA frontend lowers into. The point is that once a Triton
 * kernel reaches BIR, every backend (AMD, NVIDIA, Tensix, Metal,
 * Intel) consumes it without caring which Python decorator was
 * sitting at the top of the source file.
 *
 * Scope of this sitting:
 *   - Module skeleton: pull each @triton.jit FuncDef into the
 *     module, mark it __global__, generate BIR_PARAM instructions
 *     for each parameter, terminate the entry block with BIR_RET.
 *   - Statements: Assign (lower RHS, remember the BIR value through
 *     the AST node map), ExprStmt (lower expression for its side
 *     effect), Return.
 *   - Expressions: integer and float literals, Name references to
 *     parameters and locals, BinOp arithmetic (add, sub, mul,
 *     integer div, mod), unary negation, Call sites for the small
 *     set of scalar intrinsics this sitting recognises
 *     (tl.program_id, tl.num_programs).
 *   - Unsupported constructs get a polite diagnostic and the
 *     pass continues. For loops, conditionals, subscripts, slices,
 *     masked load/store, tile shapes, reductions, matmul, and the
 *     full intrinsic catalogue all wait their turn for sitting two
 *     and onwards.
 *
 * Type policy for this sitting: pointer parameters are typed as
 * i32* in the global address space, everything else is i32 unless
 * the source is a float literal in which case it is f32. The
 * crude policy is a deliberate sitting-one shortcut; sitting two
 * adds real type inference based on use sites and intrinsic
 * signatures. */

#include "triton.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- Helpers: Diagnostic ---- */

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

/* ---- Helpers: BIR Construction ---- */

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
 * overflow. The block's instruction count is bumped so block
 * scanners pick up the new instruction immediately. */

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
 * BIR_OPERANDS_INLINE slots; sitting one does not handle overflow
 * because the operand counts we encounter stay well below six. */

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

/* Look up or create a new BIR block as a child of the current
 * function. Sitting one only creates the entry block, but the API
 * keeps the door open for sittings that add control flow. */

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

/* ---- Type Bootstrap ---- */

static void l_types_init(tn_lower_t *L)
{
    bir_module_t *M = L->bir;
    L->t_void    = bir_type_void(M);
    L->t_i32     = bir_type_int(M, 32);
    L->t_f32     = bir_type_float(M, 32);
    L->t_ptr_i32 = bir_type_ptr(M, L->t_i32, BIR_AS_GLOBAL);
    L->t_ptr_f32 = bir_type_ptr(M, L->t_f32, BIR_AS_GLOBAL);
}

/* ---- Literal Parsing ----
 * The token text for a literal is whatever the source spelled.
 * For sitting one we accept the common forms: decimal integers
 * with optional underscores, hex / octal / binary integers with
 * the corresponding prefix, and simple floats with optional
 * exponent. Anything fancier (imaginary suffix, big literals
 * that overflow int64) falls back to zero. */

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
    /* Use a small stack buffer to feed strtod a null-terminated
     * string without copying through malloc. Anything longer than
     * the buffer is highly unusual and gets zero by default. */
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

/* ---- Expression Lowering ----
 * Each expression node either yields a BIR value or, for the
 * constructs sitting one does not yet handle, emits a diagnostic
 * and returns BIR_VAL_NONE. The walker is recursive over child
 * nodes; the recursion is shallow because Triton kernels are
 * shallow trees by design. */

static uint32_t l_expr(tn_lower_t *L, uint32_t node_idx);

/* Refuse to lower expressions whose inferred shape is rank-2 or
 * higher. The shape annotation comes from sema; what's missing is
 * the matrix-instruction codegen (MFMA on AMD, mma.sync on NVIDIA).
 * Until those exist, lowering a rank-2 tile would produce silent
 * wrong code. */

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
        /* The parameter's BIR value was emitted when we lowered the
         * function header. Look it up by its position relative to
         * cur_param_base. */
        return BIR_MAKE_VAL(L->cur_param_base + aux);
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
         * value expressions on their own; if they do it usually
         * means the kernel author wrote something we do not yet
         * lower. Report and continue. */
        l_err(L, 92, l_tok(L, node_idx),
              "bare module or intrinsic name in expression position");
        return BIR_VAL_NONE;
    case TN_SYM_UNBOUND:
    default:
        return BIR_VAL_NONE;
    }
}

/* Look up the BIR type of a value reference. For constants we read
 * the type out of the consts pool; for instructions we read it from
 * the instruction's result type slot. Sitting two needs this so
 * BinOp can dispatch between integer and floating-point operations
 * and so pointer-plus-integer can lower as a GEP. */

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

static uint32_t l_binop(tn_lower_t *L, uint32_t node_idx)
{
    const tn_node_t *n = &L->parser->nodes[node_idx];
    uint32_t lhs = l_expr(L, l_kid(L, node_idx, 0));
    uint32_t rhs = l_expr(L, l_kid(L, node_idx, 1));
    if (lhs == BIR_VAL_NONE || rhs == BIR_VAL_NONE) return BIR_VAL_NONE;

    uint32_t lhs_type = l_val_type(L, lhs);
    uint32_t rhs_type = l_val_type(L, rhs);
    int lkind = l_type_kind(L, lhs_type);
    int rkind = l_type_kind(L, rhs_type);

    /* Pointer arithmetic: ptr + i32 lowers as a BIR_GEP, the typed
     * pointer-stride flavour BIR uses for offset addressing. Only
     * ADD makes sense here; everything else on a pointer is silly. */
    if (n->flags == TN_BOP_ADD &&
        (lkind == BIR_TYPE_PTR || rkind == BIR_TYPE_PTR)) {
        uint32_t base = (lkind == BIR_TYPE_PTR) ? lhs : rhs;
        uint32_t idx  = (lkind == BIR_TYPE_PTR) ? rhs : lhs;
        uint32_t base_type = (lkind == BIR_TYPE_PTR) ? lhs_type : rhs_type;
        uint32_t inst = l_emit(L, BIR_GEP, base_type, 0);
        l_op(L, inst, base);
        l_op(L, inst, idx);
        return inst;
    }

    /* Float dispatch when either operand is a float; this is a
     * sitting-two simplification of Python's "promote int to
     * float in mixed expressions" rule. */
    if (lkind == BIR_TYPE_FLOAT || rkind == BIR_TYPE_FLOAT) {
        int op = l_bop_float((int)n->flags);
        if (op < 0) {
            l_err(L, 93, l_tok(L, node_idx),
                  "float binop not yet lowered");
            return BIR_VAL_NONE;
        }
        uint32_t inst = l_emit(L, op, L->t_f32, 0);
        l_op(L, inst, lhs);
        l_op(L, inst, rhs);
        return inst;
    }

    int op = l_bop_int((int)n->flags);
    if (op < 0) {
        l_err(L, 93, l_tok(L, node_idx),
              "integer binop not yet lowered");
        return BIR_VAL_NONE;
    }
    uint32_t inst = l_emit(L, op, L->t_i32, 0);
    l_op(L, inst, lhs);
    l_op(L, inst, rhs);
    return inst;
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

/* Lower a Call where the callee is an Attr that sema resolved to a
 * tl.* intrinsic. Sitting one recognises the scalar thread-model
 * intrinsics and stops there; the others get a polite diagnostic. */

static uint32_t l_intrinsic_call(tn_lower_t *L, uint32_t call_idx,
                                 int intrinsic_id)
{
    uint32_t nk = l_nkids(&L->parser->nodes[call_idx]);
    /* kids[0] is the callee (the Attr). Real positional args start
     * at index 1. Keyword args appear as TN_NK_KEYWORD nodes. */
    switch (intrinsic_id) {
    case TN_TLI_PROGRAM_ID: {
        /* tl.program_id(axis=N) -> BIR_BLOCK_ID with subop=N. The
         * axis can be positional or keyword; either way we look for
         * an integer literal child to read the value. */
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
         * indices. The Triton-to-CUDA mapping every backend already
         * understands is "one thread per lane", so each thread
         * sees its own lane id. Lower as BIR_THREAD_ID(0). We
         * ignore the start/stop values for sitting two; they
         * become tile-shape metadata in a later sitting. */
        return l_emit(L, BIR_THREAD_ID, L->t_i32, 0);
    }

    case TN_TLI_LOAD: {
        /* tl.load(ptr, mask=..., other=...). Sitting two does not
         * honour the mask; it is recorded in a diagnostic so the
         * kernel author knows the lowering is conservative. */
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
        /* cdiv(a, b) is the ceiling division (a + b - 1) / b. We
         * inline that arithmetic rather than introduce a dedicated
         * BIR op, on the basis that the existing optimisation
         * passes happily fold it. */
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

    default: {
        /* Everything else waits its turn. The diagnostic includes
         * the intrinsic name so the kernel author can tell at a
         * glance what is missing. */
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "intrinsic tl.%s not yet lowered in sitting one",
                 tn_intrinsic_name(intrinsic_id));
        l_err(L, 95, l_tok(L, call_idx), msg);
        return BIR_VAL_NONE;
    }
    }
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

/* ---- Statement Lowering ---- */

static void l_stmt(tn_lower_t *L, uint32_t node_idx);

static void l_assign(tn_lower_t *L, uint32_t node_idx)
{
    uint32_t value_node = l_kid(L, node_idx, 1);
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
            /* Bare string expression as a statement is a Python
             * docstring. Triton kernels routinely open with one
             * and they have no runtime meaning, so we eat them
             * silently rather than complain about strings being
             * unlowerable. */
            if (en->kind == TN_NK_LITERAL && en->flags == TN_LIT_STRING) {
                break;
            }
            (void)l_expr(L, expr);
        }
        break;
    }
    case TN_NK_PASS:
    case TN_NK_BREAK:
    case TN_NK_CONTINUE:
        /* No-ops at the BIR level until SSA-flavoured loops show up
         * in sitting three. */
        break;
    case TN_NK_BLOCK:    l_block(L, node_idx);    break;
    case TN_NK_AUG_ASSIGN: {
        /* Lower `x op= rhs` as `x = x op rhs`. We need to find the
         * declaring node of x so the resulting BIR value replaces
         * the old one in the node_val map. Only single-Name
         * targets are handled here; sitting three takes on
         * attribute and subscript targets, which require seperate
         * machinery to address the LHS slot rather than read it. */
        if (l_nkids(n) < 2) break;
        uint32_t target_idx = l_kid(L, node_idx, 0);
        uint32_t rhs_idx    = l_kid(L, node_idx, 1);
        const tn_node_t *tn = &L->parser->nodes[target_idx];
        if (tn->kind != TN_NK_NAME) {
            l_err(L, 102, l_tok(L, target_idx),
                  "AugAssign target kind not yet lowered");
            break;
        }

        uint32_t old_val = l_expr(L, target_idx);
        uint32_t rhs_val = l_expr(L, rhs_idx);
        if (old_val == BIR_VAL_NONE || rhs_val == BIR_VAL_NONE) break;

        /* Map TN_AUG_* onto TN_BOP_* so we can reuse l_bop_int /
         * l_bop_float. The codes happen to line up for the common
         * arithmetic ops because we deliberately chose the same
         * order. */
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
    case TN_NK_WHILE:
        l_err(L, 99, l_tok(L, node_idx),
              "control-flow statement waits for sitting three");
        break;
    default:
        break;
    }
}

/* ---- Function Lowering ---- */

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

    /* Collect parameters: each PARAM child becomes a BIR_PARAM.
     * Build a function type with i32* pointee for each param so
     * the backends have something to work with. */
    uint32_t nk = l_nkids(n);
    uint32_t param_types[32];
    int num_params = 0;
    for (uint32_t i = 0; i < nk && num_params < 32; i++) {
        uint32_t kid = l_kid(L, node_idx, i);
        if (P->nodes[kid].kind == TN_NK_PARAM) {
            /* Sitting two default: a parameter whose name ends in
             * "_ptr" or starts with the conventional `x_/y_/out_`
             * gets f32*, scalar params with a tl.constexpr
             * annotation stay as i32, and everything else gets
             * f32* on the assumption that Triton kernels are
             * mostly floating-point. A real type-inference pass
             * arrives in sitting three. */
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
            if (is_constexpr) {
                param_types[num_params++] = L->t_i32;
            } else if (is_ptr) {
                param_types[num_params++] = L->t_ptr_f32;
            } else {
                /* Default: assume i32 scalar for things like
                 * n_elements that look like sizes. */
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

    F->total_insts = B->num_insts;
}

/* Look at the decorators that precede a FuncDef in the AST and
 * decide whether the function is annotated @triton.jit. Sitting
 * one only lowers JIT-decorated functions; everything else is
 * skipped because we have no reason to compile it. */

static int l_is_jit(const tn_lower_t *L, uint32_t module_idx, uint32_t func_pos)
{
    /* Look one or two siblings back for a Decorator node. */
    for (uint32_t back = 1; back <= 3 && back <= func_pos; back++) {
        uint32_t sib = l_kid(L, module_idx, func_pos - back);
        const tn_node_t *sn = &L->parser->nodes[sib];
        if (sn->kind == TN_NK_DECORATOR) {
            /* The first child is a DottedName. If it contains
             * `triton.jit` we consider it a kernel; we accept the
             * bare `jit` form too in case anyone aliased. */
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
}

int tn_lower(tn_lower_t *L)
{
    bir_module_init(L->bir);
    l_types_init(L);

    /* Walk the module's top-level statements and lower each FuncDef.
     * Anything else (imports, decorators on their own, expression
     * statements) is ignored by lowering, since they do not produce
     * machine code. */
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
