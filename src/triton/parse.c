/* Triton frontend parser, sitting one: statements scaffold with
 * opaque expression spans.
 *
 * The shape of the deal in this sitting is that every Python
 * statement Triton kernels actually use gets its own AST node with
 * its own structural fields, while expressions are captured as flat
 * token ranges. That defers the expression parser to a later sitting
 * without preventing the rest of the compiler from getting structural
 * work done in the meantime: sema can still walk function defs and
 * see their parameters, and lowering can still ask which statements
 * are inside which functions.
 *
 * Recursive descent over the token stream. No left recursion, no
 * backtracking past one token of lookahead, no tables. The grammar
 * follows the Python language reference closely enough that anyone
 * with the spec open can read along, departing from it only where
 * Triton forbids something Python allows or where the parser will
 * cope better tomorrow if we keep going past a tokeniser-level
 * error today. */

#include "triton.h"

#include <string.h>
#include <stdio.h>

/* ---- Token Helpers ----
 * The parser does most of its work through a small handful of token
 * predicates and consumers, defined here once so the grammar
 * functions can read like grammar functions and not like a forest
 * of pointer arithmetic. */

static const tn_tok_t *p_peek(tn_parse_t *P, int delta)
{
    uint32_t i = P->cur + (uint32_t)delta;
    if (i >= P->lex->num_tokens) i = P->lex->num_tokens - 1;
    return &P->lex->tokens[i];
}

static int p_at(tn_parse_t *P, int kind)
{
    return p_peek(P, 0)->kind == kind;
}

static int p_eat(tn_parse_t *P, int kind)
{
    if (p_at(P, kind)) {
        P->cur++;
        return 1;
    }
    return 0;
}

static void p_advance(tn_parse_t *P)
{
    if (p_peek(P, 0)->kind != TN_TOK_EOF) P->cur++;
}

static void p_err(tn_parse_t *P, uint16_t eid, const char *msg)
{
    if (P->num_errors >= BC_MAX_ERRORS) return;
    bc_error_t *e = &P->errors[P->num_errors++];
    const tn_tok_t *t = p_peek(P, 0);
    e->eid = eid;
    e->loc.line = t->line;
    e->loc.col  = t->col;
    snprintf(e->msg, sizeof(e->msg), "%s", msg);
}

static int p_expect(tn_parse_t *P, int kind, const char *msg)
{
    if (p_eat(P, kind)) return 1;
    p_err(P, 60, msg);
    return 0;
}

/* ---- Node Helpers ----
 * Node allocation, child attachment, and the overflow trick for the
 * rare node that has more than six children. The overflow case is
 * almost entirely block nodes containing many statements, and the
 * extra_kids pool catches them without inflating the inline storage
 * for the common case. */

static uint32_t p_alloc(tn_parse_t *P, int kind)
{
    if (P->num_nodes >= TN_MAX_NODES) return 0;
    uint32_t idx = P->num_nodes++;
    tn_node_t *n = &P->nodes[idx];
    n->kind     = (uint8_t)kind;
    n->num_kids = 0;
    n->flags    = 0;
    n->tok_off  = P->cur;
    n->tok_len  = 0;
    for (int i = 0; i < TN_NODE_INLINE_KIDS; i++) n->kids[i] = 0;
    return idx;
}

static void p_finish(tn_parse_t *P, uint32_t node_idx)
{
    if (node_idx >= P->num_nodes) return;
    tn_node_t *n = &P->nodes[node_idx];
    if (P->cur > n->tok_off) n->tok_len = P->cur - n->tok_off;
}

/* Push a child onto the scratch stack. Used while a parser function
 * is collecting its children, before the node is committed. */

static void p_push_kid(tn_parse_t *P, uint32_t kid_idx)
{
    if (P->kid_scratch_top >= TN_MAX_KID_SCRATCH) {
        p_err(P, 61, "too many AST children in scratch");
        return;
    }
    P->kid_scratch[P->kid_scratch_top++] = kid_idx;
}

/* Commit the scratch range [scratch_base, kid_scratch_top) as the
 * children of node_idx, either into the inline kids array for small
 * nodes or into a contiguous extra_kids range for nodes that have
 * grown past the inline limit. The scratch stack is rewound to
 * scratch_base afterwards so the caller's own scratch state is
 * untouched. This is the central trick that fixes the interleaving
 * bug the incremental add_kid implementation suffered from: by the
 * time we are writing this node's children into extra_kids, every
 * nested node that needed extra_kids has already done its writes
 * and bumped num_extra past them. */

static void p_set_kids(tn_parse_t *P, uint32_t node_idx,
                       uint32_t scratch_base)
{
    if (node_idx >= P->num_nodes) {
        P->kid_scratch_top = scratch_base;
        return;
    }
    tn_node_t *n = &P->nodes[node_idx];
    uint32_t n_kids = P->kid_scratch_top - scratch_base;

    if (n_kids <= TN_NODE_INLINE_KIDS) {
        n->num_kids = (uint8_t)n_kids;
        for (uint32_t i = 0; i < n_kids; i++) {
            n->kids[i] = P->kid_scratch[scratch_base + i];
        }
    } else {
        if (P->num_extra + n_kids > TN_MAX_EXTRA_KIDS) {
            p_err(P, 61, "too many AST children in pool");
            n->num_kids = 0;
            P->kid_scratch_top = scratch_base;
            return;
        }
        uint32_t start = P->num_extra;
        for (uint32_t i = 0; i < n_kids; i++) {
            P->extra_kids[P->num_extra++] = P->kid_scratch[scratch_base + i];
        }
        n->num_kids = TN_NODE_KIDS_OVERFLOW;
        n->kids[0]  = start;
        n->kids[1]  = n_kids;
    }

    P->kid_scratch_top = scratch_base;
}

/* Read child i (regardless of inline vs overflow). Used by ast dump. */

static uint32_t tn_node_kid(const tn_parse_t *P, uint32_t node_idx, uint32_t i)
{
    const tn_node_t *n = &P->nodes[node_idx];
    if (n->num_kids == TN_NODE_KIDS_OVERFLOW) {
        return P->extra_kids[n->kids[0] + i];
    }
    return n->kids[i];
}

static uint32_t tn_node_nkids(const tn_node_t *n)
{
    return (n->num_kids == TN_NODE_KIDS_OVERFLOW)
           ? n->kids[1]
           : n->num_kids;
}

/* ---- NEWLINE Drain ----
 * Python's tokenizer produces NEWLINE tokens between most logical
 * lines, and the parser does not care about them once it has
 * decided which statement it is parsing. The helper below eats any
 * run of NEWLINEs in one go, which is the cheap way to avoid
 * scattering NEWLINE checks through every grammar function. */

static void p_skip_newlines(tn_parse_t *P)
{
    while (p_at(P, TN_TOK_NEWLINE)) P->cur++;
}

/* ---- Augmented Assignment Lookup ----
 * Maps a token kind onto the augmented-assignment subcode used as
 * the flags field of a TN_NK_AUG_ASSIGN node, or returns -1 if the
 * token is not an augmented assignment. The parser consults this
 * after it has consumed a target expression and is trying to decide
 * whether what follows is `=`, `op=`, or nothing in particular. */

static int p_aug_kind(int tok)
{
    switch (tok) {
    case TN_TOK_AADD:    return TN_AUG_ADD;
    case TN_TOK_ASUB:    return TN_AUG_SUB;
    case TN_TOK_AMUL:    return TN_AUG_MUL;
    case TN_TOK_ADIV:    return TN_AUG_DIV;
    case TN_TOK_AFDIV:   return TN_AUG_FDIV;
    case TN_TOK_AMOD:    return TN_AUG_MOD;
    case TN_TOK_APOW:    return TN_AUG_POW;
    case TN_TOK_AAND:    return TN_AUG_AND;
    case TN_TOK_AOR:     return TN_AUG_OR;
    case TN_TOK_AXOR:    return TN_AUG_XOR;
    case TN_TOK_ASHL:    return TN_AUG_SHL;
    case TN_TOK_ASHR:    return TN_AUG_SHR;
    case TN_TOK_AMATMUL: return TN_AUG_MATMUL;
    default:             return -1;
    }
}

/* ---- Real Expression Parser ----
 * Precedence-climbing recursive descent that follows Python's
 * operator precedence table from the language reference. Each
 * function handles one level of binding strength and recurses into
 * the next-higher level for its operands. The wrapping public
 * entry points are p_expr (allows top-level tuple folding via
 * trailing commas), p_expr_no_tuple (single expression, no comma
 * folding), and p_expr_no_in (single expression that stops at the
 * `in` keyword, used by for-loop targets where `in` is the
 * statement-level separator).
 *
 * Sitting two scope: full operator ladder, function calls with
 * positional and keyword arguments, attribute access, subscripts
 * including slicing, parenthesised expressions and tuples, list
 * displays, conditional expressions. Things we knowingly skip and
 * leave for future sittings: comprehensions, lambdas, walrus,
 * await, yield, starred unpacking targets, set and dict displays
 * beyond the empty literal. None of these appear in mainstream
 * Triton kernels and the rare kernel that uses one can be greeted
 * with a polite diagnostic when the day comes. */

static uint32_t p_expr(tn_parse_t *P);
static uint32_t p_expr_no_tuple(tn_parse_t *P);
static uint32_t p_expr_no_in(tn_parse_t *P);
static uint32_t p_ifexpr(tn_parse_t *P);
static uint32_t p_or(tn_parse_t *P);
static uint32_t p_and(tn_parse_t *P);
static uint32_t p_not(tn_parse_t *P);
static uint32_t p_cmp(tn_parse_t *P);
static uint32_t p_bor(tn_parse_t *P);
static uint32_t p_bxor(tn_parse_t *P);
static uint32_t p_band(tn_parse_t *P);
static uint32_t p_shift(tn_parse_t *P);
static uint32_t p_addsub(tn_parse_t *P);
static uint32_t p_muldiv(tn_parse_t *P);
static uint32_t p_unary(tn_parse_t *P);
static uint32_t p_pow(tn_parse_t *P);
static uint32_t p_postfix(tn_parse_t *P);
static uint32_t p_atom(tn_parse_t *P);

/* Helper: build a binary node from two operands and an op code. */

static uint32_t p_binop(tn_parse_t *P, uint32_t lhs, int op, uint32_t rhs,
                        uint32_t start_tok, int kind)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, kind);
    P->nodes[node].tok_off = start_tok;
    P->nodes[node].flags   = (uint16_t)op;
    p_push_kid(P, lhs);
    p_push_kid(P, rhs);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* Token-to-compare mapping. */

static int p_tok_to_cmp(tn_parse_t *P, int *consume_extra)
{
    *consume_extra = 0;
    int k = p_peek(P, 0)->kind;
    switch (k) {
    case TN_TOK_LT:        return TN_CMP_LT;
    case TN_TOK_LE:        return TN_CMP_LE;
    case TN_TOK_GT:        return TN_CMP_GT;
    case TN_TOK_GE:        return TN_CMP_GE;
    case TN_TOK_EQ:        return TN_CMP_EQ;
    case TN_TOK_NE:        return TN_CMP_NE;
    case TN_TOK_KW_IN:     return TN_CMP_IN;
    case TN_TOK_KW_IS:
        /* `is not` is two tokens. */
        if (p_peek(P, 1)->kind == TN_TOK_KW_NOT) {
            *consume_extra = 1;
            return TN_CMP_ISNOT;
        }
        return TN_CMP_IS;
    case TN_TOK_KW_NOT:
        /* `not in` is two tokens. */
        if (p_peek(P, 1)->kind == TN_TOK_KW_IN) {
            *consume_extra = 1;
            return TN_CMP_NOTIN;
        }
        return -1;
    default:
        return -1;
    }
}

/* ---- Top: Tuple Folding ----
 * The top-level expression call lets a comma at the same level fold
 * its left and right operands into a tuple. This is what makes
 * `a, b = c, d` legal as an assignment and `return a, b` produce a
 * tuple-valued return. Sub-expressions inside parens, brackets, or
 * a function call argument list use the no-tuple variant so commas
 * there separate their respective constructs instead. */

static uint32_t p_expr(tn_parse_t *P)
{
    uint32_t first = p_expr_no_tuple(P);
    if (!p_at(P, TN_TOK_COMMA)) return first;

    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_TUPLE);
    P->nodes[node].tok_off = P->nodes[first].tok_off;
    p_push_kid(P, first);
    while (p_eat(P, TN_TOK_COMMA)) {
        /* Trailing comma is allowed: stop if the next token starts
         * something that is plainly not an expression. */
        int k = p_peek(P, 0)->kind;
        if (k == TN_TOK_NEWLINE || k == TN_TOK_SEMI || k == TN_TOK_COLON ||
            k == TN_TOK_ASSIGN || k == TN_TOK_RPAREN || k == TN_TOK_RBRACK ||
            k == TN_TOK_RBRACE || k == TN_TOK_EOF)
            break;
        p_push_kid(P, p_expr_no_tuple(P));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_expr_no_tuple(tn_parse_t *P)
{
    return p_ifexpr(P);
}

static uint32_t p_expr_no_in(tn_parse_t *P)
{
    /* For-loop targets stop at the IN keyword. Routing through p_or
     * or any level at or above the comparison level would let p_cmp
     * happily eat the `in` as a comparison operator, leaving the
     * `for` parser without its expected separator. So we bypass the
     * comparison level entirely and start the ladder at p_bor, which
     * is the level immediately below it. For-loop targets in
     * Triton kernels are names, names with subscripts, or tuples
     * thereof, which never contain comparisons anyway. We also fold
     * top-level commas into a tuple so `for i, j in ...` works. */
    uint32_t first = p_bor(P);
    if (!p_at(P, TN_TOK_COMMA)) return first;

    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_TUPLE);
    P->nodes[node].tok_off = P->nodes[first].tok_off;
    p_push_kid(P, first);
    while (p_eat(P, TN_TOK_COMMA)) {
        if (p_at(P, TN_TOK_KW_IN)) break;
        p_push_kid(P, p_bor(P));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Conditional Expression ----
 * x if cond else y. Right-associative. The if/else keywords are
 * already known to the lexer. */

static uint32_t p_ifexpr(tn_parse_t *P)
{
    uint32_t body = p_or(P);
    if (!p_at(P, TN_TOK_KW_IF)) return body;

    uint32_t sb = P->kid_scratch_top;
    uint32_t start = P->nodes[body].tok_off;
    p_advance(P);  /* consume 'if' */
    uint32_t cond = p_or(P);
    p_expect(P, TN_TOK_KW_ELSE, "expected 'else' in conditional expression");
    uint32_t orelse = p_ifexpr(P);  /* right-associative */

    uint32_t node = p_alloc(P, TN_NK_IFEXPR);
    P->nodes[node].tok_off = start;
    p_push_kid(P, body);
    p_push_kid(P, cond);
    p_push_kid(P, orelse);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Boolean Operators ----
 * Left-associative `or` and `and`. We fold a chain of the same
 * operator into a single BoolOp with N operands, which is what
 * Python's own AST does and matches how short-circuit semantics
 * would naturally lower. */

static uint32_t p_or(tn_parse_t *P)
{
    uint32_t left = p_and(P);
    if (!p_at(P, TN_TOK_KW_OR)) return left;

    uint32_t sb = P->kid_scratch_top;
    uint32_t start = P->nodes[left].tok_off;
    p_push_kid(P, left);
    while (p_eat(P, TN_TOK_KW_OR)) {
        p_push_kid(P, p_and(P));
    }
    uint32_t node = p_alloc(P, TN_NK_BOOLOP);
    P->nodes[node].tok_off = start;
    P->nodes[node].flags   = TN_LOP_OR;
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_and(tn_parse_t *P)
{
    uint32_t left = p_not(P);
    if (!p_at(P, TN_TOK_KW_AND)) return left;

    uint32_t sb = P->kid_scratch_top;
    uint32_t start = P->nodes[left].tok_off;
    p_push_kid(P, left);
    while (p_eat(P, TN_TOK_KW_AND)) {
        p_push_kid(P, p_not(P));
    }
    uint32_t node = p_alloc(P, TN_NK_BOOLOP);
    P->nodes[node].tok_off = start;
    P->nodes[node].flags   = TN_LOP_AND;
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Boolean Not ----
 * Unary, right-associative. Note that this level sits BETWEEN `and`
 * and `cmp` in Python's precedence table, which is the only reason
 * `not a < b` parses as `not (a < b)` rather than `(not a) < b`. */

static uint32_t p_not(tn_parse_t *P)
{
    if (p_at(P, TN_TOK_KW_NOT)) {
        uint32_t sb = P->kid_scratch_top;
        uint32_t start = P->cur;
        p_advance(P);
        uint32_t operand = p_not(P);
        uint32_t node = p_alloc(P, TN_NK_UNOP);
        P->nodes[node].tok_off = start;
        P->nodes[node].flags   = TN_UOP_NOT;
        p_push_kid(P, operand);
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }
    return p_cmp(P);
}

/* ---- Comparisons ----
 * Python comparison chains (a < b < c) bind left-to-right but have
 * implicit AND semantics that we are not yet honouring. For sitting
 * two we accept a single comparison and treat any second one in a
 * chain as a fresh comparison on the previous result, which is
 * wrong semantically but at least parses to a well-formed tree. A
 * future sitting can lower chains to their proper N-ary form. */

static uint32_t p_cmp(tn_parse_t *P)
{
    uint32_t left = p_bor(P);
    int extra = 0;
    int op = p_tok_to_cmp(P, &extra);
    if (op < 0) return left;

    uint32_t start = P->nodes[left].tok_off;
    p_advance(P);
    if (extra) p_advance(P);
    uint32_t right = p_bor(P);
    return p_binop(P, left, op, right, start, TN_NK_COMPARE);
}

/* ---- Bitwise / Shift / Arithmetic ----
 * All left-associative, all parse a chain of operands separated by
 * their own operator class. The pattern below is repeated four
 * times because the operator sets differ and inlining is clearer
 * than a parameterised helper that has to look up tokens. */

static uint32_t p_bor(tn_parse_t *P)
{
    uint32_t left = p_bxor(P);
    while (p_at(P, TN_TOK_PIPE)) {
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_bxor(P);
        left = p_binop(P, left, TN_BOP_OR, right, start, TN_NK_BINOP);
    }
    return left;
}

static uint32_t p_bxor(tn_parse_t *P)
{
    uint32_t left = p_band(P);
    while (p_at(P, TN_TOK_CARET)) {
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_band(P);
        left = p_binop(P, left, TN_BOP_XOR, right, start, TN_NK_BINOP);
    }
    return left;
}

static uint32_t p_band(tn_parse_t *P)
{
    uint32_t left = p_shift(P);
    while (p_at(P, TN_TOK_AMP)) {
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_shift(P);
        left = p_binop(P, left, TN_BOP_AND, right, start, TN_NK_BINOP);
    }
    return left;
}

static uint32_t p_shift(tn_parse_t *P)
{
    uint32_t left = p_addsub(P);
    while (p_at(P, TN_TOK_SHL) || p_at(P, TN_TOK_SHR)) {
        int k = p_peek(P, 0)->kind;
        int op = (k == TN_TOK_SHL) ? TN_BOP_SHL : TN_BOP_SHR;
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_addsub(P);
        left = p_binop(P, left, op, right, start, TN_NK_BINOP);
    }
    return left;
}

static uint32_t p_addsub(tn_parse_t *P)
{
    uint32_t left = p_muldiv(P);
    while (p_at(P, TN_TOK_PLUS) || p_at(P, TN_TOK_MINUS)) {
        int k = p_peek(P, 0)->kind;
        int op = (k == TN_TOK_PLUS) ? TN_BOP_ADD : TN_BOP_SUB;
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_muldiv(P);
        left = p_binop(P, left, op, right, start, TN_NK_BINOP);
    }
    return left;
}

static uint32_t p_muldiv(tn_parse_t *P)
{
    uint32_t left = p_unary(P);
    for (;;) {
        int k = p_peek(P, 0)->kind;
        int op;
        switch (k) {
        case TN_TOK_STAR:    op = TN_BOP_MUL;    break;
        case TN_TOK_SLASH:   op = TN_BOP_DIV;    break;
        case TN_TOK_FSLASH:  op = TN_BOP_FDIV;   break;
        case TN_TOK_PERCENT: op = TN_BOP_MOD;    break;
        case TN_TOK_AT:      op = TN_BOP_MATMUL; break;
        default: return left;
        }
        uint32_t start = P->nodes[left].tok_off;
        p_advance(P);
        uint32_t right = p_unary(P);
        left = p_binop(P, left, op, right, start, TN_NK_BINOP);
    }
}

/* ---- Unary +x -x ~x ----
 * Right-associative. Stacking is rare in real code but legal, so
 * we recurse into ourselves to handle it. */

static uint32_t p_unary(tn_parse_t *P)
{
    int k = p_peek(P, 0)->kind;
    int op;
    switch (k) {
    case TN_TOK_PLUS:  op = TN_UOP_POS; break;
    case TN_TOK_MINUS: op = TN_UOP_NEG; break;
    case TN_TOK_TILDE: op = TN_UOP_INV; break;
    default: return p_pow(P);
    }
    uint32_t sb = P->kid_scratch_top;
    uint32_t start = P->cur;
    p_advance(P);
    uint32_t operand = p_unary(P);
    uint32_t node = p_alloc(P, TN_NK_UNOP);
    P->nodes[node].tok_off = start;
    P->nodes[node].flags   = (uint16_t)op;
    p_push_kid(P, operand);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Power ----
 * Right-associative: a ** b ** c parses as a ** (b ** c). The
 * idiosyncratic bit is that the LEFT operand of ** comes from a
 * unary-or-postfix expression (one above), but the RIGHT operand
 * comes from unary (one below), which lets `2 ** -3` work. */

static uint32_t p_pow(tn_parse_t *P)
{
    uint32_t left = p_postfix(P);
    if (!p_at(P, TN_TOK_DSTAR)) return left;
    uint32_t start = P->nodes[left].tok_off;
    p_advance(P);
    uint32_t right = p_unary(P);
    return p_binop(P, left, TN_BOP_POW, right, start, TN_NK_BINOP);
}

/* ---- Postfix: . [ ] ( ) ----
 * Attribute reference, subscription (including slicing), and call.
 * All left-associative, all chain after the atom. */

static uint32_t p_call_args(tn_parse_t *P, uint32_t callee, uint32_t start);
static uint32_t p_subscript(tn_parse_t *P, uint32_t base, uint32_t start);

static uint32_t p_postfix(tn_parse_t *P)
{
    uint32_t left = p_atom(P);
    for (;;) {
        int k = p_peek(P, 0)->kind;
        if (k == TN_TOK_DOT) {
            uint32_t sb = P->kid_scratch_top;
            uint32_t start = P->nodes[left].tok_off;
            p_advance(P);
            if (!p_at(P, TN_TOK_IDENT)) {
                p_err(P, 70, "expected attribute name after '.'");
                return left;
            }
            uint32_t attr_tok = P->cur;
            p_advance(P);
            uint32_t node = p_alloc(P, TN_NK_ATTR);
            P->nodes[node].tok_off = start;
            P->nodes[node].flags   = (uint16_t)(attr_tok - start);
            p_push_kid(P, left);
            p_set_kids(P, node, sb);
            p_finish(P, node);
            left = node;
        } else if (k == TN_TOK_LPAREN) {
            uint32_t start = P->nodes[left].tok_off;
            left = p_call_args(P, left, start);
        } else if (k == TN_TOK_LBRACK) {
            uint32_t start = P->nodes[left].tok_off;
            left = p_subscript(P, left, start);
        } else {
            return left;
        }
    }
}

/* Call arguments handle the mix of positional and keyword (name=value)
 * forms. We detect a keyword arg by looking ahead one token: if the
 * current is an IDENT and the next is ASSIGN, we treat it as
 * keyword. Otherwise it is a positional expression. */

static uint32_t p_call_args(tn_parse_t *P, uint32_t callee, uint32_t start)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_CALL);
    P->nodes[node].tok_off = start;
    p_push_kid(P, callee);

    p_expect(P, TN_TOK_LPAREN, "expected '(' to begin call");
    if (!p_at(P, TN_TOK_RPAREN)) {
        for (;;) {
            if (p_at(P, TN_TOK_IDENT) &&
                p_peek(P, 1)->kind == TN_TOK_ASSIGN) {
                uint32_t ksb = P->kid_scratch_top;
                uint32_t kw_start = P->cur;
                uint32_t name_tok = P->cur;
                p_advance(P);  /* name */
                p_advance(P);  /* = */
                uint32_t value = p_expr_no_tuple(P);
                uint32_t kw = p_alloc(P, TN_NK_KEYWORD);
                P->nodes[kw].tok_off = kw_start;
                P->nodes[kw].flags   = (uint16_t)(name_tok - kw_start);
                p_push_kid(P, value);
                p_set_kids(P, kw, ksb);
                p_finish(P, kw);
                p_push_kid(P, kw);
            } else {
                p_push_kid(P, p_expr_no_tuple(P));
            }
            if (!p_eat(P, TN_TOK_COMMA)) break;
            if (p_at(P, TN_TOK_RPAREN)) break;
        }
    }
    p_expect(P, TN_TOK_RPAREN, "expected ')' to end call");
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* Subscript handles both single-index `a[i]` and slicing `a[i:j:k]`,
 * plus comma-separated tuples inside brackets `a[i, j]` which Triton
 * uses extensively (offs_m[:, None] and friends). We parse each
 * comma-separated subscript element as either a slice or a regular
 * expression, then attach them all as children of SUBSCRIPT. */

static uint32_t p_slice_or_expr(tn_parse_t *P)
{
    uint32_t start = P->cur;
    int k = p_peek(P, 0)->kind;
    uint32_t lo = 0;
    int has_lo = 0;

    /* Optional lower bound. */
    if (k != TN_TOK_COLON && k != TN_TOK_COMMA &&
        k != TN_TOK_RBRACK) {
        lo = p_expr_no_tuple(P);
        has_lo = 1;
    }
    /* If no colon follows, this is a plain expression subscript. */
    if (!p_at(P, TN_TOK_COLON)) {
        return has_lo ? lo : p_atom(P);
    }

    uint32_t sb = P->kid_scratch_top;
    p_advance(P);  /* first ':' */
    uint32_t hi = 0;
    int has_hi = 0;
    k = p_peek(P, 0)->kind;
    if (k != TN_TOK_COLON && k != TN_TOK_COMMA && k != TN_TOK_RBRACK) {
        hi = p_expr_no_tuple(P);
        has_hi = 1;
    }
    uint32_t step = 0;
    int has_step = 0;
    if (p_eat(P, TN_TOK_COLON)) {
        k = p_peek(P, 0)->kind;
        if (k != TN_TOK_COMMA && k != TN_TOK_RBRACK) {
            step = p_expr_no_tuple(P);
            has_step = 1;
        }
    }

    uint32_t node = p_alloc(P, TN_NK_SLICE);
    P->nodes[node].tok_off = start;
    /* Encode the present-component bitmask in flags so consumers can
     * tell which of the three slots are actually populated. */
    P->nodes[node].flags = (uint16_t)((has_lo ? 1 : 0) |
                                       (has_hi ? 2 : 0) |
                                       (has_step ? 4 : 0));
    if (has_lo) p_push_kid(P, lo);
    if (has_hi) p_push_kid(P, hi);
    if (has_step) p_push_kid(P, step);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_subscript(tn_parse_t *P, uint32_t base, uint32_t start)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_SUBSCRIPT);
    P->nodes[node].tok_off = start;
    p_push_kid(P, base);
    p_expect(P, TN_TOK_LBRACK, "expected '[' to begin subscript");
    if (!p_at(P, TN_TOK_RBRACK)) {
        for (;;) {
            p_push_kid(P, p_slice_or_expr(P));
            if (!p_eat(P, TN_TOK_COMMA)) break;
            if (p_at(P, TN_TOK_RBRACK)) break;
        }
    }
    p_expect(P, TN_TOK_RBRACK, "expected ']' to end subscript");
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Atom ----
 * The base case of the precedence ladder: names, literals, paren
 * groups, tuples, list displays, and the bare names True / False /
 * None which the lexer hands us as their own keyword tokens. */

static uint32_t p_atom(tn_parse_t *P)
{
    int k = p_peek(P, 0)->kind;
    uint32_t start = P->cur;

    if (k == TN_TOK_IDENT) {
        uint32_t node = p_alloc(P, TN_NK_NAME);
        P->nodes[node].tok_off = start;
        p_advance(P);
        p_finish(P, node);
        return node;
    }
    if (k == TN_TOK_INT || k == TN_TOK_FLOAT || k == TN_TOK_STRING) {
        uint32_t node = p_alloc(P, TN_NK_LITERAL);
        P->nodes[node].tok_off = start;
        P->nodes[node].flags   = (uint16_t)(k == TN_TOK_INT   ? TN_LIT_INT   :
                                            k == TN_TOK_FLOAT ? TN_LIT_FLOAT :
                                                                TN_LIT_STRING);
        p_advance(P);
        p_finish(P, node);
        /* Adjacent string literals concatenate at the lexical level
         * in Python. We accept the common form of two adjacent
         * strings producing a single literal node, which lets
         * docstring-ish multi-piece strings parse. */
        while (p_at(P, TN_TOK_STRING)) {
            p_advance(P);
            P->nodes[node].tok_len = P->cur - P->nodes[node].tok_off;
        }
        return node;
    }
    if (k == TN_TOK_KW_NONE || k == TN_TOK_KW_TRUE || k == TN_TOK_KW_FALSE) {
        uint32_t node = p_alloc(P, TN_NK_LITERAL);
        P->nodes[node].tok_off = start;
        P->nodes[node].flags   = (uint16_t)(k == TN_TOK_KW_NONE  ? TN_LIT_NONE  :
                                            k == TN_TOK_KW_TRUE  ? TN_LIT_TRUE  :
                                                                   TN_LIT_FALSE);
        p_advance(P);
        p_finish(P, node);
        return node;
    }
    if (k == TN_TOK_LPAREN) {
        p_advance(P);
        if (p_eat(P, TN_TOK_RPAREN)) {
            /* Empty tuple `()`. */
            uint32_t node = p_alloc(P, TN_NK_TUPLE);
            P->nodes[node].tok_off = start;
            p_finish(P, node);
            return node;
        }
        uint32_t first = p_expr_no_tuple(P);
        if (p_at(P, TN_TOK_COMMA)) {
            uint32_t sb = P->kid_scratch_top;
            uint32_t node = p_alloc(P, TN_NK_TUPLE);
            P->nodes[node].tok_off = start;
            p_push_kid(P, first);
            while (p_eat(P, TN_TOK_COMMA)) {
                if (p_at(P, TN_TOK_RPAREN)) break;
                p_push_kid(P, p_expr_no_tuple(P));
            }
            p_set_kids(P, node, sb);
            p_expect(P, TN_TOK_RPAREN, "expected ')' to end tuple");
            p_finish(P, node);
            return node;
        }
        p_expect(P, TN_TOK_RPAREN, "expected ')' to end parenthesised expression");
        return first;
    }
    if (k == TN_TOK_LBRACK) {
        uint32_t sb = P->kid_scratch_top;
        uint32_t node = p_alloc(P, TN_NK_LIST);
        P->nodes[node].tok_off = start;
        p_advance(P);
        if (!p_at(P, TN_TOK_RBRACK)) {
            for (;;) {
                p_push_kid(P, p_expr_no_tuple(P));
                if (!p_eat(P, TN_TOK_COMMA)) break;
                if (p_at(P, TN_TOK_RBRACK)) break;
            }
        }
        p_expect(P, TN_TOK_RBRACK, "expected ']' to end list display");
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }

    /* Anything else is something this sitting does not yet handle.
     * Produce an EXPR_SPAN as a graceful fallback so the parser
     * continues making progress and a diagnostic is emitted. */
    p_err(P, 71, "unrecognised expression");
    uint32_t node = p_alloc(P, TN_NK_EXPR_SPAN);
    P->nodes[node].tok_off = start;
    if (!p_at(P, TN_TOK_NEWLINE) && !p_at(P, TN_TOK_EOF))
        p_advance(P);
    p_finish(P, node);
    return node;
}

/* p_expr_span is kept as a compatibility shim: anywhere the old
 * statement-level code asked for an opaque span, the new
 * tuple-folding p_expr is the right replacement. The two extra
 * flags it took (allow_comma, allow_in) are now expressed by
 * choosing between p_expr, p_expr_no_tuple, and p_expr_no_in. */

static uint32_t p_expr_span(tn_parse_t *P, int allow_comma, int allow_in)
{
    if (!allow_in)   return p_expr_no_in(P);
    if (!allow_comma) return p_expr_no_tuple(P);
    return p_expr(P);
}

/* ---- Dotted Name ----
 * A run of identifiers connected by dots, used by import statements
 * and decorators. We record the first and last token and leave the
 * interior structure flat, because nothing downstream actually needs
 * to walk the dots one by one. */

static uint32_t p_dotted_name(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_DOTTED_NAME);
    if (!p_at(P, TN_TOK_IDENT)) {
        p_err(P, 62, "expected identifier");
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }
    p_advance(P);
    while (p_at(P, TN_TOK_DOT)) {
        p_advance(P);
        if (!p_at(P, TN_TOK_IDENT)) {
            p_err(P, 63, "expected identifier after '.'");
            break;
        }
        p_advance(P);
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Suite ----
 * A statement block as Python understands it: a NEWLINE, then an
 * INDENT, then one or more statements, then a DEDENT. The simple
 * single-line form (`if cond: pass`) is not supported in sitting
 * one because Triton kernels almost never use it and supporting it
 * properly requires the full statement grammar to be reachable in
 * two distinct contexts. */

static uint32_t p_stmt(tn_parse_t *P);

static uint32_t p_suite(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_BLOCK);
    p_skip_newlines(P);
    if (!p_expect(P, TN_TOK_INDENT, "expected indented block")) {
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }
    uint32_t guard = 0;
    while (!p_at(P, TN_TOK_DEDENT) && !p_at(P, TN_TOK_EOF) &&
           guard < TN_MAX_NODES) {
        uint32_t s = p_stmt(P);
        if (s != 0) p_push_kid(P, s);
        p_skip_newlines(P);
        guard++;
    }
    p_eat(P, TN_TOK_DEDENT);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Compound Statements ---- */

static uint32_t p_if_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_IF);
    p_advance(P);  /* consume 'if' */
    p_push_kid(P, p_expr_span(P, 1, 1));
    p_expect(P, TN_TOK_COLON, "expected ':' after if condition");
    p_push_kid(P, p_suite(P));

    while (p_at(P, TN_TOK_KW_ELIF)) {
        p_advance(P);
        p_push_kid(P, p_expr_span(P, 1, 1));
        p_expect(P, TN_TOK_COLON, "expected ':' after elif condition");
        p_push_kid(P, p_suite(P));
    }
    if (p_at(P, TN_TOK_KW_ELSE)) {
        p_advance(P);
        p_expect(P, TN_TOK_COLON, "expected ':' after else");
        p_push_kid(P, p_suite(P));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_for_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_FOR);
    p_advance(P);  /* consume 'for' */
    p_push_kid(P, p_expr_span(P, 1, 0));   /* stop at 'in' */
    p_expect(P, TN_TOK_KW_IN, "expected 'in' in for statement");
    p_push_kid(P, p_expr_span(P, 1, 1));
    p_expect(P, TN_TOK_COLON, "expected ':' after for header");
    p_push_kid(P, p_suite(P));
    if (p_at(P, TN_TOK_KW_ELSE)) {
        p_advance(P);
        p_expect(P, TN_TOK_COLON, "expected ':' after else");
        p_push_kid(P, p_suite(P));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_while_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_WHILE);
    p_advance(P);  /* consume 'while' */
    p_push_kid(P, p_expr_span(P, 1, 1));
    p_expect(P, TN_TOK_COLON, "expected ':' after while condition");
    p_push_kid(P, p_suite(P));
    if (p_at(P, TN_TOK_KW_ELSE)) {
        p_advance(P);
        p_expect(P, TN_TOK_COLON, "expected ':' after else");
        p_push_kid(P, p_suite(P));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Function Definition ----
 * The full Python form is `def NAME ( params ) [ -> ret ] : suite`,
 * where each parameter is `NAME [: anno] [= default]`. Triton
 * grants us the simplification that kernels are top-level functions
 * and that the only decorator we will see in practice is
 * @triton.jit. We still parse arbitrary decorators because the
 * grammar gets cleaner that way and the cost is a single dotted
 * name plus an optional parenthesised argument span. */

static uint32_t p_param(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_PARAM);

    /* Skip any leading '*' or '**' for var-positional and var-keyword
     * parameters. We do not produce a separate AST shape for them in
     * sitting one; they are rare in Triton kernels and the parser
     * just records them as part of the param's token range. */
    if (p_at(P, TN_TOK_STAR) || p_at(P, TN_TOK_DSTAR)) p_advance(P);

    if (!p_at(P, TN_TOK_IDENT)) {
        p_err(P, 64, "expected parameter name");
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }
    p_advance(P);

    /* Optional type annotation: `: expr`. */
    if (p_at(P, TN_TOK_COLON)) {
        p_advance(P);
        p_push_kid(P, p_expr_span(P, 0, 1));
    }

    /* Optional default value: `= expr`. */
    if (p_at(P, TN_TOK_ASSIGN)) {
        p_advance(P);
        p_push_kid(P, p_expr_span(P, 0, 1));
    }

    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_funcdef(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_FUNCDEF);
    p_advance(P);  /* consume 'def' */

    if (!p_expect(P, TN_TOK_IDENT, "expected function name")) {
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }

    p_expect(P, TN_TOK_LPAREN, "expected '(' after function name");
    if (!p_at(P, TN_TOK_RPAREN)) {
        for (;;) {
            p_push_kid(P, p_param(P));
            if (!p_eat(P, TN_TOK_COMMA)) break;
            /* Allow trailing comma. */
            if (p_at(P, TN_TOK_RPAREN)) break;
        }
    }
    p_expect(P, TN_TOK_RPAREN, "expected ')' after parameter list");

    /* Optional return annotation: `-> expr`. */
    if (p_at(P, TN_TOK_ARROW)) {
        p_advance(P);
        p_push_kid(P, p_expr_span(P, 1, 1));
    }

    p_expect(P, TN_TOK_COLON, "expected ':' after function header");
    p_push_kid(P, p_suite(P));
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_decorator(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_DECORATOR);
    p_advance(P);  /* consume '@' */
    p_push_kid(P, p_dotted_name(P));

    /* Optional call args. We parse them as an opaque expression
     * span between the parens so the actual argument list waits
     * its turn until the real expression parser shows up. */
    if (p_at(P, TN_TOK_LPAREN)) {
        p_advance(P);
        if (!p_at(P, TN_TOK_RPAREN)) {
            p_push_kid(P, p_expr_span(P, 1, 1));
        }
        p_expect(P, TN_TOK_RPAREN, "expected ')' to close decorator args");
    }
    p_expect(P, TN_TOK_NEWLINE, "expected newline after decorator");
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Import Statements ----
 * Two flavours: the bare `import X [as Y]` and the qualified
 * `from X import Y, Z` form. Both parse into the same broad shape
 * with the dotted name on one side and the imported names on the
 * other, and we keep them as separate node kinds because sema is
 * going to want to treat them differently. */

static uint32_t p_import_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_IMPORT);
    p_advance(P);  /* consume 'import' */
    for (;;) {
        p_push_kid(P, p_dotted_name(P));
        if (p_eat(P, TN_TOK_KW_AS)) {
            if (!p_expect(P, TN_TOK_IDENT,
                          "expected identifier after 'as'")) break;
        }
        if (!p_eat(P, TN_TOK_COMMA)) break;
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_from_import_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_IMPORT_FROM);
    p_advance(P);  /* consume 'from' */
    p_push_kid(P, p_dotted_name(P));
    p_expect(P, TN_TOK_KW_IMPORT, "expected 'import' in from statement");
    if (p_eat(P, TN_TOK_STAR)) {
        /* `from X import *`. Nothing more to capture. */
    } else {
        for (;;) {
            if (!p_at(P, TN_TOK_IDENT)) {
                p_err(P, 65, "expected identifier after 'import'");
                break;
            }
            p_advance(P);
            if (p_eat(P, TN_TOK_KW_AS)) {
                if (!p_expect(P, TN_TOK_IDENT,
                              "expected identifier after 'as'")) break;
            }
            if (!p_eat(P, TN_TOK_COMMA)) break;
        }
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Simple Statements ----
 * Assignment vs expression-statement is disambiguated by looking at
 * what follows the first expression: an `=` makes it an assignment,
 * an augmented-assign token makes it AUG_ASSIGN, and anything else
 * (newline, semi) means the expression was a statement in its own
 * right. We rely on the fact that p_expr_span stops at any
 * assignment token. */

static uint32_t p_assign_or_expr_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t start = P->cur;
    uint32_t left = p_expr_span(P, 1, 1);

    if (p_at(P, TN_TOK_ASSIGN)) {
        uint32_t node = p_alloc(P, TN_NK_ASSIGN);
        P->nodes[node].tok_off = start;
        p_advance(P);
        uint32_t value = p_expr_span(P, 1, 1);
        p_push_kid(P, left);
        p_push_kid(P, value);
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }

    int aug = p_aug_kind(p_peek(P, 0)->kind);
    if (aug >= 0) {
        uint32_t node = p_alloc(P, TN_NK_AUG_ASSIGN);
        P->nodes[node].tok_off = start;
        P->nodes[node].flags   = (uint16_t)aug;
        p_advance(P);
        uint32_t value = p_expr_span(P, 1, 1);
        p_push_kid(P, left);
        p_push_kid(P, value);
        p_set_kids(P, node, sb);
        p_finish(P, node);
        return node;
    }

    /* Plain expression statement. */
    uint32_t node = p_alloc(P, TN_NK_EXPR_STMT);
    P->nodes[node].tok_off = start;
    p_push_kid(P, left);
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

static uint32_t p_return_stmt(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t node = p_alloc(P, TN_NK_RETURN);
    p_advance(P);  /* consume 'return' */
    int k = p_peek(P, 0)->kind;
    if (k != TN_TOK_NEWLINE && k != TN_TOK_SEMI && k != TN_TOK_EOF) {
        p_push_kid(P, p_expr_span(P, 1, 1));
    }
    p_set_kids(P, node, sb);
    p_finish(P, node);
    return node;
}

/* ---- Top-Level Statement Dispatcher ----
 * Decides what kind of statement we are looking at and delegates to
 * the appropriate grammar function. Recovery on error advances at
 * least one token and skips to the next NEWLINE so we do not get
 * stuck on the same problem twice. */

static uint32_t p_stmt(tn_parse_t *P)
{
    int k = p_peek(P, 0)->kind;
    switch (k) {
    case TN_TOK_AT:           return p_decorator(P);
    case TN_TOK_KW_DEF:       return p_funcdef(P);
    case TN_TOK_KW_IF:        return p_if_stmt(P);
    case TN_TOK_KW_FOR:       return p_for_stmt(P);
    case TN_TOK_KW_WHILE:     return p_while_stmt(P);
    case TN_TOK_KW_RETURN:    return p_return_stmt(P);
    case TN_TOK_KW_PASS:      { uint32_t n = p_alloc(P, TN_NK_PASS);
                                p_advance(P); p_finish(P, n); return n; }
    case TN_TOK_KW_BREAK:     { uint32_t n = p_alloc(P, TN_NK_BREAK);
                                p_advance(P); p_finish(P, n); return n; }
    case TN_TOK_KW_CONTINUE:  { uint32_t n = p_alloc(P, TN_NK_CONTINUE);
                                p_advance(P); p_finish(P, n); return n; }
    case TN_TOK_KW_IMPORT:    return p_import_stmt(P);
    case TN_TOK_KW_FROM:      return p_from_import_stmt(P);
    case TN_TOK_NEWLINE:      p_advance(P); return 0;
    default:                  return p_assign_or_expr_stmt(P);
    }
}

/* ---- Public Entry Points ---- */

void tn_parse_init(tn_parse_t *P, const tn_lex_t *L)
{
    memset(P, 0, sizeof(*P));
    P->lex = L;
    /* Reserve node index 0 as a sentinel "no node" so that zero
     * initialisation of a parent's kids[] does not accidentally
     * point at a real child. */
    uint32_t sentinel = p_alloc(P, TN_NK_PASS);
    (void)sentinel;
}

int tn_parse(tn_parse_t *P)
{
    uint32_t sb = P->kid_scratch_top;
    uint32_t mod = p_alloc(P, TN_NK_MODULE);
    P->root = mod;
    p_skip_newlines(P);
    uint32_t guard = 0;
    while (!p_at(P, TN_TOK_EOF) && guard < TN_MAX_NODES) {
        uint32_t s = p_stmt(P);
        if (s != 0) p_push_kid(P, s);
        p_skip_newlines(P);
        guard++;
    }
    p_set_kids(P, mod, sb);
    p_finish(P, mod);
    return (P->num_errors > 0) ? BC_ERR_TRITON : BC_OK;
}

/* ---- AST Dump ----
 * Indented walk of the tree, one node per line. The point of this
 * is the human eye, not any downstream consumer, so the format is
 * deliberately readable rather than machine parseable. */

static const char *tn_nk_names[TN_NK_COUNT] = {
    "Module", "Import", "ImportFrom", "FuncDef", "Decorator",
    "Param", "Block", "Assign", "AugAssign", "ExprStmt",
    "If", "For", "While", "Return", "Pass", "Break", "Continue",
    "DottedName", "ExprSpan",
    "Name", "Literal", "Tuple", "BinOp", "UnOp", "BoolOp",
    "Compare", "Call", "Keyword", "Attr", "Subscript", "Slice",
    "IfExpr", "List"
};

static const char *tn_bop_names[TN_BOP_COUNT] = {
    "+", "-", "*", "/", "//", "%", "**", "@",
    "&", "|", "^", "<<", ">>"
};

static const char *tn_uop_names[TN_UOP_COUNT] = {
    "+", "-", "~", "not"
};

static const char *tn_lop_names[TN_LOP_COUNT] = {
    "and", "or"
};

static const char *tn_cmp_names[TN_CMP_COUNT] = {
    "<", "<=", ">", ">=", "==", "!=",
    "is", "is not", "in", "not in"
};

static const char *tn_lit_names[TN_LIT_COUNT] = {
    "int", "float", "string", "None", "True", "False"
};

const char *tn_nk_name(int kind)
{
    if (kind < 0 || kind >= TN_NK_COUNT) return "?";
    return tn_nk_names[kind];
}

static void tn_ast_dump_node(const tn_parse_t *P, uint32_t idx,
                             int depth, FILE *out)
{
    if (idx >= P->num_nodes) return;
    const tn_node_t *n = &P->nodes[idx];
    for (int i = 0; i < depth; i++) fprintf(out, "  ");
    fprintf(out, "%s", tn_nk_name(n->kind));

    /* For nodes that wrap a single named identity (function defs,
     * params, modules, names), print the first identifier token so
     * the dump is not just a row of bracket-faced node kinds. */
    if (n->kind == TN_NK_FUNCDEF || n->kind == TN_NK_PARAM ||
        n->kind == TN_NK_DOTTED_NAME || n->kind == TN_NK_NAME) {
        uint32_t tok = n->tok_off;
        while (tok < P->lex->num_tokens &&
               P->lex->tokens[tok].kind != TN_TOK_IDENT) tok++;
        if (tok < P->lex->num_tokens) {
            const tn_tok_t *t = &P->lex->tokens[tok];
            char text[64];
            tn_tok_text(P->lex, t, text, sizeof(text));
            fprintf(out, " '%s'", text);
        }
    }

    if (n->kind == TN_NK_LITERAL) {
        if (n->flags < TN_LIT_COUNT)
            fprintf(out, " %s", tn_lit_names[n->flags]);
        if (n->tok_off < P->lex->num_tokens) {
            char text[64];
            tn_tok_text(P->lex, &P->lex->tokens[n->tok_off],
                        text, sizeof(text));
            fprintf(out, " '%s'", text);
        }
    }

    if (n->kind == TN_NK_BINOP && n->flags < TN_BOP_COUNT) {
        fprintf(out, " %s", tn_bop_names[n->flags]);
    }
    if (n->kind == TN_NK_UNOP && n->flags < TN_UOP_COUNT) {
        fprintf(out, " %s", tn_uop_names[n->flags]);
    }
    if (n->kind == TN_NK_BOOLOP && n->flags < TN_LOP_COUNT) {
        fprintf(out, " %s", tn_lop_names[n->flags]);
    }
    if (n->kind == TN_NK_COMPARE && n->flags < TN_CMP_COUNT) {
        fprintf(out, " %s", tn_cmp_names[n->flags]);
    }

    if (n->kind == TN_NK_ATTR) {
        /* tok_off + flags points at the attribute name. */
        uint32_t attr_tok = n->tok_off + n->flags;
        if (attr_tok < P->lex->num_tokens) {
            char text[64];
            tn_tok_text(P->lex, &P->lex->tokens[attr_tok],
                        text, sizeof(text));
            fprintf(out, " .%s", text);
        }
    }

    if (n->kind == TN_NK_KEYWORD) {
        uint32_t name_tok = n->tok_off + n->flags;
        if (name_tok < P->lex->num_tokens) {
            char text[64];
            tn_tok_text(P->lex, &P->lex->tokens[name_tok],
                        text, sizeof(text));
            fprintf(out, " %s=", text);
        }
    }

    if (n->kind == TN_NK_SLICE) {
        fprintf(out, " %s%s%s",
                (n->flags & 1) ? "lo " : "",
                (n->flags & 2) ? "hi " : "",
                (n->flags & 4) ? "step" : "");
    }

    if (n->kind == TN_NK_EXPR_SPAN) {
        fprintf(out, " [%u..%u]", n->tok_off, n->tok_off + n->tok_len);
    }

    if (n->kind == TN_NK_AUG_ASSIGN) {
        static const char *aug_names[TN_AUG_COUNT] = {
            "+=", "-=", "*=", "/=", "//=", "%=", "**=",
            "&=", "|=", "^=", "<<=", ">>=", "@="
        };
        if (n->flags < TN_AUG_COUNT)
            fprintf(out, " %s", aug_names[n->flags]);
    }

    fprintf(out, "\n");

    uint32_t nk = tn_node_nkids(n);
    for (uint32_t i = 0; i < nk; i++) {
        tn_ast_dump_node(P, tn_node_kid(P, idx, i), depth + 1, out);
    }
}

void tn_ast_dump(const tn_parse_t *P, FILE *out)
{
    if (P->root == 0) {
        fprintf(out, "(empty)\n");
        return;
    }
    tn_ast_dump_node(P, P->root, 0, out);
    fprintf(out, "\n%u nodes, %d error(s)\n",
            P->num_nodes, P->num_errors);
}
