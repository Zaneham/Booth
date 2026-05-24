/* Triton frontend semantic analysis, sitting one.
 *
 * The pass walks the AST the parser produces and, for every name
 * reference it can find, decides what that name actually means: a
 * parameter, a local variable, an imported module alias, or a tl.*
 * intrinsic. Anything that does not bind gets a polite diagnostic
 * and the pass keeps going, so a kernel with three unbound names
 * yields three messages rather than just the first.
 *
 * What we deliberately defer to a later sitting: tile shape
 * inference, constexpr value propagation, type checking of
 * intrinsic call sites, error reporting for arity mismatches, and
 * the full module-resolution rules Python uses (we get away with a
 * single hardcoded set of import aliases because real Triton
 * kernels only use a handful). */

#include "triton.h"

#include <string.h>
#include <stdio.h>

/* ---- Intrinsic Table ----
 * The hardcoded list of tl.* names sema knows about. New
 * intrinsics arrive as one entry per line. The lookup is a linear
 * scan, which is fine when the table is a hundred entries and the
 * kernel touches a dozen of them; if it ever gets unweildy a
 * sorted-binary-search lookup is the obvious upgrade. */

typedef struct {
    const char *name;
    int         id;
    int         is_type;    /* 1 if this is a dtype, 0 for a function */
} tn_intrinsic_entry_t;

static const tn_intrinsic_entry_t tn_intrinsics[] = {
    {"program_id",      TN_TLI_PROGRAM_ID,    0},
    {"num_programs",    TN_TLI_NUM_PROGRAMS,  0},
    {"load",            TN_TLI_LOAD,          0},
    {"store",           TN_TLI_STORE,         0},
    {"make_block_ptr",  TN_TLI_MAKE_BLOCK_PTR,0},
    {"advance",         TN_TLI_ADVANCE,       0},
    {"arange",          TN_TLI_ARANGE,        0},
    {"zeros",           TN_TLI_ZEROS,         0},
    {"zeros_like",      TN_TLI_ZEROS_LIKE,    0},
    {"full",            TN_TLI_FULL,          0},
    {"broadcast_to",    TN_TLI_BROADCAST_TO,  0},
    {"reshape",         TN_TLI_RESHAPE,       0},
    {"trans",           TN_TLI_TRANS,         0},
    {"where",           TN_TLI_WHERE,         0},
    {"sum",             TN_TLI_SUM,           0},
    {"max",             TN_TLI_MAX,           0},
    {"min",             TN_TLI_MIN,           0},
    {"argmax",          TN_TLI_ARGMAX,        0},
    {"argmin",          TN_TLI_ARGMIN,        0},
    {"dot",             TN_TLI_DOT,           0},
    {"exp",             TN_TLI_EXP,           0},
    {"exp2",            TN_TLI_EXP2,          0},
    {"log",             TN_TLI_LOG,           0},
    {"log2",            TN_TLI_LOG2,          0},
    {"sin",             TN_TLI_SIN,           0},
    {"cos",             TN_TLI_COS,           0},
    {"tan",             TN_TLI_TAN,           0},
    {"tanh",            TN_TLI_TANH,          0},
    {"sqrt",            TN_TLI_SQRT,          0},
    {"rsqrt",           TN_TLI_RSQRT,         0},
    {"abs",             TN_TLI_ABS,           0},
    {"floor",           TN_TLI_FLOOR,         0},
    {"ceil",            TN_TLI_CEIL,          0},
    {"erf",             TN_TLI_ERF,           0},
    {"maximum",         TN_TLI_MAXIMUM,       0},
    {"minimum",         TN_TLI_MINIMUM,       0},
    {"fdiv",            TN_TLI_FDIV,          0},
    {"cdiv",            TN_TLI_CDIV,          0},
    {"static_assert",   TN_TLI_STATIC_ASSERT, 0},
    {"static_print",    TN_TLI_STATIC_PRINT,  0},
    {"device_assert",   TN_TLI_DEVICE_ASSERT, 0},
    {"device_print",    TN_TLI_DEVICE_PRINT,  0},

    /* The constexpr marker is technically a type but we list it
     * here because it shows up wherever a parameter annotation
     * would. */
    {"constexpr",       TN_TLI_CONSTEXPR,     1},

    /* Numeric types. */
    {"float16",         TN_TLI_FLOAT16,       1},
    {"float32",         TN_TLI_FLOAT32,       1},
    {"float64",         TN_TLI_FLOAT64,       1},
    {"bfloat16",        TN_TLI_BFLOAT16,      1},
    {"int1",            TN_TLI_INT1,          1},
    {"int8",            TN_TLI_INT8,          1},
    {"int16",           TN_TLI_INT16,         1},
    {"int32",           TN_TLI_INT32,         1},
    {"int64",           TN_TLI_INT64,         1},
    {"uint8",           TN_TLI_UINT8,         1},
    {"uint16",          TN_TLI_UINT16,        1},
    {"uint32",          TN_TLI_UINT32,        1},
    {"uint64",          TN_TLI_UINT64,        1},
    {NULL, 0, 0}
};

static int tn_intrinsic_lookup(const char *name, uint32_t len, int *is_type)
{
    for (int i = 0; tn_intrinsics[i].name != NULL; i++) {
        const char *n = tn_intrinsics[i].name;
        uint32_t nl = (uint32_t)strlen(n);
        if (nl == len && memcmp(n, name, len) == 0) {
            if (is_type) *is_type = tn_intrinsics[i].is_type;
            return tn_intrinsics[i].id;
        }
    }
    return TN_TLI_NONE;
}

/* Intrinsic name table, for the AST dump. The simple linear scan
 * over a known-small set is good enough; nothing here is hot. */

static const char *tn_intrinsic_names[TN_TLI_COUNT] = {0};

static void tn_intrinsic_names_init(void)
{
    static int built = 0;
    if (built) return;
    built = 1;
    for (int i = 0; tn_intrinsics[i].name != NULL; i++) {
        int id = tn_intrinsics[i].id;
        if (id > 0 && id < TN_TLI_COUNT)
            tn_intrinsic_names[id] = tn_intrinsics[i].name;
    }
}

const char *tn_intrinsic_name(int id)
{
    tn_intrinsic_names_init();
    if (id <= 0 || id >= TN_TLI_COUNT) return "?";
    const char *n = tn_intrinsic_names[id];
    return n ? n : "?";
}

/* ---- Symbol Kind Names (for diagnostics) ---- */

static const char *tn_sym_kind_names[TN_SYM_KIND_COUNT] = {
    "unbound", "param", "local", "loopvar", "module",
    "intrinsic", "type"
};

const char *tn_sym_kind_name(int kind)
{
    if (kind < 0 || kind >= TN_SYM_KIND_COUNT) return "?";
    return tn_sym_kind_names[kind];
}

/* ---- Scope Stack ----
 * Each scope tracks the range of symbol indices it owns. Lookup
 * walks scopes from innermost (top) to outermost (bottom). When we
 * pop a scope, the symbols belonging to it stay in the array but
 * become unreachable through lookup; this is cheaper than shrinking
 * and we never run out of slots in practice. */

static void s_push_scope(tn_sema_t *S)
{
    if (S->num_scopes >= TN_MAX_SCOPES) return;
    tn_scope_t *sc = &S->scopes[S->num_scopes++];
    sc->start = S->num_syms;
    sc->end   = S->num_syms;
}

static void s_pop_scope(tn_sema_t *S)
{
    if (S->num_scopes == 0) return;
    /* Roll back the symbol cursor so the outer scope can reuse the
     * indices. This keeps symbol storage bounded over the whole
     * program even for kernels with many local variables. */
    S->num_syms = S->scopes[S->num_scopes - 1].start;
    S->num_scopes--;
}

static void s_bind(tn_sema_t *S, uint32_t name_off, uint16_t name_len,
                   int kind, uint32_t aux, uint32_t decl_node)
{
    if (S->num_syms >= TN_MAX_SYMBOLS) return;
    if (S->num_scopes == 0) return;
    tn_sym_t *sym = &S->syms[S->num_syms++];
    sym->name_off  = name_off;
    sym->name_len  = name_len;
    sym->kind      = (uint8_t)kind;
    sym->pad       = 0;
    sym->aux       = aux;
    sym->decl_node = decl_node;
    S->scopes[S->num_scopes - 1].end = S->num_syms;
}

/* Python builtins that real Triton kernels legitimately reach for,
 * even though kernels live in a heavily restricted Python subset.
 * `range` is the obvious one (every for-loop uses it), the type
 * constructors come up in conversions, and the small numeric
 * helpers appear here and there. Treated as a fixed table the
 * lookup consults before falling back to user-declared symbols. */

typedef struct { const char *name; uint8_t kind; uint32_t aux; }
                                                            tn_builtin_t;

static const tn_builtin_t tn_py_builtins[] = {
    {"range", TN_SYM_INTRINSIC, 0},   /* aux=0 means "builtin range" */
    {"len",   TN_SYM_INTRINSIC, 0},
    {"min",   TN_SYM_INTRINSIC, 0},
    {"max",   TN_SYM_INTRINSIC, 0},
    {"abs",   TN_SYM_INTRINSIC, 0},
    {"int",   TN_SYM_TYPE,      0},
    {"float", TN_SYM_TYPE,      0},
    {"bool",  TN_SYM_TYPE,      0},
    {"tuple", TN_SYM_TYPE,      0},
    {"list",  TN_SYM_TYPE,      0},
    {NULL, 0, 0}
};

static int s_lookup_builtin(const char *name, uint32_t len,
                            int *kind, uint32_t *aux)
{
    for (int i = 0; tn_py_builtins[i].name != NULL; i++) {
        const char *n = tn_py_builtins[i].name;
        uint32_t nl = (uint32_t)strlen(n);
        if (nl == len && memcmp(n, name, len) == 0) {
            *kind = tn_py_builtins[i].kind;
            *aux  = tn_py_builtins[i].aux;
            return 1;
        }
    }
    return 0;
}

static const tn_sym_t *s_lookup(const tn_sema_t *S,
                                const char *src, uint32_t off, uint16_t len)
{
    /* Search innermost scope first. */
    for (int sc = S->num_scopes - 1; sc >= 0; sc--) {
        const tn_scope_t *scope = &S->scopes[sc];
        for (int i = (int)scope->end - 1; i >= (int)scope->start; i--) {
            const tn_sym_t *sym = &S->syms[i];
            if (sym->name_len != len) continue;
            if (memcmp(src + sym->name_off, src + off, len) == 0)
                return sym;
        }
    }
    return NULL;
}

/* ---- Diagnostic Helper ---- */

static void s_err(tn_sema_t *S, uint16_t eid, const tn_tok_t *t,
                  const char *msg)
{
    if (S->num_errors >= BC_MAX_ERRORS) return;
    bc_error_t *e = &S->errors[S->num_errors++];
    e->eid = eid;
    e->loc.line = t ? t->line : 0;
    e->loc.col  = t ? t->col  : 0;
    snprintf(e->msg, sizeof(e->msg), "%s", msg);
}

/* ---- Per-Node Annotation ---- */

static void s_annotate(tn_sema_t *S, uint32_t node_idx,
                       int kind, uint32_t aux)
{
    if (node_idx >= TN_MAX_NODES) return;
    S->node_sym_kind[node_idx] = (uint8_t)kind;
    S->node_sym_aux[node_idx]  = aux;
}

/* ---- AST Walk ----
 * The walk handles each node kind it cares about and recurses into
 * children for everything else. Statements that introduce bindings
 * (Assign, For target, function parameters) call s_bind to add
 * symbols to the current scope; expressions that reference names
 * (Name, Attr-on-module) call s_lookup or the intrinsic table. */

static void s_walk(tn_sema_t *S, uint32_t node_idx);
static int  s_const_int(const tn_sema_t *S, uint32_t node_idx);

/* Read a child index regardless of inline/overflow storage. Mirrors
 * the parser's helper of the same shape. */

static uint32_t s_kid(const tn_sema_t *S, uint32_t node_idx, uint32_t i)
{
    const tn_node_t *n = &S->parser->nodes[node_idx];
    if (n->num_kids == TN_NODE_KIDS_OVERFLOW) {
        return S->parser->extra_kids[n->kids[0] + i];
    }
    return n->kids[i];
}

static uint32_t s_nkids(const tn_node_t *n)
{
    return (n->num_kids == TN_NODE_KIDS_OVERFLOW)
           ? n->kids[1]
           : n->num_kids;
}

/* Return a pointer to the source identifier text for a Name (or
 * other naming node) by walking from tok_off until we find an
 * IDENT token. */

static const tn_tok_t *s_name_tok(const tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx >= P->num_nodes) return NULL;
    const tn_node_t *n = &P->nodes[node_idx];
    uint32_t t = n->tok_off;
    while (t < P->lex->num_tokens &&
           P->lex->tokens[t].kind != TN_TOK_IDENT) {
        t++;
    }
    if (t >= P->lex->num_tokens) return NULL;
    return &P->lex->tokens[t];
}

/* Walk an Import statement, binding each module alias into the
 * current scope. For `import triton`, the alias is `triton`. For
 * `import triton.language as tl`, the alias is `tl` and the parser
 * stored it as the next IDENT after the `as` keyword. */

static void s_handle_import(tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    /* Walk the token range looking for IDENT runs separated by `.`
     * and watching for `as` keywords. For each dotted-name we hit,
     * decide whether the binding is the leading IDENT or the
     * trailing alias after `as`. */
    uint32_t t = n->tok_off;
    uint32_t end = n->tok_off + n->tok_len;
    while (t < end) {
        if (t >= P->lex->num_tokens) break;
        int k = P->lex->tokens[t].kind;
        if (k != TN_TOK_IDENT) { t++; continue; }

        /* Scan the dotted name. */
        uint32_t first_ident = t;
        char modbuf[64];
        uint32_t mbp = 0;
        while (t < end && t < P->lex->num_tokens) {
            int kk = P->lex->tokens[t].kind;
            if (kk == TN_TOK_IDENT) {
                const tn_tok_t *tk = &P->lex->tokens[t];
                for (uint32_t c = 0; c < tk->len && mbp + 1 < sizeof(modbuf); c++)
                    modbuf[mbp++] = P->lex->src[tk->off + c];
                t++;
                continue;
            }
            if (kk == TN_TOK_DOT) {
                if (mbp + 1 < sizeof(modbuf)) modbuf[mbp++] = '.';
                t++;
                continue;
            }
            break;
        }
        modbuf[mbp] = '\0';

        int mod_id = TN_MOD_NONE;
        if (strcmp(modbuf, "triton") == 0)           mod_id = TN_MOD_TRITON;
        else if (strcmp(modbuf, "triton.language") == 0) mod_id = TN_MOD_TL;
        else if (strcmp(modbuf, "math") == 0)        mod_id = TN_MOD_MATH;

        /* Skip ws-equivalent tokens and check for `as alias`. */
        const tn_tok_t *alias_tok = NULL;
        if (t < end && t < P->lex->num_tokens &&
            P->lex->tokens[t].kind == TN_TOK_KW_AS) {
            t++;
            if (t < end && t < P->lex->num_tokens &&
                P->lex->tokens[t].kind == TN_TOK_IDENT) {
                alias_tok = &P->lex->tokens[t];
                t++;
            }
        }

        /* Bind. The name we bind is either the leading IDENT of the
         * dotted name (top-level alias) or the explicit `as` alias. */
        const tn_tok_t *bind_tok = alias_tok
                                   ? alias_tok
                                   : &P->lex->tokens[first_ident];
        s_bind(S, bind_tok->off, (uint16_t)bind_tok->len,
               TN_SYM_MODULE, (uint32_t)mod_id, node_idx);

        /* Eat trailing comma if any. */
        if (t < end && t < P->lex->num_tokens &&
            P->lex->tokens[t].kind == TN_TOK_COMMA) t++;
    }
}

/* For an Attr node, decide if it is a tl.* (or math.*) intrinsic
 * reference and annotate accordingly. The check is: kids[0] is a
 * Name that resolved to a TL or MATH module symbol, and the
 * attribute name (token n->tok_off + n->flags) is in the
 * intrinsic table. */

static void s_resolve_attr(tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    if (n->kind != TN_NK_ATTR) return;

    uint32_t base = (n->num_kids > 0) ? n->kids[0] : 0;
    if (base == 0 || base >= P->num_nodes) return;
    if (S->node_sym_kind[base] != TN_SYM_MODULE) return;
    int mod_id = (int)S->node_sym_aux[base];
    if (mod_id != TN_MOD_TL && mod_id != TN_MOD_MATH) return;

    uint32_t attr_tok = n->tok_off + n->flags;
    if (attr_tok >= P->lex->num_tokens) return;
    const tn_tok_t *t = &P->lex->tokens[attr_tok];
    if (t->kind != TN_TOK_IDENT) return;

    int is_type = 0;
    int id = tn_intrinsic_lookup(P->lex->src + t->off,
                                 t->len, &is_type);
    if (id == TN_TLI_NONE) {
        char nb[64];
        uint32_t nl = t->len < 60 ? t->len : 60;
        memcpy(nb, P->lex->src + t->off, nl);
        nb[nl] = '\0';
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "unknown intrinsic: %s.%s",
                 mod_id == TN_MOD_TL ? "tl" : "math", nb);
        s_err(S, 80, t, msg);
        s_annotate(S, node_idx, TN_SYM_UNBOUND, 0);
        return;
    }
    s_annotate(S, node_idx, is_type ? TN_SYM_TYPE : TN_SYM_INTRINSIC,
               (uint32_t)id);
}

/* For an Assign, the LHS is a target. In sitting one we only handle
 * single-Name targets, which is what real Triton kernels overwhelmingly
 * use. Tuple unpacking (a, b = ...) gets the polite TODO treatment. */

static void s_bind_assign_target(tn_sema_t *S, uint32_t target_idx,
                                  uint32_t assign_node)
{
    const tn_parse_t *P = S->parser;
    if (target_idx >= P->num_nodes) return;
    const tn_node_t *t = &P->nodes[target_idx];

    if (t->kind == TN_NK_NAME) {
        const tn_tok_t *tk = s_name_tok(S, target_idx);
        if (!tk) return;
        /* Bind the local with aux = the assign node that introduced
         * it. The lowering pass uses aux to find the BIR value the
         * RHS produced, so this has to point at the declaring node
         * even on a re-assign. */
        if (!s_lookup(S, P->lex->src, tk->off, (uint16_t)tk->len)) {
            s_bind(S, tk->off, (uint16_t)tk->len,
                   TN_SYM_LOCAL, assign_node, assign_node);
        }
        s_annotate(S, target_idx, TN_SYM_LOCAL, assign_node);
        return;
    }

    if (t->kind == TN_NK_TUPLE) {
        uint32_t nk = s_nkids(t);
        for (uint32_t i = 0; i < nk; i++) {
            s_bind_assign_target(S, s_kid(S, target_idx, i), assign_node);
        }
        return;
    }

    /* Subscript and attribute targets (a[i] = ..., a.b = ...) do not
     * introduce new bindings; the base name should already be in
     * scope. Walk the base for resolution. */
    s_walk(S, target_idx);
}

/* Walk a single AST node, recursing into children and handling the
 * structural roles each node kind plays in name resolution. */

static void s_walk(tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx == 0 || node_idx >= P->num_nodes) return;
    const tn_node_t *n = &P->nodes[node_idx];
    uint32_t nk = s_nkids(n);

    switch (n->kind) {
    case TN_NK_MODULE: {
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;
    }

    case TN_NK_IMPORT:
    case TN_NK_IMPORT_FROM:
        s_handle_import(S, node_idx);
        return;

    case TN_NK_FUNCDEF: {
        s_push_scope(S);
        /* Children are params then Block. Bind params, then walk the
         * body which itself contains the inner statements. */
        for (uint32_t i = 0; i < nk; i++) {
            uint32_t kid = s_kid(S, node_idx, i);
            const tn_node_t *kn = &P->nodes[kid];
            if (kn->kind == TN_NK_PARAM) {
                const tn_tok_t *tk = s_name_tok(S, kid);
                if (tk) {
                    s_bind(S, tk->off, (uint16_t)tk->len,
                           TN_SYM_PARAM, i, kid);
                    s_annotate(S, kid, TN_SYM_PARAM, i);
                }
                uint32_t pkn = s_nkids(kn);
                for (uint32_t j = 0; j < pkn; j++) {
                    s_walk(S, s_kid(S, kid, j));
                }
                /* `: tl.constexpr = <int>` -> stash the value on the
                 * param node so name refs and shape inference can
                 * pick it up. Constexpr-ness comes from any annotation
                 * child whose sema resolved to TN_TLI_CONSTEXPR; the
                 * default is the last child if it's an int literal. */
                int is_constexpr = 0;
                int default_val = -1;
                for (uint32_t j = 0; j < pkn; j++) {
                    uint32_t pk = s_kid(S, kid, j);
                    if (pk >= P->num_nodes) continue;
                    if (S->node_sym_kind[pk] == TN_SYM_TYPE &&
                        S->node_sym_aux[pk] == TN_TLI_CONSTEXPR) {
                        is_constexpr = 1;
                    }
                    /* Walk the annotation subtree too in case
                     * tl.constexpr sits inside an Attr node. */
                    const tn_node_t *pkn_node = &P->nodes[pk];
                    uint32_t cck = s_nkids(pkn_node);
                    for (uint32_t cc = 0; cc < cck; cc++) {
                        uint32_t ckid = s_kid(S, pk, cc);
                        if (ckid < P->num_nodes &&
                            S->node_sym_kind[ckid] == TN_SYM_TYPE &&
                            S->node_sym_aux[ckid] == TN_TLI_CONSTEXPR) {
                            is_constexpr = 1;
                        }
                    }
                    int lv = s_const_int(S, pk);
                    if (lv >= 0) default_val = lv;
                }
                if (is_constexpr && default_val >= 0) {
                    S->node_const_val[kid] = default_val;
                }
            } else {
                s_walk(S, kid);
            }
        }
        s_pop_scope(S);
        return;
    }

    case TN_NK_DECORATOR:
        /* The decorator's dotted-name and any args sit as children;
         * walk them for resolution but do not bind anything. */
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_BLOCK:
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_ASSIGN: {
        /* kids[0] = target, kids[1] = value. Walk value first so the
         * RHS resolves against the OUTER scope (Python rebinds at
         * the next statement boundary, not at the assignment
         * point), then bind the target. */
        uint32_t target = s_kid(S, node_idx, 0);
        uint32_t value  = (nk > 1) ? s_kid(S, node_idx, 1) : 0;
        if (value) s_walk(S, value);
        s_bind_assign_target(S, target, node_idx);
        return;
    }

    case TN_NK_AUG_ASSIGN:
        /* a += b: a must already be bound (Python's rule), so we
         * walk both sides without introducing a new binding. */
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_FOR: {
        /* kids: target, iter, body, optional else. */
        if (nk >= 2) s_walk(S, s_kid(S, node_idx, 1));  /* iter first */
        if (nk >= 1) {
            uint32_t target = s_kid(S, node_idx, 0);
            const tn_node_t *tn = &P->nodes[target];
            if (tn->kind == TN_NK_NAME) {
                const tn_tok_t *tk = s_name_tok(S, target);
                if (tk) {
                    if (!s_lookup(S, P->lex->src, tk->off,
                                  (uint16_t)tk->len)) {
                        s_bind(S, tk->off, (uint16_t)tk->len,
                               TN_SYM_LOOPVAR, node_idx, node_idx);
                    }
                    s_annotate(S, target, TN_SYM_LOOPVAR, node_idx);
                }
            } else if (tn->kind == TN_NK_TUPLE) {
                uint32_t tk = s_nkids(tn);
                for (uint32_t i = 0; i < tk; i++) {
                    uint32_t kid = s_kid(S, target, i);
                    const tn_node_t *kn = &P->nodes[kid];
                    if (kn->kind == TN_NK_NAME) {
                        const tn_tok_t *kt = s_name_tok(S, kid);
                        if (kt) {
                            if (!s_lookup(S, P->lex->src, kt->off,
                                          (uint16_t)kt->len)) {
                                s_bind(S, kt->off, (uint16_t)kt->len,
                                       TN_SYM_LOOPVAR, node_idx, node_idx);
                            }
                            s_annotate(S, kid, TN_SYM_LOOPVAR, node_idx);
                        }
                    }
                }
            }
        }
        for (uint32_t i = 2; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;
    }

    case TN_NK_NAME: {
        const tn_tok_t *tk = s_name_tok(S, node_idx);
        if (!tk) return;
        /* Check user-declared symbols first so a kernel can legitimately
         * shadow a builtin if it wants to. Then fall back to the
         * Python builtin table. */
        const tn_sym_t *sym = s_lookup(S, P->lex->src,
                                       tk->off, (uint16_t)tk->len);
        if (sym) {
            s_annotate(S, node_idx, sym->kind, sym->aux);
            /* Constexpr param default flows through Name references. */
            if (sym->decl_node < TN_MAX_NODES &&
                S->node_const_val[sym->decl_node] >= 0) {
                S->node_const_val[node_idx] =
                    S->node_const_val[sym->decl_node];
            }
            return;
        }
        int bkind;
        uint32_t baux;
        if (s_lookup_builtin(P->lex->src + tk->off, tk->len,
                             &bkind, &baux)) {
            s_annotate(S, node_idx, bkind, baux);
            return;
        }
        char nb[64];
        uint32_t nl = tk->len < 60 ? tk->len : 60;
        memcpy(nb, P->lex->src + tk->off, nl);
        nb[nl] = '\0';
        char msg[128];
        snprintf(msg, sizeof(msg), "unbound name: %s", nb);
        s_err(S, 81, tk, msg);
        s_annotate(S, node_idx, TN_SYM_UNBOUND, 0);
        return;
    }

    case TN_NK_ATTR:
        /* Walk the base first so its resolution is in place before
         * we decide whether this Attr is an intrinsic reference. */
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        s_resolve_attr(S, node_idx);
        return;

    /* Catch-all for the rest: just recurse into children. */
    default:
        for (uint32_t i = 0; i < nk; i++) {
            s_walk(S, s_kid(S, node_idx, i));
        }
        return;
    }
}

/* ---- Tile Shape Inference ----
 * Post-order walk over expression nodes. Separate from the sym-
 * resolution walk because that one is top-down for scoping. Constexpr
 * value propagation is deferred to a later sitting; constexpr params
 * in tile-size contexts yield dynamic dims (-1) for now. */

static tn_shape_t s_scalar(int dtype)
{
    tn_shape_t sh = {0};
    sh.rank  = 0;
    sh.dtype = (uint8_t)dtype;
    return sh;
}

static tn_shape_t s_vec(int dim, int dtype)
{
    tn_shape_t sh = {0};
    sh.rank    = 1;
    sh.dtype   = (uint8_t)dtype;
    sh.dims[0] = dim;
    return sh;
}

static tn_shape_t s_mat(int outer, int inner, int dtype)
{
    tn_shape_t sh = {0};
    sh.rank    = 2;
    sh.dtype   = (uint8_t)dtype;
    sh.dims[0] = outer;
    sh.dims[1] = inner;
    return sh;
}

/* Compile-time integer value, or -1 if dynamic. Resolves int
 * literals directly and follows Name refs to their constexpr param
 * default when one is recorded. */

static int s_const_int(const tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx == 0 || node_idx >= P->num_nodes) return -1;
    if (S->node_const_val[node_idx] >= 0) return S->node_const_val[node_idx];
    const tn_node_t *n = &P->nodes[node_idx];
    if (n->kind != TN_NK_LITERAL || n->flags != TN_LIT_INT) return -1;
    if (n->tok_off >= P->lex->num_tokens) return -1;
    const tn_tok_t *t = &P->lex->tokens[n->tok_off];
    const char *s = P->lex->src + t->off;
    long v = 0;
    int base = 10;
    uint32_t p = 0;
    if (t->len > 2 && s[0] == '0') {
        if (s[1] == 'x' || s[1] == 'X') { base = 16; p = 2; }
        else if (s[1] == 'o' || s[1] == 'O') { base = 8;  p = 2; }
        else if (s[1] == 'b' || s[1] == 'B') { base = 2;  p = 2; }
    }
    for (; p < t->len; p++) {
        char c = s[p];
        if (c == '_') continue;
        int d = -1;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        if (d < 0 || d >= base) return -1;
        v = v * base + d;
        if (v > 0x7FFFFFFF) return -1;          /* dim overflow guard */
    }
    return (int)v;
}

/* Numpy broadcast of a single dim: size-1 expands to the other,
 * unknowns stay unknown, mismatches stay unknown. */

static int s_bcast_dim(int a, int b)
{
    if (a == b)        return a;
    if (a == 1)        return b;
    if (b == 1)        return a;
    if (a == -1)       return -1;
    if (b == -1)       return -1;
    return -1;                                  /* mismatched constants */
}

/* Widening rank for promotion (enum order is not widening order). */

static int s_dtype_rank(int id)
{
    switch (id) {
    case TN_TLI_INT1:    return 1;
    case TN_TLI_INT8:    case TN_TLI_UINT8:    return 2;
    case TN_TLI_INT16:   case TN_TLI_UINT16:   return 3;
    case TN_TLI_INT32:   case TN_TLI_UINT32:   return 4;
    case TN_TLI_INT64:   case TN_TLI_UINT64:   return 5;
    case TN_TLI_FLOAT16: case TN_TLI_BFLOAT16: return 10;
    case TN_TLI_FLOAT32:                       return 11;
    case TN_TLI_FLOAT64:                       return 12;
    default:                                   return 0;
    }
}

static int s_promote_dtype(int a, int b)
{
    if (a == 0)  return b;
    if (b == 0)  return a;
    if (a == b)  return a;
    int ra = s_dtype_rank(a);
    int rb = s_dtype_rank(b);
    if (ra != rb) return ra > rb ? a : b;
    /* Equal rank: bf16 beats f16 (wider exponent), uint wins
     * over signed (lowerer's float-vs-int dispatch is signedness-agnostic). */
    if (a == TN_TLI_BFLOAT16 || b == TN_TLI_BFLOAT16) return TN_TLI_BFLOAT16;
    if (a == TN_TLI_UINT8  || b == TN_TLI_UINT8)  return TN_TLI_UINT8;
    if (a == TN_TLI_UINT16 || b == TN_TLI_UINT16) return TN_TLI_UINT16;
    if (a == TN_TLI_UINT32 || b == TN_TLI_UINT32) return TN_TLI_UINT32;
    if (a == TN_TLI_UINT64 || b == TN_TLI_UINT64) return TN_TLI_UINT64;
    return a;
}

/* Numpy broadcast: align trailing dims, expand size-1, widen dtype. */

static tn_shape_t s_broadcast(tn_shape_t a, tn_shape_t b)
{
    tn_shape_t out = {0};
    int dt = s_promote_dtype(a.dtype, b.dtype);
    out.dtype = (uint8_t)dt;

    if (a.rank == 0 && b.rank == 0) return out;
    if (a.rank == 0) { out = b; out.dtype = (uint8_t)dt; return out; }
    if (b.rank == 0) { out = a; out.dtype = (uint8_t)dt; return out; }

    int rank = a.rank > b.rank ? a.rank : b.rank;
    out.rank = (uint8_t)rank;
    if (rank == 1) {
        out.dims[0] = s_bcast_dim(a.dims[0], b.dims[0]);
    } else {
        int a0 = (a.rank == 2) ? a.dims[0] : 1;
        int a1 = (a.rank == 2) ? a.dims[1] : a.dims[0];
        int b0 = (b.rank == 2) ? b.dims[0] : 1;
        int b1 = (b.rank == 2) ? b.dims[1] : b.dims[0];
        out.dims[0] = s_bcast_dim(a0, b0);
        out.dims[1] = s_bcast_dim(a1, b1);
    }
    return out;
}

static int s_intrinsic_dtype(int id)
{
    switch (id) {
    case TN_TLI_FLOAT16: case TN_TLI_FLOAT32: case TN_TLI_FLOAT64:
    case TN_TLI_BFLOAT16:
    case TN_TLI_INT1:    case TN_TLI_INT8:    case TN_TLI_INT16:
    case TN_TLI_INT32:   case TN_TLI_INT64:
    case TN_TLI_UINT8:   case TN_TLI_UINT16:  case TN_TLI_UINT32:
    case TN_TLI_UINT64:
        return id;
    default:
        return 0;
    }
}

/* dtype= kw arg (tl.float32 binds as TN_SYM_TYPE). */

static int s_call_dtype_arg(const tn_sema_t *S, uint32_t call_idx)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[call_idx];
    uint32_t nk = (n->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? n->kids[1] : n->num_kids;
    for (uint32_t i = 1; i < nk; i++) {
        uint32_t kid = s_kid(S, call_idx, i);
        const tn_node_t *kn = &P->nodes[kid];
        if (kn->kind != TN_NK_KEYWORD) continue;
        const tn_tok_t *kt = NULL;
        uint32_t tk = kn->tok_off;
        while (tk < P->lex->num_tokens &&
               P->lex->tokens[tk].kind != TN_TOK_IDENT) tk++;
        if (tk < P->lex->num_tokens) kt = &P->lex->tokens[tk];
        if (!kt) continue;
        if (kt->len == 5 &&
            memcmp(P->lex->src + kt->off, "dtype", 5) == 0) {
            uint32_t value = s_kid(S, kid, 0);
            if (value >= P->num_nodes) return 0;
            if (S->node_sym_kind[value] != TN_SYM_TYPE) return 0;
            return s_intrinsic_dtype((int)S->node_sym_aux[value]);
        }
    }
    return 0;
}

/* Tile-shape tuple arg (e.g. tl.zeros((M, N), ...)). out_real_rank
 * reports the source-level rank so callers can diagnose rank > 2. */

static int s_call_shape_arg(tn_sema_t *S, uint32_t call_idx,
                            int *out_dims, int *out_real_rank)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[call_idx];
    if (out_real_rank) *out_real_rank = 0;
    if (n->num_kids == 0) return 0;
    uint32_t arg = s_kid(S, call_idx, 1);
    if (arg == 0 || arg >= P->num_nodes) return 0;
    const tn_node_t *an = &P->nodes[arg];

    if (an->kind == TN_NK_LITERAL && an->flags == TN_LIT_INT) {
        int v = s_const_int(S, arg);
        out_dims[0] = v < 0 ? -1 : v;
        out_dims[1] = 0;
        if (out_real_rank) *out_real_rank = 1;
        return 1;
    }
    if (an->kind == TN_NK_NAME) {
        out_dims[0] = -1;
        out_dims[1] = 0;
        if (out_real_rank) *out_real_rank = 1;
        return 1;
    }

    if (an->kind == TN_NK_TUPLE) {
        uint32_t tnk = (an->num_kids == TN_NODE_KIDS_OVERFLOW)
                       ? an->kids[1] : an->num_kids;
        if (out_real_rank) *out_real_rank = (int)tnk;
        if (tnk == 1) {
            int v = s_const_int(S, s_kid(S, arg, 0));
            out_dims[0] = v < 0 ? -1 : v;
            out_dims[1] = 0;
            return 1;
        }
        if (tnk >= 2) {
            int av = s_const_int(S, s_kid(S, arg, 0));
            int bv = s_const_int(S, s_kid(S, arg, 1));
            out_dims[0] = av < 0 ? -1 : av;
            out_dims[1] = bv < 0 ? -1 : bv;
            if (tnk > 2) {
                const tn_tok_t *t = NULL;
                if (an->tok_off < P->lex->num_tokens)
                    t = &P->lex->tokens[an->tok_off];
                s_err(S, 83, t,
                      "rank-3+ tile shapes are not modelled yet "
                      "(only rank-1 and rank-2 tiles are tracked)");
            }
            return 2;
        }
    }
    return 0;
}

/* int drops the axis, slice keeps it, None inserts a size-1 axis. */

enum { S_IX_INT = 0, S_IX_SLICE, S_IX_NONE };

static int s_subscript_ix_kind(const tn_sema_t *S, uint32_t kid)
{
    const tn_parse_t *P = S->parser;
    if (kid >= P->num_nodes) return S_IX_INT;
    const tn_node_t *kn = &P->nodes[kid];
    if (kn->kind == TN_NK_SLICE)   return S_IX_SLICE;
    if (kn->kind == TN_NK_LITERAL && kn->flags == TN_LIT_NONE)
                                   return S_IX_NONE;
    return S_IX_INT;
}

static tn_shape_t s_reshape_subscript(const tn_sema_t *S,
                                      uint32_t node_idx,
                                      tn_shape_t base)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    uint32_t nk = (n->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? n->kids[1] : n->num_kids;
    if (nk < 2) return base;

    int out_dims[4] = {0};
    int out_rank = 0;
    int base_ax = 0;

    for (uint32_t i = 1; i < nk && out_rank < 4; i++) {
        int kind = s_subscript_ix_kind(S, s_kid(S, node_idx, i));
        if (kind == S_IX_NONE) {
            out_dims[out_rank++] = 1;
        } else if (kind == S_IX_SLICE) {
            int dim = (base_ax < base.rank) ? base.dims[base_ax] : -1;
            out_dims[out_rank++] = dim;
            base_ax++;
        } else {
            base_ax++;
        }
    }

    tn_shape_t out = {0};
    out.dtype = base.dtype;
    if (out_rank == 0) return s_scalar(base.dtype);
    if (out_rank == 1) return s_vec(out_dims[0], base.dtype);
    if (out_rank == 2) return s_mat(out_dims[0], out_dims[1], base.dtype);
    /* rank >= 3: truncate to rank-2 (not modelled yet). */
    out.rank = 2;
    out.dims[0] = out_dims[0];
    out.dims[1] = out_dims[1];
    return out;
}

static tn_shape_t s_infer_expr(tn_sema_t *S, uint32_t node_idx);

static void s_set_shape(tn_sema_t *S, uint32_t node_idx, tn_shape_t sh)
{
    if (node_idx >= TN_MAX_NODES) return;
    S->node_shape[node_idx] = sh;
}

/* Shape rule for an intrinsic Call. */

static tn_shape_t s_call_shape(tn_sema_t *S, uint32_t call_idx, int id)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *cn = &P->nodes[call_idx];
    uint32_t nk = (cn->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? cn->kids[1] : cn->num_kids;

    switch (id) {
    case TN_TLI_PROGRAM_ID:
    case TN_TLI_NUM_PROGRAMS:
        return s_scalar(TN_TLI_INT32);

    case TN_TLI_ARANGE: {
        /* (start, stop) -> vec[stop-start] when both literal. */
        int dim = -1;
        if (nk >= 3) {
            uint32_t a = s_kid(S, call_idx, 1);
            uint32_t b = s_kid(S, call_idx, 2);
            int av = s_const_int(S, a);
            int bv = s_const_int(S, b);
            if (av >= 0 && bv >= 0 && bv >= av) dim = bv - av;
        }
        return s_vec(dim, TN_TLI_INT32);
    }

    case TN_TLI_ZEROS:
    case TN_TLI_FULL: {
        int dims[2] = {0, 0};
        int rank = s_call_shape_arg(S, call_idx, dims, NULL);
        int dtype = s_call_dtype_arg(S, call_idx);
        if (rank == 0) return s_scalar(dtype);
        if (rank == 1) return s_vec(dims[0], dtype);
        return s_mat(dims[0], dims[1], dtype);
    }

    case TN_TLI_ZEROS_LIKE: {
        if (nk < 2) return s_scalar(0);
        return s_infer_expr(S, s_kid(S, call_idx, 1));
    }

    case TN_TLI_BROADCAST_TO: {
        int dims[2] = {0, 0};
        int rank = s_call_shape_arg(S, call_idx, dims, NULL);
        int dtype = 0;
        if (nk >= 2) {
            tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
            dtype = base.dtype;
        }
        if (rank == 1) return s_vec(dims[0], dtype);
        if (rank == 2) return s_mat(dims[0], dims[1], dtype);
        return s_scalar(dtype);
    }

    case TN_TLI_RESHAPE: {
        int dims[2] = {0, 0};
        int rank = s_call_shape_arg(S, call_idx, dims, NULL);
        int dtype = 0;
        if (nk >= 2) {
            tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
            dtype = base.dtype;
        }
        if (rank == 1) return s_vec(dims[0], dtype);
        if (rank == 2) return s_mat(dims[0], dims[1], dtype);
        return s_scalar(dtype);
    }

    case TN_TLI_TRANS: {
        if (nk < 2) return s_scalar(0);
        tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
        if (base.rank == 2) {
            return s_mat(base.dims[1], base.dims[0], base.dtype);
        }
        return base;
    }

    case TN_TLI_DOT: {
        /* [M, K] @ [K, N] -> [M, N]; K is the kernel author's
         * problem to get right. */
        if (nk < 3) return s_scalar(0);
        tn_shape_t a = s_infer_expr(S, s_kid(S, call_idx, 1));
        tn_shape_t b = s_infer_expr(S, s_kid(S, call_idx, 2));
        int dtype = s_call_dtype_arg(S, call_idx);
        if (dtype == 0) dtype = s_promote_dtype(a.dtype, b.dtype);
        if (a.rank == 2 && b.rank == 2) {
            return s_mat(a.dims[0], b.dims[1], dtype);
        }
        return s_mat(-1, -1, dtype);
    }

    case TN_TLI_WHERE: {
        if (nk < 4) return s_scalar(0);
        tn_shape_t t = s_infer_expr(S, s_kid(S, call_idx, 2));
        tn_shape_t f = s_infer_expr(S, s_kid(S, call_idx, 3));
        return s_broadcast(t, f);
    }

    case TN_TLI_SUM: case TN_TLI_MAX: case TN_TLI_MIN:
    case TN_TLI_ARGMAX: case TN_TLI_ARGMIN: {
        /* Reductions drop one axis. Resolving an axis= kw statically
         * is for a later sitting; for now rank-2 collapses to vec[?]. */
        if (nk < 2) return s_scalar(0);
        tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
        if (base.rank == 0) return base;
        if (base.rank == 1) return s_scalar(base.dtype);
        /* rank 2 with no axis given: collapse to scalar. With an
         * axis kw the result keeps one dim, dynamic for now. */
        return s_vec(-1, base.dtype);
    }

    case TN_TLI_LOAD: {
        /* Shape follows the pointer expr; pointee dtype defaults to
         * f32 until we have a real pointer-type system. */
        if (nk < 2) return s_scalar(0);
        tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
        if (base.dtype == 0) base.dtype = TN_TLI_FLOAT32;
        return base;
    }

    case TN_TLI_STORE:
        return s_scalar(0);

    case TN_TLI_MAKE_BLOCK_PTR:
    case TN_TLI_ADVANCE:
        return s_scalar(0);

    case TN_TLI_EXP: case TN_TLI_EXP2: case TN_TLI_LOG: case TN_TLI_LOG2:
    case TN_TLI_SIN: case TN_TLI_COS: case TN_TLI_TAN: case TN_TLI_TANH:
    case TN_TLI_SQRT: case TN_TLI_RSQRT: case TN_TLI_ABS:
    case TN_TLI_FLOOR: case TN_TLI_CEIL: case TN_TLI_ERF: {
        if (nk < 2) return s_scalar(0);
        return s_infer_expr(S, s_kid(S, call_idx, 1));
    }

    case TN_TLI_MAXIMUM: case TN_TLI_MINIMUM:
    case TN_TLI_FDIV: {
        if (nk < 3) return s_scalar(0);
        tn_shape_t a = s_infer_expr(S, s_kid(S, call_idx, 1));
        tn_shape_t b = s_infer_expr(S, s_kid(S, call_idx, 2));
        return s_broadcast(a, b);
    }

    case TN_TLI_CDIV:
        return s_scalar(TN_TLI_INT32);

    default:
        return s_scalar(0);
    }
}

static tn_shape_t s_infer_expr(tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx == 0 || node_idx >= P->num_nodes) return s_scalar(0);
    const tn_node_t *n = &P->nodes[node_idx];

    /* Cached: shape rules like DOT and WHERE recurse into args. */
    tn_shape_t cached = S->node_shape[node_idx];
    if (cached.rank != 0 || cached.dtype != 0) return cached;

    tn_shape_t result = s_scalar(0);
    uint32_t nk = (n->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? n->kids[1] : n->num_kids;

    switch (n->kind) {
    case TN_NK_LITERAL: {
        int dt = 0;
        switch (n->flags) {
        case TN_LIT_INT:   dt = TN_TLI_INT32; break;
        case TN_LIT_FLOAT: dt = TN_TLI_FLOAT32; break;
        case TN_LIT_TRUE:  case TN_LIT_FALSE: dt = TN_TLI_INT1; break;
        default: dt = 0; break;
        }
        result = s_scalar(dt);
        break;
    }

    case TN_NK_NAME: {
        int kind = S->node_sym_kind[node_idx];
        if (kind == TN_SYM_LOCAL || kind == TN_SYM_LOOPVAR) {
            /* The declaring node's shape was stashed by s_infer_stmt. */
            uint32_t decl = S->node_sym_aux[node_idx];
            if (decl < TN_MAX_NODES) {
                tn_shape_t ds = S->node_shape[decl];
                if (ds.rank != 0 || ds.dtype != 0) {
                    result = ds;
                    break;
                }
            }
        }
        result = s_scalar(0);
        break;
    }

    case TN_NK_BINOP: {
        if (nk < 2) { result = s_scalar(0); break; }
        tn_shape_t a = s_infer_expr(S, s_kid(S, node_idx, 0));
        tn_shape_t b = s_infer_expr(S, s_kid(S, node_idx, 1));
        result = s_broadcast(a, b);
        break;
    }

    case TN_NK_UNOP: {
        if (nk < 1) { result = s_scalar(0); break; }
        result = s_infer_expr(S, s_kid(S, node_idx, 0));
        break;
    }

    case TN_NK_BOOLOP: {
        if (nk < 2) { result = s_scalar(TN_TLI_INT1); break; }
        tn_shape_t a = s_infer_expr(S, s_kid(S, node_idx, 0));
        tn_shape_t b = s_infer_expr(S, s_kid(S, node_idx, 1));
        result = s_broadcast(a, b);
        result.dtype = TN_TLI_INT1;
        break;
    }

    case TN_NK_COMPARE: {
        if (nk < 2) { result = s_scalar(TN_TLI_INT1); break; }
        tn_shape_t a = s_infer_expr(S, s_kid(S, node_idx, 0));
        tn_shape_t b = s_infer_expr(S, s_kid(S, node_idx, 1));
        result = s_broadcast(a, b);
        result.dtype = TN_TLI_INT1;
        break;
    }

    case TN_NK_CALL: {
        if (nk == 0) { result = s_scalar(0); break; }
        uint32_t callee = s_kid(S, node_idx, 0);
        if (S->node_sym_kind[callee] == TN_SYM_INTRINSIC) {
            result = s_call_shape(S, node_idx,
                                  (int)S->node_sym_aux[callee]);
        } else {
            for (uint32_t i = 1; i < nk; i++) {
                (void)s_infer_expr(S, s_kid(S, node_idx, i));
            }
            result = s_scalar(0);
        }
        break;
    }

    case TN_NK_SUBSCRIPT: {
        if (nk == 0) { result = s_scalar(0); break; }
        tn_shape_t base = s_infer_expr(S, s_kid(S, node_idx, 0));
        result = s_reshape_subscript(S, node_idx, base);
        break;
    }

    case TN_NK_IFEXPR: {
        /* x if cond else y: result is broadcast(x, y). */
        if (nk < 3) { result = s_scalar(0); break; }
        tn_shape_t t = s_infer_expr(S, s_kid(S, node_idx, 0));
        tn_shape_t f = s_infer_expr(S, s_kid(S, node_idx, 2));
        (void)s_infer_expr(S, s_kid(S, node_idx, 1));    /* condition */
        result = s_broadcast(t, f);
        break;
    }

    case TN_NK_ATTR:
        if (S->node_sym_kind[node_idx] == TN_SYM_TYPE) {
            result = s_scalar((int)S->node_sym_aux[node_idx]);
        } else {
            result = s_scalar(0);
        }
        break;

    case TN_NK_TUPLE:
    case TN_NK_LIST:
        for (uint32_t i = 0; i < nk; i++) {
            (void)s_infer_expr(S, s_kid(S, node_idx, i));
        }
        result = s_scalar(0);
        break;

    default:
        result = s_scalar(0);
        break;
    }

    s_set_shape(S, node_idx, result);
    return result;
}

/* Walk statements; the RHS shape of an Assign is stashed on the
 * Assign node so later Name references can pick it up. */

static void s_infer_stmt(tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx == 0 || node_idx >= P->num_nodes) return;
    const tn_node_t *n = &P->nodes[node_idx];
    uint32_t nk = (n->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? n->kids[1] : n->num_kids;

    switch (n->kind) {
    case TN_NK_MODULE:
    case TN_NK_BLOCK:
    case TN_NK_FUNCDEF:
        for (uint32_t i = 0; i < nk; i++) {
            s_infer_stmt(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_ASSIGN:
        if (nk >= 2) {
            tn_shape_t v = s_infer_expr(S, s_kid(S, node_idx, 1));
            s_set_shape(S, node_idx, v);
        }
        return;

    case TN_NK_AUG_ASSIGN:
        for (uint32_t i = 0; i < nk; i++) {
            (void)s_infer_expr(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_EXPR_STMT:
    case TN_NK_RETURN:
        for (uint32_t i = 0; i < nk; i++) {
            (void)s_infer_expr(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_IF:
    case TN_NK_WHILE:
    case TN_NK_FOR:
        for (uint32_t i = 0; i < nk; i++) {
            uint32_t kid = s_kid(S, node_idx, i);
            if (kid == 0 || kid >= P->num_nodes) continue;
            const tn_node_t *kn = &P->nodes[kid];
            if (kn->kind == TN_NK_BLOCK) {
                s_infer_stmt(S, kid);
            } else {
                (void)s_infer_expr(S, kid);
            }
        }
        return;

    default:
        return;
    }
}

/* ---- Public API ---- */

int tn_shape_format(tn_shape_t sh, char *buf, int bufsize)
{
    if (!buf || bufsize <= 0) return 0;
    if (sh.rank == 0) {
        if (sh.dtype == 0) return snprintf(buf, (size_t)bufsize, "scalar");
        return snprintf(buf, (size_t)bufsize, "scalar:%s",
                        tn_intrinsic_name(sh.dtype));
    }
    if (sh.rank == 1) {
        char d[16];
        if (sh.dims[0] < 0) snprintf(d, sizeof(d), "?");
        else                snprintf(d, sizeof(d), "%d", sh.dims[0]);
        if (sh.dtype == 0)
            return snprintf(buf, (size_t)bufsize, "vec[%s]", d);
        return snprintf(buf, (size_t)bufsize, "vec[%s]:%s",
                        d, tn_intrinsic_name(sh.dtype));
    }
    if (sh.rank == 2) {
        char d0[16], d1[16];
        if (sh.dims[0] < 0) snprintf(d0, sizeof(d0), "?");
        else                snprintf(d0, sizeof(d0), "%d", sh.dims[0]);
        if (sh.dims[1] < 0) snprintf(d1, sizeof(d1), "?");
        else                snprintf(d1, sizeof(d1), "%d", sh.dims[1]);
        if (sh.dtype == 0)
            return snprintf(buf, (size_t)bufsize, "mat[%s, %s]", d0, d1);
        return snprintf(buf, (size_t)bufsize, "mat[%s, %s]:%s",
                        d0, d1, tn_intrinsic_name(sh.dtype));
    }
    return snprintf(buf, (size_t)bufsize, "?");
}

void tn_sema_init(tn_sema_t *S, const tn_parse_t *P)
{
    memset(S, 0, sizeof(*S));
    S->parser = P;
    for (uint32_t i = 0; i < TN_MAX_NODES; i++) S->node_const_val[i] = -1;
}

int tn_sema(tn_sema_t *S)
{
    if (!S->parser) return BC_ERR_TRITON;
    s_push_scope(S);

    /* Python builtins are looked up via the tn_py_builtins table at
     * Name resolution time, not pre-seeded into the symbol table.
     * The table-based lookup avoids the awkwardness of fabricating
     * source-buffer offsets for names that do not appear in the
     * input. See s_lookup_builtin above. */

    if (S->parser->root != 0) {
        s_walk(S, S->parser->root);
        s_infer_stmt(S, S->parser->root);
    }

    s_pop_scope(S);
    return (S->num_errors > 0) ? BC_ERR_TRITON : BC_OK;
}

/* ---- AST + Annotation Dump ----
 * Same shape as the parser's tn_ast_dump but adds the resolution
 * info next to every Name and intrinsic-bound Attr. The point is
 * the human reader: a kernel author can run --triton --sema and
 * confirm at a glance that every name they wrote actually bound to
 * what they expected. */

static void s_dump_node(const tn_sema_t *S, uint32_t idx,
                        int depth, FILE *out)
{
    const tn_parse_t *P = S->parser;
    if (idx == 0 || idx >= P->num_nodes) return;
    const tn_node_t *n = &P->nodes[idx];

    for (int i = 0; i < depth; i++) fprintf(out, "  ");
    fprintf(out, "%s", tn_nk_name(n->kind));

    if (n->kind == TN_NK_NAME || n->kind == TN_NK_FUNCDEF ||
        n->kind == TN_NK_PARAM || n->kind == TN_NK_DOTTED_NAME) {
        uint32_t tok = n->tok_off;
        while (tok < P->lex->num_tokens &&
               P->lex->tokens[tok].kind != TN_TOK_IDENT) tok++;
        if (tok < P->lex->num_tokens) {
            char text[64];
            tn_tok_text(P->lex, &P->lex->tokens[tok],
                        text, sizeof(text));
            fprintf(out, " '%s'", text);
        }
    }

    if (n->kind == TN_NK_NAME) {
        int kind = S->node_sym_kind[idx];
        if (kind == TN_SYM_INTRINSIC || kind == TN_SYM_TYPE) {
            fprintf(out, " -> %s(%s)",
                    tn_sym_kind_name(kind),
                    tn_intrinsic_name((int)S->node_sym_aux[idx]));
        } else if (kind == TN_SYM_MODULE) {
            static const char *mod_names[TN_MOD_COUNT] = {
                "none", "triton", "tl", "math"
            };
            uint32_t m = S->node_sym_aux[idx];
            fprintf(out, " -> module(%s)",
                    m < TN_MOD_COUNT ? mod_names[m] : "?");
        } else if (kind != 0) {
            fprintf(out, " -> %s", tn_sym_kind_name(kind));
        }
    }

    if (n->kind == TN_NK_ATTR) {
        uint32_t at = n->tok_off + n->flags;
        if (at < P->lex->num_tokens) {
            char text[64];
            tn_tok_text(P->lex, &P->lex->tokens[at], text, sizeof(text));
            fprintf(out, " .%s", text);
        }
        int kind = S->node_sym_kind[idx];
        if (kind == TN_SYM_INTRINSIC || kind == TN_SYM_TYPE) {
            fprintf(out, " -> %s(%s)",
                    tn_sym_kind_name(kind),
                    tn_intrinsic_name((int)S->node_sym_aux[idx]));
        }
    }

    /* Tile shape, when set. */
    {
        tn_shape_t sh = S->node_shape[idx];
        if (sh.rank != 0 || sh.dtype != 0) {
            char sbuf[64];
            tn_shape_format(sh, sbuf, sizeof(sbuf));
            fprintf(out, " : %s", sbuf);
        }
    }

    fprintf(out, "\n");

    uint32_t nk = s_nkids(n);
    for (uint32_t i = 0; i < nk; i++) {
        s_dump_node(S, s_kid(S, idx, i), depth + 1, out);
    }
}

void tn_sema_dump(const tn_sema_t *S, FILE *out)
{
    if (!S->parser || S->parser->root == 0) {
        fprintf(out, "(empty)\n");
        return;
    }
    s_dump_node(S, S->parser->root, 0, out);
    fprintf(out, "\n%u symbols across %d scope%s, %d error(s)\n",
            S->num_syms, S->num_scopes,
            S->num_scopes == 1 ? "" : "s",
            S->num_errors);
}
