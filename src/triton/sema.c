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
                /* Walk the param's annotation and default for name
                 * resolution within them. */
                uint32_t pkn = s_nkids(kn);
                for (uint32_t j = 0; j < pkn; j++) {
                    s_walk(S, s_kid(S, kid, j));
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
 * A second pass over the AST, post-order over expression nodes.
 * The walk is independent of the sym-resolution walk above because
 * shape inference needs bottom-up flow and the sym walk does top-
 * down scoping; trying to fuse them produces a tangle. Two short
 * walks beats one knotty one.
 *
 * The general principle: every expression node gets a shape, where
 * shape is (rank, dims, dtype). Statements leave the shape at its
 * zero-initialised default. Unknown / unhandled expressions get a
 * scalar shape with dtype 0 (TN_TLI_NONE) so downstream code does
 * not need to worry about uninitialised reads.
 *
 * Constexpr value propagation is deferred. Constexpr parameters in
 * tile-size contexts (tl.arange, tl.zeros) yield dynamic dims (-1)
 * for now. The rank and rough shape carry through, which is enough
 * to dispatch lowering between scalar / vector / matrix variants. */

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

/* Read an integer literal AST node and return its value, or -1 if
 * the node is not a compile-time integer the sema pass can resolve.
 * The only forms recognised in sitting one are literal int tokens;
 * constexpr-parameter propagation is a later concern. */

static int s_const_int(const tn_sema_t *S, uint32_t node_idx)
{
    const tn_parse_t *P = S->parser;
    if (node_idx == 0 || node_idx >= P->num_nodes) return -1;
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

/* Broadcast a single dimension under numpy rules: if either side is
 * 1, the result is the other side; otherwise the dims must agree.
 * A -1 (dynamic) on either side returns -1: the runtime is the
 * authority and we cannot decide statically. */

static int s_bcast_dim(int a, int b)
{
    if (a == b)        return a;
    if (a == 1)        return b;
    if (b == 1)        return a;
    if (a == -1)       return -1;
    if (b == -1)       return -1;
    return -1;                                  /* mismatched constants */
}

static int s_promote_dtype(int a, int b)
{
    /* If either side is float, the result is float of the wider
     * width; otherwise it is the wider integer type. The current
     * sema has no signed/unsigned rank so we keep this simple. */
    if (a == 0) return b;
    if (b == 0) return a;
    if (a == b) return a;
    /* Float wins. */
    int a_float = (a == TN_TLI_FLOAT16 || a == TN_TLI_FLOAT32 ||
                   a == TN_TLI_FLOAT64 || a == TN_TLI_BFLOAT16);
    int b_float = (b == TN_TLI_FLOAT16 || b == TN_TLI_FLOAT32 ||
                   b == TN_TLI_FLOAT64 || b == TN_TLI_BFLOAT16);
    if (a_float && !b_float) return a;
    if (b_float && !a_float) return b;
    /* Same family, pick the higher enum which happens to track the
     * widening order in the intrinsic enum as written. */
    return a > b ? a : b;
}

/* Broadcast two shapes following the numpy rules every Triton
 * kernel author has internalised: align trailing dims, expand
 * size-1 dims, mismatches are an error. We do not emit an error
 * for the mismatch case in sitting one; the lowerer will fail
 * with a clearer message when it can see the actual operation
 * being attempted. */

static tn_shape_t s_broadcast(tn_shape_t a, tn_shape_t b)
{
    tn_shape_t out = {0};
    out.dtype = (uint8_t)s_promote_dtype(a.dtype, b.dtype);

    if (a.rank == 0 && b.rank == 0) {
        out.rank = 0;
        return out;
    }
    if (a.rank == 0) { out = b; out.dtype = (uint8_t)s_promote_dtype(a.dtype, b.dtype); return out; }
    if (b.rank == 0) { out = a; out.dtype = (uint8_t)s_promote_dtype(a.dtype, b.dtype); return out; }

    /* Both rank >= 1. Align trailing dims; the lower-rank side is
     * implicitly padded with size-1 leading dims. */
    int rank = a.rank > b.rank ? a.rank : b.rank;
    out.rank = (uint8_t)rank;
    if (rank == 1) {
        out.dims[0] = s_bcast_dim(a.dims[0], b.dims[0]);
    } else {
        /* rank 2: pad the shorter to [1, x]. */
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
    /* Map an intrinsic dtype id back to itself if it is one of the
     * numeric types; otherwise zero. This is what tl.float32 in a
     * call site argument resolves to via the Attr lookup. */
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

/* Read a dtype= keyword argument out of a call, returning the
 * intrinsic dtype id or 0 if unspecified. The dtype= value is an
 * Attr (tl.float32) whose sema resolved kind == TN_SYM_TYPE. */

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
        /* The keyword's name is the first IDENT token of the node. */
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

/* Read a (size_outer, size_inner) tile-shape tuple argument, e.g.
 * the first positional of tl.zeros((BLOCK_M, BLOCK_N), ...). Returns
 * the rank actually present (0, 1, or 2) and writes the dim values
 * (or -1 dynamic) into out_dims[0..1]. Anything we cannot statically
 * decode lands as a dynamic dim. */

static int s_call_shape_arg(const tn_sema_t *S, uint32_t call_idx,
                            int *out_dims)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[call_idx];
    if (n->num_kids == 0) return 0;
    uint32_t arg = s_kid(S, call_idx, 1);
    if (arg == 0 || arg >= P->num_nodes) return 0;
    const tn_node_t *an = &P->nodes[arg];

    /* Single-int positional like tl.zeros(N, dtype=...). */
    if (an->kind == TN_NK_LITERAL && an->flags == TN_LIT_INT) {
        int v = s_const_int(S, arg);
        out_dims[0] = v < 0 ? -1 : v;
        out_dims[1] = 0;
        return 1;
    }
    if (an->kind == TN_NK_NAME) {
        /* Treat any non-literal name as dynamic. Constexpr propagation
         * lands in a later sitting. */
        out_dims[0] = -1;
        out_dims[1] = 0;
        return 1;
    }

    if (an->kind == TN_NK_TUPLE) {
        uint32_t tnk = (an->num_kids == TN_NODE_KIDS_OVERFLOW)
                       ? an->kids[1] : an->num_kids;
        if (tnk == 1) {
            uint32_t k = s_kid(S, arg, 0);
            int v = s_const_int(S, k);
            out_dims[0] = v < 0 ? -1 : v;
            out_dims[1] = 0;
            return 1;
        }
        if (tnk >= 2) {
            uint32_t a = s_kid(S, arg, 0);
            uint32_t b = s_kid(S, arg, 1);
            int av = s_const_int(S, a);
            int bv = s_const_int(S, b);
            out_dims[0] = av < 0 ? -1 : av;
            out_dims[1] = bv < 0 ? -1 : bv;
            return 2;
        }
    }
    return 0;
}

/* Detect the `x[None]` / `x[:, None]` / `x[None, :]` reshape pattern.
 * Returns the rank of the resulting tile and writes broadcast dims to
 * out_dims. The base shape is passed in via base.
 *
 * Encoding observed in the parser: SUBSCRIPT kids[0] is the base
 * tile, kids[1..] are the index components. A `:` slice is a
 * TN_NK_SLICE node with flags = 0 (no lo/hi/step present); a `None`
 * is a TN_NK_LITERAL with flags = TN_LIT_NONE. */

static int s_subscript_is_bare_slice(const tn_sema_t *S, uint32_t kid)
{
    const tn_parse_t *P = S->parser;
    if (kid >= P->num_nodes) return 0;
    const tn_node_t *kn = &P->nodes[kid];
    return kn->kind == TN_NK_SLICE && kn->flags == 0;
}

static int s_subscript_is_none(const tn_sema_t *S, uint32_t kid)
{
    const tn_parse_t *P = S->parser;
    if (kid >= P->num_nodes) return 0;
    const tn_node_t *kn = &P->nodes[kid];
    return kn->kind == TN_NK_LITERAL && kn->flags == TN_LIT_NONE;
}

static tn_shape_t s_reshape_subscript(const tn_sema_t *S,
                                      uint32_t node_idx,
                                      tn_shape_t base)
{
    const tn_parse_t *P = S->parser;
    const tn_node_t *n = &P->nodes[node_idx];
    uint32_t nk = (n->num_kids == TN_NODE_KIDS_OVERFLOW)
                  ? n->kids[1] : n->num_kids;
    /* kids[0] = base, the rest are index components. */
    if (nk < 2) return base;

    /* Special-case the two patterns that appear in real kernels.
     * The first kid after the base is `:` or `None`, ditto the
     * second if present. */
    int n_idx = (int)nk - 1;
    uint32_t k0 = s_kid(S, node_idx, 1);
    uint32_t k1 = (nk > 2) ? s_kid(S, node_idx, 2) : 0;

    if (n_idx == 1) {
        /* x[None]: rank goes from 1 to 2 with leading 1. */
        if (s_subscript_is_none(S, k0) && base.rank == 1) {
            return s_mat(1, base.dims[0], base.dtype);
        }
        /* x[i] with a single integer index: drop one dim. */
        return s_scalar(base.dtype);
    }
    if (n_idx == 2 && base.rank == 1) {
        int k0_none  = s_subscript_is_none(S, k0);
        int k1_none  = s_subscript_is_none(S, k1);
        int k0_slice = s_subscript_is_bare_slice(S, k0);
        int k1_slice = s_subscript_is_bare_slice(S, k1);
        if (k0_slice && k1_none) return s_mat(base.dims[0], 1, base.dtype);
        if (k0_none  && k1_slice) return s_mat(1, base.dims[0], base.dtype);
    }
    /* Anything else: pass through unchanged, sitting two refines. */
    return base;
}

static tn_shape_t s_infer_expr(tn_sema_t *S, uint32_t node_idx);

static void s_set_shape(tn_sema_t *S, uint32_t node_idx, tn_shape_t sh)
{
    if (node_idx >= TN_MAX_NODES) return;
    S->node_shape[node_idx] = sh;
}

/* Shape rule for an intrinsic Call. Pulled out so the Call handler
 * stays readable; the dispatch is large but each case is small. */

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
        /* tl.arange(start, stop): rank-1 tile of size (stop - start)
         * when both are integer literals; dynamic otherwise. */
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
        int rank = s_call_shape_arg(S, call_idx, dims);
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
        int rank = s_call_shape_arg(S, call_idx, dims);
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
        int rank = s_call_shape_arg(S, call_idx, dims);
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
        /* tl.dot(A, B): if A is [M, K] and B is [K, N], result is
         * [M, N]. We trust the kernel author on K and just take the
         * outer dims. */
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
        /* Reductions drop one axis. Without an axis= keyword we
         * collapse the whole tile to a scalar; with one we drop
         * just that axis. Resolving the axis= keyword statically is
         * a sitting-two refinement. */
        if (nk < 2) return s_scalar(0);
        tn_shape_t base = s_infer_expr(S, s_kid(S, call_idx, 1));
        if (base.rank == 0) return base;
        if (base.rank == 1) return s_scalar(base.dtype);
        /* rank 2 with no axis given: collapse to scalar. With an
         * axis kw the result keeps one dim, dynamic for now. */
        return s_vec(-1, base.dtype);
    }

    case TN_TLI_LOAD: {
        if (nk < 2) return s_scalar(0);
        /* Shape follows the pointer expression's shape. The dtype
         * is the pointee, which we cannot recover here without a
         * type system; default to TN_TLI_FLOAT32 for now. */
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

    /* If we have already computed this node's shape, return it. The
     * walker can reach the same node twice through shape rules that
     * recurse into their arguments (DOT, WHERE, etc.). */
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
            /* Copy shape from the declaring node, which by virtue of
             * the post-order walk has already been inferred for any
             * assignment that appeared earlier in source order. */
            uint32_t decl = S->node_sym_aux[node_idx];
            if (decl < TN_MAX_NODES) {
                tn_shape_t ds = S->node_shape[decl];
                /* If the declaring node is an Assign, its shape was
                 * stashed under the Assign itself by s_infer_stmt. */
                if (ds.rank != 0 || ds.dtype != 0) {
                    result = ds;
                    break;
                }
            }
            result = s_scalar(0);
        } else if (kind == TN_SYM_PARAM) {
            /* Without type annotations we cannot know the dtype.
             * Pointer-ending names (..._ptr) get pointer-shaped
             * treatment downstream; for shape inference itself
             * they remain scalar with an unspecified dtype. */
            result = s_scalar(0);
        } else {
            result = s_scalar(0);
        }
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
        int kind = S->node_sym_kind[callee];
        if (kind == TN_SYM_INTRINSIC) {
            int id = (int)S->node_sym_aux[callee];
            result = s_call_shape(S, node_idx, id);
        } else {
            /* Non-intrinsic call: walk children so their shapes are
             * still annotated, then leave the call result scalar. */
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
        /* tl.float32 and friends: the dtype is the aux, but the
         * Attr itself appears in expression position only as part
         * of a Call callee. Leave it scalar with the dtype set so
         * dump output is informative. */
        if (S->node_sym_kind[node_idx] == TN_SYM_TYPE) {
            result = s_scalar((int)S->node_sym_aux[node_idx]);
        } else {
            result = s_scalar(0);
        }
        break;

    case TN_NK_TUPLE:
    case TN_NK_LIST:
        /* No tile-shape meaning; walk children for their own shapes
         * but report scalar/0 for the tuple itself. */
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

/* Walk statements, inferring shapes for any expression children and
 * propagating the RHS shape of an Assign onto the Assign node itself
 * so Name references reading from this binding pick it up. */

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

    case TN_NK_ASSIGN: {
        /* kids[0]=target, kids[1]=value. Stash the value's shape on
         * the Assign node so Name resolution can find it via the
         * sym aux pointer. */
        if (nk >= 2) {
            tn_shape_t v = s_infer_expr(S, s_kid(S, node_idx, 1));
            s_set_shape(S, node_idx, v);
        }
        return;
    }

    case TN_NK_AUG_ASSIGN: {
        /* a += b: walk both sides for their shapes; the binding's
         * shape was set when it was first introduced. */
        for (uint32_t i = 0; i < nk; i++) {
            (void)s_infer_expr(S, s_kid(S, node_idx, i));
        }
        return;
    }

    case TN_NK_EXPR_STMT:
    case TN_NK_RETURN:
        for (uint32_t i = 0; i < nk; i++) {
            (void)s_infer_expr(S, s_kid(S, node_idx, i));
        }
        return;

    case TN_NK_IF:
    case TN_NK_WHILE:
    case TN_NK_FOR:
        /* Conditions and iterators are expressions; suites are
         * statements. Walk both kinds without inspecting the node
         * shape further. */
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

    /* Append the inferred tile shape for expression nodes. We skip
     * the rank-0 / dtype-0 default since printing "scalar" on every
     * statement and structural node would just be noise. */
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
