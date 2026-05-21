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

/* ---- Public API ---- */

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
