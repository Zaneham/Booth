#include "sema.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

/*
 * Semantic analysis for Booth.
 * Two passes: collect declarations, then type-check function bodies.
 * Writes node_types[] side table so the BIR lowerer can stop guessing.
 */

/* ---- AST Navigation (same helpers as bir_lower.c) ---- */

static const ast_node_t *ND(const sema_ctx_t *S, uint32_t i)
{
    return &S->P->nodes[i];
}

static uint32_t child_at(const sema_ctx_t *S, uint32_t node, int n)
{
    uint32_t c = ND(S, node)->first_child;
    for (int i = 0; i < n && c; i++)
        c = ND(S, c)->next_sibling;
    return c;
}

/* Resolve a text offset — normally into src, but synthetic names
 * (anonymous struct/union) live in the parser's anon_buf. */
static const char *txt_ptr(const sema_ctx_t *S, uint32_t off)
{
    if (off >= BC_ANON_BASE)
        return S->P->anon_buf + (off - BC_ANON_BASE);
    return S->src + off;
}

static void get_text(const sema_ctx_t *S, uint32_t node, char *buf, int sz)
{
    const ast_node_t *n = ND(S, node);
    int len = (int)n->d.text.len;
    if (len >= sz) len = sz - 1;
    memcpy(buf, txt_ptr(S, n->d.text.offset), (size_t)len);
    buf[len] = '\0';
}

static int text_eq(const sema_ctx_t *S, uint32_t node, const char *s)
{
    const ast_node_t *n = ND(S, node);
    int len = (int)n->d.text.len;
    return (int)strlen(s) == len
        && memcmp(txt_ptr(S, n->d.text.offset), s, (size_t)len) == 0;
}

/* ---- Error Reporting ---- */

static void sema_error(sema_ctx_t *S, uint32_t node,
                       bc_eid_t eid, ...)
{
    if (S->num_errors >= SEMA_MAX_ERRORS) return;
    bc_error_t *e = &S->errors[S->num_errors++];
    if (node) {
        const ast_node_t *n = ND(S, node);
        e->loc.line = n->line;
        e->loc.col  = (uint16_t)n->col;
    } else {
        e->loc.line = 0;
        e->loc.col  = 0;
    }
    e->loc.offset = 0;
    e->code = BC_ERR_SEMA;
    e->eid  = (uint16_t)eid;
    va_list ap;
    va_start(ap, eid);
    vsnprintf(e->msg, sizeof(e->msg), bc_efmt(eid), ap);
    va_end(ap);
}

/* ---- Type Pool ---- */

static uint32_t intern_type(sema_ctx_t *S, stype_kind_t kind,
                            uint8_t quals, uint16_t width,
                            uint32_t inner, uint32_t extra)
{
    for (uint32_t i = 0; i < S->num_types; i++) {
        const stype_t *t = &S->types[i];
        if (t->kind == kind && t->qualifiers == quals
            && t->width == width && t->inner == inner && t->extra == extra)
            return i;
    }
    if (S->num_types >= SEMA_MAX_TYPES) return 0;
    uint32_t idx = S->num_types++;
    stype_t *t = &S->types[idx];
    t->kind       = (uint8_t)kind;
    t->qualifiers = quals;
    t->width      = width;
    t->inner      = inner;
    t->extra      = extra;
    return idx;
}

static uint32_t st_void(sema_ctx_t *S)     { return intern_type(S, STYPE_VOID,    0, 0, 0, 0); }
static uint32_t st_bool(sema_ctx_t *S)     { return intern_type(S, STYPE_BOOL,    0, 0, 0, 0); }
static uint32_t st_char(sema_ctx_t *S)     { return intern_type(S, STYPE_CHAR,    0, 0, 0, 0); }
static uint32_t st_schar(sema_ctx_t *S)    { return intern_type(S, STYPE_SCHAR,   0, 0, 0, 0); }
static uint32_t st_uchar(sema_ctx_t *S)    { return intern_type(S, STYPE_UCHAR,   0, 0, 0, 0); }
static uint32_t st_short(sema_ctx_t *S)    { return intern_type(S, STYPE_SHORT,   0, 0, 0, 0); }
static uint32_t st_ushort(sema_ctx_t *S)   { return intern_type(S, STYPE_USHORT,  0, 0, 0, 0); }
static uint32_t st_int(sema_ctx_t *S)      { return intern_type(S, STYPE_INT,     0, 0, 0, 0); }
static uint32_t st_uint(sema_ctx_t *S)     { return intern_type(S, STYPE_UINT,    0, 0, 0, 0); }
static uint32_t st_long(sema_ctx_t *S)     { return intern_type(S, STYPE_LONG,    0, 0, 0, 0); }
static uint32_t st_ulong(sema_ctx_t *S)    { return intern_type(S, STYPE_ULONG,   0, 0, 0, 0); }
static uint32_t st_llong(sema_ctx_t *S)    { return intern_type(S, STYPE_LLONG,   0, 0, 0, 0); }
static uint32_t st_ullong(sema_ctx_t *S)   { return intern_type(S, STYPE_ULLONG,  0, 0, 0, 0); }
static uint32_t st_float(sema_ctx_t *S)    { return intern_type(S, STYPE_FLOAT,   0, 0, 0, 0); }
static uint32_t st_double(sema_ctx_t *S)   { return intern_type(S, STYPE_DOUBLE,  0, 0, 0, 0); }
static uint32_t st_ldouble(sema_ctx_t *S)  { return intern_type(S, STYPE_LDOUBLE, 0, 0, 0, 0); }
static uint32_t st_error(sema_ctx_t *S)    { return intern_type(S, STYPE_ERROR,   0, 0, 0, 0); }

static uint32_t st_ptr(sema_ctx_t *S, uint32_t pointee)
{
    return intern_type(S, STYPE_PTR, 0, 0, pointee, 0);
}

static uint32_t st_array(sema_ctx_t *S, uint32_t elem, uint16_t count)
{
    return intern_type(S, STYPE_ARRAY, 0, count, elem, 0);
}

static uint32_t st_struct(sema_ctx_t *S, uint32_t struct_idx)
{
    return intern_type(S, STYPE_STRUCT, 0, 0, 0, struct_idx);
}

static uint32_t st_enum(sema_ctx_t *S)
{
    return intern_type(S, STYPE_ENUM, 0, 0, 0, 0);
}

static uint32_t st_func(sema_ctx_t *S, uint32_t ret_type,
                         const uint32_t *param_types, int nparams)
{
    uint32_t params_start = S->num_params;
    for (int i = 0; i < nparams; i++) {
        if (S->num_params < SEMA_MAX_PARAMS)
            S->param_pool[S->num_params++] = param_types[i];
    }
    return intern_type(S, STYPE_FUNC, 0, (uint16_t)nparams,
                       ret_type, params_start);
}

/* ---- Type Queries ---- */

static int is_error(const sema_ctx_t *S, uint32_t t)
{
    return t < S->num_types && S->types[t].kind == STYPE_ERROR;
}

static int is_integer(const sema_ctx_t *S, uint32_t t)
{
    if (t >= S->num_types) return 0;
    uint8_t k = S->types[t].kind;
    return k == STYPE_BOOL || k == STYPE_CHAR || k == STYPE_SCHAR
        || k == STYPE_UCHAR || k == STYPE_SHORT || k == STYPE_USHORT
        || k == STYPE_INT || k == STYPE_UINT || k == STYPE_LONG
        || k == STYPE_ULONG || k == STYPE_LLONG || k == STYPE_ULLONG
        || k == STYPE_ENUM;
}

static int is_floating(const sema_ctx_t *S, uint32_t t)
{
    if (t >= S->num_types) return 0;
    uint8_t k = S->types[t].kind;
    return k == STYPE_FLOAT || k == STYPE_DOUBLE || k == STYPE_LDOUBLE
        || k == STYPE_HALF || k == STYPE_BF16;
}

static int is_arithmetic(const sema_ctx_t *S, uint32_t t)
{
    return is_integer(S, t) || is_floating(S, t);
}

static int is_scalar(const sema_ctx_t *S, uint32_t t)
{
    if (t >= S->num_types) return 0;
    return is_arithmetic(S, t) || S->types[t].kind == STYPE_PTR;
}

static int is_unsigned_int(const sema_ctx_t *S, uint32_t t)
{
    if (t >= S->num_types) return 0;
    uint8_t k = S->types[t].kind;
    return k == STYPE_BOOL || k == STYPE_UCHAR || k == STYPE_USHORT
        || k == STYPE_UINT || k == STYPE_ULONG || k == STYPE_ULLONG;
}

static int is_ptr(const sema_ctx_t *S, uint32_t t)
{
    return t < S->num_types && S->types[t].kind == STYPE_PTR;
}

static uint32_t ptr_pointee(const sema_ctx_t *S, uint32_t t)
{
    if (!is_ptr(S, t)) return 0;
    return S->types[t].inner;
}

static int is_void_type(const sema_ctx_t *S, uint32_t t)
{
    return t < S->num_types && S->types[t].kind == STYPE_VOID;
}

static int is_struct(const sema_ctx_t *S, uint32_t t)
{
    return t < S->num_types && S->types[t].kind == STYPE_STRUCT;
}

static int is_array(const sema_ctx_t *S, uint32_t t)
{
    return t < S->num_types && S->types[t].kind == STYPE_ARRAY;
}

/* ---- Integer Conversion Rank (C11 6.3.1.1) ---- */

static int int_rank(const sema_ctx_t *S, uint32_t t)
{
    if (t >= S->num_types) return 0;
    switch (S->types[t].kind) {
    case STYPE_BOOL:                   return 1;
    case STYPE_CHAR: case STYPE_SCHAR:
    case STYPE_UCHAR:                  return 2;
    case STYPE_SHORT: case STYPE_USHORT: return 3;
    case STYPE_INT: case STYPE_UINT:
    case STYPE_ENUM:                   return 4;
    case STYPE_LONG: case STYPE_ULONG:  return 5;
    case STYPE_LLONG: case STYPE_ULLONG:return 6;
    default: return 0;
    }
}

/* ---- Integer Promotion (C11 6.3.1.1) ---- */

static uint32_t integer_promote(sema_ctx_t *S, uint32_t t)
{
    if (!is_integer(S, t)) return t;
    if (int_rank(S, t) < int_rank(S, st_int(S)))
        return st_int(S);
    return t;
}

/* ---- Usual Arithmetic Conversions (C11 6.3.1.8) ---- */

static uint32_t usual_arith_conv(sema_ctx_t *S, uint32_t a, uint32_t b)
{
    if (is_error(S, a) || is_error(S, b)) return st_error(S);

    /* If either is long double → long double */
    if (a < S->num_types && S->types[a].kind == STYPE_LDOUBLE) return a;
    if (b < S->num_types && S->types[b].kind == STYPE_LDOUBLE) return b;
    /* If either is double → double */
    if (a < S->num_types && S->types[a].kind == STYPE_DOUBLE) return a;
    if (b < S->num_types && S->types[b].kind == STYPE_DOUBLE) return b;
    /* If either is float → float */
    if (a < S->num_types && S->types[a].kind == STYPE_FLOAT) return a;
    if (b < S->num_types && S->types[b].kind == STYPE_FLOAT) return b;
    /* If either is half → half */
    if (a < S->num_types && S->types[a].kind == STYPE_HALF) return a;
    if (b < S->num_types && S->types[b].kind == STYPE_HALF) return b;
    /* If either is bf16 → bf16 */
    if (a < S->num_types && S->types[a].kind == STYPE_BF16) return a;
    if (b < S->num_types && S->types[b].kind == STYPE_BF16) return b;

    /* Integer promotions on both */
    a = integer_promote(S, a);
    b = integer_promote(S, b);

    /* Same type — done */
    if (a == b) return a;

    int a_unsigned = is_unsigned_int(S, a);
    int b_unsigned = is_unsigned_int(S, b);

    /* Same signedness → larger rank wins */
    if (a_unsigned == b_unsigned)
        return int_rank(S, a) >= int_rank(S, b) ? a : b;

    /* Different signedness */
    uint32_t u = a_unsigned ? a : b;
    uint32_t s = a_unsigned ? b : a;
    int u_rank = int_rank(S, u);
    int s_rank = int_rank(S, s);

    if (u_rank >= s_rank) return u;

    if (s_rank > u_rank) return s;
    return u;
}

/* ---- Symbol Table ---- */

static void push_scope(sema_ctx_t *S)
{
    if (S->scope_depth < SEMA_MAX_SCOPES)
        S->scope_stack[S->scope_depth++] = S->num_syms;
}

static void pop_scope(sema_ctx_t *S)
{
    if (S->scope_depth > 0)
        S->num_syms = S->scope_stack[--S->scope_depth];
}

static void add_sym(sema_ctx_t *S, const char *name, uint32_t type,
                    uint32_t ast_node, sym_kind_t kind, uint16_t cuda)
{
    if (S->num_syms >= SEMA_MAX_SYMS) return;
    /* Copy name first — source may overlap with destination (both inside S) */
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%s", name);
    sema_sym_t *sym = &S->syms[S->num_syms++];
    memcpy(sym->name, tmp, sizeof(sym->name));
    sym->type       = type;
    sym->ast_node   = ast_node;
    sym->kind       = (uint8_t)kind;
    sym->scope      = (uint8_t)(S->scope_depth > 0 ? S->scope_depth - 1 : 0);
    sym->cuda_flags = cuda;
}

static const sema_sym_t *find_sym(const sema_ctx_t *S, const char *name)
{
    for (int i = (int)S->num_syms - 1; i >= 0; i--)
        if (strcmp(S->syms[i].name, name) == 0)
            return &S->syms[i];
    return NULL;
}

/* ---- Struct Lookup ---- */

static sema_struct_t *find_struct(const sema_ctx_t *S, const char *name)
{
    for (uint32_t i = 0; i < S->num_structs; i++)
        if (strcmp(S->structs[i].name, name) == 0)
            return (sema_struct_t *)&S->structs[i];
    return NULL;
}

static sema_struct_t *find_struct_by_idx(const sema_ctx_t *S, uint32_t idx)
{
    if (idx < S->num_structs)
        return (sema_struct_t *)&S->structs[idx];
    return NULL;
}

/* ---- Resolve AST Type Specifier → stype ---- */

static uint32_t resolve_typespec(sema_ctx_t *S, uint32_t node, int ptr_depth)
{
    if (!node) return st_int(S);
    const ast_node_t *n = ND(S, node);
    if (n->type != AST_TYPE_SPEC) return st_int(S);

    uint32_t base;
    int kind = n->d.btype.kind;
    int is_unsigned = n->d.btype.is_unsigned;

    switch (kind) {
    case TYPE_VOID:    base = st_void(S);    break;
    case TYPE_BOOL:    base = st_bool(S);    break;
    case TYPE_CHAR:
        base = is_unsigned ? st_uchar(S) : st_char(S);
        break;
    case TYPE_SHORT:
        base = is_unsigned ? st_ushort(S) : st_short(S);
        break;
    case TYPE_INT:
        base = is_unsigned ? st_uint(S) : st_int(S);
        break;
    case TYPE_LONG:
        base = is_unsigned ? st_ulong(S) : st_long(S);
        break;
    case TYPE_LLONG:
        base = is_unsigned ? st_ullong(S) : st_llong(S);
        break;
    case TYPE_FLOAT:   base = st_float(S);   break;
    case TYPE_DOUBLE:  base = st_double(S);  break;
    case TYPE_LDOUBLE: base = st_ldouble(S); break;
    case TYPE_UNSIGNED:
        base = st_uint(S);
        break;
    case TYPE_SIGNED:
        base = st_int(S);
        break;

    case TYPE_NAME: {
        char tname[128] = {0};
        uint32_t nc = n->first_child;
        if (nc) get_text(S, nc, tname, sizeof(tname));

        if (strcmp(tname, "size_t") == 0)       { base = st_ulong(S); break; }
        if (strcmp(tname, "ptrdiff_t") == 0)    { base = st_long(S); break; }
        if (strcmp(tname, "uint32_t") == 0)     { base = st_uint(S); break; }
        if (strcmp(tname, "int32_t") == 0)      { base = st_int(S); break; }
        if (strcmp(tname, "uint64_t") == 0)     { base = st_ullong(S); break; }
        if (strcmp(tname, "int64_t") == 0)      { base = st_llong(S); break; }
        if (strcmp(tname, "uint16_t") == 0)     { base = st_ushort(S); break; }
        if (strcmp(tname, "int16_t") == 0)      { base = st_short(S); break; }
        if (strcmp(tname, "uint8_t") == 0)      { base = st_uchar(S); break; }
        if (strcmp(tname, "int8_t") == 0)       { base = st_schar(S); break; }
        if (strcmp(tname, "__half") == 0
            || strcmp(tname, "half") == 0
            || strcmp(tname, "_Float16") == 0)   { base = intern_type(S, STYPE_HALF, 0, 0, 0, 0); break; }
        if (strcmp(tname, "__nv_bfloat16") == 0
            || strcmp(tname, "nv_bfloat16") == 0
            || strcmp(tname, "__bfloat16") == 0
            || strcmp(tname, "__hip_bfloat16") == 0) { base = intern_type(S, STYPE_BF16, 0, 0, 0, 0); break; }

        /* ---- Vector types (CUDA vector_types.h) ---- */
        {
            static const struct { const char *name; stype_kind_t elem; uint16_t lanes; } vec_types[] = {
                {"float2",STYPE_FLOAT,2},{"float3",STYPE_FLOAT,3},{"float4",STYPE_FLOAT,4},
                {"double2",STYPE_DOUBLE,2},{"double3",STYPE_DOUBLE,3},{"double4",STYPE_DOUBLE,4},
                {"int2",STYPE_INT,2},{"int3",STYPE_INT,3},{"int4",STYPE_INT,4},
                {"uint2",STYPE_UINT,2},{"uint3",STYPE_UINT,3},{"uint4",STYPE_UINT,4},
                {"char2",STYPE_CHAR,2},{"char3",STYPE_CHAR,3},{"char4",STYPE_CHAR,4},
                {"uchar2",STYPE_UCHAR,2},{"uchar3",STYPE_UCHAR,3},{"uchar4",STYPE_UCHAR,4},
                {"short2",STYPE_SHORT,2},{"short3",STYPE_SHORT,3},{"short4",STYPE_SHORT,4},
                {"ushort2",STYPE_USHORT,2},{"ushort3",STYPE_USHORT,3},{"ushort4",STYPE_USHORT,4},
                {"long2",STYPE_LONG,2},{"long3",STYPE_LONG,3},{"long4",STYPE_LONG,4},
                {"ulong2",STYPE_ULONG,2},{"ulong3",STYPE_ULONG,3},{"ulong4",STYPE_ULONG,4},
                {"longlong2",STYPE_LLONG,2},{"ulonglong2",STYPE_ULLONG,2},
                {"dim3",STYPE_UINT,3},
                {NULL,0,0}
            };
            int vec_found = 0;
            for (int vi = 0; vec_types[vi].name; vi++) {
                if (strcmp(tname, vec_types[vi].name) == 0) {
                    uint32_t elem_t = intern_type(S, vec_types[vi].elem, 0, 0, 0, 0);
                    base = intern_type(S, STYPE_VECTOR, 0, vec_types[vi].lanes, elem_t, 0);
                    vec_found = 1;
                    break;
                }
            }
            if (vec_found) break;
        }

        const sema_sym_t *sym = find_sym(S, tname);
        if (sym) {
            if (sym->kind == SYM_TYPEDEF) {
                base = sym->type;
                break;
            }
            if (sym->kind == SYM_STRUCT) {
                base = sym->type;
                break;
            }
        }

        /* Unresolved name — default to int and soldier on */
        base = st_int(S);
        break;
    }

    case TYPE_STRUCT: case TYPE_UNION: case TYPE_CLASS: {
        char sname[128] = {0};
        uint32_t nc = n->first_child;
        if (nc) get_text(S, nc, sname, sizeof(sname));

        sema_struct_t *sd = find_struct(S, sname);
        if (sd) {
            uint32_t si = (uint32_t)(sd - S->structs);
            base = st_struct(S, si);
        } else {
            base = st_int(S);
        }
        break;
    }

    case TYPE_ENUM:
        base = st_enum(S);
        break;

    case TYPE_AUTO:
        base = st_int(S);
        break;

    default:
        base = st_int(S);
        break;
    }

    for (int i = 0; i < ptr_depth; i++)
        base = st_ptr(S, base);

    return base;
}

/* ---- Forward Declarations ---- */

static uint32_t check_expr(sema_ctx_t *S, uint32_t node);
static void     check_stmt(sema_ctx_t *S, uint32_t node);
static void     check_block_stmts(sema_ctx_t *S, uint32_t node);
static void     collect_struct_def(sema_ctx_t *S, uint32_t node);

/* ---- Annotate Helper ---- */

static uint32_t annotate(sema_ctx_t *S, uint32_t node, uint32_t type)
{
    if (node && node < BC_MAX_NODES)
        S->node_types[node] = type;
    return type;
}

/* ---- Operator Name Helpers ---- */

/* Normalize "operator +" → "operator+" (strip whitespace after 'operator') */
static void normalize_op_name(const char *raw, char *out, int outsz)
{
    int ri = 0, oi = 0;
    /* Copy "operator" prefix as-is */
    while (ri < 8 && raw[ri] && oi < outsz - 1)
        out[oi++] = raw[ri++];
    /* Skip whitespace */
    while (raw[ri] == ' ' || raw[ri] == '\t') ri++;
    /* Copy the rest */
    while (raw[ri] && oi < outsz - 1)
        out[oi++] = raw[ri++];
    out[oi] = '\0';
}

/* Build "operator+" from a token type */
static void op_name_from_tok(int tok, char *out, int outsz)
{
    const char *sym;
    switch (tok) {
    case TOK_PLUS: sym = "+"; break;
    case TOK_MINUS: sym = "-"; break;
    case TOK_STAR: sym = "*"; break;
    case TOK_SLASH: sym = "/"; break;
    case TOK_PERCENT: sym = "%"; break;
    case TOK_AMP: sym = "&"; break;
    case TOK_PIPE: sym = "|"; break;
    case TOK_CARET: sym = "^"; break;
    case TOK_SHL: sym = "<<"; break;
    case TOK_SHR: sym = ">>"; break;
    case TOK_EQ: sym = "=="; break;
    case TOK_NE: sym = "!="; break;
    case TOK_LT: sym = "<"; break;
    case TOK_GT: sym = ">"; break;
    case TOK_LE: sym = "<="; break;
    case TOK_GE: sym = ">="; break;
    default: sym = "?"; break;
    }
    snprintf(out, (size_t)outsz, "operator%s", sym);
}

/* ---- CUDA Builtin Type Resolution ---- */

static int try_cuda_member_type(sema_ctx_t *S, uint32_t node, uint32_t *out_type)
{
    const ast_node_t *n = ND(S, node);
    if (n->type != AST_MEMBER) return 0;

    uint32_t obj = n->first_child;
    uint32_t fld = obj ? ND(S, obj)->next_sibling : 0;
    if (!obj || !fld) return 0;
    if (ND(S, obj)->type != AST_IDENT) return 0;
    if (ND(S, fld)->type != AST_IDENT) return 0;

    if (text_eq(S, obj, "threadIdx") || text_eq(S, obj, "blockIdx")
        || text_eq(S, obj, "blockDim") || text_eq(S, obj, "gridDim"))
    {
        if (text_eq(S, fld, "x") || text_eq(S, fld, "y")
            || text_eq(S, fld, "z"))
        {
            *out_type = st_uint(S);
            return 1;
        }
    }
    return 0;
}

/* ---- CUDA Builtin Function Types ---- */

typedef struct {
    const char *name;
    int         nargs;      /* -1 = variadic */
    int         ret_void;   /* 1 = returns void */
    int         ret_arg;    /* -1 = fixed ret type. >=0 = return type matches arg[N] */
} cuda_builtin_t;

static const cuda_builtin_t cuda_builtins[] = {
    {"__syncthreads",    0, 1, -1},
    {"__threadfence",    0, 1, -1},
    {"atomicAdd",        2, 0,  1},
    {"atomicSub",        2, 0,  1},
    {"atomicAnd",        2, 0,  1},
    {"atomicOr",         2, 0,  1},
    {"atomicXor",        2, 0,  1},
    {"atomicMin",        2, 0,  1},
    {"atomicMax",        2, 0,  1},
    {"atomicExch",       2, 0,  1},
    {"atomicCAS",        3, 0,  1},
    {"__ballot_sync",    2, 0, -1},
    {"__any_sync",       2, 0, -1},
    {"__all_sync",       2, 0, -1},
    {"__shfl_sync",     -1, 0,  1},
    {"__shfl_up_sync",  -1, 0,  1},
    {"__shfl_down_sync",-1, 0,  1},
    {"__shfl_xor_sync", -1, 0,  1},
    {"__umul64hi",       2, 0,  0},  /* high 64 of a 64x64 product — ZK field arith */
    {"sqrtf",1,0,0},{"sqrt",1,0,0},{"__fsqrt_rn",1,0,0},{"rsqrtf",1,0,0},{"__frsqrt_rn",1,0,0},
    {"__frcp_rn",1,0,0},{"expf",1,0,0},{"__expf",1,0,0},{"exp2f",1,0,0},
    {"logf",1,0,0},{"__logf",1,0,0},{"log2f",1,0,0},{"__log2f",1,0,0},
    {"log10f",1,0,0},{"sinf",1,0,0},{"__sinf",1,0,0},{"cosf",1,0,0},
    {"__cosf",1,0,0},{"tanf",1,0,0},{"fabsf",1,0,0},{"fabs",1,0,0},
    {"floorf",1,0,0},{"ceilf",1,0,0},{"truncf",1,0,0},{"roundf",1,0,0},
    {"rintf",1,0,0},{"tanhf",1,0,0},{"copysignf",2,0,0},
    {"fmaxf",2,0,0},{"fminf",2,0,0},{"fmodf",2,0,0},{"powf",2,0,0},{"__powf",2,0,0},
    {NULL, 0, 0, 0}
};

/* ---- Expression Type Checking ---- */

static int64_t parse_int_value(const char *s, int len)
{
    char buf[64];
    int n = len > 63 ? 63 : len;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    /* Strip suffixes */
    while (n > 0 && (buf[n-1]=='u'||buf[n-1]=='U'||
                     buf[n-1]=='l'||buf[n-1]=='L'))
        buf[--n] = '\0';
    return strtoll(buf, NULL, 0);
}

static int parse_float_is_f32(const char *s, int len)
{
    return len > 0 && (s[len-1] == 'f' || s[len-1] == 'F');
}

static uint32_t check_expr(sema_ctx_t *S, uint32_t node)
{
    if (!node) return st_void(S);
    const ast_node_t *n = ND(S, node);

    switch (n->type) {
    case AST_NONE:
        return annotate(S, node, st_void(S));

    case AST_INT_LIT: {
        const char *s = S->src + n->d.text.offset;
        int len = (int)n->d.text.len;
        int has_u = 0, has_l = 0, has_ll = 0;
        int i = len;
        while (i > 0 && (s[i-1]=='u'||s[i-1]=='U'||s[i-1]=='l'||s[i-1]=='L')) {
            if (s[i-1]=='u'||s[i-1]=='U') has_u = 1;
            if (s[i-1]=='l'||s[i-1]=='L') {
                if (i-2>=0 && (s[i-2]=='l'||s[i-2]=='L')) { has_ll = 1; i--; }
                else has_l = 1;
            }
            i--;
        }

        uint32_t t;
        if (has_ll && has_u) t = st_ullong(S);
        else if (has_ll)     t = st_llong(S);
        else if (has_l && has_u) t = st_ulong(S);
        else if (has_l)      t = st_long(S);
        else if (has_u)      t = st_uint(S);
        else                 t = st_int(S);

        return annotate(S, node, t);
    }

    case AST_FLOAT_LIT: {
        int is_f32 = parse_float_is_f32(S->src + n->d.text.offset,
                                         (int)n->d.text.len);
        uint32_t t = is_f32 ? st_float(S) : st_double(S);
        return annotate(S, node, t);
    }

    case AST_BOOL_LIT:
        return annotate(S, node, st_bool(S));

    case AST_NULL_LIT:
        return annotate(S, node, st_ptr(S, st_void(S)));

    case AST_CHAR_LIT:
        return annotate(S, node, st_char(S));

    case AST_STRING_LIT:
        return annotate(S, node, st_ptr(S, st_char(S)));

    case AST_IDENT: {
        char name[128];
        get_text(S, node, name, sizeof(name));

        const sema_sym_t *sym = find_sym(S, name);
        if (sym) {
            return annotate(S, node, sym->type);
        }

        return annotate(S, node, st_int(S));
    }

    case AST_PAREN:
        if (n->first_child) {
            uint32_t inner = check_expr(S, n->first_child);
            return annotate(S, node, inner);
        }
        return annotate(S, node, st_void(S));

    case AST_MEMBER: {
        uint32_t cuda_t;
        if (try_cuda_member_type(S, node, &cuda_t))
            return annotate(S, node, cuda_t);

        uint32_t obj_n = n->first_child;
        uint32_t fld_n = obj_n ? ND(S, obj_n)->next_sibling : 0;
        if (!obj_n || !fld_n)
            return annotate(S, node, st_error(S));

        uint32_t obj_t = check_expr(S, obj_n);
        if (is_error(S, obj_t))
            return annotate(S, node, st_error(S));

        uint32_t struct_t = obj_t;
        if (n->d.member.is_arrow) {
            if (!is_ptr(S, obj_t)) {
                sema_error(S, node, BC_E070);
                return annotate(S, node, st_error(S));
            }
            struct_t = ptr_pointee(S, obj_t);
        }

        if (is_struct(S, struct_t)) {
            uint32_t si = S->types[struct_t].extra;
            sema_struct_t *sd = find_struct_by_idx(S, si);
            if (sd) {
                char fname[128];
                get_text(S, fld_n, fname, sizeof(fname));
                for (int fi = 0; fi < sd->num_fields; fi++) {
                    if (strcmp(sd->field_names[fi], fname) == 0) {
                        return annotate(S, node, sd->field_types[fi]);
                    }
                }
                sema_error(S, fld_n, BC_E071,
                           fname, sd->name);
            }
        }

        /* Vector member access: .x .y .z .w */
        if (struct_t < S->num_types && S->types[struct_t].kind == STYPE_VECTOR) {
            char fname[128];
            get_text(S, fld_n, fname, sizeof(fname));
            uint16_t width = S->types[struct_t].width;
            int lane = -1;
            if (strcmp(fname, "x") == 0)      lane = 0;
            else if (strcmp(fname, "y") == 0) lane = 1;
            else if (strcmp(fname, "z") == 0) lane = 2;
            else if (strcmp(fname, "w") == 0) lane = 3;
            if (lane >= 0 && lane < (int)width)
                return annotate(S, node, S->types[struct_t].inner);
            sema_error(S, fld_n, BC_E072, fname);
        }
        return annotate(S, node, st_int(S));
    }

    case AST_SUBSCRIPT: {
        uint32_t base_n = n->first_child;
        uint32_t idx_n  = base_n ? ND(S, base_n)->next_sibling : 0;
        if (!base_n || !idx_n)
            return annotate(S, node, st_error(S));

        uint32_t base_t = check_expr(S, base_n);
        uint32_t idx_t  = check_expr(S, idx_n);
        (void)idx_t;

        if (is_error(S, base_t))
            return annotate(S, node, st_error(S));

        if (is_ptr(S, base_t))
            return annotate(S, node, ptr_pointee(S, base_t));
        if (is_array(S, base_t))
            return annotate(S, node, S->types[base_t].inner);
        return annotate(S, node, st_int(S));
    }

    case AST_BINARY: {
        int op = n->d.oper.op;
        uint32_t lhs_n = n->first_child;
        uint32_t rhs_n = lhs_n ? ND(S, lhs_n)->next_sibling : 0;

        if (op == TOK_ASSIGN) {
            uint32_t lhs_t = check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);
            (void)rhs_t;
            return annotate(S, node, lhs_t);
        }

        if (op == TOK_PLUS_EQ || op == TOK_MINUS_EQ || op == TOK_STAR_EQ
            || op == TOK_SLASH_EQ || op == TOK_PERCENT_EQ
            || op == TOK_AMP_EQ || op == TOK_PIPE_EQ || op == TOK_CARET_EQ
            || op == TOK_SHL_EQ || op == TOK_SHR_EQ)
        {
            uint32_t lhs_t = check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);
            (void)rhs_t;
            return annotate(S, node, lhs_t);
        }

        if (op == TOK_EQ || op == TOK_NE || op == TOK_LT || op == TOK_GT
            || op == TOK_LE || op == TOK_GE)
        {
            uint32_t lhs_t = check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);
            (void)lhs_t; (void)rhs_t;
            return annotate(S, node, st_bool(S));
        }

        if (op == TOK_LAND || op == TOK_LOR) {
            uint32_t lhs_t = check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);
            (void)lhs_t; (void)rhs_t;
            return annotate(S, node, st_bool(S));
        }

        if (op == TOK_COMMA) {
            check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);
            return annotate(S, node, rhs_t);
        }

        {
            uint32_t lhs_t = check_expr(S, lhs_n);
            uint32_t rhs_t = check_expr(S, rhs_n);

            if (is_ptr(S, lhs_t) && is_integer(S, rhs_t)
                && (op == TOK_PLUS || op == TOK_MINUS))
            {
                return annotate(S, node, lhs_t);
            }
            if (is_integer(S, lhs_t) && is_ptr(S, rhs_t) && op == TOK_PLUS)
                return annotate(S, node, rhs_t);

            if (is_ptr(S, lhs_t) && is_ptr(S, rhs_t) && op == TOK_MINUS)
                return annotate(S, node, st_long(S));

            if (op == TOK_SHL || op == TOK_SHR)
                return annotate(S, node, integer_promote(S, lhs_t));

            /* Operator overloading: struct/vector operand → look up operator func */
            if ((is_struct(S, lhs_t) || (lhs_t < S->num_types && S->types[lhs_t].kind == STYPE_VECTOR))
                || (is_struct(S, rhs_t) || (rhs_t < S->num_types && S->types[rhs_t].kind == STYPE_VECTOR)))
            {
                char oname[32];
                op_name_from_tok(op, oname, sizeof(oname));
                const sema_sym_t *osym = find_sym(S, oname);
                if (osym && osym->kind == SYM_FUNC) {
                    uint32_t ft = osym->type;
                    if (ft < S->num_types && S->types[ft].kind == STYPE_FUNC)
                        return annotate(S, node, S->types[ft].inner);
                    return annotate(S, node, lhs_t);
                }
            }

            uint32_t result = usual_arith_conv(S, lhs_t, rhs_t);
            return annotate(S, node, result);
        }
    }

    case AST_UNARY_PREFIX: {
        int op = n->d.oper.op;
        uint32_t operand = n->first_child;

        if (op == TOK_PLUS || op == TOK_MINUS) {
            uint32_t t = check_expr(S, operand);
            if (is_integer(S, t)) t = integer_promote(S, t);
            return annotate(S, node, t);
        }

        if (op == TOK_BANG) {
            check_expr(S, operand);
            return annotate(S, node, st_bool(S));
        }

        if (op == TOK_TILDE) {
            uint32_t t = check_expr(S, operand);
            t = integer_promote(S, t);
            return annotate(S, node, t);
        }

        if (op == TOK_AMP) {
            uint32_t t = check_expr(S, operand);
            return annotate(S, node, st_ptr(S, t));
        }

        if (op == TOK_STAR) {
            uint32_t t = check_expr(S, operand);
            if (is_ptr(S, t))
                return annotate(S, node, ptr_pointee(S, t));
            return annotate(S, node, st_int(S));
        }

        if (op == TOK_INC || op == TOK_DEC) {
            uint32_t t = check_expr(S, operand);
            return annotate(S, node, t);
        }

        return annotate(S, node, check_expr(S, operand));
    }

    case AST_UNARY_POSTFIX: {
        int op = n->d.oper.op;
        uint32_t operand = n->first_child;
        if (op == TOK_INC || op == TOK_DEC) {
            uint32_t t = check_expr(S, operand);
            return annotate(S, node, t);
        }
        return annotate(S, node, check_expr(S, operand));
    }

    case AST_TERNARY: {
        uint32_t cond_n = n->first_child;
        uint32_t then_n = cond_n ? ND(S, cond_n)->next_sibling : 0;
        uint32_t else_n = then_n ? ND(S, then_n)->next_sibling : 0;

        check_expr(S, cond_n);
        uint32_t then_t = check_expr(S, then_n);
        uint32_t else_t = check_expr(S, else_n);

        if (is_error(S, then_t)) return annotate(S, node, else_t);
        if (is_error(S, else_t)) return annotate(S, node, then_t);

        if (is_ptr(S, then_t) && is_ptr(S, else_t))
            return annotate(S, node, then_t);

        if (is_arithmetic(S, then_t) && is_arithmetic(S, else_t)) {
            uint32_t unified = usual_arith_conv(S, then_t, else_t);
            return annotate(S, node, unified);
        }

        return annotate(S, node, then_t);
    }

    case AST_CALL: {
        uint32_t callee_n = n->first_child;
        if (!callee_n) return annotate(S, node, st_int(S));

        char cname[128];
        get_text(S, callee_n, cname, sizeof(cname));

        uint32_t arg_types[16];
        int nargs = 0;
        uint32_t arg = ND(S, callee_n)->next_sibling;
        while (arg && nargs < 16) {
            arg_types[nargs++] = check_expr(S, arg);
            arg = ND(S, arg)->next_sibling;
        }

        /* ---- Cooperative groups: namespace::func() and handle.method() ---- */
        if (ND(S, callee_n)->type == AST_SCOPE_RES) {
            uint32_t ns_n = ND(S, callee_n)->first_child;
            uint32_t fn_n = ns_n ? ND(S, ns_n)->next_sibling : 0;
            if (ns_n && fn_n && text_eq(S, ns_n, "cooperative_groups")) {
                if (text_eq(S, fn_n, "this_thread_block"))
                    return annotate(S, node, st_int(S)); /* phantom handle */
                if (text_eq(S, fn_n, "this_grid"))
                    return annotate(S, node, st_int(S));
            }
        }
        if (ND(S, callee_n)->type == AST_MEMBER) {
            /* tb.sync(), tb.thread_rank(), tb.size() on coop group handles */
            uint32_t fld_n = ND(S, callee_n)->first_child
                ? ND(S, ND(S, callee_n)->first_child)->next_sibling : 0;
            if (fld_n && ND(S, fld_n)->type == AST_IDENT) {
                if (text_eq(S, fld_n, "sync"))
                    return annotate(S, node, st_void(S));
                if (text_eq(S, fld_n, "thread_rank"))
                    return annotate(S, node, st_int(S));
                if (text_eq(S, fld_n, "size"))
                    return annotate(S, node, st_int(S));
            }
        }

        /* ---- Vector constructors: make_float2/3/4, make_int2/3/4 etc. ---- */
        {
            static const struct { const char *name; stype_kind_t elem; uint16_t lanes; } vec_ctors[] = {
                {"make_float2",STYPE_FLOAT,2},{"make_float3",STYPE_FLOAT,3},{"make_float4",STYPE_FLOAT,4},
                {"make_double2",STYPE_DOUBLE,2},{"make_double3",STYPE_DOUBLE,3},{"make_double4",STYPE_DOUBLE,4},
                {"make_int2",STYPE_INT,2},{"make_int3",STYPE_INT,3},{"make_int4",STYPE_INT,4},
                {"make_uint2",STYPE_UINT,2},{"make_uint3",STYPE_UINT,3},{"make_uint4",STYPE_UINT,4},
                {"make_char2",STYPE_CHAR,2},{"make_char3",STYPE_CHAR,3},{"make_char4",STYPE_CHAR,4},
                {"make_uchar2",STYPE_UCHAR,2},{"make_uchar3",STYPE_UCHAR,3},{"make_uchar4",STYPE_UCHAR,4},
                {"make_short2",STYPE_SHORT,2},{"make_short3",STYPE_SHORT,3},{"make_short4",STYPE_SHORT,4},
                {"make_ushort2",STYPE_USHORT,2},{"make_ushort3",STYPE_USHORT,3},{"make_ushort4",STYPE_USHORT,4},
                {"make_long2",STYPE_LONG,2},{"make_long3",STYPE_LONG,3},{"make_long4",STYPE_LONG,4},
                {"make_ulong2",STYPE_ULONG,2},{"make_ulong3",STYPE_ULONG,3},{"make_ulong4",STYPE_ULONG,4},
                {NULL,0,0}
            };
            for (int vi = 0; vec_ctors[vi].name; vi++) {
                if (strcmp(cname, vec_ctors[vi].name) == 0) {
                    uint32_t elem_t = intern_type(S, vec_ctors[vi].elem, 0, 0, 0, 0);
                    uint32_t vt = intern_type(S, STYPE_VECTOR, 0, vec_ctors[vi].lanes, elem_t, 0);
                    if (nargs != (int)vec_ctors[vi].lanes)
                        sema_error(S, node, BC_E073,
                                   cname, (int)vec_ctors[vi].lanes, nargs);
                    return annotate(S, node, vt);
                }
            }
        }

        /* ---- Half conversion builtins ---- */
        if (strcmp(cname, "__float2half") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, "__float2half", nargs);
            return annotate(S, node, intern_type(S, STYPE_HALF, 0, 0, 0, 0));
        }
        if (strcmp(cname, "__half2float") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, "__half2float", nargs);
            return annotate(S, node, st_float(S));
        }

        /* ---- BF16 conversion builtins ---- */
        if (strcmp(cname, "__float2bfloat16") == 0
            || strcmp(cname, "__float2bfloat16_rn") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, cname, nargs);
            return annotate(S, node, intern_type(S, STYPE_BF16, 0, 0, 0, 0));
        }
        if (strcmp(cname, "__bfloat162float") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, "__bfloat162float", nargs);
            return annotate(S, node, st_float(S));
        }
        /* ---- Bit-cast builtins ---- */
        if (strcmp(cname, "__int_as_float") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, "__int_as_float", nargs);
            return annotate(S, node, st_float(S));
        }
        if (strcmp(cname, "__float_as_int") == 0) {
            if (nargs != 1)
                sema_error(S, node, BC_E074, "__float_as_int", nargs);
            return annotate(S, node, st_int(S));
        }

        /* ---- MFMA intrinsics (CDNA matrix ops) ---- */
        if (strncmp(cname, "__builtin_amdgcn_mfma_", 22) == 0) {
            if (nargs != 3)
                sema_error(S, node, BC_E075, cname, nargs);
            /* Return type = accumulator type (arg[2]) */
            uint32_t rt = (nargs >= 3) ? arg_types[2] : st_float(S);
            return annotate(S, node, rt);
        }

        for (int i = 0; cuda_builtins[i].name; i++) {
            if (strcmp(cname, cuda_builtins[i].name) != 0) continue;
            const cuda_builtin_t *b = &cuda_builtins[i];

            if (b->nargs >= 0 && nargs != b->nargs) {
                sema_error(S, node, BC_E073,
                           cname, b->nargs, nargs);
            }

            uint32_t rt;
            if (b->ret_void) {
                rt = st_void(S);
            } else if (b->ret_arg >= 0 && b->ret_arg < nargs) {
                rt = arg_types[b->ret_arg];
                /* Atomics: ret = value type (arg[1]), not the pointer */
                if (b->ret_arg == 1 && nargs >= 2) {
                    rt = arg_types[1];
                }
            } else if (strcmp(cname, "__ballot_sync") == 0) {
                rt = st_uint(S);
            } else if (strcmp(cname, "__any_sync") == 0
                    || strcmp(cname, "__all_sync") == 0) {
                rt = st_int(S);
            } else {
                rt = st_int(S);
            }

            return annotate(S, node, rt);
        }

        const sema_sym_t *sym = find_sym(S, cname);
        if (sym && sym->kind == SYM_FUNC) {
            uint32_t ft = sym->type;
            if (ft < S->num_types && S->types[ft].kind == STYPE_FUNC) {
                int expected = (int)S->types[ft].width;
                if (expected > 0 && nargs != expected) {
                    sema_error(S, node, BC_E073,
                               cname, expected, nargs);
                }
                return annotate(S, node, S->types[ft].inner);
            }
            return annotate(S, node, sym->type);
        }

        return annotate(S, node, st_int(S));
    }

    case AST_LAUNCH: {
        uint32_t callee_n = n->first_child;
        uint32_t c = callee_n ? ND(S, callee_n)->next_sibling : 0;
        while (c) {
            check_expr(S, c);
            c = ND(S, c)->next_sibling;
        }
        return annotate(S, node, st_void(S));
    }

    case AST_CAST: {
        uint32_t type_n = n->first_child;
        uint32_t expr_n = type_n ? ND(S, type_n)->next_sibling : 0;
        int pdepth = n->d.oper.flags;

        uint32_t dst_t = resolve_typespec(S, type_n, pdepth);
        if (expr_n) check_expr(S, expr_n);

        return annotate(S, node, dst_t);
    }

    case AST_SIZEOF: {
        uint32_t inner = n->first_child;
        if (inner) check_expr(S, inner);
        return annotate(S, node, st_ulong(S));
    }

    case AST_INIT_LIST: {
        uint32_t el = n->first_child;
        while (el) {
            check_expr(S, el);
            el = ND(S, el)->next_sibling;
        }
        return annotate(S, node, st_void(S));
    }

    case AST_SCOPE_RES: {
        return annotate(S, node, st_int(S));
    }

    case AST_TEMPLATE_ARGS: {
        return annotate(S, node, st_void(S));
    }

    default:
        return annotate(S, node, st_int(S));
    }
}

/* ---- Statement Checking ---- */

static void check_var_decl(sema_ctx_t *S, uint32_t node)
{
    const ast_node_t *n = ND(S, node);

    if (n->qualifiers & QUAL_TYPEDEF) return;

    uint32_t type_n = child_at(S, node, 0);
    uint32_t name_n = child_at(S, node, 1);
    if (!name_n || ND(S, name_n)->type != AST_IDENT) return;

    /* Anonymous struct/union embedded by parser? Register it now,
     * before resolve_typespec goes looking for the name. Like
     * filing the birth certificate before enrolling the kid. */
    uint32_t peek = ND(S, name_n)->next_sibling;
    if (peek && ND(S, peek)->type == AST_STRUCT_DEF)
        collect_struct_def(S, peek);

    char name[128];
    get_text(S, name_n, name, sizeof(name));

    uint32_t var_type = resolve_typespec(S, type_n, n->d.oper.flags);

    /* Parser stores bracket-dimension count in d.oper.op */
    int pdims = n->d.oper.op;
    uint32_t next = ND(S, name_n)->next_sibling;
    /* Skip embedded struct_def (handled above) */
    if (next && ND(S, next)->type == AST_STRUCT_DEF)
        next = ND(S, next)->next_sibling;
    if (pdims > 0 && next && ND(S, next)->type == AST_INT_LIT) {
        int64_t count = parse_int_value(S->src + ND(S, next)->d.text.offset,
                                        (int)ND(S, next)->d.text.len);
        uint32_t arr_t = st_array(S, var_type, (uint16_t)count);
        annotate(S, node, arr_t);
        add_sym(S, name, st_ptr(S, var_type), node, SYM_VAR, n->cuda_flags);
        uint32_t init_n = ND(S, next)->next_sibling;
        if (init_n) check_expr(S, init_n);
        return;
    }
    if (pdims > 0 && next && ND(S, next)->type == AST_IDENT) {
        char aname[128];
        get_text(S, next, aname, sizeof(aname));
        const sema_sym_t *as = find_sym(S, aname);
        int base_kind = (type_n && ND(S, type_n)->type == AST_TYPE_SPEC)
                        ? ND(S, type_n)->d.btype.kind : -1;
        if (as && as->kind == SYM_ENUM_CONST
            && base_kind != TYPE_ENUM && base_kind != TYPE_STRUCT
            && base_kind != TYPE_NAME)
        {
            annotate(S, node, st_array(S, var_type, 0));
            add_sym(S, name, st_ptr(S, var_type), node, SYM_VAR, n->cuda_flags);
            uint32_t init_n = ND(S, next)->next_sibling;
            if (init_n) check_expr(S, init_n);
            return;
        }
    }

    annotate(S, node, var_type);
    add_sym(S, name, var_type, node, SYM_VAR, n->cuda_flags);

    uint32_t init_n = next;
    if (init_n && ND(S, init_n)->type != AST_NONE) {
        check_expr(S, init_n);
    }
}

static void check_stmt(sema_ctx_t *S, uint32_t node)
{
    if (!node) return;
    const ast_node_t *n = ND(S, node);

    switch (n->type) {
    case AST_EXPR_STMT:
        if (n->first_child)
            check_expr(S, n->first_child);
        break;

    case AST_BLOCK:
        push_scope(S);
        check_block_stmts(S, node);
        pop_scope(S);
        break;

    case AST_VAR_DECL:
        check_var_decl(S, node);
        break;

    case AST_RETURN: {
        if (n->first_child) {
            uint32_t ret_t = check_expr(S, n->first_child);
            if (S->cur_ret_type && !is_error(S, ret_t)
                && !is_void_type(S, S->cur_ret_type))
            {
                annotate(S, node, ret_t);
            }
        } else {
            annotate(S, node, st_void(S));
        }
        break;
    }

    case AST_IF: {
        uint32_t cond_n = child_at(S, node, 0);
        uint32_t then_n = child_at(S, node, 1);
        uint32_t else_n = child_at(S, node, 2);

        if (cond_n) {
            uint32_t ct = check_expr(S, cond_n);
            if (!is_scalar(S, ct) && !is_error(S, ct))
                sema_error(S, cond_n, BC_E076);
        }
        check_stmt(S, then_n);
        if (else_n) check_stmt(S, else_n);
        break;
    }

    case AST_FOR: {
        uint32_t init_n = child_at(S, node, 0);
        uint32_t cond_n = child_at(S, node, 1);
        uint32_t incr_n = child_at(S, node, 2);
        uint32_t body_n = child_at(S, node, 3);

        push_scope(S);

        if (init_n && ND(S, init_n)->type != AST_NONE) {
            if (ND(S, init_n)->type == AST_VAR_DECL)
                check_var_decl(S, init_n);
            else
                check_expr(S, init_n);
        }
        if (cond_n && ND(S, cond_n)->type != AST_NONE) {
            uint32_t ct = check_expr(S, cond_n);
            if (!is_scalar(S, ct) && !is_error(S, ct))
                sema_error(S, cond_n, BC_E077);
        }
        if (incr_n && ND(S, incr_n)->type != AST_NONE)
            check_expr(S, incr_n);
        if (body_n) check_stmt(S, body_n);

        pop_scope(S);
        break;
    }

    case AST_WHILE: {
        uint32_t cond_n = child_at(S, node, 0);
        uint32_t body_n = child_at(S, node, 1);

        if (cond_n) {
            uint32_t ct = check_expr(S, cond_n);
            if (!is_scalar(S, ct) && !is_error(S, ct))
                sema_error(S, cond_n, BC_E078);
        }
        if (body_n) check_stmt(S, body_n);
        break;
    }

    case AST_DO_WHILE: {
        uint32_t body_n = child_at(S, node, 0);
        uint32_t cond_n = child_at(S, node, 1);

        if (body_n) check_stmt(S, body_n);
        if (cond_n) {
            uint32_t ct = check_expr(S, cond_n);
            if (!is_scalar(S, ct) && !is_error(S, ct))
                sema_error(S, cond_n, BC_E079);
        }
        break;
    }

    case AST_SWITCH: {
        uint32_t cond_n = child_at(S, node, 0);
        uint32_t body_n = child_at(S, node, 1);

        if (cond_n) {
            uint32_t ct = check_expr(S, cond_n);
            if (!is_integer(S, ct) && !is_error(S, ct))
                sema_error(S, cond_n, BC_E080);
        }
        if (body_n) {
            uint32_t c = ND(S, body_n)->first_child;
            while (c) {
                if (ND(S, c)->type == AST_CASE) {
                    uint32_t val_n = ND(S, c)->first_child;
                    if (val_n) check_expr(S, val_n);
                } else if (ND(S, c)->type == AST_DEFAULT) {
                } else {
                    check_stmt(S, c);
                }
                c = ND(S, c)->next_sibling;
            }
        }
        break;
    }

    case AST_BREAK:
    case AST_CONTINUE:
    case AST_PP_DIRECTIVE:
        break;

    case AST_GOTO:
    case AST_LABEL:
        break;

    default:
        break;
    }
}

static void check_block_stmts(sema_ctx_t *S, uint32_t block_node)
{
    uint32_t c = ND(S, block_node)->first_child;
    while (c) {
        check_stmt(S, c);
        c = ND(S, c)->next_sibling;
    }
}

/* ---- Pass 1: Collect Declarations ---- */

static void collect_struct_def(sema_ctx_t *S, uint32_t node)
{
    if (S->num_structs >= SEMA_MAX_STRUCTS) return;

    uint32_t type_n = ND(S, node)->first_child;
    if (!type_n) return;

    sema_struct_t *sd = &S->structs[S->num_structs];
    memset(sd, 0, sizeof(*sd));

    uint32_t name_n = ND(S, type_n)->first_child;
    if (name_n && ND(S, name_n)->type == AST_IDENT)
        get_text(S, name_n, sd->name, sizeof(sd->name));

    uint32_t member = ND(S, type_n)->next_sibling;
    while (member && sd->num_fields < SEMA_MAX_FIELDS) {
        if (ND(S, member)->type == AST_VAR_DECL) {
            uint32_t ft_n = ND(S, member)->first_child;
            uint32_t fn_n = ft_n ? ND(S, ft_n)->next_sibling : 0;

            uint32_t ft = resolve_typespec(S, ft_n, ND(S, member)->d.oper.flags);

            if (fn_n && ND(S, fn_n)->type == AST_IDENT) {
                sd->field_types[sd->num_fields] = ft;
                get_text(S, fn_n, sd->field_names[sd->num_fields],
                         sizeof(sd->field_names[0]));
                sd->num_fields++;

                uint32_t extra = ND(S, fn_n)->next_sibling;
                while (extra && ND(S, extra)->type == AST_IDENT
                       && sd->num_fields < SEMA_MAX_FIELDS) {
                    sd->field_types[sd->num_fields] = ft;
                    get_text(S, extra, sd->field_names[sd->num_fields],
                             sizeof(sd->field_names[0]));
                    sd->num_fields++;
                    extra = ND(S, extra)->next_sibling;
                }
            }
        }
        member = ND(S, member)->next_sibling;
    }

    if (sd->num_fields > 0 || sd->name[0]) {
        uint32_t si = S->num_structs++;
        uint32_t st = st_struct(S, si);
        add_sym(S, sd->name, st, node, SYM_STRUCT, 0);
    }
}

static void collect_enum_def(sema_ctx_t *S, uint32_t node)
{
    uint32_t c = ND(S, node)->first_child;
    if (c) c = ND(S, c)->next_sibling;

    while (c) {
        if (ND(S, c)->type == AST_ENUMERATOR) {
            char name[128];
            get_text(S, c, name, sizeof(name));
            add_sym(S, name, st_int(S), c, SYM_ENUM_CONST, 0);
        }
        c = ND(S, c)->next_sibling;
    }
}

static void collect_typedef_def(sema_ctx_t *S, uint32_t node)
{
    if (!(ND(S, node)->qualifiers & QUAL_TYPEDEF)) return;

    uint32_t type_n = child_at(S, node, 0);
    uint32_t name_n = child_at(S, node, 1);
    if (!name_n || ND(S, name_n)->type != AST_IDENT) return;

    char name[128];
    get_text(S, name_n, name, sizeof(name));
    uint32_t resolved = resolve_typespec(S, type_n, ND(S, node)->d.oper.flags);
    add_sym(S, name, resolved, node, SYM_TYPEDEF, 0);
}

static void collect_func_decl(sema_ctx_t *S, uint32_t node)
{
    uint32_t type_n = child_at(S, node, 0);
    uint32_t name_n = child_at(S, node, 1);
    if (!name_n) return;

    char fname_raw[128], fname[128];
    get_text(S, name_n, fname_raw, sizeof(fname_raw));
    /* Normalize operator names: "operator +" → "operator+" */
    if (strncmp(fname_raw, "operator", 8) == 0 && fname_raw[8] != '\0')
        normalize_op_name(fname_raw, fname, sizeof(fname));
    else
        memcpy(fname, fname_raw, sizeof(fname));

    int ret_ptr = ND(S, node)->d.oper.flags;
    uint32_t ret_t = resolve_typespec(S, type_n, ret_ptr);

    uint32_t param_types[32];
    int nparams = 0;
    uint32_t c = ND(S, node)->first_child;
    while (c) {
        if (ND(S, c)->type == AST_PARAM && nparams < 32) {
            uint32_t pt_n = ND(S, c)->first_child;
            int pdepth = ND(S, c)->d.oper.flags;
            param_types[nparams++] = resolve_typespec(S, pt_n, pdepth);
        }
        c = ND(S, c)->next_sibling;
    }

    uint32_t ft = st_func(S, ret_t, param_types, nparams);
    add_sym(S, fname, ft, node, SYM_FUNC, ND(S, node)->cuda_flags);
}

static void collect_global_var(sema_ctx_t *S, uint32_t node)
{
    const ast_node_t *n = ND(S, node);
    if (n->qualifiers & QUAL_TYPEDEF) return;

    uint32_t type_n = child_at(S, node, 0);
    uint32_t name_n = child_at(S, node, 1);
    if (!name_n || ND(S, name_n)->type != AST_IDENT) return;

    char name[128];
    get_text(S, name_n, name, sizeof(name));

    uint32_t var_t = resolve_typespec(S, type_n, n->d.oper.flags);
    add_sym(S, name, var_t, node, SYM_VAR, n->cuda_flags);
}

static void collect_template(sema_ctx_t *S, uint32_t node)
{
    uint32_t c = ND(S, node)->first_child;
    while (c) {
        if (ND(S, c)->type == AST_FUNC_DEF || ND(S, c)->type == AST_FUNC_DECL) {
            collect_func_decl(S, c);
            return;
        }
        c = ND(S, c)->next_sibling;
    }
}

/* ---- Pass 2: Check Function Bodies ---- */

static uint32_t find_func_body(const sema_ctx_t *S, uint32_t func_def)
{
    uint32_t c = ND(S, func_def)->first_child;
    while (c) {
        if (ND(S, c)->type == AST_BLOCK) return c;
        c = ND(S, c)->next_sibling;
    }
    return 0;
}

static void check_func_def(sema_ctx_t *S, uint32_t node)
{
    uint32_t type_n = child_at(S, node, 0);
    int ret_ptr = ND(S, node)->d.oper.flags;
    uint32_t ret_t = resolve_typespec(S, type_n, ret_ptr);
    uint16_t cuda = ND(S, node)->cuda_flags;

    if ((cuda & CUDA_GLOBAL) && !is_void_type(S, ret_t))
        sema_error(S, node, BC_E081);

    S->cur_ret_type = ret_t;

    push_scope(S);

    uint32_t c = ND(S, node)->first_child;
    while (c) {
        if (ND(S, c)->type == AST_PARAM) {
            uint32_t pt_n = ND(S, c)->first_child;
            int pdepth = ND(S, c)->d.oper.flags;
            uint32_t pt = resolve_typespec(S, pt_n, pdepth);

            uint32_t pname_n = 0;
            uint32_t pc = ND(S, c)->first_child;
            while (pc) {
                if (ND(S, pc)->type == AST_IDENT) { pname_n = pc; break; }
                pc = ND(S, pc)->next_sibling;
            }
            if (pname_n) {
                char pname[128];
                get_text(S, pname_n, pname, sizeof(pname));
                add_sym(S, pname, pt, c, SYM_PARAM, 0);
            }
        }
        c = ND(S, c)->next_sibling;
    }

    uint32_t body = find_func_body(S, node);
    if (body) {
        push_scope(S);
        check_block_stmts(S, body);
        pop_scope(S);
    }

    pop_scope(S);
    S->cur_ret_type = 0;
}

/* ---- Initialization ---- */

void sema_init(sema_ctx_t *S, const parser_t *P, uint32_t root)
{
    (void)root;
    memset(S, 0, sizeof(*S));
    S->P   = P;
    S->src = P->src;

    st_void(S);    /* 0 */
    st_bool(S);    /* 1 */
    st_char(S);    /* 2 */
    st_schar(S);   /* 3 */
    st_uchar(S);   /* 4 */
    st_short(S);   /* 5 */
    st_ushort(S);  /* 6 */
    st_int(S);     /* 7 */
    st_uint(S);    /* 8 */
    st_long(S);    /* 9 */
    st_ulong(S);   /* 10 */
    st_llong(S);   /* 11 */
    st_ullong(S);  /* 12 */
    st_float(S);   /* 13 */
    st_double(S);  /* 14 */
    st_ldouble(S); /* 15 */
    st_error(S);   /* 16 */
}

/* ---- Main Entry Point ---- */

int sema_check(sema_ctx_t *S, uint32_t root)
{
    /* Pass 1: declarations */
    uint32_t c = S->P->nodes[root].first_child;
    while (c) {
        const ast_node_t *n = ND(S, c);
        switch (n->type) {
        case AST_STRUCT_DEF:
            collect_struct_def(S, c);
            break;
        case AST_ENUM_DEF:
            collect_enum_def(S, c);
            break;
        case AST_VAR_DECL:
            if (n->qualifiers & QUAL_TYPEDEF)
                collect_typedef_def(S, c);
            else
                collect_global_var(S, c);
            break;
        case AST_FUNC_DEF:
        case AST_FUNC_DECL:
            collect_func_decl(S, c);
            break;
        case AST_TEMPLATE_DECL:
            collect_template(S, c);
            break;
        default:
            break;
        }
        c = S->P->nodes[c].next_sibling;
    }

    /* Pass 2: function bodies */
    c = S->P->nodes[root].first_child;
    while (c) {
        const ast_node_t *n = ND(S, c);
        if (n->type == AST_FUNC_DEF)
            check_func_def(S, c);
        if (n->type == AST_TEMPLATE_DECL) {
            uint32_t fc = n->first_child;
            while (fc) {
                if (ND(S, fc)->type == AST_FUNC_DEF)
                    check_func_def(S, fc);
                fc = ND(S, fc)->next_sibling;
            }
        }
        c = S->P->nodes[c].next_sibling;
    }

    return S->num_errors > 0 ? BC_ERR_SEMA : BC_OK;
}

/* ---- Type Name for Display ---- */

static const char *stype_kind_names[] = {
    "void", "bool",
    "char", "signed char", "unsigned char",
    "short", "unsigned short",
    "int", "unsigned int",
    "long", "unsigned long",
    "long long", "unsigned long long",
    "float", "double", "long double",
    "__half", "__bfloat16",
    "ptr", "array", "struct", "enum", "func", "vector",
    "<error>"
};

/*
 * Iterative type printer for non-func types.
 * Peels ptr/array wrappers, prints base name, appends suffixes in reverse.
 * Func types encountered mid-chain get the dignified treatment of "func(...)".
 */
static int print_flat_type(const sema_ctx_t *S, uint32_t tidx, char *buf, int size)
{
    if (size <= 0) return 0;
    if (tidx >= S->num_types) return snprintf(buf, (size_t)size, "?");

    #define FLAT_MAX_WRAP 32
    struct { uint8_t kind; uint16_t width; } wrap[FLAT_MAX_WRAP];
    int nw = 0;
    uint32_t cur = tidx;

    for (int g = 0; g < FLAT_MAX_WRAP && cur < S->num_types; g++) {
        uint8_t k = S->types[cur].kind;
        if ((k == STYPE_PTR || k == STYPE_ARRAY) && nw < FLAT_MAX_WRAP) {
            wrap[nw].kind = k;
            wrap[nw].width = S->types[cur].width;
            nw++;
            cur = S->types[cur].inner;
        } else {
            break;
        }
    }

    int n = 0;
    if (cur >= S->num_types) {
        n = snprintf(buf, (size_t)size, "?");
    } else {
        const stype_t *b = &S->types[cur];
        if (b->kind == STYPE_STRUCT) {
            sema_struct_t *sd = find_struct_by_idx(S, b->extra);
            n = (sd && sd->name[0])
                ? snprintf(buf, (size_t)size, "struct %s", sd->name)
                : snprintf(buf, (size_t)size, "struct <anon>");
        } else if (b->kind == STYPE_FUNC) {
            n = snprintf(buf, (size_t)size, "func(...)");
        } else if (b->kind < STYPE_COUNT) {
            n = snprintf(buf, (size_t)size, "%s", stype_kind_names[b->kind]);
        } else {
            n = snprintf(buf, (size_t)size, "<kind:%d>", b->kind);
        }
    }

    for (int i = nw - 1; i >= 0 && n < size; i--) {
        if (wrap[i].kind == STYPE_PTR)
            n += snprintf(buf + n, (size_t)(size - n), "*");
        else
            n += snprintf(buf + n, (size_t)(size - n), "[%u]", wrap[i].width);
    }

    return n;
}

int stype_str(const sema_ctx_t *S, uint32_t tidx, char *buf, int size)
{
    if (size <= 0) return 0;
    if (tidx >= S->num_types) return snprintf(buf, (size_t)size, "<invalid>");
    const stype_t *t = &S->types[tidx];

    if (t->kind != STYPE_FUNC)
        return print_flat_type(S, tidx, buf, size);

    /* Function type: func(params) -> ret */
    int n = snprintf(buf, (size_t)size, "func(");
    int nparams = (int)t->width;
    uint32_t start = t->extra;
    for (int i = 0; i < nparams && i < 16
             && (start + (uint32_t)i) < S->num_params; i++) {
        if (i > 0 && n < size)
            n += snprintf(buf + n, (size_t)(size - n), ", ");
        if (n < size)
            n += print_flat_type(S, S->param_pool[start + (uint32_t)i],
                                 buf + n, size - n);
    }
    if (n < size)
        n += snprintf(buf + n, (size_t)(size - n), ") -> ");
    if (n < size)
        n += print_flat_type(S, t->inner, buf + n, size - n);
    return n;
}

/* ---- Dump: Print Annotated Nodes ---- */

void sema_dump(const sema_ctx_t *S, uint32_t root)
{
    printf("; Sema: %u types interned, %u symbols, %u structs, %d error(s)\n",
           S->num_types, S->num_syms, S->num_structs, S->num_errors);

    printf("; Symbols:\n");
    for (uint32_t i = 0; i < S->num_syms; i++) {
        const sema_sym_t *sym = &S->syms[i];
        char tbuf[256];
        stype_str(S, sym->type, tbuf, sizeof(tbuf));
        const char *kind_names[] = {
            "var", "func", "param", "typedef",
            "struct", "enum", "enum_const", "template"
        };
        printf(";   %-20s %-10s %s\n",
               sym->name,
               sym->kind < 8 ? kind_names[sym->kind] : "?",
               tbuf);
    }

    printf("; Annotated expressions:\n");
    int count = 0;
    for (uint32_t i = 1; i < S->P->num_nodes && i < BC_MAX_NODES; i++) {
        if (S->node_types[i] == 0) continue;
        const ast_node_t *n = &S->P->nodes[i];
        char tbuf[256];
        stype_str(S, S->node_types[i], tbuf, sizeof(tbuf));

        printf(";   node[%u] %s @ %u:%u → %s\n",
               i, ast_type_name(n->type), n->line, n->col, tbuf);
        count++;
    }
    printf("; %d nodes annotated\n", count);
    (void)root;
}
