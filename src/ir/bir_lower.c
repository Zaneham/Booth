#include "bir_lower.h"
#include "amdgpu.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/*
 * AST-to-BIR lowering.
 *
 * Strategy:
 *   - Parameters: direct SSA values (BIR_PARAM)
 *   - Local variables: alloca + store/load (mem2reg cleans up later)
 *   - Templates: store definitions, deduce types at call sites, instantiate
 *   - Device functions only; host scanned for launches
 *   - CUDA builtins (threadIdx.x etc.) detected as member access patterns
 */

/* ---- Limits ---- */

#define MAX_SYMS        1024
#define MAX_STRUCTS     64
#define MAX_ENUMS       256
#define MAX_TYPEDEFS    64
#define MAX_TEMPLATES   32
#define MAX_SCOPES      64
#define MAX_LOOPS       32

/* ---- Internal Types ---- */

typedef struct {
    char        name[128];
    uint32_t    ref;        /* param: BIR_MAKE_VAL(idx). local: alloca inst idx */
    uint32_t    type;       /* BIR type index of the value (not pointer) */
    int         is_alloca;
} sym_t;

typedef struct {
    char        name[64];
    uint32_t    field_types[16];
    char        field_names[16][64];
    int         num_fields;
    uint32_t    bir_type;
} struct_def_t;

typedef struct {
    char        name[64];
    int64_t     value;
} enum_val_t;

typedef struct {
    char        name[64];
    uint32_t    bir_type;
} typedef_def_t;

typedef struct {
    uint32_t    ast;        /* AST_TEMPLATE_DECL node index */
    char        name[64];   /* function name inside */
} template_def_t;

typedef struct {
    char        name[64];
    uint32_t    type;       /* BIR type index */
    int64_t     ival;       /* for non-type params */
    int         is_type;    /* 1 = typename, 0 = int param */
} binding_t;

typedef struct {
    const parser_t  *P;
    bir_module_t    *M;
    const char      *src;

    sym_t           syms[MAX_SYMS];
    int             nsyms;
    int             scope_stack[MAX_SCOPES];
    int             scope_depth;

    struct_def_t    structs[MAX_STRUCTS];
    int             nstructs;
    enum_val_t      enums[MAX_ENUMS];
    int             nenums;
    typedef_def_t   typedefs[MAX_TYPEDEFS];
    int             ntypedefs;

    template_def_t  templates[MAX_TEMPLATES];
    int             ntemplates;
    binding_t       bindings[8];
    int             nbindings;

    /* Current function state */
    uint32_t        cur_func;
    uint32_t        cur_block;
    uint32_t        base_inst;

    /* Loop stack */
    uint32_t        break_tgt[MAX_LOOPS];
    uint32_t        cont_tgt[MAX_LOOPS];
    int             loop_depth;

    /* Switch exit block */
    uint32_t        switch_exit;

    /* Labels for goto/label */
    struct { char name[64]; uint32_t block; } labels[256];
    int             nlabels;

    const sema_ctx_t *sema;   /* NULL if sema didn't run */

    uint32_t        cur_node;   /* AST node being lowered — for source loc tracking */

    bc_error_t      errors[BC_MAX_ERRORS];
    int             nerrors;
} lower_t;

/* ---- AST Navigation ---- */

static const ast_node_t *ND(const lower_t *L, uint32_t i)
{
    return &L->P->nodes[i];
}

static uint32_t child_at(const lower_t *L, uint32_t node, int n)
{
    uint32_t c = ND(L, node)->first_child;
    for (int i = 0; i < n && c; i++)
        c = ND(L, c)->next_sibling;
    return c;
}

static const char *lw_tptr(const lower_t *L, uint32_t off)
{
    if (off >= BC_ANON_BASE)
        return L->P->anon_buf + (off - BC_ANON_BASE);
    return L->src + off;
}

static void get_text(const lower_t *L, uint32_t node, char *buf, int sz)
{
    const ast_node_t *n = ND(L, node);
    int len = (int)n->d.text.len;
    if (len >= sz) len = sz - 1;
    memcpy(buf, lw_tptr(L, n->d.text.offset), (size_t)len);
    buf[len] = '\0';
}

static int text_eq(const lower_t *L, uint32_t node, const char *s)
{
    const ast_node_t *n = ND(L, node);
    int len = (int)n->d.text.len;
    return (int)strlen(s) == len
        && memcmp(lw_tptr(L, n->d.text.offset), s, (size_t)len) == 0;
}

static void lower_error(lower_t *L, uint32_t node, bc_eid_t eid, ...)
{
    if (L->nerrors < BC_MAX_ERRORS) {
        bc_error_t *e = &L->errors[L->nerrors++];
        if (node) {
            const ast_node_t *n = ND(L, node);
            e->loc.line = n->line;
            e->loc.col  = n->col;
        }
        e->loc.offset = 0;
        e->code = BC_ERR_LOWER;
        e->eid  = (uint16_t)eid;
        va_list ap;
        va_start(ap, eid);
        vsnprintf(e->msg, sizeof(e->msg), bc_efmt(eid), ap);
        va_end(ap);
    }
}

/* ---- Scope ---- */

static void push_scope(lower_t *L)
{
    if (L->scope_depth < MAX_SCOPES)
        L->scope_stack[L->scope_depth++] = L->nsyms;
}

static void pop_scope(lower_t *L)
{
    if (L->scope_depth > 0)
        L->nsyms = L->scope_stack[--L->scope_depth];
}

static void add_sym(lower_t *L, const char *name, uint32_t ref,
                    uint32_t type, int is_alloca)
{
    if (L->nsyms >= MAX_SYMS) return;
    sym_t *s = &L->syms[L->nsyms++];
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->ref = ref;
    s->type = type;
    s->is_alloca = is_alloca;
}

static sym_t *find_sym(lower_t *L, const char *name)
{
    for (int i = L->nsyms - 1; i >= 0; i--)
        if (strcmp(L->syms[i].name, name) == 0)
            return &L->syms[i];
    return NULL;
}

/* ---- Lookups ---- */

static int find_enum(lower_t *L, const char *name, int64_t *val)
{
    for (int i = 0; i < L->nenums; i++)
        if (strcmp(L->enums[i].name, name) == 0) {
            *val = L->enums[i].value;
            return 1;
        }
    return 0;
}

static int find_typedef(lower_t *L, const char *name, uint32_t *type)
{
    for (int i = 0; i < L->ntypedefs; i++)
        if (strcmp(L->typedefs[i].name, name) == 0) {
            *type = L->typedefs[i].bir_type;
            return 1;
        }
    return 0;
}

static int find_binding(lower_t *L, const char *name, uint32_t *type)
{
    for (int i = 0; i < L->nbindings; i++)
        if (L->bindings[i].is_type
            && strcmp(L->bindings[i].name, name) == 0) {
            *type = L->bindings[i].type;
            return 1;
        }
    return 0;
}

static int find_binding_int(lower_t *L, const char *name, int64_t *val)
{
    for (int i = 0; i < L->nbindings; i++)
        if (!L->bindings[i].is_type
            && strcmp(L->bindings[i].name, name) == 0) {
            *val = L->bindings[i].ival;
            return 1;
        }
    return 0;
}

static template_def_t *find_template(lower_t *L, const char *name)
{
    for (int i = 0; i < L->ntemplates; i++)
        if (strcmp(L->templates[i].name, name) == 0)
            return &L->templates[i];
    return NULL;
}

/* ---- Operator Name Helpers ---- */

static void normalize_op_name(const char *raw, char *out, int outsz)
{
    int ri = 0, oi = 0;
    while (ri < 8 && raw[ri] && oi < outsz - 1)
        out[oi++] = raw[ri++];
    while (raw[ri] == ' ' || raw[ri] == '\t') ri++;
    while (raw[ri] && oi < outsz - 1)
        out[oi++] = raw[ri++];
    out[oi] = '\0';
}

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

/* ---- Instruction Emission ---- */

static uint32_t emit(lower_t *L, uint16_t op, uint32_t type,
                     uint8_t nops, uint8_t subop)
{
    if (L->M->num_insts >= BIR_MAX_INSTS) return 0;
    uint32_t idx = L->M->num_insts++;
    bir_inst_t *I = &L->M->insts[idx];
    memset(I, 0, sizeof(*I));
    I->op   = op;
    I->type = type;
    I->num_operands = nops;
    I->subop = subop;
    L->M->blocks[L->cur_block].num_insts++;
    /* Leave a trail of breadcrumbs back to the source */
    if (L->cur_node && L->cur_node < L->P->num_nodes)
        L->M->inst_lines[idx] = ND(L, L->cur_node)->line;
    return idx;
}

static void set_op(lower_t *L, uint32_t inst, int slot, uint32_t val)
{
    L->M->insts[inst].operands[slot] = val;
}

/* ---- Block Management ---- */

static uint32_t new_block(lower_t *L, const char *name)
{
    if (L->M->num_blocks >= BIR_MAX_BLOCKS) return 0;
    uint32_t idx = L->M->num_blocks++;
    bir_block_t *B = &L->M->blocks[idx];
    B->name = bir_add_string(L->M, name, (uint32_t)strlen(name));
    B->first_inst = L->M->num_insts;
    B->num_insts  = 0;
    /* Count blocks in current function */
    L->M->funcs[L->cur_func].num_blocks++;
    return idx;
}

static void set_block(lower_t *L, uint32_t bidx)
{
    L->cur_block = bidx;
    if (L->M->blocks[bidx].num_insts == 0)
        L->M->blocks[bidx].first_inst = L->M->num_insts;
}

/* Check if current block already has a terminator */
static int block_terminated(const lower_t *L)
{
    const bir_block_t *B = &L->M->blocks[L->cur_block];
    if (B->num_insts == 0) return 0;
    uint32_t last = B->first_inst + B->num_insts - 1;
    uint16_t op = L->M->insts[last].op;
    return op == BIR_BR || op == BIR_BR_COND || op == BIR_RET
        || op == BIR_UNREACHABLE || op == BIR_SWITCH;
}

/* ---- Label Lookup (Goto/Label) ---- */

static uint32_t find_or_create_label(lower_t *L, const char *name)
{
    for (int i = 0; i < L->nlabels; i++)
        if (strcmp(L->labels[i].name, name) == 0)
            return L->labels[i].block;
    if (L->nlabels >= 256) {
        lower_error(L, 0, BC_E100);
        return L->cur_block;
    }
    int idx = L->nlabels++;
    snprintf(L->labels[idx].name, sizeof(L->labels[0].name), "%s", name);
    L->labels[idx].block = new_block(L, name);
    return L->labels[idx].block;
}

/* ---- Type Resolution ---- */

static uint32_t resolve_basic(lower_t *L, int kind)
{
    switch (kind) {
    case TYPE_VOID:     return bir_type_void(L->M);
    case TYPE_BOOL:     return bir_type_int(L->M, 1);
    case TYPE_CHAR:     return bir_type_int(L->M, 8);
    case TYPE_SHORT:    return bir_type_int(L->M, 16);
    case TYPE_INT:      return bir_type_int(L->M, 32);
    case TYPE_LONG:     return bir_type_int(L->M, 64);
    case TYPE_LLONG:    return bir_type_int(L->M, 64);
    case TYPE_FLOAT:    return bir_type_float(L->M, 32);
    case TYPE_DOUBLE:   return bir_type_float(L->M, 64);
    case TYPE_LDOUBLE:  return bir_type_float(L->M, 64);
    case TYPE_UNSIGNED: return bir_type_int(L->M, 32);
    case TYPE_SIGNED:   return bir_type_int(L->M, 32);
    default:            return bir_type_int(L->M, 32);
    }
}

static uint32_t resolve_type(lower_t *L, uint32_t node, int ptr_depth,
                             uint16_t cuda)
{
    uint32_t base;
    if (!node) return bir_type_int(L->M, 32);
    const ast_node_t *n = ND(L, node);

    if (n->type != AST_TYPE_SPEC)
        return bir_type_int(L->M, 32);

    int kind = n->d.btype.kind;

    if (kind == TYPE_NAME) {
        char name[128] = {0};
        uint32_t nc = n->first_child;
        if (nc) get_text(L, nc, name, sizeof(name));

        uint32_t bound;
        if (find_binding(L, name, &bound))
            base = bound;
        else if (find_typedef(L, name, &bound))
            base = bound;
        else if (strcmp(name, "size_t") == 0)
            base = bir_type_int(L->M, 64);
        else if (strcmp(name, "__half") == 0 || strcmp(name, "half") == 0
                 || strcmp(name, "_Float16") == 0)
            base = bir_type_float(L->M, 16);
        else if (strcmp(name, "__nv_bfloat16") == 0
                 || strcmp(name, "nv_bfloat16") == 0
                 || strcmp(name, "__bfloat16") == 0
                 || strcmp(name, "__hip_bfloat16") == 0)
            base = bir_type_bfloat(L->M);
        else {
            /* ---- Vector types → synthetic structs ---- */
            static const struct { const char *name; int elem; int lanes; } vec_map[] = {
                {"float2",TYPE_FLOAT,2},{"float3",TYPE_FLOAT,3},{"float4",TYPE_FLOAT,4},
                {"double2",TYPE_DOUBLE,2},{"double3",TYPE_DOUBLE,3},{"double4",TYPE_DOUBLE,4},
                {"int2",TYPE_INT,2},{"int3",TYPE_INT,3},{"int4",TYPE_INT,4},
                {"uint2",TYPE_INT,2},{"uint3",TYPE_INT,3},{"uint4",TYPE_INT,4},
                {"char2",TYPE_CHAR,2},{"char3",TYPE_CHAR,3},{"char4",TYPE_CHAR,4},
                {"uchar2",TYPE_CHAR,2},{"uchar3",TYPE_CHAR,3},{"uchar4",TYPE_CHAR,4},
                {"short2",TYPE_SHORT,2},{"short3",TYPE_SHORT,3},{"short4",TYPE_SHORT,4},
                {"ushort2",TYPE_SHORT,2},{"ushort3",TYPE_SHORT,3},{"ushort4",TYPE_SHORT,4},
                {"long2",TYPE_LONG,2},{"long3",TYPE_LONG,3},{"long4",TYPE_LONG,4},
                {"ulong2",TYPE_LONG,2},{"ulong3",TYPE_LONG,3},{"ulong4",TYPE_LONG,4},
                {"longlong2",TYPE_LLONG,2},{"ulonglong2",TYPE_LLONG,2},
                {"dim3",TYPE_INT,3},
                {NULL,0,0}
            };
            int vfound = 0;
            for (int vi = 0; vec_map[vi].name; vi++) {
                if (strcmp(name, vec_map[vi].name) != 0) continue;
                uint32_t et = resolve_basic(L, vec_map[vi].elem);
                int lanes = vec_map[vi].lanes;
                uint32_t fields[4] = {et, et, et, et};
                uint32_t st = bir_type_struct(L->M, fields, lanes);
                /* Register as struct_def so member access works */
                if (L->nstructs < MAX_STRUCTS) {
                    static const char *xyzw[] = {"x","y","z","w"};
                    struct_def_t *sd = &L->structs[L->nstructs];
                    memset(sd, 0, sizeof(*sd));
                    snprintf(sd->name, sizeof(sd->name), "%s", name);
                    sd->num_fields = lanes;
                    sd->bir_type = st;
                    for (int fi = 0; fi < lanes; fi++) {
                        sd->field_types[fi] = et;
                        snprintf(sd->field_names[fi], sizeof(sd->field_names[0]),
                                 "%s", xyzw[fi]);
                    }
                    L->nstructs++;
                }
                base = st;
                vfound = 1;
                break;
            }
            if (!vfound) {
                /* Check user-defined structs (C++ style: Vec2 v; without 'struct') */
                for (int si = 0; si < L->nstructs; si++) {
                    if (strcmp(L->structs[si].name, name) == 0) {
                        base = L->structs[si].bir_type;
                        vfound = 1;
                        break;
                    }
                }
                if (!vfound)
                    base = bir_type_int(L->M, 32);
            }
        }
    } else if (kind == TYPE_STRUCT || kind == TYPE_UNION) {
        char name[128] = {0};
        uint32_t nc = n->first_child;
        if (nc) get_text(L, nc, name, sizeof(name));
        base = bir_type_int(L->M, 32); /* fallback */
        for (int i = 0; i < L->nstructs; i++) {
            if (strcmp(L->structs[i].name, name) == 0) {
                base = L->structs[i].bir_type;
                break;
            }
        }
    } else if (kind == TYPE_ENUM) {
        base = bir_type_int(L->M, 32);
    } else {
        base = resolve_basic(L, kind);
    }

    for (int i = 0; i < ptr_depth; i++) {
        int as = BIR_AS_GLOBAL;
        if (cuda & CUDA_SHARED)   as = BIR_AS_SHARED;
        if (cuda & CUDA_CONSTANT) as = BIR_AS_CONSTANT;
        base = bir_type_ptr(L->M, base, as);
    }

    return base;
}

/* ---- Type Queries ---- */

static uint32_t ref_type(const lower_t *L, uint32_t ref)
{
    if (ref == BIR_VAL_NONE) return 0;
    if (BIR_VAL_IS_CONST(ref)) {
        uint32_t ci = BIR_VAL_INDEX(ref);
        return ci < L->M->num_consts ? L->M->consts[ci].type : 0;
    }
    uint32_t ii = BIR_VAL_INDEX(ref);
    return ii < L->M->num_insts ? L->M->insts[ii].type : 0;
}

static int is_float_type(const lower_t *L, uint32_t t)
{
    if (t >= L->M->num_types) return 0;
    uint8_t k = L->M->types[t].kind;
    return k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT;
}

/* Implicit type coercion: insert conversion if val's type != dst.
 * Handles int->float, float->float (width change), int widening.
 * Used by binary ops to promote operands (C usual arithmetic). */
static uint32_t coerce_to(lower_t *L, uint32_t val, uint32_t dst_t,
                          int src_unsigned)
{
    uint32_t src_t = ref_type(L, val);
    if (src_t == dst_t) return val;
    if (src_t >= L->M->num_types || dst_t >= L->M->num_types) return val;

    int sf = is_float_type(L, src_t), df = is_float_type(L, dst_t);
    uint16_t cop;
    if (sf && df) {
        cop = (L->M->types[dst_t].width > L->M->types[src_t].width)
              ? BIR_FPEXT : BIR_FPTRUNC;
    } else if (!sf && df) {
        cop = src_unsigned ? BIR_UITOFP : BIR_SITOFP;
    } else if (sf && !df) {
        cop = src_unsigned ? BIR_FPTOUI : BIR_FPTOSI;
    } else {
        int sw = L->M->types[src_t].width;
        int dw = L->M->types[dst_t].width;
        if (dw > sw)      cop = src_unsigned ? BIR_ZEXT : BIR_SEXT;
        else if (dw < sw) cop = BIR_TRUNC;
        else              return val; /* same width int — no conversion */
    }
    uint32_t inst = emit(L, cop, dst_t, 1, 0);
    set_op(L, inst, 0, val);
    return BIR_MAKE_VAL(inst);
}

static int is_ptr_type(const lower_t *L, uint32_t t)
{
    return t < L->M->num_types && L->M->types[t].kind == BIR_TYPE_PTR;
}

static uint32_t ptr_inner(const lower_t *L, uint32_t t)
{
    if (is_ptr_type(L, t)) return L->M->types[t].inner;
    return 0;
}

static int is_void_type(const lower_t *L, uint32_t t)
{
    return t < L->M->num_types && L->M->types[t].kind == BIR_TYPE_VOID;
}

/* ---- Sema Signedness Query ---- */

static int node_is_unsigned(const lower_t *L, uint32_t ast_node)
{
    if (!L->sema || !ast_node || ast_node >= BC_MAX_NODES) return 0;
    uint32_t st = L->sema->node_types[ast_node];
    if (!st || st >= L->sema->num_types) return 0;
    uint8_t k = L->sema->types[st].kind;
    return k == STYPE_BOOL || k == STYPE_UCHAR || k == STYPE_USHORT
        || k == STYPE_UINT || k == STYPE_ULONG || k == STYPE_ULLONG;
}

/* Width of an int-lit as typed by sema. Fallback 32 when sema was
   skipped, so --ast / --ir without --sema still lower sensibly. */
static int ilit_w(const lower_t *L, uint32_t ast_node)
{
    if (!L->sema || !ast_node || ast_node >= BC_MAX_NODES) return 32;
    uint32_t st = L->sema->node_types[ast_node];
    if (!st || st >= L->sema->num_types) return 32;
    uint8_t k = L->sema->types[st].kind;
    if (k == STYPE_LONG  || k == STYPE_ULONG
     || k == STYPE_LLONG || k == STYPE_ULLONG)
        return 64;
    return 32;
}

/* ---- Literal Parsing ---- */

static int64_t parse_int_text(const char *s, int len)
{
    char buf[64];
    int n = len > 63 ? 63 : len;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    while (n > 0 && (buf[n-1]=='u'||buf[n-1]=='U'||
                     buf[n-1]=='l'||buf[n-1]=='L'))
        buf[--n] = '\0';
    return strtoll(buf, NULL, 0);
}

static double parse_float_text(const char *s, int len, int *is_f32)
{
    char buf[64];
    int n = len > 63 ? 63 : len;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    *is_f32 = 0;
    if (n > 0 && (buf[n-1]=='f'||buf[n-1]=='F')) {
        buf[--n] = '\0';
        *is_f32 = 1;
    }
    return strtod(buf, NULL);
}

/* ---- Operator Mapping ---- */

static int bin_op_code(int tok, int is_fp, int is_unsigned)
{
    switch (tok) {
    case TOK_PLUS:    return is_fp ? BIR_FADD : BIR_ADD;
    case TOK_MINUS:   return is_fp ? BIR_FSUB : BIR_SUB;
    case TOK_STAR:    return is_fp ? BIR_FMUL : BIR_MUL;
    case TOK_SLASH:   return is_fp ? BIR_FDIV : (is_unsigned ? BIR_UDIV : BIR_SDIV);
    case TOK_PERCENT: return is_fp ? BIR_FREM : (is_unsigned ? BIR_UREM : BIR_SREM);
    case TOK_AMP:     return BIR_AND;
    case TOK_PIPE:    return BIR_OR;
    case TOK_CARET:   return BIR_XOR;
    case TOK_SHL:     return BIR_SHL;
    case TOK_SHR:     return is_unsigned ? BIR_LSHR : BIR_ASHR;
    default:          return -1;
    }
}

static int cmp_pred(int tok, int is_fp, int is_unsigned)
{
    if (is_fp) {
        switch (tok) {
        case TOK_EQ: return BIR_FCMP_OEQ;
        case TOK_NE: return BIR_FCMP_ONE;
        case TOK_LT: return BIR_FCMP_OLT;
        case TOK_LE: return BIR_FCMP_OLE;
        case TOK_GT: return BIR_FCMP_OGT;
        case TOK_GE: return BIR_FCMP_OGE;
        default:     return BIR_FCMP_OEQ;
        }
    }
    switch (tok) {
    case TOK_EQ: return BIR_ICMP_EQ;
    case TOK_NE: return BIR_ICMP_NE;
    case TOK_LT: return is_unsigned ? BIR_ICMP_ULT : BIR_ICMP_SLT;
    case TOK_LE: return is_unsigned ? BIR_ICMP_ULE : BIR_ICMP_SLE;
    case TOK_GT: return is_unsigned ? BIR_ICMP_UGT : BIR_ICMP_SGT;
    case TOK_GE: return is_unsigned ? BIR_ICMP_UGE : BIR_ICMP_SGE;
    default:     return BIR_ICMP_EQ;
    }
}

static int is_cmp_tok(int t)
{
    return t==TOK_EQ||t==TOK_NE||t==TOK_LT||t==TOK_GT||t==TOK_LE||t==TOK_GE;
}

static int compound_base(int tok)
{
    switch (tok) {
    case TOK_PLUS_EQ:    return TOK_PLUS;
    case TOK_MINUS_EQ:   return TOK_MINUS;
    case TOK_STAR_EQ:    return TOK_STAR;
    case TOK_SLASH_EQ:   return TOK_SLASH;
    case TOK_PERCENT_EQ: return TOK_PERCENT;
    case TOK_AMP_EQ:     return TOK_AMP;
    case TOK_PIPE_EQ:    return TOK_PIPE;
    case TOK_CARET_EQ:   return TOK_CARET;
    case TOK_SHL_EQ:     return TOK_SHL;
    case TOK_SHR_EQ:     return TOK_SHR;
    default:             return -1;
    }
}

static int is_compound_assign(int tok)
{
    return compound_base(tok) >= 0;
}

/* ---- Forward Declarations ---- */

static uint32_t lower_expr(lower_t *L, uint32_t node);
static uint32_t lower_lvalue(lower_t *L, uint32_t node);
static void     lower_stmt(lower_t *L, uint32_t node);
static void     lower_block_stmts(lower_t *L, uint32_t node);
static void     collect_struct(lower_t *L, uint32_t node);
static void     lower_func_body(lower_t *L, uint32_t func_def,
                                uint16_t cuda_flags, const char *name_override);

/* ---- CUDA Builtin Detection ---- */

static int try_cuda_builtin(lower_t *L, uint32_t node, uint32_t *out)
{
    const ast_node_t *n = ND(L, node);
    if (n->type != AST_MEMBER) return 0;

    uint32_t obj = n->first_child;
    uint32_t fld = obj ? ND(L, obj)->next_sibling : 0;
    if (!obj || !fld) return 0;
    if (ND(L, obj)->type != AST_IDENT) return 0;
    if (ND(L, fld)->type != AST_IDENT) return 0;

    uint16_t op;
    if      (text_eq(L, obj, "threadIdx")) op = BIR_THREAD_ID;
    else if (text_eq(L, obj, "blockIdx"))  op = BIR_BLOCK_ID;
    else if (text_eq(L, obj, "blockDim"))  op = BIR_BLOCK_DIM;
    else if (text_eq(L, obj, "gridDim"))   op = BIR_GRID_DIM;
    else return 0;

    uint8_t dim = 0;
    if      (text_eq(L, fld, "y")) dim = 1;
    else if (text_eq(L, fld, "z")) dim = 2;

    uint32_t t_i32 = bir_type_int(L->M, 32);
    uint32_t inst = emit(L, op, t_i32, 0, dim);
    *out = BIR_MAKE_VAL(inst);
    return 1;
}

/* ---- Expression Lowering ---- */

static uint32_t lower_expr(lower_t *L, uint32_t node)
{
    if (!node) return BIR_VAL_NONE;
    L->cur_node = node;
    const ast_node_t *n = ND(L, node);

    switch (n->type) {
    case AST_NONE:
        return BIR_VAL_NONE;

    case AST_INT_LIT: {
        int64_t val = parse_int_text(L->src + n->d.text.offset,
                                     (int)n->d.text.len);
        uint32_t t = bir_type_int(L->M, ilit_w(L, node));
        return BIR_MAKE_CONST(bir_const_int(L->M, t, val));
    }

    case AST_FLOAT_LIT: {
        int is_f32;
        double val = parse_float_text(L->src + n->d.text.offset,
                                      (int)n->d.text.len, &is_f32);
        uint32_t t = is_f32 ? bir_type_float(L->M, 32)
                            : bir_type_float(L->M, 64);
        return BIR_MAKE_CONST(bir_const_float(L->M, t, val));
    }

    case AST_BOOL_LIT: {
        uint32_t t = bir_type_int(L->M, 1);
        return BIR_MAKE_CONST(bir_const_int(L->M, t, n->d.ival));
    }

    case AST_NULL_LIT: {
        uint32_t pt = bir_type_ptr(L->M, bir_type_int(L->M, 8),
                                   BIR_AS_GENERIC);
        return BIR_MAKE_CONST(bir_const_null(L->M, pt));
    }

    case AST_CHAR_LIT: {
        /* 'c' → i8 constant */
        int64_t val = 0;
        if (n->d.text.len >= 3) /* 'X' */
            val = (int64_t)(unsigned char)L->src[n->d.text.offset + 1];
        uint32_t t = bir_type_int(L->M, 8);
        return BIR_MAKE_CONST(bir_const_int(L->M, t, val));
    }

    case AST_IDENT: {
        char name[128];
        get_text(L, node, name, sizeof(name));

        /* Builtin constant: warpSize (HIP) */
        if (strcmp(name, "warpSize") == 0 && L->sema) {
            int wave_size = AMD_WAVE_SIZE;
            if (L->sema->amd_target == AMD_TARGET_GFX90A || L->sema->amd_target == AMD_TARGET_GFX942)
                wave_size = AMD_WAVE64;
            uint32_t t = bir_type_int(L->M, 32);
            return BIR_MAKE_CONST(bir_const_int(L->M, t, wave_size));
        }

        /* Enum constant? */
        int64_t eval;
        if (find_enum(L, name, &eval))
            return BIR_MAKE_CONST(bir_const_int(L->M,
                bir_type_int(L->M, 32), eval));

        /* Template non-type binding? */
        int64_t bval;
        if (find_binding_int(L, name, &bval))
            return BIR_MAKE_CONST(bir_const_int(L->M,
                bir_type_int(L->M, 32), bval));

        sym_t *s = find_sym(L, name);
        if (!s) {
            /* Check file-scope globals (__shared__, __device__, __constant__) */
            for (uint32_t gi = 0; gi < L->M->num_globals; gi++) {
                bir_global_t *G = &L->M->globals[gi];
                if (G->name < L->M->string_len
                    && strcmp(&L->M->strings[G->name], name) == 0) {
                    int adrspc = G->addrspace;
                    uint32_t ptr_t = bir_type_ptr(L->M, G->type, adrspc);
                    if (G->cuda_flags & CUDA_SHARED) {
                        uint32_t sa = emit(L, BIR_SHARED_ALLOC, ptr_t, 0, 0);
                        add_sym(L, name, sa, G->type, 1);
                    } else {
                        uint32_t gr = emit(L, BIR_GLOBAL_REF, ptr_t, 0,
                                           (uint8_t)gi);
                        add_sym(L, name, gr, G->type, 1);
                    }
                    s = find_sym(L, name);
                    break;
                }
            }
        }
        if (!s) {
            lower_error(L, node, BC_E101);
            return BIR_VAL_NONE;
        }
        if (s->is_alloca) {
            /* Arrays and structs decay to pointer — return alloca addr
             * directly.  Loading from an array alloca reads uninit
             * scratch.  Loading a whole struct produces a bulk load
             * that isel can't decompose into per-field dwords — the
             * assignment path needs the pointer to fire its per-field
             * copy decomposition.  Without this, fbank[slot] = site
             * stores only field 0 and the rest read back as whatever
             * the silicon had for breakfast. */
            if (s->type < L->M->num_types &&
                (L->M->types[s->type].kind == BIR_TYPE_ARRAY ||
                 L->M->types[s->type].kind == BIR_TYPE_STRUCT))
                return BIR_MAKE_VAL(s->ref);
            uint32_t inst = emit(L, BIR_LOAD, s->type, 1, 0);
            set_op(L, inst, 0, BIR_MAKE_VAL(s->ref));
            return BIR_MAKE_VAL(inst);
        }
        return s->ref; /* param — direct SSA value */
    }

    case AST_PAREN:
        return lower_expr(L, n->first_child);

    case AST_MEMBER: {
        uint32_t result;
        if (try_cuda_builtin(L, node, &result))
            return result;

        /* Struct member: get address, then load */
        uint32_t addr = lower_lvalue(L, node);
        if (addr == BIR_VAL_NONE) return BIR_VAL_NONE;
        uint32_t pt = ref_type(L, addr);
        uint32_t et = ptr_inner(L, pt);
        if (!et) et = bir_type_int(L->M, 32);
        uint32_t inst = emit(L, BIR_LOAD, et, 1, 0);
        set_op(L, inst, 0, addr);
        return BIR_MAKE_VAL(inst);
    }

    case AST_SUBSCRIPT: {
        uint32_t base_n = n->first_child;
        uint32_t idx_n  = ND(L, base_n)->next_sibling;

        uint32_t base_v = lower_expr(L, base_n);
        uint32_t idx_v  = lower_expr(L, idx_n);
        uint32_t bt     = ref_type(L, base_v);
        uint32_t et     = ptr_inner(L, bt);
        if (!et) et = bir_type_int(L->M, 32);

        /* Array decay: ptr(T[N]) → ptr(T) so GEP strides by element */
        uint32_t gep_t = bt;
        if (et < L->M->num_types &&
            L->M->types[et].kind == BIR_TYPE_ARRAY) {
            uint32_t arr_el = L->M->types[et].inner;
            uint8_t as = is_ptr_type(L, bt) ? L->M->types[bt].addrspace : 0;
            gep_t = bir_type_ptr(L->M, arr_el, as);
            et = arr_el;
        }

        uint32_t gep = emit(L, BIR_GEP, gep_t, 2, 0);
        set_op(L, gep, 0, base_v);
        set_op(L, gep, 1, idx_v);

        /* Multi-dim array or struct: yield pointer, don't load */
        if (et < L->M->num_types &&
            (L->M->types[et].kind == BIR_TYPE_ARRAY ||
             L->M->types[et].kind == BIR_TYPE_STRUCT))
            return BIR_MAKE_VAL(gep);

        uint32_t ld = emit(L, BIR_LOAD, et, 1, 0);
        set_op(L, ld, 0, BIR_MAKE_VAL(gep));
        return BIR_MAKE_VAL(ld);
    }

    case AST_BINARY: {
        int op = n->d.oper.op;
        uint32_t lhs_n = n->first_child;
        uint32_t rhs_n = ND(L, lhs_n)->next_sibling;

        /* Assignment */
        if (op == TOK_ASSIGN) {
            uint32_t ptr = lower_lvalue(L, lhs_n);

            /* Bare init list: pixel = { r, g, b } → field-by-field store */
            if (ND(L, rhs_n)->type == AST_INIT_LIST) {
                uint32_t pt = ref_type(L, ptr);
                uint32_t st = ptr_inner(L, pt);
                struct_def_t *sd = NULL;
                for (int si = 0; si < L->nstructs; si++) {
                    if (L->structs[si].bir_type == st) { sd = &L->structs[si]; break; }
                }
                if (sd) {
                    uint32_t el = ND(L, rhs_n)->first_child;
                    for (int fi = 0; fi < sd->num_fields && el; fi++) {
                        uint32_t val = lower_expr(L, el);
                        uint32_t fpt = bir_type_ptr(L->M, sd->field_types[fi], BIR_AS_PRIVATE);
                        uint32_t ci  = BIR_MAKE_CONST(bir_const_int(L->M, bir_type_int(L->M, 32), fi));
                        uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
                        set_op(L, gep, 0, ptr);
                        set_op(L, gep, 1, ci);
                        uint32_t s = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                        set_op(L, s, 0, val);
                        set_op(L, s, 1, BIR_MAKE_VAL(gep));
                        el = ND(L, el)->next_sibling;
                    }
                    /* Return the LHS pointer dereferenced, in case used as expr */
                    uint32_t ld = emit(L, BIR_LOAD, st, 1, 0);
                    set_op(L, ld, 0, ptr);
                    return BIR_MAKE_VAL(ld);
                }
            }

            uint32_t val = lower_expr(L, rhs_n);

            /* Struct copy: p = arr[i] where RHS is a pointer to a
             * struct in foreign memory (global/shared). The subscript
             * lowering yielded a GEP, not a loaded value — sensible
             * for member access, catastrophic for whole-struct copy.
             * Without this the alloca gets a raw pointer dumped into
             * its first 8 bytes and the remaining fields read back
             * as whatever the silicon was dreaming about. */
            uint32_t vt = ref_type(L, val);
            uint32_t dt = ptr_inner(L, ref_type(L, ptr));
            if (is_ptr_type(L, vt) && dt &&
                dt < L->M->num_types &&
                L->M->types[dt].kind == BIR_TYPE_STRUCT) {
                struct_def_t *sd = NULL;
                for (int si = 0; si < L->nstructs; si++) {
                    if (L->structs[si].bir_type == dt) {
                        sd = &L->structs[si]; break;
                    }
                }
                if (sd) {
                    uint8_t src_as = L->M->types[vt].addrspace;
                    uint8_t dst_as = L->M->types[ref_type(L, ptr)].addrspace;
                    uint32_t t32 = bir_type_int(L->M, 32);
                    for (int fi = 0; fi < sd->num_fields && fi < 16; fi++) {
                        uint32_t ci = BIR_MAKE_CONST(
                            bir_const_int(L->M, t32, fi));
                        uint32_t spt = bir_type_ptr(L->M,
                            sd->field_types[fi], src_as);
                        uint32_t sg = emit(L, BIR_GEP, spt, 2, 0);
                        set_op(L, sg, 0, val);
                        set_op(L, sg, 1, ci);
                        uint32_t ld = emit(L, BIR_LOAD,
                            sd->field_types[fi], 1, 0);
                        set_op(L, ld, 0, BIR_MAKE_VAL(sg));
                        uint32_t dpt = bir_type_ptr(L->M,
                            sd->field_types[fi], dst_as);
                        uint32_t dg = emit(L, BIR_GEP, dpt, 2, 0);
                        set_op(L, dg, 0, ptr);
                        set_op(L, dg, 1, ci);
                        uint32_t s = emit(L, BIR_STORE,
                            bir_type_void(L->M), 2, 0);
                        set_op(L, s, 0, BIR_MAKE_VAL(ld));
                        set_op(L, s, 1, BIR_MAKE_VAL(dg));
                    }
                    uint32_t ld = emit(L, BIR_LOAD, dt, 1, 0);
                    set_op(L, ld, 0, ptr);
                    return BIR_MAKE_VAL(ld);
                }
            }

            uint32_t t_void = bir_type_void(L->M);
            uint32_t st = emit(L, BIR_STORE, t_void, 2, 0);
            set_op(L, st, 0, val);
            set_op(L, st, 1, ptr);
            return val;
        }

        /* Compound assignment (+=, *=, etc.) */
        if (is_compound_assign(op)) {
            uint32_t ptr = lower_lvalue(L, lhs_n);
            uint32_t pt  = ref_type(L, ptr);
            uint32_t et  = ptr_inner(L, pt);
            if (!et) et  = bir_type_int(L->M, 32);

            uint32_t old = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, old, 0, ptr);
            uint32_t rhs = lower_expr(L, rhs_n);

            int base = compound_base(op);
            int fp   = is_float_type(L, et);
            int opc  = bin_op_code(base, fp, node_is_unsigned(L, node));

            uint32_t res = emit(L, (uint16_t)opc, et, 2, 0);
            set_op(L, res, 0, BIR_MAKE_VAL(old));
            set_op(L, res, 1, rhs);

            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, BIR_MAKE_VAL(res));
            set_op(L, st, 1, ptr);
            return BIR_MAKE_VAL(res);
        }

        /* Comparison */
        if (is_cmp_tok(op)) {
            uint32_t lhs = lower_expr(L, lhs_n);
            uint32_t rhs = lower_expr(L, rhs_n);
            uint32_t lt  = ref_type(L, lhs);
            int fp       = is_float_type(L, lt);
            uint32_t t1  = bir_type_int(L->M, 1);
            int pred     = cmp_pred(op, fp, node_is_unsigned(L, lhs_n));
            uint32_t inst = emit(L, fp ? BIR_FCMP : BIR_ICMP,
                                 t1, 2, (uint8_t)pred);
            set_op(L, inst, 0, lhs);
            set_op(L, inst, 1, rhs);
            return BIR_MAKE_VAL(inst);
        }

        /* Short-circuit && via alloca+store+load (mem2reg cleans up) */
        if (op == TOK_LAND) {
            uint32_t t1 = bir_type_int(L->M, 1);
            uint32_t pt = bir_type_ptr(L->M, t1, BIR_AS_PRIVATE);
            uint32_t al = emit(L, BIR_ALLOCA, pt, 0, 0);
            uint32_t f  = BIR_MAKE_CONST(bir_const_int(L->M, t1, 0));
            uint32_t s0 = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, s0, 0, f);
            set_op(L, s0, 1, BIR_MAKE_VAL(al));
            uint32_t lhs   = lower_expr(L, lhs_n);
            uint32_t rhs_b = new_block(L, "land.rhs");
            uint32_t end_b = new_block(L, "land.end");
            uint32_t br    = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, lhs);
            set_op(L, br, 1, rhs_b);
            set_op(L, br, 2, end_b);
            set_op(L, br, 3, end_b);
            set_block(L, rhs_b);
            uint32_t rhs = lower_expr(L, rhs_n);
            uint32_t s1  = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, s1, 0, rhs);
            set_op(L, s1, 1, BIR_MAKE_VAL(al));
            if (!block_terminated(L)) {
                uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
                set_op(L, j, 0, end_b);
            }
            set_block(L, end_b);
            uint32_t ld = emit(L, BIR_LOAD, t1, 1, 0);
            set_op(L, ld, 0, BIR_MAKE_VAL(al));
            return BIR_MAKE_VAL(ld);
        }
        /* Short-circuit || via alloca+store+load */
        if (op == TOK_LOR) {
            uint32_t t1 = bir_type_int(L->M, 1);
            uint32_t pt = bir_type_ptr(L->M, t1, BIR_AS_PRIVATE);
            uint32_t al = emit(L, BIR_ALLOCA, pt, 0, 0);
            uint32_t tr = BIR_MAKE_CONST(bir_const_int(L->M, t1, 1));
            uint32_t s0 = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, s0, 0, tr);
            set_op(L, s0, 1, BIR_MAKE_VAL(al));
            uint32_t lhs   = lower_expr(L, lhs_n);
            uint32_t rhs_b = new_block(L, "lor.rhs");
            uint32_t end_b = new_block(L, "lor.end");
            uint32_t br    = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, lhs);
            set_op(L, br, 1, end_b);  /* true → skip RHS */
            set_op(L, br, 2, rhs_b);  /* false → eval RHS */
            set_op(L, br, 3, end_b);
            set_block(L, rhs_b);
            uint32_t rhs = lower_expr(L, rhs_n);
            uint32_t s1  = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, s1, 0, rhs);
            set_op(L, s1, 1, BIR_MAKE_VAL(al));
            if (!block_terminated(L)) {
                uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
                set_op(L, j, 0, end_b);
            }
            set_block(L, end_b);
            uint32_t ld = emit(L, BIR_LOAD, t1, 1, 0);
            set_op(L, ld, 0, BIR_MAKE_VAL(al));
            return BIR_MAKE_VAL(ld);
        }

        /* Comma */
        if (op == TOK_COMMA) {
            lower_expr(L, lhs_n);
            return lower_expr(L, rhs_n);
        }

        /* Regular binary arithmetic/bitwise */
        {
            uint32_t lhs = lower_expr(L, lhs_n);
            uint32_t rhs = lower_expr(L, rhs_n);
            uint32_t lt  = ref_type(L, lhs);

            /* Operator overloading: struct-typed operands → call operator func */
            if (lt < L->M->num_types
                && L->M->types[lt].kind == BIR_TYPE_STRUCT) {
                char oname[32];
                op_name_from_tok(op, oname, sizeof(oname));
                /* Look up operator function in module */
                for (uint32_t fi = 0; fi < L->M->num_funcs; fi++) {
                    if (L->M->funcs[fi].name < L->M->string_len
                        && strcmp(&L->M->strings[L->M->funcs[fi].name],
                                 oname) == 0) {
                        uint32_t ftype = L->M->funcs[fi].type;
                        uint32_t ret_t = L->M->types[ftype].inner;
                        uint32_t inst = emit(L, BIR_CALL, ret_t, 3, 0);
                        set_op(L, inst, 0, fi);
                        set_op(L, inst, 1, lhs);
                        set_op(L, inst, 2, rhs);
                        return BIR_MAKE_VAL(inst);
                    }
                }
            }

            uint32_t rt  = ref_type(L, rhs);

            /* Pointer arithmetic. In C `ptr + n` steps n ELEMENTS, not n
             * bytes, so it has to lower to a GEP (which the backends scale by
             * the pointee size) rather than a plain add. A raw BIR_ADD treats
             * the index as bytes and drops everything past the first element in
             * the wrong place -- subscripts already do the right thing, but
             * explicit `a + i*8` did not until now. */
            if (op == TOK_PLUS || op == TOK_MINUS) {
                int lp = is_ptr_type(L, lt), rp = is_ptr_type(L, rt);
                if ((lp && !rp) || (op == TOK_PLUS && !lp && rp)) {
                    uint32_t ptr = lp ? lhs : rhs;
                    uint32_t idx = lp ? rhs : lhs;
                    uint32_t pt  = lp ? lt  : rt;
                    if (op == TOK_MINUS) {  /* ptr - n is ptr + (-n) */
                        uint32_t it  = ref_type(L, idx);
                        uint32_t z   = BIR_MAKE_CONST(bir_const_int(L->M, it, 0));
                        uint32_t neg = emit(L, BIR_SUB, it, 2, 0);
                        set_op(L, neg, 0, z); set_op(L, neg, 1, idx);
                        idx = BIR_MAKE_VAL(neg);
                    }
                    uint32_t gep = emit(L, BIR_GEP, pt, 2, 0);
                    set_op(L, gep, 0, ptr);
                    set_op(L, gep, 1, idx);
                    return BIR_MAKE_VAL(gep);
                }
            }

            /* Usual arithmetic conversion: promote both operands
             * to the wider/float type. C says int*double → double,
             * not int*double → garbage. Without this, backends get
             * mixed-type ops (mul.u32 with an f64 register) and
             * the PTX JIT has strong opinions about that. */
            uint32_t res_t = lt;
            int lf = is_float_type(L, lt), rf = is_float_type(L, rt);
            if (lf && !rf) {
                rhs = coerce_to(L, rhs, lt, node_is_unsigned(L, rhs_n));
                res_t = lt;
            } else if (!lf && rf) {
                lhs = coerce_to(L, lhs, rt, node_is_unsigned(L, lhs_n));
                res_t = rt;
            } else if (lf && rf) {
                /* Both float: promote narrower to wider */
                if (lt < L->M->num_types && rt < L->M->num_types
                    && L->M->types[rt].width > L->M->types[lt].width) {
                    lhs = coerce_to(L, lhs, rt, 0);
                    res_t = rt;
                } else if (lt < L->M->num_types && rt < L->M->num_types
                           && L->M->types[lt].width > L->M->types[rt].width) {
                    rhs = coerce_to(L, rhs, lt, 0);
                    res_t = lt;
                }
            }
            int fp       = is_float_type(L, res_t);
            int opc      = bin_op_code(op, fp, node_is_unsigned(L, node));
            if (opc < 0) {
                lower_error(L, node, BC_E102);
                return lhs;
            }
            uint32_t inst = emit(L, (uint16_t)opc, res_t, 2, 0);
            set_op(L, inst, 0, lhs);
            set_op(L, inst, 1, rhs);
            return BIR_MAKE_VAL(inst);
        }
    }

    case AST_UNARY_PREFIX: {
        int op = n->d.oper.op;
        uint32_t operand = n->first_child;

        if (op == TOK_PLUS)
            return lower_expr(L, operand);

        if (op == TOK_MINUS) {
            uint32_t val = lower_expr(L, operand);
            uint32_t t   = ref_type(L, val);
            if (is_float_type(L, t)) {
                uint32_t z = BIR_MAKE_CONST(bir_const_float(L->M, t, 0.0));
                uint32_t inst = emit(L, BIR_FSUB, t, 2, 0);
                set_op(L, inst, 0, z);
                set_op(L, inst, 1, val);
                return BIR_MAKE_VAL(inst);
            }
            uint32_t z = BIR_MAKE_CONST(bir_const_int(L->M, t, 0));
            uint32_t inst = emit(L, BIR_SUB, t, 2, 0);
            set_op(L, inst, 0, z);
            set_op(L, inst, 1, val);
            return BIR_MAKE_VAL(inst);
        }

        if (op == TOK_BANG) {
            uint32_t val = lower_expr(L, operand);
            uint32_t t   = ref_type(L, val);
            uint32_t t1  = bir_type_int(L->M, 1);
            uint32_t z   = BIR_MAKE_CONST(bir_const_int(L->M, t, 0));
            uint32_t inst = emit(L, BIR_ICMP, t1, 2, BIR_ICMP_EQ);
            set_op(L, inst, 0, val);
            set_op(L, inst, 1, z);
            return BIR_MAKE_VAL(inst);
        }

        if (op == TOK_TILDE) {
            uint32_t val = lower_expr(L, operand);
            uint32_t t   = ref_type(L, val);
            uint32_t neg = BIR_MAKE_CONST(bir_const_int(L->M, t, -1));
            uint32_t inst = emit(L, BIR_XOR, t, 2, 0);
            set_op(L, inst, 0, val);
            set_op(L, inst, 1, neg);
            return BIR_MAKE_VAL(inst);
        }

        if (op == TOK_AMP)
            return lower_lvalue(L, operand);

        if (op == TOK_STAR) {
            uint32_t ptr = lower_expr(L, operand);
            uint32_t pt  = ref_type(L, ptr);
            uint32_t et  = ptr_inner(L, pt);
            if (!et) et  = bir_type_int(L->M, 32);
            uint32_t inst = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, inst, 0, ptr);
            return BIR_MAKE_VAL(inst);
        }

        if (op == TOK_INC || op == TOK_DEC) {
            uint32_t ptr = lower_lvalue(L, operand);
            uint32_t pt  = ref_type(L, ptr);
            uint32_t et  = ptr_inner(L, pt);
            if (!et) et  = bir_type_int(L->M, 32);

            uint32_t old = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, old, 0, ptr);

            uint32_t one, res;
            if (is_float_type(L, et)) {
                one = BIR_MAKE_CONST(bir_const_float(L->M, et, 1.0));
                res = emit(L, op==TOK_INC ? BIR_FADD : BIR_FSUB, et, 2, 0);
            } else {
                one = BIR_MAKE_CONST(bir_const_int(L->M, et, 1));
                res = emit(L, op==TOK_INC ? BIR_ADD : BIR_SUB, et, 2, 0);
            }
            set_op(L, res, 0, BIR_MAKE_VAL(old));
            set_op(L, res, 1, one);

            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, BIR_MAKE_VAL(res));
            set_op(L, st, 1, ptr);
            return BIR_MAKE_VAL(res); /* pre-inc returns new value */
        }

        lower_error(L, node, BC_E103);
        return BIR_VAL_NONE;
    }

    case AST_UNARY_POSTFIX: {
        int op = n->d.oper.op;
        uint32_t operand = n->first_child;

        if (op == TOK_INC || op == TOK_DEC) {
            uint32_t ptr = lower_lvalue(L, operand);
            uint32_t pt  = ref_type(L, ptr);
            uint32_t et  = ptr_inner(L, pt);
            if (!et) et  = bir_type_int(L->M, 32);

            uint32_t old = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, old, 0, ptr);

            uint32_t one, res;
            if (is_float_type(L, et)) {
                one = BIR_MAKE_CONST(bir_const_float(L->M, et, 1.0));
                res = emit(L, op==TOK_INC ? BIR_FADD : BIR_FSUB, et, 2, 0);
            } else {
                one = BIR_MAKE_CONST(bir_const_int(L->M, et, 1));
                res = emit(L, op==TOK_INC ? BIR_ADD : BIR_SUB, et, 2, 0);
            }
            set_op(L, res, 0, BIR_MAKE_VAL(old));
            set_op(L, res, 1, one);

            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, BIR_MAKE_VAL(res));
            set_op(L, st, 1, ptr);
            return BIR_MAKE_VAL(old); /* post-inc returns old value */
        }
        lower_error(L, node, BC_E104);
        return BIR_VAL_NONE;
    }

    case AST_TERNARY: {
        uint32_t cond_n = n->first_child;
        uint32_t then_n = ND(L, cond_n)->next_sibling;
        uint32_t else_n = ND(L, then_n)->next_sibling;

        uint32_t cond = lower_expr(L, cond_n);
        uint32_t tv   = lower_expr(L, then_n);
        uint32_t ev   = lower_expr(L, else_n);
        uint32_t rt   = ref_type(L, tv);

        uint32_t inst = emit(L, BIR_SELECT, rt, 3, 0);
        set_op(L, inst, 0, cond);
        set_op(L, inst, 1, tv);
        set_op(L, inst, 2, ev);
        return BIR_MAKE_VAL(inst);
    }

    case AST_CALL: {
        uint32_t callee_n = n->first_child;

        /* ---- Cooperative groups: phantom types meet real hardware ---- */
        if (ND(L, callee_n)->type == AST_SCOPE_RES) {
            uint32_t ns_n = ND(L, callee_n)->first_child;
            uint32_t fn_n = ns_n ? ND(L, ns_n)->next_sibling : 0;
            if (ns_n && fn_n && text_eq(L, ns_n, "cooperative_groups")) {
                if (text_eq(L, fn_n, "this_thread_block")) {
                    /* Returns a phantom handle. The GPU doesn't care about
                       your type system — it's just a thread block. */
                    uint32_t i32 = bir_type_int(L->M, 32);
                    return BIR_MAKE_CONST(bir_const_int(L->M, i32, 0));
                }
                if (text_eq(L, fn_n, "this_grid")) {
                    uint32_t i32 = bir_type_int(L->M, 32);
                    return BIR_MAKE_CONST(bir_const_int(L->M, i32, 0));
                }
            }
        }
        if (ND(L, callee_n)->type == AST_MEMBER) {
            uint32_t obj_n = ND(L, callee_n)->first_child;
            uint32_t fld_n = obj_n ? ND(L, obj_n)->next_sibling : 0;
            if (fld_n && ND(L, fld_n)->type == AST_IDENT) {
                if (text_eq(L, fld_n, "sync")) {
                    /* .sync() → barrier. All threads, please stop and wait politely. */
                    uint32_t inst = emit(L, BIR_BARRIER, bir_type_void(L->M), 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
                if (text_eq(L, fld_n, "thread_rank")) {
                    /* .thread_rank() → threadIdx.x. A thread by any other name. */
                    uint32_t i32 = bir_type_int(L->M, 32);
                    uint32_t inst = emit(L, BIR_THREAD_ID, i32, 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
                if (text_eq(L, fld_n, "size")) {
                    /* .size() → blockDim.x. How many of us are there? */
                    uint32_t i32 = bir_type_int(L->M, 32);
                    uint32_t inst = emit(L, BIR_BLOCK_DIM, i32, 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
            }
        }

        char cname[128];
        get_text(L, callee_n, cname, sizeof(cname));

        /* ---- CUDA builtin functions ---- */

        /* Barriers: zero args, void return */
        if (strcmp(cname, "__syncthreads") == 0) {
            uint32_t inst = emit(L, BIR_BARRIER, bir_type_void(L->M), 0, 0);
            return BIR_MAKE_VAL(inst);
        }
        if (strcmp(cname, "__threadfence") == 0) {
            uint32_t inst = emit(L, BIR_BARRIER, bir_type_void(L->M), 0, 1);
            return BIR_MAKE_VAL(inst);
        }

        /* Atomics: 2-arg (ptr, val) */
        {
            static const struct { const char *n; uint16_t op; } atab[] = {
                {"atomicAdd", BIR_ATOMIC_ADD}, {"atomicSub", BIR_ATOMIC_SUB},
                {"atomicAnd", BIR_ATOMIC_AND}, {"atomicOr",  BIR_ATOMIC_OR},
                {"atomicXor", BIR_ATOMIC_XOR}, {"atomicMin", BIR_ATOMIC_MIN},
                {"atomicMax", BIR_ATOMIC_MAX}, {"atomicExch",BIR_ATOMIC_XCHG},
            };
            int matched = 0;
            for (int bi = 0; bi < 8; bi++) {
                if (strcmp(cname, atab[bi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0 = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t a1 = lower_expr(L, an);
                uint32_t rt = ref_type(L, a1);
                uint32_t inst = emit(L, atab[bi].op, rt, 2, BIR_ORDER_RELAXED);
                set_op(L, inst, 0, a0);
                set_op(L, inst, 1, a1);
                return BIR_MAKE_VAL(inst);
            }
            (void)matched;
        }

        /* atomicCAS: 3-arg (ptr, cmp, val) */
        if (strcmp(cname, "atomicCAS") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            an = ND(L, an)->next_sibling;
            uint32_t a1 = lower_expr(L, an);
            an = ND(L, an)->next_sibling;
            uint32_t a2 = lower_expr(L, an);
            uint32_t rt = ref_type(L, a1);
            uint32_t inst = emit(L, BIR_ATOMIC_CAS, rt, 3, BIR_ORDER_RELAXED);
            set_op(L, inst, 0, a0);
            set_op(L, inst, 1, a1);
            set_op(L, inst, 2, a2);
            return BIR_MAKE_VAL(inst);
        }

        /* __umul64hi(a, b): high 64 bits of a 64x64 unsigned product.
           Keystone for ZK field arithmetic — Montgomery multiply is
           mul-lo + mul-hi + add-with-carry. */
        if (strcmp(cname, "__umul64hi") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            an = ND(L, an)->next_sibling;
            uint32_t a1 = lower_expr(L, an);
            uint32_t rt = bir_type_int(L->M, 64);
            uint32_t inst = emit(L, BIR_UMULHI, rt, 2, 0);
            set_op(L, inst, 0, a0);
            set_op(L, inst, 1, a1);
            return BIR_MAKE_VAL(inst);
        }

        /* Warp shuffle: up to 4-arg (mask, val, lane/delta, [width]) */
        {
            static const struct { const char *n; uint16_t op; } stab[] = {
                {"__shfl_sync",      BIR_SHFL},
                {"__shfl_up_sync",   BIR_SHFL_UP},
                {"__shfl_down_sync", BIR_SHFL_DOWN},
                {"__shfl_xor_sync",  BIR_SHFL_XOR},
            };
            for (int bi = 0; bi < 4; bi++) {
                if (strcmp(cname, stab[bi].n) != 0) continue;
                uint32_t sa[4];
                int sn = 0;
                uint32_t an = ND(L, callee_n)->next_sibling;
                while (an && sn < 4) {
                    sa[sn++] = lower_expr(L, an);
                    an = ND(L, an)->next_sibling;
                }
                uint32_t rt = sn >= 2 ? ref_type(L, sa[1])
                                      : bir_type_int(L->M, 32);
                uint32_t inst = emit(L, stab[bi].op, rt, (uint8_t)sn, 0);
                for (int j = 0; j < sn; j++)
                    set_op(L, inst, j, sa[j]);
                return BIR_MAKE_VAL(inst);
            }
        }

        /* Warp vote: 2-arg (mask, pred) */
        {
            static const struct { const char *n; uint16_t op; } vtab[] = {
                {"__ballot_sync", BIR_BALLOT},
                {"__any_sync",    BIR_VOTE_ANY},
                {"__all_sync",    BIR_VOTE_ALL},
            };
            for (int bi = 0; bi < 3; bi++) {
                if (strcmp(cname, vtab[bi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0 = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t a1 = lower_expr(L, an);
                uint32_t rt = (vtab[bi].op == BIR_BALLOT)
                    ? bir_type_int(L->M, 32) : bir_type_int(L->M, 1);
                uint32_t inst = emit(L, vtab[bi].op, rt, 2, 0);
                set_op(L, inst, 0, a0);
                set_op(L, inst, 1, a1);
                return BIR_MAKE_VAL(inst);
            }
        }

        /* ---- Vector constructors: make_float2/3/4, make_int2/3/4 etc. ---- */
        if (strncmp(cname, "make_", 5) == 0) {
            /* Try to resolve as vector type name (without "make_" prefix) */
            uint32_t vst = resolve_type(L, 0, 0, 0); /* dummy, overwritten */
            char vname[64];
            { int vl = (int)strlen(cname + 5);
              if (vl > 63) vl = 63;
              memcpy(vname, cname + 5, (size_t)vl);
              vname[vl] = '\0'; }
            /* Find struct_def for the vector type */
            struct_def_t *vsd = NULL;
            for (int si = 0; si < L->nstructs; si++) {
                if (strcmp(L->structs[si].name, vname) == 0) {
                    vsd = &L->structs[si]; break;
                }
            }
            if (!vsd) {
                /* Force-resolve so it gets registered */
                /* Force-resolve inline to register the struct_def */
                static const struct { const char *name; int elem; int lanes; } vc_map[] = {
                    {"float2",TYPE_FLOAT,2},{"float3",TYPE_FLOAT,3},{"float4",TYPE_FLOAT,4},
                    {"double2",TYPE_DOUBLE,2},{"double3",TYPE_DOUBLE,3},{"double4",TYPE_DOUBLE,4},
                    {"int2",TYPE_INT,2},{"int3",TYPE_INT,3},{"int4",TYPE_INT,4},
                    {"uint2",TYPE_INT,2},{"uint3",TYPE_INT,3},{"uint4",TYPE_INT,4},
                    {"char2",TYPE_CHAR,2},{"char3",TYPE_CHAR,3},{"char4",TYPE_CHAR,4},
                    {"uchar2",TYPE_CHAR,2},{"uchar3",TYPE_CHAR,3},{"uchar4",TYPE_CHAR,4},
                    {"short2",TYPE_SHORT,2},{"short3",TYPE_SHORT,3},{"short4",TYPE_SHORT,4},
                    {"ushort2",TYPE_SHORT,2},{"ushort3",TYPE_SHORT,3},{"ushort4",TYPE_SHORT,4},
                    {"long2",TYPE_LONG,2},{"long3",TYPE_LONG,3},{"long4",TYPE_LONG,4},
                    {"ulong2",TYPE_LONG,2},{"ulong3",TYPE_LONG,3},{"ulong4",TYPE_LONG,4},
                    {NULL,0,0}
                };
                for (int vi = 0; vc_map[vi].name; vi++) {
                    if (strcmp(vname, vc_map[vi].name) != 0) continue;
                    uint32_t et = resolve_basic(L, vc_map[vi].elem);
                    int lanes = vc_map[vi].lanes;
                    uint32_t fields[4] = {et, et, et, et};
                    vst = bir_type_struct(L->M, fields, lanes);
                    if (L->nstructs < MAX_STRUCTS) {
                        static const char *xyzw[] = {"x","y","z","w"};
                        vsd = &L->structs[L->nstructs];
                        memset(vsd, 0, sizeof(*vsd));
                        snprintf(vsd->name, sizeof(vsd->name), "%s", vname);
                        vsd->num_fields = lanes;
                        vsd->bir_type = vst;
                        for (int fi = 0; fi < lanes; fi++) {
                            vsd->field_types[fi] = et;
                            snprintf(vsd->field_names[fi],
                                     sizeof(vsd->field_names[0]),
                                     "%s", xyzw[fi]);
                        }
                        L->nstructs++;
                    }
                    break;
                }
            }
            if (vsd) {
                /* Alloca struct, GEP+store each arg, load result */
                uint32_t st = vsd->bir_type;
                uint32_t pt = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
                uint32_t al = emit(L, BIR_ALLOCA, pt, 0, 0);

                uint32_t an = ND(L, callee_n)->next_sibling;
                for (int fi = 0; fi < vsd->num_fields && an; fi++) {
                    uint32_t val = lower_expr(L, an);
                    uint32_t fpt = bir_type_ptr(L->M, vsd->field_types[fi],
                                                BIR_AS_PRIVATE);
                    uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                        bir_type_int(L->M, 32), fi));
                    uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
                    set_op(L, gep, 0, BIR_MAKE_VAL(al));
                    set_op(L, gep, 1, ci);
                    uint32_t sv = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                    set_op(L, sv, 0, val);
                    set_op(L, sv, 1, BIR_MAKE_VAL(gep));
                    an = ND(L, an)->next_sibling;
                }
                uint32_t ld = emit(L, BIR_LOAD, st, 1, 0);
                set_op(L, ld, 0, BIR_MAKE_VAL(al));
                return BIR_MAKE_VAL(ld);
            }
        }

        /* ---- Half conversion builtins ---- */
        if (strcmp(cname, "__float2half") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t f16 = bir_type_float(L->M, 16);
            uint32_t inst = emit(L, BIR_FPTRUNC, f16, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }
        if (strcmp(cname, "__half2float") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t f32 = bir_type_float(L->M, 32);
            uint32_t inst = emit(L, BIR_FPEXT, f32, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }

        /* ---- BF16 conversion builtins ---- */
        if (strcmp(cname, "__float2bfloat16") == 0
            || strcmp(cname, "__float2bfloat16_rn") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t bf16 = bir_type_bfloat(L->M);
            uint32_t inst = emit(L, BIR_FPTRUNC, bf16, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }
        if (strcmp(cname, "__bfloat162float") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t f32 = bir_type_float(L->M, 32);
            uint32_t inst = emit(L, BIR_FPEXT, f32, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }
        /* ---- Bit-cast builtins ---- */
        if (strcmp(cname, "__int_as_float") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t f32 = bir_type_float(L->M, 32);
            uint32_t inst = emit(L, BIR_BITCAST, f32, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }
        if (strcmp(cname, "__float_as_int") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t val = lower_expr(L, an);
            uint32_t i32 = bir_type_int(L->M, 32);
            uint32_t inst = emit(L, BIR_BITCAST, i32, 1, 0);
            set_op(L, inst, 0, val);
            return BIR_MAKE_VAL(inst);
        }

        /* ---- Math builtins: unary ---- */
        {
            static const struct { const char *n; uint16_t op; } mt1[] = {
                {"sqrtf",BIR_SQRT},{"sqrt",BIR_SQRT},{"__fsqrt_rn",BIR_SQRT},
                {"rsqrtf",BIR_RSQ},{"__frsqrt_rn",BIR_RSQ},
                {"__frcp_rn",BIR_RCP},
                {"exp2f",BIR_EXP2},{"log2f",BIR_LOG2},{"__log2f",BIR_LOG2},
                {"fabsf",BIR_FABS},{"fabs",BIR_FABS},
                {"floorf",BIR_FLOOR},{"ceilf",BIR_CEIL},
                {"truncf",BIR_FTRUNC},{"roundf",BIR_RNDNE},{"rintf",BIR_RNDNE},
            };
            for (int mi = 0; mi < 16; mi++) {
                if (strcmp(cname, mt1[mi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t rt = ref_type(L, v);
                uint32_t r = emit(L, mt1[mi].op, rt, 1, 0);
                set_op(L, r, 0, v);
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- Math builtins: binary ---- */
        {
            static const struct { const char *n; uint16_t op; } mt2[] = {
                {"fmaxf",BIR_FMAX},{"fminf",BIR_FMIN},{"fmodf",BIR_FREM},
            };
            for (int mi = 0; mi < 3; mi++) {
                if (strcmp(cname, mt2[mi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0 = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t a1 = lower_expr(L, an);
                uint32_t rt = ref_type(L, a0);
                uint32_t r = emit(L, mt2[mi].op, rt, 2, 0);
                set_op(L, r, 0, a0);
                set_op(L, r, 1, a1);
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- Math builtins: compound (scaling constants) ---- */
        {
            uint32_t f32 = bir_type_float(L->M, 32);
            /* expf(x) = exp2(x * log2(e)) */
            if (strcmp(cname, "expf") == 0 || strcmp(cname, "__expf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.4426950408889634));
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k);
                uint32_t r = emit(L, BIR_EXP2, f32, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(m));
                return BIR_MAKE_VAL(r);
            }
            /* logf(x) = log2(x) * ln(2) */
            if (strcmp(cname, "logf") == 0 || strcmp(cname, "__logf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t lg = emit(L, BIR_LOG2, f32, 1, 0);
                set_op(L, lg, 0, v);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.6931471805599453));
                uint32_t r = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(lg)); set_op(L, r, 1, k);
                return BIR_MAKE_VAL(r);
            }
            /* log10f(x) = log2(x) * log10(2) */
            if (strcmp(cname, "log10f") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t lg = emit(L, BIR_LOG2, f32, 1, 0);
                set_op(L, lg, 0, v);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.30102999566398114));
                uint32_t r = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(lg)); set_op(L, r, 1, k);
                return BIR_MAKE_VAL(r);
            }
            /* sinf(x) = hw_sin(x / 2pi) */
            if (strcmp(cname, "sinf") == 0 || strcmp(cname, "__sinf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.15915494309189535));
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k);
                uint32_t r = emit(L, BIR_SIN, f32, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(m));
                return BIR_MAKE_VAL(r);
            }
            /* cosf(x) = hw_cos(x / 2pi) */
            if (strcmp(cname, "cosf") == 0 || strcmp(cname, "__cosf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.15915494309189535));
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k);
                uint32_t r = emit(L, BIR_COS, f32, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(m));
                return BIR_MAKE_VAL(r);
            }
            /* tanf(x) = sin(t) / cos(t), t = x / 2pi */
            if (strcmp(cname, "tanf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.15915494309189535));
                uint32_t t = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, t, 0, v); set_op(L, t, 1, k);
                uint32_t s = emit(L, BIR_SIN, f32, 1, 0);
                set_op(L, s, 0, BIR_MAKE_VAL(t));
                uint32_t c = emit(L, BIR_COS, f32, 1, 0);
                set_op(L, c, 0, BIR_MAKE_VAL(t));
                uint32_t r = emit(L, BIR_FDIV, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(s)); set_op(L, r, 1, BIR_MAKE_VAL(c));
                return BIR_MAKE_VAL(r);
            }
            /* powf(x,y) = exp2(y * log2(x)) */
            if (strcmp(cname, "powf") == 0 || strcmp(cname, "__powf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t x = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t y = lower_expr(L, an);
                uint32_t lg = emit(L, BIR_LOG2, f32, 1, 0);
                set_op(L, lg, 0, x);
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, y); set_op(L, m, 1, BIR_MAKE_VAL(lg));
                uint32_t r = emit(L, BIR_EXP2, f32, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(m));
                return BIR_MAKE_VAL(r);
            }
            /* tanhf(x) = (e2-1)/(e2+1), e2 = exp2(2x * log2e) */
            if (strcmp(cname, "tanhf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k2 = BIR_MAKE_CONST(bir_const_float(L->M, f32, 2.8853900817779268));
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k2);
                uint32_t e2 = emit(L, BIR_EXP2, f32, 1, 0);
                set_op(L, e2, 0, BIR_MAKE_VAL(m));
                uint32_t one = BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.0));
                uint32_t nm = emit(L, BIR_FSUB, f32, 2, 0);
                set_op(L, nm, 0, BIR_MAKE_VAL(e2)); set_op(L, nm, 1, one);
                uint32_t dn = emit(L, BIR_FADD, f32, 2, 0);
                set_op(L, dn, 0, BIR_MAKE_VAL(e2)); set_op(L, dn, 1, one);
                uint32_t r = emit(L, BIR_FDIV, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(nm)); set_op(L, r, 1, BIR_MAKE_VAL(dn));
                return BIR_MAKE_VAL(r);
            }
            /* copysignf(x,y) = (x & 0x7FFFFFFF) | (y & 0x80000000) */
            if (strcmp(cname, "copysignf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t xv = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t yv = lower_expr(L, an);
                uint32_t i32 = bir_type_int(L->M, 32);
                uint32_t bx = emit(L, BIR_BITCAST, i32, 1, 0);
                set_op(L, bx, 0, xv);
                uint32_t by = emit(L, BIR_BITCAST, i32, 1, 0);
                set_op(L, by, 0, yv);
                uint32_t mk = BIR_MAKE_CONST(bir_const_int(L->M, i32, 0x7FFFFFFF));
                uint32_t sb = BIR_MAKE_CONST(bir_const_int(L->M, i32, (int64_t)0x80000000u));
                uint32_t ax = emit(L, BIR_AND, i32, 2, 0);
                set_op(L, ax, 0, BIR_MAKE_VAL(bx)); set_op(L, ax, 1, mk);
                uint32_t ay = emit(L, BIR_AND, i32, 2, 0);
                set_op(L, ay, 0, BIR_MAKE_VAL(by)); set_op(L, ay, 1, sb);
                uint32_t o = emit(L, BIR_OR, i32, 2, 0);
                set_op(L, o, 0, BIR_MAKE_VAL(ax)); set_op(L, o, 1, BIR_MAKE_VAL(ay));
                uint32_t r = emit(L, BIR_BITCAST, f32, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(o));
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- MFMA intrinsics (CDNA matrix multiply) ---- */
        if (strncmp(cname, "__builtin_amdgcn_mfma_", 22) == 0) {
            static const struct { const char *sfx; uint8_t var; } mfma_tab[] = {
                {"f32_4x4x4_f16",       0},
                {"f32_16x16x16_f16",     1},
                {"f32_32x32x8_f16",      2},
                {"f32_4x4x4_bf16_1k",    3},
                {"f32_16x16x16_bf16_1k", 4},
                {"f32_32x32x8_bf16_1k",  5},
                {"f32_4x4x1_f32",        6},
                {"f32_16x16x4_f32",      7},
                {"f32_32x32x2_f32",      8},
                {"i32_4x4x4_i8",         9},
                {"i32_16x16x16_i8",     10},
                {"i32_32x32x8_i8",      11},
                /* FP8/BF8 mixed-precision (gfx942) */
                {"f32_16x16x32_fp8_fp8", 12},
                {"f32_16x16x32_fp8_bf8", 13},
                {"f32_16x16x32_bf8_fp8", 14},
                {"f32_16x16x32_bf8_bf8", 15},
                {"f32_32x32x16_fp8_fp8", 16},
                {"f32_32x32x16_fp8_bf8", 17},
                {"f32_32x32x16_bf8_fp8", 18},
                {"f32_32x32x16_bf8_bf8", 19},
                /* F64 matrix */
                {"f64_4x4x4f64",        20},
                {"f64_16x16x4f64",      21},
            };
            const char *sfx = cname + 22;
            for (int mi = 0; mi < 22; mi++) {
                if (strcmp(sfx, mfma_tab[mi].sfx) != 0) continue;
                /* 3 args: A, B, C(accum) */
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0 = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t a1 = lower_expr(L, an);
                an = ND(L, an)->next_sibling;
                uint32_t a2 = lower_expr(L, an);
                uint32_t rt = ref_type(L, a2); /* return type = accum type */
                uint32_t r = emit(L, BIR_MFMA, rt, 3, mfma_tab[mi].var);
                set_op(L, r, 0, a0);
                set_op(L, r, 1, a1);
                set_op(L, r, 2, a2);
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- __ockl_* thread model (tinygrad's preferred API) ---- */
        if (strncmp(cname, "__ockl_get_", 11) == 0) {
            static const struct { const char *sfx; uint16_t op; } ockl[] = {
                {"local_id",   BIR_THREAD_ID},
                {"group_id",   BIR_BLOCK_ID},
                {"local_size", BIR_BLOCK_DIM},
                {"num_groups", BIR_GRID_DIM},
            };
            const char *rest = cname + 11;
            for (int oi = 0; oi < 4; oi++) {
                if (strcmp(rest, ockl[oi].sfx) != 0) continue;
                /* arg is literal dim (0/1/2) */
                uint32_t an = ND(L, callee_n)->next_sibling;
                int dim = 0;
                if (an && ND(L, an)->type == AST_INT_LIT)
                    dim = (int)parse_int_text(
                        L->src + ND(L, an)->d.text.offset,
                        (int)ND(L, an)->d.text.len);
                uint32_t i32 = bir_type_int(L->M, 32);
                uint32_t r = emit(L, ockl[oi].op, i32, 0, (uint8_t)dim);
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- __ocml_* math builtins (AMD's libm naming) ---- */
        if (strncmp(cname, "__ocml_", 7) == 0) {
            const char *rest = cname + 7;
            /* Unary direct: no prescale needed */
            static const struct { const char *n; size_t len; uint16_t op; } ou[] = {
                {"exp2",  4, BIR_EXP2},  {"log2",  4, BIR_LOG2},
                {"sqrt",  4, BIR_SQRT},  {"fabs",  4, BIR_FABS},
                {"floor", 5, BIR_FLOOR}, {"ceil",  4, BIR_CEIL},
                {"trunc", 5, BIR_FTRUNC},{"rint",  4, BIR_RNDNE},
            };
            for (int oi = 0; oi < 8; oi++) {
                if (strncmp(rest, ou[oi].n, ou[oi].len) == 0
                    && rest[ou[oi].len] == '_') {
                    uint32_t an = ND(L, callee_n)->next_sibling;
                    uint32_t v = lower_expr(L, an);
                    uint32_t rt = ref_type(L, v);
                    uint32_t r = emit(L, ou[oi].op, rt, 1, 0);
                    set_op(L, r, 0, v);
                    return BIR_MAKE_VAL(r);
                }
            }
            /* Binary direct */
            static const struct { const char *n; size_t len; uint16_t op; } ob[] = {
                {"fmax", 4, BIR_FMAX}, {"fmin", 4, BIR_FMIN},
            };
            for (int oi = 0; oi < 2; oi++) {
                if (strncmp(rest, ob[oi].n, ob[oi].len) == 0
                    && rest[ob[oi].len] == '_') {
                    uint32_t an = ND(L, callee_n)->next_sibling;
                    uint32_t a0 = lower_expr(L, an);
                    an = ND(L, an)->next_sibling;
                    uint32_t a1 = lower_expr(L, an);
                    uint32_t rt = ref_type(L, a0);
                    uint32_t r = emit(L, ob[oi].op, rt, 2, 0);
                    set_op(L, r, 0, a0); set_op(L, r, 1, a1);
                    return BIR_MAKE_VAL(r);
                }
            }
            /* sin/cos: HW wants input in turns, so prescale by 1/(2pi) */
            if (strncmp(rest, "sin_", 4) == 0 || strncmp(rest, "cos_", 4) == 0) {
                uint16_t sop = (rest[0] == 's') ? BIR_SIN : BIR_COS;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t rt = ref_type(L, v);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, rt, 0.15915494309189535));
                uint32_t m = emit(L, BIR_FMUL, rt, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k);
                uint32_t r = emit(L, sop, rt, 1, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(m));
                return BIR_MAKE_VAL(r);
            }
        }

        /* ---- Regular function call ---- */

        /* Find function in module */
        uint32_t fi = 0;
        int found = 0;
        for (uint32_t i = 0; i < L->M->num_funcs; i++) {
            if (L->M->funcs[i].name < L->M->string_len
                && strcmp(&L->M->strings[L->M->funcs[i].name], cname) == 0) {
                fi = i;
                found = 1;
                break;
            }
        }
        if (!found) {
            lower_error(L, node, BC_E105);
            return BIR_VAL_NONE;
        }

        uint32_t ftype = L->M->funcs[fi].type;
        uint32_t ret_t = L->M->types[ftype].inner;

        /* Lower arguments */
        uint32_t args[16];
        int nargs = 0;
        uint32_t arg = ND(L, callee_n)->next_sibling;
        while (arg && nargs < 16) {
            args[nargs++] = lower_expr(L, arg);
            arg = ND(L, arg)->next_sibling;
        }

        if (1 + nargs <= BIR_OPERANDS_INLINE) {
            uint32_t inst = emit(L, BIR_CALL, ret_t, (uint8_t)(1+nargs), 0);
            set_op(L, inst, 0, fi);
            for (int i = 0; i < nargs; i++)
                set_op(L, inst, 1+i, args[i]);
            return BIR_MAKE_VAL(inst);
        }
        /* Overflow mode: pack into extra_operands */
        {
            uint32_t extra_start = L->M->num_extra_ops;
            if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                L->M->extra_operands[L->M->num_extra_ops++] = fi;
            for (int i = 0; i < nargs; i++)
                if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                    L->M->extra_operands[L->M->num_extra_ops++] = args[i];
            uint32_t total = L->M->num_extra_ops - extra_start;
            uint32_t inst = emit(L, BIR_CALL, ret_t, BIR_OPERANDS_OVERFLOW, 0);
            set_op(L, inst, 0, extra_start);
            set_op(L, inst, 1, total);
            return BIR_MAKE_VAL(inst);
        }
    }

    case AST_CAST: {
        uint32_t type_n = n->first_child;
        uint32_t expr_n = ND(L, type_n)->next_sibling;
        int pdepth      = n->d.oper.flags;

        /* C++ aggregate init: Type{expr, ...} → alloca + field stores + load */
        if (expr_n && ND(L, expr_n)->type == AST_INIT_LIST) {
            uint32_t st = resolve_type(L, type_n, pdepth, 0);
            struct_def_t *sd = NULL;
            for (int si = 0; si < L->nstructs; si++) {
                if (L->structs[si].bir_type == st) { sd = &L->structs[si]; break; }
            }
            if (sd) {
                uint32_t ptr_t = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
                uint32_t alloca = emit(L, BIR_ALLOCA, ptr_t, 0, 0);
                uint32_t el = ND(L, expr_n)->first_child;
                for (int fi = 0; fi < sd->num_fields && el; fi++) {
                    uint32_t val = lower_expr(L, el);
                    uint32_t fpt = bir_type_ptr(L->M, sd->field_types[fi],
                                                BIR_AS_PRIVATE);
                    uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                        bir_type_int(L->M, 32), fi));
                    uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
                    set_op(L, gep, 0, BIR_MAKE_VAL(alloca));
                    set_op(L, gep, 1, ci);
                    uint32_t store = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                    set_op(L, store, 0, val);
                    set_op(L, store, 1, BIR_MAKE_VAL(gep));
                    el = ND(L, el)->next_sibling;
                }
                uint32_t ld = emit(L, BIR_LOAD, st, 1, 0);
                set_op(L, ld, 0, BIR_MAKE_VAL(alloca));
                return BIR_MAKE_VAL(ld);
            }
        }

        uint32_t dst_t  = resolve_type(L, type_n, pdepth, 0);
        uint32_t val    = lower_expr(L, expr_n);
        uint32_t src_t  = ref_type(L, val);

        if (src_t == dst_t) return val;

        int sf = is_float_type(L, src_t), df = is_float_type(L, dst_t);
        int sp = is_ptr_type(L, src_t),   dp = is_ptr_type(L, dst_t);
        int src_uns = node_is_unsigned(L, expr_n);
        int dst_uns = node_is_unsigned(L, node);

        uint16_t cop;
        if (sf && df) {
            cop = (L->M->types[dst_t].width > L->M->types[src_t].width)
                  ? BIR_FPEXT : BIR_FPTRUNC;
        } else if (sf && !df)  cop = dst_uns ? BIR_FPTOUI : BIR_FPTOSI;
        else if (!sf && df)    cop = src_uns ? BIR_UITOFP : BIR_SITOFP;
        else if (sp && !dp)    cop = BIR_PTRTOINT;
        else if (!sp && dp)    cop = BIR_INTTOPTR;
        else if (sp && dp)     cop = BIR_BITCAST;
        else {
            int sw = L->M->types[src_t].width;
            int dw = L->M->types[dst_t].width;
            if (dw > sw)      cop = src_uns ? BIR_ZEXT : BIR_SEXT;
            else if (dw < sw) cop = BIR_TRUNC;
            else              cop = BIR_BITCAST;
        }

        uint32_t inst = emit(L, cop, dst_t, 1, 0);
        set_op(L, inst, 0, val);
        return BIR_MAKE_VAL(inst);
    }

    case AST_SIZEOF: {
        uint32_t inner = n->first_child;
        int64_t sz = 4;
        if (inner && ND(L, inner)->type == AST_TYPE_SPEC) {
            switch (ND(L, inner)->d.btype.kind) {
            case TYPE_CHAR:  sz = 1; break;
            case TYPE_SHORT: sz = 2; break;
            case TYPE_LONG: case TYPE_LLONG: sz = 8; break;
            case TYPE_DOUBLE: case TYPE_LDOUBLE: sz = 8; break;
            default: sz = 4; break;
            }
        }
        uint32_t t = bir_type_int(L->M, 64);
        return BIR_MAKE_CONST(bir_const_int(L->M, t, sz));
    }

    case AST_STRING_LIT: {
        /* Decode the source span (with surrounding quotes and C
         * escape sequences) into raw bytes, intern them in the
         * strings table, create a __constant__ global whose
         * initializer is BIR_CONST_BYTES pointing at those bytes,
         * and return a BIR_GLOBAL_REF to the global. Each backend
         * is responsible for materialising the bytes in its
         * binary's read-only region. */
        const char *src = L->src + ND(L, node)->d.text.offset;
        uint32_t srclen = ND(L, node)->d.text.len;

        /* Adjacent string literals are merged by the parser into a
         * single AST_STRING_LIT whose source span covers both. We
         * walk the span and decode each "..." segment, skipping
         * whitespace between them. */
        char bytes[2048];
        uint32_t blen = 0;
        uint32_t p = 0;
        while (p < srclen) {
            while (p < srclen && (src[p] == ' ' || src[p] == '\t' ||
                                  src[p] == '\n' || src[p] == '\r')) p++;
            if (p >= srclen) break;
            if (src[p] != '"') { p++; continue; }
            p++;  /* opening quote */
            while (p < srclen && src[p] != '"' && blen + 1 < sizeof(bytes)) {
                if (src[p] != '\\') { bytes[blen++] = src[p++]; continue; }
                p++;
                if (p >= srclen) break;
                char c = src[p++];
                switch (c) {
                case 'n':  bytes[blen++] = '\n'; break;
                case 't':  bytes[blen++] = '\t'; break;
                case 'r':  bytes[blen++] = '\r'; break;
                case '0':  bytes[blen++] = '\0'; break;
                case '\\': bytes[blen++] = '\\'; break;
                case '"':  bytes[blen++] = '"';  break;
                case '\'': bytes[blen++] = '\''; break;
                case 'a':  bytes[blen++] = '\a'; break;
                case 'b':  bytes[blen++] = '\b'; break;
                case 'f':  bytes[blen++] = '\f'; break;
                case 'v':  bytes[blen++] = '\v'; break;
                case 'x': {
                    /* \xHH hex escape. Read 1-2 hex digits. */
                    int val = 0, hd = 0;
                    while (p < srclen && hd < 2) {
                        char h = src[p];
                        int d = -1;
                        if (h >= '0' && h <= '9') d = h - '0';
                        else if (h >= 'a' && h <= 'f') d = 10 + (h - 'a');
                        else if (h >= 'A' && h <= 'F') d = 10 + (h - 'A');
                        if (d < 0) break;
                        val = val * 16 + d;
                        p++; hd++;
                    }
                    bytes[blen++] = (char)val;
                    break;
                }
                default:   bytes[blen++] = c; break;
                }
            }
            if (p < srclen && src[p] == '"') p++;  /* closing quote */
        }
        bytes[blen] = '\0';

        /* Stash the bytes in the strings table and create a global
         * for them. We never deduplicate; two source-distinct
         * literals stay distinct, which avoids the pointer-identity
         * surprise dedup would create. */
        uint32_t off = bir_add_string(L->M, bytes, blen);
        uint32_t i8  = bir_type_int(L->M, 8);
        uint32_t arr = bir_type_array(L->M, i8, blen + 1u);
        uint32_t ptr = bir_type_ptr(L->M, arr, BIR_AS_CONSTANT);

        if (L->M->num_globals >= BIR_MAX_GLOBALS) {
            lower_error(L, node, BC_E106);
            return BIR_VAL_NONE;
        }
        uint32_t gi = L->M->num_globals++;
        bir_global_t *G = &L->M->globals[gi];
        char gname[24];
        snprintf(gname, sizeof(gname), ".str.%u", gi);
        G->name = bir_add_string(L->M, gname, (uint32_t)strlen(gname));
        G->type = arr;
        G->initializer = BIR_MAKE_CONST(
            bir_const_bytes(L->M, arr, off, blen + 1u));
        G->cuda_flags = CUDA_CONSTANT;
        G->addrspace = BIR_AS_CONSTANT;
        G->is_const = 1;

        uint32_t inst = emit(L, BIR_GLOBAL_REF, ptr, 0, (uint8_t)(gi & 0xff));
        return BIR_MAKE_VAL(inst);
    }

    default:
        lower_error(L, node, BC_E106);
        return BIR_VAL_NONE;
    }
}

/* ---- L-Value Lowering ---- */

static uint32_t lower_lvalue(lower_t *L, uint32_t node)
{
    if (!node) return BIR_VAL_NONE;
    const ast_node_t *n = ND(L, node);

    switch (n->type) {
    case AST_IDENT: {
        char name[128];
        get_text(L, node, name, sizeof(name));
        sym_t *s = find_sym(L, name);
        if (!s) {
            lower_error(L, node, BC_E107);
            return BIR_VAL_NONE;
        }
        if (s->is_alloca)
            return BIR_MAKE_VAL(s->ref);
        lower_error(L, node, BC_E108);
        return BIR_VAL_NONE;
    }

    case AST_SUBSCRIPT: {
        uint32_t base_n = n->first_child;
        uint32_t idx_n  = ND(L, base_n)->next_sibling;

        uint32_t base_v = lower_expr(L, base_n);
        uint32_t idx_v  = lower_expr(L, idx_n);
        uint32_t bt     = ref_type(L, base_v);

        /* Array decay: ptr(T[N]) → ptr(T) for correct GEP stride */
        uint32_t gep_t = bt;
        uint32_t et = ptr_inner(L, bt);
        if (et && et < L->M->num_types &&
            L->M->types[et].kind == BIR_TYPE_ARRAY) {
            uint8_t as = is_ptr_type(L, bt) ? L->M->types[bt].addrspace : 0;
            gep_t = bir_type_ptr(L->M, L->M->types[et].inner, as);
        }

        uint32_t gep = emit(L, BIR_GEP, gep_t, 2, 0);
        set_op(L, gep, 0, base_v);
        set_op(L, gep, 1, idx_v);
        return BIR_MAKE_VAL(gep);
    }

    case AST_UNARY_PREFIX:
        if (n->d.oper.op == TOK_STAR)
            return lower_expr(L, n->first_child);
        lower_error(L, node, BC_E109);
        return BIR_VAL_NONE;

    case AST_MEMBER: {
        uint32_t obj = n->first_child;
        uint32_t fld = ND(L, obj)->next_sibling;
        char fname[128];
        get_text(L, fld, fname, sizeof(fname));

        uint32_t obj_ptr;
        if (n->d.member.is_arrow)
            obj_ptr = lower_expr(L, obj);
        else
            obj_ptr = lower_lvalue(L, obj);
        if (obj_ptr == BIR_VAL_NONE) return BIR_VAL_NONE;

        uint32_t pt = ref_type(L, obj_ptr);
        uint32_t st = ptr_inner(L, pt);

        /* Preserve source address space: local alloca → private,
         * global array element → global. Don't assume private —
         * cells[i].mat needs a global load, not a scratch read. */
        uint8_t src_as = L->M->types[pt].addrspace;
        for (int si = 0; si < L->nstructs; si++) {
            if (L->structs[si].bir_type != st) continue;
            for (int fi = 0; fi < L->structs[si].num_fields; fi++) {
                if (strcmp(L->structs[si].field_names[fi], fname) != 0)
                    continue;
                uint32_t ft  = L->structs[si].field_types[fi];
                uint32_t fpt = bir_type_ptr(L->M, ft, src_as);
                uint32_t idx = BIR_MAKE_CONST(bir_const_int(L->M,
                    bir_type_int(L->M, 32), fi));
                uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
                set_op(L, gep, 0, obj_ptr);
                set_op(L, gep, 1, idx);
                return BIR_MAKE_VAL(gep);
            }
        }
        lower_error(L, node, BC_E110);
        return BIR_VAL_NONE;
    }

    case AST_PAREN:
        return lower_lvalue(L, n->first_child);

    default:
        lower_error(L, node, BC_E111);
        return BIR_VAL_NONE;
    }
}

/* ---- Statement Lowering ---- */

static void lower_var_decl(lower_t *L, uint32_t node)
{
    L->cur_node = node;
    const ast_node_t *n = ND(L, node);
    uint32_t type_n = child_at(L, node, 0);
    uint32_t name_n = child_at(L, node, 1);
    if (!name_n || ND(L, name_n)->type != AST_IDENT) return;

    /* Embedded anonymous struct/union def? Register before resolve. */
    uint32_t andef = ND(L, name_n)->next_sibling;
    if (andef && ND(L, andef)->type == AST_STRUCT_DEF)
        collect_struct(L, andef);

    char name[128];
    get_text(L, name_n, name, sizeof(name));

    uint32_t elem_t = resolve_type(L, type_n, n->d.oper.flags, n->cuda_flags);

    /* Collect array dimensions: float a[16][16] → dims={16,16}, ndim=2.
     * Parser stores the bracket-dimension count in d.oper.op so we don't
     * accidentally eat `int step = 4096` as `int step[4096]`. */
    uint32_t next = ND(L, name_n)->next_sibling;
    /* Skip embedded anonymous struct_def (sema already registered it) */
    if (next && ND(L, next)->type == AST_STRUCT_DEF)
        next = ND(L, next)->next_sibling;
    int is_array = 0;
    uint32_t dims[8];
    int ndim = 0;
    int max_dims = n->d.oper.op;  /* how many [N] the parser saw */

    while (next && ndim < 8 && ndim < max_dims) {
        if (ND(L, next)->type == AST_INT_LIT) {
            dims[ndim++] = (uint32_t)parse_int_text(
                L->src + ND(L, next)->d.text.offset,
                (int)ND(L, next)->d.text.len);
            is_array = 1;
            next = ND(L, next)->next_sibling;
        } else if (ND(L, next)->type == AST_IDENT) {
            char aname[128];
            get_text(L, next, aname, sizeof(aname));
            int64_t aval;
            if (find_enum(L, aname, &aval) || find_binding_int(L, aname, &aval)) {
                dims[ndim++] = (uint32_t)aval;
                is_array = 1;
                next = ND(L, next)->next_sibling;
            } else break;
        } else break;
    }
    uint32_t arr_count = ndim > 0 ? dims[0] : 0;

    /* ---- __shared__ variables: LDS allocation, no initializer ---- */
    if (n->cuda_flags & CUDA_SHARED) {
        if (is_array) {
            uint32_t arr_t = elem_t;
            for (int d = ndim - 1; d >= 0; d--)
                arr_t = bir_type_array(L->M, arr_t, dims[d]);
            uint32_t ptr_t = bir_type_ptr(L->M, arr_t, BIR_AS_SHARED);
            uint32_t sa = emit(L, BIR_SHARED_ALLOC, ptr_t, 0, 0);
            add_sym(L, name, sa, arr_t, 1);
        } else {
            uint32_t ptr_t = bir_type_ptr(L->M, elem_t, BIR_AS_SHARED);
            uint32_t sa = emit(L, BIR_SHARED_ALLOC, ptr_t, 0, 0);
            add_sym(L, name, sa, elem_t, 1);
        }
        return;
    }

    if (is_array) {
        uint32_t arr_t = elem_t;
        for (int d = ndim - 1; d >= 0; d--)
            arr_t = bir_type_array(L->M, arr_t, dims[d]);
        uint32_t ptr_t = bir_type_ptr(L->M, arr_t, BIR_AS_PRIVATE);
        uint32_t alloca = emit(L, BIR_ALLOCA, ptr_t, 0, 0);
        add_sym(L, name, alloca, arr_t, 1);
        /* Array initializer list: int arr[3] = {1, 2, 3}; */
        if (next && ND(L, next)->type == AST_INIT_LIST) {
            uint32_t init_n = next;
            uint32_t ept = bir_type_ptr(L->M, elem_t, BIR_AS_PRIVATE);
            uint32_t el = ND(L, init_n)->first_child;
            uint32_t idx = 0;
            while (el && idx < arr_count) {
                uint32_t val = lower_expr(L, el);
                uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                    bir_type_int(L->M, 32), (int64_t)idx));
                uint32_t gep = emit(L, BIR_GEP, ept, 2, 0);
                set_op(L, gep, 0, BIR_MAKE_VAL(alloca));
                set_op(L, gep, 1, ci);
                uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                set_op(L, st, 0, val);
                set_op(L, st, 1, BIR_MAKE_VAL(gep));
                idx++;
                el = ND(L, el)->next_sibling;
            }
        }
        return;
    }

    /* Scalar / pointer variable */
    uint32_t ptr_t = bir_type_ptr(L->M, elem_t, BIR_AS_PRIVATE);
    uint32_t alloca = emit(L, BIR_ALLOCA, ptr_t, 0, 0);
    add_sym(L, name, alloca, elem_t, 1);

    /* Initializer: skip type and name, skip struct_def, find init expr */
    uint32_t init_n = ND(L, name_n)->next_sibling;
    if (init_n && ND(L, init_n)->type == AST_STRUCT_DEF)
        init_n = ND(L, init_n)->next_sibling;
    if (init_n && ND(L, init_n)->type == AST_INIT_LIST) {
        /* Struct initializer: Vec3 v = {1.0f, 2.0f, 3.0f}; */
        struct_def_t *sd = NULL;
        for (int si = 0; si < L->nstructs; si++) {
            if (L->structs[si].bir_type == elem_t) { sd = &L->structs[si]; break; }
        }
        if (sd) {
            uint32_t el = ND(L, init_n)->first_child;
            for (int fi = 0; fi < sd->num_fields && el; fi++) {
                uint32_t val = lower_expr(L, el);
                uint32_t fpt = bir_type_ptr(L->M, sd->field_types[fi],
                                            BIR_AS_PRIVATE);
                uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                    bir_type_int(L->M, 32), fi));
                uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
                set_op(L, gep, 0, BIR_MAKE_VAL(alloca));
                set_op(L, gep, 1, ci);
                uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                set_op(L, st, 0, val);
                set_op(L, st, 1, BIR_MAKE_VAL(gep));
                el = ND(L, el)->next_sibling;
            }
        }
    } else if (init_n && ND(L, init_n)->type != AST_NONE) {
        uint32_t init_v = lower_expr(L, init_n);

        /* Struct copy from pointer: Part p = parts[tid];
         * lower_expr returned a GEP (ptr to struct in global/shared),
         * not the struct value. Field-by-field copy required, or we'd
         * dump 8 bytes of raw pointer into the alloca and the remaining
         * fields would contain whatever the silicon was dreaming about. */
        uint32_t vt = ref_type(L, init_v);
        int did_copy = 0;
        if (is_ptr_type(L, vt) &&
            elem_t < L->M->num_types &&
            L->M->types[elem_t].kind == BIR_TYPE_STRUCT) {
            struct_def_t *sd = NULL;
            for (int si = 0; si < L->nstructs; si++) {
                if (L->structs[si].bir_type == elem_t) {
                    sd = &L->structs[si]; break;
                }
            }
            if (sd) {
                uint8_t src_as = L->M->types[vt].addrspace;
                uint32_t t32 = bir_type_int(L->M, 32);
                for (int fi = 0; fi < sd->num_fields && fi < 16; fi++) {
                    uint32_t ci = BIR_MAKE_CONST(
                        bir_const_int(L->M, t32, fi));
                    uint32_t spt = bir_type_ptr(L->M,
                        sd->field_types[fi], src_as);
                    uint32_t sg = emit(L, BIR_GEP, spt, 2, 0);
                    set_op(L, sg, 0, init_v);
                    set_op(L, sg, 1, ci);
                    uint32_t ld = emit(L, BIR_LOAD,
                        sd->field_types[fi], 1, 0);
                    set_op(L, ld, 0, BIR_MAKE_VAL(sg));
                    uint32_t dpt = bir_type_ptr(L->M,
                        sd->field_types[fi], BIR_AS_PRIVATE);
                    uint32_t dg = emit(L, BIR_GEP, dpt, 2, 0);
                    set_op(L, dg, 0, BIR_MAKE_VAL(alloca));
                    set_op(L, dg, 1, ci);
                    uint32_t s = emit(L, BIR_STORE,
                        bir_type_void(L->M), 2, 0);
                    set_op(L, s, 0, BIR_MAKE_VAL(ld));
                    set_op(L, s, 1, BIR_MAKE_VAL(dg));
                }
                did_copy = 1;
            }
        }
        if (!did_copy) {
            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, init_v);
            set_op(L, st, 1, BIR_MAKE_VAL(alloca));
        }
    }
}

static void lower_stmt(lower_t *L, uint32_t node)
{
    if (!node) return;
    L->cur_node = node;
    const ast_node_t *n = ND(L, node);

    switch (n->type) {
    case AST_EXPR_STMT:
        if (n->first_child) lower_expr(L, n->first_child);
        break;

    case AST_BLOCK:
        push_scope(L);
        lower_block_stmts(L, node);
        pop_scope(L);
        break;

    case AST_VAR_DECL:
        lower_var_decl(L, node);
        break;

    case AST_RETURN: {
        uint32_t t_void = bir_type_void(L->M);
        if (n->first_child) {
            uint32_t val  = lower_expr(L, n->first_child);
            uint32_t inst = emit(L, BIR_RET, t_void, 1, 0);
            set_op(L, inst, 0, val);
        } else {
            emit(L, BIR_RET, t_void, 0, 0);
        }
        break;
    }

    case AST_IF: {
        uint32_t cond_n = child_at(L, node, 0);
        uint32_t then_n = child_at(L, node, 1);
        uint32_t else_n = child_at(L, node, 2);

        uint32_t cond = lower_expr(L, cond_n);

        uint32_t then_b = new_block(L, "if.then");
        uint32_t else_b = else_n ? new_block(L, "if.else") : 0;
        uint32_t end_b  = new_block(L, "if.end");

        uint32_t br = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
        set_op(L, br, 0, cond);
        set_op(L, br, 1, then_b);
        set_op(L, br, 2, else_n ? else_b : end_b);
        set_op(L, br, 3, end_b);

        /* Then */
        set_block(L, then_b);
        lower_stmt(L, then_n);
        if (!block_terminated(L)) {
            uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, j, 0, end_b);
        }

        /* Else */
        if (else_n) {
            set_block(L, else_b);
            lower_stmt(L, else_n);
            if (!block_terminated(L)) {
                uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
                set_op(L, j, 0, end_b);
            }
        }

        set_block(L, end_b);
        break;
    }

    case AST_FOR: {
        /* Children: init, cond, incr, body (always 4) */
        uint32_t init_n = child_at(L, node, 0);
        uint32_t cond_n = child_at(L, node, 1);
        uint32_t incr_n = child_at(L, node, 2);
        uint32_t body_n = child_at(L, node, 3);

        push_scope(L);

        /* Init */
        if (init_n && ND(L, init_n)->type != AST_NONE) {
            if (ND(L, init_n)->type == AST_VAR_DECL)
                lower_var_decl(L, init_n);
            else
                lower_expr(L, init_n);
        }

        uint32_t cond_b = new_block(L, "for.cond");
        uint32_t body_b = new_block(L, "for.body");
        uint32_t incr_b = new_block(L, "for.inc");
        uint32_t end_b  = new_block(L, "for.end");

        /* Push loop targets */
        if (L->loop_depth < MAX_LOOPS) {
            L->break_tgt[L->loop_depth] = end_b;
            L->cont_tgt[L->loop_depth]  = incr_b;
            L->loop_depth++;
        }

        /* Branch to condition */
        uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
        set_op(L, j, 0, cond_b);

        /* Cond */
        set_block(L, cond_b);
        if (cond_n && ND(L, cond_n)->type != AST_NONE) {
            uint32_t cv = lower_expr(L, cond_n);
            uint32_t br = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, cv);
            set_op(L, br, 1, body_b);
            set_op(L, br, 2, end_b);
            set_op(L, br, 3, end_b);
        } else {
            /* Infinite loop: for(;;) */
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, body_b);
        }

        /* Body */
        set_block(L, body_b);
        if (body_n) lower_stmt(L, body_n);
        if (!block_terminated(L)) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, incr_b);
        }

        /* Incr */
        set_block(L, incr_b);
        if (incr_n && ND(L, incr_n)->type != AST_NONE)
            lower_expr(L, incr_n);
        {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, cond_b);
        }

        /* Pop loop targets */
        if (L->loop_depth > 0) L->loop_depth--;

        set_block(L, end_b);
        pop_scope(L);
        break;
    }

    case AST_WHILE: {
        uint32_t cond_n = child_at(L, node, 0);
        uint32_t body_n = child_at(L, node, 1);

        uint32_t cond_b = new_block(L, "while.cond");
        uint32_t body_b = new_block(L, "while.body");
        uint32_t end_b  = new_block(L, "while.end");

        if (L->loop_depth < MAX_LOOPS) {
            L->break_tgt[L->loop_depth] = end_b;
            L->cont_tgt[L->loop_depth]  = cond_b;
            L->loop_depth++;
        }

        uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
        set_op(L, j, 0, cond_b);

        set_block(L, cond_b);
        {
            uint32_t cv = lower_expr(L, cond_n);
            uint32_t br = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, cv);
            set_op(L, br, 1, body_b);
            set_op(L, br, 2, end_b);
            set_op(L, br, 3, end_b);
        }

        set_block(L, body_b);
        if (body_n) lower_stmt(L, body_n);
        if (!block_terminated(L)) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, cond_b);
        }

        if (L->loop_depth > 0) L->loop_depth--;
        set_block(L, end_b);
        break;
    }

    case AST_DO_WHILE: {
        uint32_t body_n = child_at(L, node, 0);
        uint32_t cond_n = child_at(L, node, 1);

        uint32_t body_b = new_block(L, "do.body");
        uint32_t cond_b = new_block(L, "do.cond");
        uint32_t end_b  = new_block(L, "do.end");

        if (L->loop_depth < MAX_LOOPS) {
            L->break_tgt[L->loop_depth] = end_b;
            L->cont_tgt[L->loop_depth]  = cond_b;
            L->loop_depth++;
        }

        {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, body_b);
        }

        set_block(L, body_b);
        if (body_n) lower_stmt(L, body_n);
        if (!block_terminated(L)) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, cond_b);
        }

        set_block(L, cond_b);
        {
            uint32_t cv = lower_expr(L, cond_n);
            uint32_t br = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, cv);
            set_op(L, br, 1, body_b);
            set_op(L, br, 2, end_b);
            set_op(L, br, 3, end_b);
        }

        if (L->loop_depth > 0) L->loop_depth--;
        set_block(L, end_b);
        break;
    }

    case AST_SWITCH: {
        uint32_t cond_n = child_at(L, node, 0);
        uint32_t body_n = child_at(L, node, 1);

        uint32_t cond_v = lower_expr(L, cond_n);
        uint32_t end_b  = new_block(L, "switch.end");
        uint32_t old_exit = L->switch_exit;
        L->switch_exit = end_b;

        /* Save break target */
        int old_loop = L->loop_depth;
        if (L->loop_depth < MAX_LOOPS) {
            L->break_tgt[L->loop_depth] = end_b;
            L->loop_depth++;
        }

        /* Scan body for case/default to build jump table */
        /* Lower as if-else chain for simplicity */
        uint32_t default_b = end_b;
        uint32_t c = body_n ? ND(L, body_n)->first_child : 0;

        /* Two passes: first create blocks, then lower */
        /* Build case list */
        typedef struct { uint32_t ast; uint32_t block; int is_default; } case_t;
        case_t cases[64];
        int ncases = 0;

        uint32_t scan = c;
        int case_idx = 0;
        while (scan && ncases < 64) {
            if (ND(L, scan)->type == AST_CASE || ND(L, scan)->type == AST_DEFAULT) {
                char bname[32];
                if (ND(L, scan)->type == AST_DEFAULT)
                    snprintf(bname, sizeof(bname), "switch.default");
                else
                    snprintf(bname, sizeof(bname), "switch.case.%d", case_idx++);
                cases[ncases].ast = scan;
                cases[ncases].block = new_block(L, bname);
                cases[ncases].is_default = (ND(L, scan)->type == AST_DEFAULT);
                if (cases[ncases].is_default)
                    default_b = cases[ncases].block;
                ncases++;
            }
            scan = ND(L, scan)->next_sibling;
        }

        /* Emit BIR_SWITCH in overflow mode */
        {
            uint32_t extra_start = L->M->num_extra_ops;
            /* Pack: cond_val, default_block, (case_const, target_block)... */
            if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                L->M->extra_operands[L->M->num_extra_ops++] = cond_v;
            if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                L->M->extra_operands[L->M->num_extra_ops++] = default_b;
            for (int i = 0; i < ncases; i++) {
                if (cases[i].is_default) continue;
                uint32_t case_val_n = ND(L, cases[i].ast)->first_child;
                uint32_t cv = lower_expr(L, case_val_n);
                if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                    L->M->extra_operands[L->M->num_extra_ops++] = cv;
                if (L->M->num_extra_ops < BIR_MAX_EXTRA_OPS)
                    L->M->extra_operands[L->M->num_extra_ops++] = cases[i].block;
            }
            uint32_t total = L->M->num_extra_ops - extra_start;
            uint32_t sw = emit(L, BIR_SWITCH, bir_type_void(L->M),
                               BIR_OPERANDS_OVERFLOW, 0);
            set_op(L, sw, 0, extra_start);
            set_op(L, sw, 1, total);
        }

        /* Lower case bodies */
        uint32_t stmt_node = c;
        int cur_case = -1;
        while (stmt_node) {
            if (ND(L, stmt_node)->type == AST_CASE ||
                ND(L, stmt_node)->type == AST_DEFAULT) {
                /* Find this case's block */
                for (int i = 0; i < ncases; i++) {
                    if (cases[i].ast == stmt_node) {
                        cur_case = i;
                        set_block(L, cases[i].block);
                        break;
                    }
                }
            } else if (cur_case >= 0) {
                lower_stmt(L, stmt_node);
            }
            stmt_node = ND(L, stmt_node)->next_sibling;
        }

        /* Terminate last case if needed */
        if (!block_terminated(L)) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, end_b);
        }

        L->switch_exit = old_exit;
        L->loop_depth = old_loop;
        set_block(L, end_b);
        break;
    }

    case AST_BREAK:
        if (L->loop_depth > 0) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, L->break_tgt[L->loop_depth - 1]);
        }
        break;

    case AST_CONTINUE:
        if (L->loop_depth > 0) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, L->cont_tgt[L->loop_depth - 1]);
        }
        break;

    case AST_LABEL: {
        char lname[64];
        uint32_t ln = n->first_child;
        if (ln && ND(L, ln)->type == AST_IDENT)
            get_text(L, ln, lname, sizeof(lname));
        else
            get_text(L, node, lname, sizeof(lname));
        uint32_t lbl_b = find_or_create_label(L, lname);
        if (!block_terminated(L)) {
            uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
            set_op(L, br, 0, lbl_b);
        }
        set_block(L, lbl_b);
        break;
    }

    case AST_GOTO: {
        char lname[64];
        uint32_t gn = n->first_child;
        if (gn && ND(L, gn)->type == AST_IDENT)
            get_text(L, gn, lname, sizeof(lname));
        else
            get_text(L, node, lname, sizeof(lname));
        uint32_t lbl_b = find_or_create_label(L, lname);
        uint32_t br = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
        set_op(L, br, 0, lbl_b);
        break;
    }

    case AST_PP_DIRECTIVE:
        /* Preprocessor directives skipped */
        break;

    default:
        /* Ignore other node types (struct defs, enum defs inside functions, etc.) */
        break;
    }
}

static void lower_block_stmts(lower_t *L, uint32_t block_node)
{
    uint32_t c = ND(L, block_node)->first_child;
    while (c) {
        if (ND(L, c)->type == AST_LABEL) {
            /* Labels always processed — they create new unterminated blocks */
            lower_stmt(L, c);
        } else if (!block_terminated(L)) {
            if (ND(L, c)->type == AST_VAR_DECL)
                lower_var_decl(L, c);
            else
                lower_stmt(L, c);
        }
        c = ND(L, c)->next_sibling;
    }
}

/* ---- Function Lowering ---- */

/*
 * Find the body (AST_BLOCK) node among func_def's children.
 * Children are: type_spec, name, [params...], block
 */
static uint32_t find_func_body(const lower_t *L, uint32_t func_def)
{
    uint32_t c = ND(L, func_def)->first_child;
    while (c) {
        if (ND(L, c)->type == AST_BLOCK) return c;
        c = ND(L, c)->next_sibling;
    }
    return 0;
}

/*
 * Count and collect function parameters from AST_PARAM children.
 * Parameters appear after type_spec and name, before body.
 */
static int collect_params(const lower_t *L, uint32_t func_def,
                          uint32_t *out, int max)
{
    int count = 0;
    uint32_t c = ND(L, func_def)->first_child;
    while (c) {
        if (ND(L, c)->type == AST_PARAM && count < max)
            out[count++] = c;
        c = ND(L, c)->next_sibling;
    }
    return count;
}

static void lower_func_body(lower_t *L, uint32_t func_def,
                            uint16_t cuda_flags, const char *name_override)
{
    /* Get return type */
    uint32_t type_n = child_at(L, func_def, 0);
    uint32_t name_n = child_at(L, func_def, 1);
    int ret_ptr = ND(L, func_def)->d.oper.flags;

    char fname_raw[128], fname[128];
    if (name_override)
        snprintf(fname, sizeof(fname), "%s", name_override);
    else {
        get_text(L, name_n, fname_raw, sizeof(fname_raw));
        /* Normalize operator names: "operator +" → "operator+" */
        if (strncmp(fname_raw, "operator", 8) == 0 && fname_raw[8] != '\0')
            normalize_op_name(fname_raw, fname, sizeof(fname));
        else
            memcpy(fname, fname_raw, sizeof(fname));
    }

    uint32_t ret_t = resolve_type(L, type_n, ret_ptr, 0);

    /* Collect parameters */
    uint32_t param_nodes[32];
    int nparams = collect_params(L, func_def, param_nodes, 32);

    /* Resolve param types and build function type */
    uint32_t param_types[32];
    for (int i = 0; i < nparams; i++) {
        const ast_node_t *pn = ND(L, param_nodes[i]);
        uint32_t pt_type_n = pn->first_child;
        int pdepth = pn->d.oper.flags; /* stored by parser */
        uint16_t p_cuda = pn->cuda_flags;
        param_types[i] = resolve_type(L, pt_type_n, pdepth, p_cuda);
    }

    /* Create function type */
    uint32_t fn_type = bir_type_func(L->M, ret_t, param_types, nparams);

    /* Create function */
    if (L->M->num_funcs >= BIR_MAX_FUNCS) return;
    uint32_t fi = L->M->num_funcs++;
    L->cur_func = fi;

    bir_func_t *F = &L->M->funcs[fi];
    memset(F, 0, sizeof(*F));
    F->name = bir_add_string(L->M, fname, (uint32_t)strlen(fname));
    F->type = fn_type;
    F->cuda_flags = cuda_flags;
    F->num_params = (uint16_t)nparams;
    F->first_block = L->M->num_blocks;
    F->num_blocks = 0;
    F->launch_bounds_max = ND(L, func_def)->launch_bounds_max;
    F->launch_bounds_min = ND(L, func_def)->launch_bounds_min;

    /* Reset labels for this function */
    L->nlabels = 0;

    /* Entry block */
    uint32_t entry = new_block(L, "entry");
    set_block(L, entry);
    L->base_inst = L->M->num_insts;

    /* Emit PARAM instructions */
    push_scope(L);
    for (int i = 0; i < nparams; i++) {
        uint32_t inst = emit(L, BIR_PARAM, param_types[i], 0, (uint8_t)i);
        /* Get param name */
        uint32_t pname_n = 0;
        uint32_t pc = ND(L, param_nodes[i])->first_child;
        while (pc) {
            if (ND(L, pc)->type == AST_IDENT) { pname_n = pc; break; }
            pc = ND(L, pc)->next_sibling;
        }
        if (pname_n) {
            char pname[128];
            get_text(L, pname_n, pname, sizeof(pname));
            /* Promote all params to allocas so they're reassignable.
               mem2reg cleans up the ones that are never written. */
            uint32_t pt = bir_type_ptr(L->M, param_types[i], BIR_AS_PRIVATE);
            uint32_t al = emit(L, BIR_ALLOCA, pt, 0, 0);
            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, BIR_MAKE_VAL(inst));
            set_op(L, st, 1, BIR_MAKE_VAL(al));
            add_sym(L, pname, al, param_types[i], 1);
        }
    }

    /* Lower body */
    uint32_t body_n = find_func_body(L, func_def);
    if (body_n) {
        push_scope(L);
        lower_block_stmts(L, body_n);
        pop_scope(L);
    }

    /* Add implicit ret void if no terminator */
    if (!block_terminated(L)) {
        if (is_void_type(L, ret_t))
            emit(L, BIR_RET, bir_type_void(L->M), 0, 0);
        else
            emit(L, BIR_UNREACHABLE, bir_type_void(L->M), 0, 0);
    }

    pop_scope(L);

    /* Set total_insts */
    F->total_insts = L->M->num_insts - L->base_inst;
}

/* ---- Declaration Collection (Pass 1) ---- */

static void collect_struct(lower_t *L, uint32_t node)
{
    if (L->nstructs >= MAX_STRUCTS) return;

    uint32_t type_n = ND(L, node)->first_child;
    if (!type_n) return;

    struct_def_t *sd = &L->structs[L->nstructs];
    memset(sd, 0, sizeof(*sd));

    /* Name from type_spec child */
    uint32_t name_n = ND(L, type_n)->first_child;
    if (name_n && ND(L, name_n)->type == AST_IDENT)
        get_text(L, name_n, sd->name, sizeof(sd->name));

    /* Collect fields: children after type_n */
    uint32_t member = ND(L, type_n)->next_sibling;
    while (member && sd->num_fields < 16) {
        if (ND(L, member)->type == AST_VAR_DECL) {
            uint32_t ft_n = ND(L, member)->first_child;
            uint32_t fn_n = ft_n ? ND(L, ft_n)->next_sibling : 0;

            uint32_t ft = resolve_type(L, ft_n, ND(L, member)->d.oper.flags, 0);

            if (fn_n && ND(L, fn_n)->type == AST_IDENT) {
                /* Handle comma-separated fields: float x, y, z;
                   Parser puts all names as siblings within one var_decl */
                sd->field_types[sd->num_fields] = ft;
                get_text(L, fn_n, sd->field_names[sd->num_fields],
                         sizeof(sd->field_names[0]));
                sd->num_fields++;

                /* Additional idents in the same var_decl */
                uint32_t extra = ND(L, fn_n)->next_sibling;
                while (extra && ND(L, extra)->type == AST_IDENT
                       && sd->num_fields < 16) {
                    sd->field_types[sd->num_fields] = ft;
                    get_text(L, extra, sd->field_names[sd->num_fields],
                             sizeof(sd->field_names[0]));
                    sd->num_fields++;
                    extra = ND(L, extra)->next_sibling;
                }

                /* Check for chained var_decl siblings (different types).
                 * Each has its own type_spec — resolve it, or we'd
                 * paint every field the same colour as the first. */
                uint32_t chain = ND(L, member)->next_sibling;
                while (chain && ND(L, chain)->type == AST_VAR_DECL) {
                    uint32_t cft_n = ND(L, chain)->first_child;
                    uint32_t cfn = cft_n ? ND(L, cft_n)->next_sibling : 0;
                    if (cfn && ND(L, cfn)->type == AST_IDENT
                        && sd->num_fields < 16) {
                        uint32_t cft = resolve_type(L, cft_n,
                            ND(L, chain)->d.oper.flags, 0);
                        sd->field_types[sd->num_fields] = cft;
                        get_text(L, cfn, sd->field_names[sd->num_fields],
                                 sizeof(sd->field_names[0]));
                        sd->num_fields++;
                    }
                    member = chain; /* advance past chained */
                    chain = ND(L, chain)->next_sibling;
                }
            }
        }
        member = ND(L, member)->next_sibling;
    }

    /* Create BIR struct type */
    if (sd->num_fields > 0) {
        sd->bir_type = bir_type_struct(L->M, sd->field_types, sd->num_fields);
        L->nstructs++;
    }
}

static void collect_enum(lower_t *L, uint32_t node)
{
    uint32_t c = ND(L, node)->first_child;
    /* Skip type_spec child */
    if (c) c = ND(L, c)->next_sibling;

    int64_t next_val = 0;
    while (c && L->nenums < MAX_ENUMS) {
        if (ND(L, c)->type == AST_ENUMERATOR) {
            char name[64];
            get_text(L, c, name, sizeof(name));

            /* Check for explicit value */
            if (ND(L, c)->first_child) {
                const ast_node_t *val_n = ND(L, ND(L, c)->first_child);
                if (val_n->type == AST_INT_LIT)
                    next_val = parse_int_text(L->src + val_n->d.text.offset,
                                              (int)val_n->d.text.len);
            }

            snprintf(L->enums[L->nenums].name,
                     sizeof(L->enums[0].name), "%s", name);
            L->enums[L->nenums].value = next_val;
            L->nenums++;
            next_val++;
        }
        c = ND(L, c)->next_sibling;
    }
}

static void collect_typedef(lower_t *L, uint32_t node)
{
    /* typedef TYPE NAME;
       In our AST this is a VAR_DECL with QUAL_TYPEDEF */
    if (!(ND(L, node)->qualifiers & QUAL_TYPEDEF)) return;
    if (L->ntypedefs >= MAX_TYPEDEFS) return;

    uint32_t type_n = child_at(L, node, 0);
    uint32_t name_n = child_at(L, node, 1);
    if (!name_n || ND(L, name_n)->type != AST_IDENT) return;

    /* If this is "typedef struct { ... } name;" the parser stashed
     * the AST_STRUCT_DEF as child[2] of the var_decl (after type
     * and name). The struct itself was given a synthetic anonymous
     * name by the parser, so collect_struct would record a name no
     * source code ever looks up. Register the struct under the
     * typedef name before falling through to the typedef record,
     * so any later "name f; f.field" lookup finds it via L->structs.
     *
     * Looking for the AST_STRUCT_DEF among the var_decl's children
     * rather than as a sibling of name_n because that is where the
     * parser actually puts it; this is the bug that bit the
     * soft-float compile and is now under test. */
    {
        uint32_t cc = ND(L, node)->first_child;
        while (cc) {
            if (ND(L, cc)->type == AST_STRUCT_DEF) {
                int prev_n = L->nstructs;
                collect_struct(L, cc);
                /* Add a second entry under the typedef name that
                 * shares the underlying bir_type. The synthetic
                 * anon entry stays so resolve_type for the typedef
                 * target finds it during its own TYPE_STRUCT walk;
                 * the typedef-named entry is what later member
                 * access via "name v; v.field" looks up by name. */
                if (L->nstructs > prev_n &&
                    L->nstructs < MAX_STRUCTS) {
                    struct_def_t *src = &L->structs[prev_n];
                    struct_def_t *dst = &L->structs[L->nstructs];
                    *dst = *src;
                    get_text(L, name_n, dst->name, sizeof(dst->name));
                    L->nstructs++;
                }
                break;
            }
            cc = ND(L, cc)->next_sibling;
        }
    }

    typedef_def_t *td = &L->typedefs[L->ntypedefs++];
    get_text(L, name_n, td->name, sizeof(td->name));
    td->bir_type = resolve_type(L, type_n, ND(L, node)->d.oper.flags, 0);
}

static void collect_global_var(lower_t *L, uint32_t node)
{
    uint16_t cuda = ND(L, node)->cuda_flags;
    if (!(cuda & (CUDA_SHARED | CUDA_CONSTANT | CUDA_DEVICE))) return;
    if (L->M->num_globals >= BIR_MAX_GLOBALS) return;

    uint32_t type_n = child_at(L, node, 0);
    uint32_t name_n = child_at(L, node, 1);
    if (!name_n || ND(L, name_n)->type != AST_IDENT) return;

    char gname[128];
    get_text(L, name_n, gname, sizeof(gname));

    uint32_t elem_t = resolve_type(L, type_n, ND(L, node)->d.oper.flags, cuda);

    /* Check for array size */
    uint32_t next = ND(L, name_n)->next_sibling;
    int has_arr_size = 0;
    if (next && ND(L, next)->type == AST_INT_LIT) {
        has_arr_size = 1;
        uint32_t count = (uint32_t)parse_int_text(
            L->src + ND(L, next)->d.text.offset,
            (int)ND(L, next)->d.text.len);
        elem_t = bir_type_array(L->M, elem_t, count);
    } else if (next && ND(L, next)->type == AST_IDENT) {
        char aname[128];
        get_text(L, next, aname, sizeof(aname));
        int64_t aval;
        if (find_enum(L, aname, &aval) || find_binding_int(L, aname, &aval)) {
            has_arr_size = 1;
            elem_t = bir_type_array(L->M, elem_t, (uint32_t)aval);
        }
    }

    uint32_t gi = L->M->num_globals++;
    bir_global_t *G = &L->M->globals[gi];
    G->name = bir_add_string(L->M, gname, (uint32_t)strlen(gname));
    G->type = elem_t;
    G->initializer = BIR_VAL_NONE;
    /* Check for literal initializer (skip past array size if present) */
    {
        uint32_t ic = has_arr_size ? ND(L, next)->next_sibling : next;
        if (ic && ND(L, ic)->type == AST_INT_LIT) {
            int64_t v = parse_int_text(L->src + ND(L, ic)->d.text.offset,
                                       (int)ND(L, ic)->d.text.len);
            G->initializer = BIR_MAKE_CONST(
                bir_const_int(L->M, bir_type_int(L->M, 32), v));
        } else if (ic && ND(L, ic)->type == AST_FLOAT_LIT) {
            int is_f32;
            double v = parse_float_text(L->src + ND(L, ic)->d.text.offset,
                                        (int)ND(L, ic)->d.text.len, &is_f32);
            uint32_t ft = is_f32 ? bir_type_float(L->M, 32)
                                 : bir_type_float(L->M, 64);
            G->initializer = BIR_MAKE_CONST(bir_const_float(L->M, ft, v));
        }
    }
    G->cuda_flags = cuda;
    G->is_const = (cuda & CUDA_CONSTANT) ? 1 : 0;

    if (cuda & CUDA_SHARED)        G->addrspace = BIR_AS_SHARED;
    else if (cuda & CUDA_CONSTANT) G->addrspace = BIR_AS_CONSTANT;
    else                           G->addrspace = BIR_AS_GLOBAL;
}

static void collect_template(lower_t *L, uint32_t node)
{
    if (L->ntemplates >= MAX_TEMPLATES) return;

    /* Find the function def inside */
    uint32_t c = ND(L, node)->first_child;
    uint32_t func_n = 0;
    while (c) {
        if (ND(L, c)->type == AST_FUNC_DEF || ND(L, c)->type == AST_FUNC_DECL) {
            func_n = c;
            break;
        }
        c = ND(L, c)->next_sibling;
    }
    if (!func_n) return;

    /* Get function name */
    uint32_t fn_name = child_at(L, func_n, 1);
    if (!fn_name) return;

    template_def_t *td = &L->templates[L->ntemplates++];
    td->ast = node;
    get_text(L, fn_name, td->name, sizeof(td->name));
}

/* ---- Template Instantiation ---- */

/*
 * Scan host main() for kernel launches and instantiate templates.
 * For scale<<<4, 256>>>(d_data, 2.0f, 1024):
 *   - callee = "scale"
 *   - Deduce T from literal arguments
 */
static void scan_launches(lower_t *L, uint32_t node)
{
    if (!node) return;
    const ast_node_t *n = ND(L, node);

    if (n->type == AST_LAUNCH) {
        uint32_t callee_n = n->first_child;
        if (!callee_n || ND(L, callee_n)->type != AST_IDENT) goto recurse;

        char cname[128];
        get_text(L, callee_n, cname, sizeof(cname));

        template_def_t *tmpl = find_template(L, cname);
        if (!tmpl) goto recurse;

        /* Deduce template type from launch arguments.
           Skip grid and block args (children 1 and 2 of launch).
           Actual args start after that. */
        uint32_t arg = ND(L, callee_n)->next_sibling; /* grid */
        if (arg) arg = ND(L, arg)->next_sibling;      /* block */
        /* Optional shared mem and stream */
        /* Skip to actual function args: after >>> comes () */
        /* In AST_LAUNCH, children after grid,block,[smem],[stream] are args */
        /* We need to find the function args. Let's skip 2 more possible. */
        if (arg) arg = ND(L, arg)->next_sibling; /* first real arg, or smem */

        /* Collect template params from AST_TEMPLATE_DECL */
        uint32_t tmpl_node = tmpl->ast;
        uint32_t tp = ND(L, tmpl_node)->first_child;
        int ntparams = 0;
        binding_t new_bindings[8];

        while (tp && ND(L, tp)->type == AST_TEMPLATE_PARAM && ntparams < 8) {
            int is_type = (ND(L, tp)->d.oper.flags == 0);
            uint32_t tpname_n = 0;
            uint32_t tpc = ND(L, tp)->first_child;
            while (tpc) {
                if (ND(L, tpc)->type == AST_IDENT) { tpname_n = tpc; break; }
                tpc = ND(L, tpc)->next_sibling;
            }

            new_bindings[ntparams].is_type = is_type;
            new_bindings[ntparams].type = 0;
            new_bindings[ntparams].ival = 0;
            if (tpname_n)
                get_text(L, tpname_n, new_bindings[ntparams].name,
                         sizeof(new_bindings[0].name));
            else
                new_bindings[ntparams].name[0] = '\0';

            ntparams++;
            tp = ND(L, tp)->next_sibling;
        }

        /* Deduce types from arguments.
           For typename T: look at float literal args (2.0f → float) */
        /* Find the function def to match params with template params */
        uint32_t func_n = 0;
        {
            uint32_t fc = ND(L, tmpl_node)->first_child;
            while (fc) {
                if (ND(L, fc)->type == AST_FUNC_DEF) { func_n = fc; break; }
                fc = ND(L, fc)->next_sibling;
            }
        }

        if (func_n) {
            /* Match function params against launch args to deduce T */
            uint32_t fparam_nodes[16];
            int nfparams = collect_params(L, func_n, fparam_nodes, 16);

            /* For each function param, check if its type is a template param */
            /* Then look at the corresponding launch arg to deduce */
            for (int i = 0; i < nfparams && arg; i++) {
                uint32_t fpt = ND(L, fparam_nodes[i])->first_child;
                if (fpt && ND(L, fpt)->type == AST_TYPE_SPEC
                    && ND(L, fpt)->d.btype.kind == TYPE_NAME) {
                    /* Named type — might be a template param */
                    uint32_t fpt_name = ND(L, fpt)->first_child;
                    if (fpt_name) {
                        char tname[64];
                        get_text(L, fpt_name, tname, sizeof(tname));
                        for (int b = 0; b < ntparams; b++) {
                            if (new_bindings[b].is_type
                                && strcmp(new_bindings[b].name, tname) == 0
                                && new_bindings[b].type == 0) {
                                /* Deduce from arg */
                                if (ND(L, arg)->type == AST_FLOAT_LIT) {
                                    int is_f32;
                                    parse_float_text(
                                        L->src + ND(L, arg)->d.text.offset,
                                        (int)ND(L, arg)->d.text.len, &is_f32);
                                    new_bindings[b].type = is_f32
                                        ? bir_type_float(L->M, 32)
                                        : bir_type_float(L->M, 64);
                                } else if (ND(L, arg)->type == AST_INT_LIT) {
                                    new_bindings[b].type = bir_type_int(L->M, 32);
                                } else if (ND(L, arg)->type == AST_IDENT) {
                                    /* Look up variable type */
                                    /* For pointers like d_data (float*),
                                       the template param T maps to pointee */
                                    /* Can't easily deduce without host sym table.
                                       Default to float for now. */
                                    new_bindings[b].type = bir_type_float(L->M, 32);
                                }
                            }
                        }
                    }
                }
                arg = ND(L, arg)->next_sibling;
            }
        }

        /* Default unresolved type bindings to float */
        for (int i = 0; i < ntparams; i++) {
            if (new_bindings[i].is_type && new_bindings[i].type == 0)
                new_bindings[i].type = bir_type_float(L->M, 32);
        }

        /* Build mangled name */
        char mangled[256];
        snprintf(mangled, sizeof(mangled), "%s", cname);
        /* For template instantiation, we could add <float> suffix
           but keep it simple — just use the base name if unique */

        /* Check if already instantiated */
        int already = 0;
        for (uint32_t i = 0; i < L->M->num_funcs; i++) {
            if (L->M->funcs[i].name < L->M->string_len
                && strcmp(&L->M->strings[L->M->funcs[i].name], mangled) == 0) {
                already = 1;
                break;
            }
        }

        if (!already && func_n) {
            /* Set bindings and instantiate */
            int old_nb = L->nbindings;
            for (int i = 0; i < ntparams && L->nbindings < 8; i++)
                L->bindings[L->nbindings++] = new_bindings[i];

            uint16_t cuda = ND(L, func_n)->cuda_flags;
            lower_func_body(L, func_n, cuda, mangled);

            L->nbindings = old_nb;
        }

        return;
    }

recurse:
    {
        uint32_t c = n->first_child;
        while (c) {
            scan_launches(L, c);
            c = ND(L, c)->next_sibling;
        }
    }
}

/* ---- Top-Level Entry Point ---- */

int bir_lower(const parser_t *P, uint32_t ast_root, bir_module_t *M,
              const sema_ctx_t *sema,
              bc_error_t *out_errs, int *out_nerrs)
{
    static lower_t L_storage; /* large struct — static to avoid stack overflow */
    lower_t *L = &L_storage;
    memset(L, 0, sizeof(*L));
    L->P    = P;
    L->M    = M;
    L->src  = P->src;
    L->sema = sema;

    bir_module_init(M);

    /* Pass 1: collect declarations */
    uint32_t c = P->nodes[ast_root].first_child;
    while (c) {
        const ast_node_t *n = &P->nodes[c];
        switch (n->type) {
        case AST_STRUCT_DEF:
            collect_struct(L, c);
            break;
        case AST_ENUM_DEF:
            collect_enum(L, c);
            break;
        case AST_VAR_DECL:
            if (n->qualifiers & QUAL_TYPEDEF)
                collect_typedef(L, c);
            else if (n->cuda_flags & (CUDA_SHARED|CUDA_CONSTANT|CUDA_DEVICE))
                collect_global_var(L, c);
            break;
        case AST_TEMPLATE_DECL:
            collect_template(L, c);
            break;
        default:
            break;
        }
        c = P->nodes[c].next_sibling;
    }

    /* Pass 2: lower non-template device functions */
    c = P->nodes[ast_root].first_child;
    while (c) {
        const ast_node_t *n = &P->nodes[c];
        if (n->type == AST_FUNC_DEF) {
            uint16_t cuda = n->cuda_flags;
            if (cuda & (CUDA_GLOBAL | CUDA_DEVICE | CUDA_HOST)) {
                lower_func_body(L, c, cuda, NULL);
            }
            /* Pure host functions (no CUDA qualifier) skipped for lowering
               but scanned for launches below */
        }
        c = P->nodes[c].next_sibling;
    }

    /* Pass 3: scan host functions for kernel launches → instantiate templates */
    c = P->nodes[ast_root].first_child;
    while (c) {
        const ast_node_t *n = &P->nodes[c];
        if (n->type == AST_FUNC_DEF && !(n->cuda_flags & (CUDA_GLOBAL|CUDA_DEVICE))) {
            /* Host function — scan for launches */
            uint32_t body = find_func_body(L, c);
            if (body) scan_launches(L, body);
        }
        c = P->nodes[c].next_sibling;
    }

    /* Copy errors out for main.c to display */
    if (out_errs && out_nerrs) {
        int n = L->nerrors < BC_MAX_ERRORS ? L->nerrors : BC_MAX_ERRORS;
        memcpy(out_errs, L->errors, (size_t)n * sizeof(bc_error_t));
        *out_nerrs = n;
    }

    if (L->nerrors > 0)
        return BC_ERR_LOWER;

    return BC_OK;
}
