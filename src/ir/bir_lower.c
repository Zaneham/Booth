#include "bir_lower.h"
#include "parser.h"
#include "sema.h"
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
#define MAX_STRUCTS     256
#define MAX_FIELDS      64
#define MAX_ENUMS       1024
#define MAX_TYPEDEFS    256
#define MAX_TDSN        1024
#define MAX_ALIAS       64
#define MAX_TEMPLATES   256
#define MAX_PKELM       32
#define MAX_FWPK        64
#define MAX_FPACKS      4
#define MAX_INSTS       16384
#define MAX_TPARM       8
#define MAX_BIND        64
#define MAX_TDEP        7
#define MAX_NEST        8
#define MAX_FWB         64
#define MAX_PARM        64
#define MAX_FWD         16
#define MAX_CGV         32
#define MAX_SCOPES      64
#define MAX_LOOPS       32
#define MAX_CEXPS       512
#define MAX_CFNS        128
#define MAX_TLV         8192
#define MAX_CDEP        12
#define CF_DEEP         24
#define MAX_CLST        64
#define MAX_CINS        128
#define MAX_CARG        8
#define MAX_CNST        24
#define MAX_VTPL        64
#define MAX_SDM         512
#define MAX_BASE        4
#define SD_QUE          8
#define CI_SEP          '$'
#define CF_PARM         8
#define CF_DEPTH        64
#define CE_NAMEZ        192
#define CE_MAXDIM       (1 << 24)
#define LO_MAXALG       4096
#define LO_MAX_DIM      4
#define CE_NEST         16
#define CE_STMT         256
#define CF_BODY         2048
#define MAX_CAGF        12
#define MAX_CAGG        64
#define TM_DEEP         6
#define AI_NEST         8
#define AI_STORES       4096
#define UC_NEST         256
#define OV_CAND         16
#define MAX_LAMS        1024
#define MAX_TAGS        32
#define MAX_GLAM        16
#define LM_CAP          16
#define LM_PARM         8
#define LM_EXCL         64
#define LM_NEST         512
#define MAX_VTY         48

/* ---- Internal Types ---- */

typedef struct {
    char        name[128];
    uint32_t    ref;        /* param: BIR_MAKE_VAL(idx). local: alloca inst idx */
    uint32_t    type;       /* BIR type index of the value (not pointer) */
    int         is_alloca;
    int         lam;
    int         wfrg;
} sym_t;

typedef struct {
    char        name[64];
    uint32_t    field_types[MAX_FIELDS];
    char        field_names[MAX_FIELDS][64];
    int         num_fields;
    uint32_t    bir_type;
    uint16_t    algn;
    uint16_t    falgn[MAX_FIELDS];
    uint8_t     anon;
    uint8_t     uni;
    uint8_t     fanon[MAX_FIELDS];
    uint32_t    node;
    uint16_t    bad;
    char        badn[64];
    char        base[MAX_BASE][64];
    uint8_t     nbase;
    uint8_t     bopq;
    uint8_t     fwd;
} struct_def_t;

typedef struct {
    char        name[128];
    int64_t     value;
} enum_val_t;

typedef struct {
    uint8_t     kind;
    uint16_t    bits;
    uint8_t     lanes;
    uint32_t    ty;
} vtreg_t;

typedef struct {
    int64_t     ival;
    double      fval;
    int         isf;
    uint32_t    fnd;
    uint16_t    agg;
    uint16_t    fsy;
} cval_t;

typedef struct {
    int64_t     ival;
    double      fval;
    uint32_t    fnd;
    uint16_t    fsy;
    uint8_t     isf;
} cagf_t;

typedef struct {
    int         sdi;
    int         nf;
    cagf_t      f[MAX_CAGF];
} cagg_t;

typedef struct {
    char        name[CE_NAMEZ];
    cval_t      v;
    uint32_t    type;   /* BIR type the declaration gave it */
    uint8_t     bad;    /* initialiser would not fold */
    uint8_t     hag;
    cagg_t      ag;
} cexp_t;

typedef struct {
    char        cls[64];
    char        nm[64];
    cval_t      v;
    uint32_t    type;
    uint8_t     bad;    /* 1 no initialiser, 2 mutable, 3 would not fold */
    uint8_t     arr;
} sdm_t;

typedef struct {
    char        name[64];
    uint32_t    bir_type;
    uint16_t    bad;
    char        badn[64];
    char        snm[64];
} typedef_def_t;

typedef struct {
    uint32_t    ast;        /* AST_TEMPLATE_DECL node index */
    char        name[64];   /* function name inside */
} template_def_t;

typedef struct {
    uint32_t    tdcl;
    uint32_t    use;
    char        name[64];
} alia_t;

typedef struct {
    uint32_t    ast;
    uint32_t    sdef;
    uint32_t    spec;
    char        name[64];
} cltd_t;

typedef struct {
    uint32_t    tdcl;
    uint32_t    ty;
    uint32_t    init;
    int         spec;
    char        name[64];
} vtpl_t;

typedef struct {
    char        name[64];
    uint32_t    type;       /* BIR type index */
    int64_t     ival;       /* for non-type params */
    int         is_type;    /* 1 = typename, 0 = int param */
    int         is_pack;
    int         is_fn;
    char        fnm[64];
    uint32_t    dflt;
    int         fixed;
    int         guess;      /* deduction fell back, the type is not known */
    int         npk;
    int         pk_nt;
    uint32_t    pk_t[MAX_PKELM];
    int64_t     pk_v[MAX_PKELM];
} binding_t;

typedef struct {
    char        name[64];
    int         n;
} fpack_t;

typedef struct {
    uint32_t    ast;
    uint32_t    fidx;
    uint32_t    cst;
    uint32_t    ret;
    char        name[BIR_SYM_MAX];
    int         ncap;
    char        cnm[LM_CAP][64];
    uint32_t    cty[LM_CAP];
    uint8_t     crf[LM_CAP];
    int         np;
    uint32_t    pty[LM_PARM];
    uint32_t    pbs[LM_PARM];
    char        pnm[LM_PARM][64];
    uint32_t    rfm, rfq;
    uint32_t    amsk, tmsk;
    int64_t     tval[LM_PARM];
    int         gsrc;
    int         tdo;
    int         tdn;
    uint8_t     gen;
    uint8_t     done;
} lam_t;

typedef struct {
    char        name[64];
    int64_t     val;
} tgent_t;

typedef struct {
    char        name[64];
    uint32_t    ast;
    uint8_t     busy;
} glent_t;

typedef struct {
    struct { char name[64]; uint32_t arg; } b[MAX_FWB];
    int         nb;
    char        pn[64];
    uint32_t    pk[MAX_FWPK];
    int         npk;
} fwfr_t;

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
    vtreg_t         vty[MAX_VTY];
    int             nvty;
    int             nzst;
    enum_val_t      enums[MAX_ENUMS];
    int             nenums;
    cexp_t          cexps[MAX_CEXPS];
    int             ncexp;
    cagg_t          cags[MAX_CAGG];
    int             ncag;
    sdm_t           sdms[MAX_SDM];
    int             nsdm;
    char            scls[64];
    uint16_t        sdbad;
    uint8_t         sdarr;
    uint32_t        sdty;
    char            sdbnm[144];
    int             cexp_stack[MAX_SCOPES];
    int             tdst[MAX_SCOPES];
    int             atdp;
    struct { char name[64]; uint32_t ast; } cfns[MAX_CFNS];
    int             ncfn;
    uint32_t        tlv[MAX_TLV];
    int             ntlv;
    int             cdep;
    int             sdep;
    int             cbrk;
    int             fwovf;
    typedef_def_t   typedefs[MAX_TYPEDEFS];
    int             ntypedefs;
    typedef_def_t   tdsn[MAX_TDSN];
    int             ntdsn;
    int             ntdf;
    alia_t          alia[MAX_ALIAS];
    int             nalia;
    int             adep;
    binding_t       anb[MAX_TDEP][MAX_TPARM];

    template_def_t  templates[MAX_TEMPLATES];
    int             ntemplates;
    cltd_t          clss[MAX_CLST];
    int             nclss;
    vtpl_t          vtpl[MAX_VTPL];
    int             nvtpl;
    binding_t       cnb[MAX_CNST][MAX_TPARM];
    binding_t       cab[MAX_CNST][MAX_CARG];
    uint8_t         ccv[MAX_CNST][MAX_CLST][MAX_CARG];
    int             cidep;
    struct { char key[256]; char sym[128]; } cins[MAX_CINS];
    int             ncins;
    char            covr[144];
    int             clow;
    binding_t       bindings[MAX_BIND];
    int             nbindings;
    int             nbbase;
    binding_t       snb[MAX_TPARM];
    binding_t       pnb[MAX_NEST][MAX_TPARM];
    int             pdep;
    binding_t       tnb[MAX_TPARM];
    binding_t       xnb[MAX_NEST][MAX_TPARM];
    int             xdep;
    int             tsc;
    struct { char key[256]; char sym[128]; } inst[MAX_INSTS];
    int             ninst;
    int             tdep;

    char            pnms[MAX_PARM][64];

    fwfr_t          fwf[MAX_FWD];
    int             nfwf;
    fpack_t         fpk[MAX_FPACKS];
    int             nfpk;
    struct { char name[64]; int idx; } pkact[MAX_FPACKS];
    int             npkact;

    /* Current function state */
    char            fnnm[128];
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

    uint16_t        tu;

    uint32_t        cur_node;   /* AST node being lowered — for source loc tracking */

    bc_error_t      errors[BC_MAX_ERRORS];
    int             nerrors;

    lam_t           lams[MAX_LAMS];
    char            lex[LM_EXCL][64];
    int             nlam;
    int             ldrn;

    tgent_t         tags[MAX_TAGS];
    int             ntags;
    glent_t         glms[MAX_GLAM];
    int             nglm;

    int             wpend;
    uint16_t        rtbad;
    int             pkbad;
    int             rtunk;
    char            rtbadn[64];
    uint16_t        rtalgn;
    char            rtsnm[64];
    int             rtcg;

    char            cgnv[MAX_CGV][64];
    int             ncgv;

    uint32_t        mrec;

    uint32_t        srdst;
    uint32_t        srdty;
} lower_t;

/* ---- Names ---- */

static void ncpy(char *d, size_t n, const char *s)
{
    size_t k = strlen(s);

    if (k >= n) k = n - 1;
    memcpy(d, s, k);
    d[k] = '\0';
}

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

static uint32_t unfw(const lower_t *L, uint32_t node)
{
    KA_GUARD(g, CE_NEST);
    while (node && g--) {
        const ast_node_t *n = ND(L, node);
        uint32_t cn, l, r, a;

        if (n->type == AST_PAREN) { node = n->first_child; continue; }
        if (n->type != AST_CALL) break;
        cn = n->first_child;
        if (!cn || ND(L, cn)->type != AST_SCOPE_RES) break;
        l = ND(L, cn)->first_child;
        r = l ? ND(L, l)->next_sibling : 0;
        if (!r || ND(L, r)->next_sibling) break;
        if (ND(L, l)->type != AST_IDENT || ND(L, r)->type != AST_IDENT) break;
        if (!text_eq(L, l, "std")) break;
        if (!text_eq(L, r, "forward") && !text_eq(L, r, "move")) break;
        a = ND(L, cn)->next_sibling;
        if (!a || ND(L, a)->next_sibling) break;
        node = a;
    }
    return node;
}

static int fwstd(const lower_t *L, uint32_t cn, char *out, int oz)
{
    uint32_t l, r;

    if (!cn || ND(L, cn)->type != AST_SCOPE_RES) return 0;
    l = ND(L, cn)->first_child;
    r = l ? ND(L, l)->next_sibling : 0;
    if (!r || ND(L, r)->next_sibling) return 0;
    if (ND(L, l)->type != AST_IDENT || ND(L, r)->type != AST_IDENT) return 0;
    if (!text_eq(L, l, "std")) return 0;
    if (text_eq(L, r, "forward")) return snprintf(out, (size_t)oz,
                                                  "std::forward") > 0;
    if (text_eq(L, r, "move")) return snprintf(out, (size_t)oz,
                                               "std::move") > 0;
    return 0;
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


static uint32_t sym_fx(const lower_t *L, const char *name)
{
    for (uint32_t i = 0; i < L->M->num_funcs; i++)
        if (L->M->funcs[i].name < L->M->string_len
            && strcmp(&L->M->strings[L->M->funcs[i].name], name) == 0)
            return i;
    return BIR_SYM_NONE;
}

static uint32_t sym_gx(const lower_t *L, const char *name)
{
    for (uint32_t i = 0; i < L->M->num_globals; i++)
        if (L->M->globals[i].name < L->M->string_len
            && strcmp(&L->M->strings[L->M->globals[i].name], name) == 0)
            return i;
    return BIR_SYM_NONE;
}

static void ccomp(lower_t *L, int bas)
{
    int w = bas;

    for (int i = bas; i < L->ncexp; i++)
        if (strstr(L->cexps[i].name, "::"))
            L->cexps[w++] = L->cexps[i];
    L->ncexp = w;
}

/* ---- Scope ---- */

static void push_scope(lower_t *L)
{
    if (L->scope_depth < MAX_SCOPES) {
        L->cexp_stack[L->scope_depth] = L->ncexp;
        L->tdst[L->scope_depth] = L->ntypedefs;
        L->scope_stack[L->scope_depth++] = L->nsyms;
    }
}

static void pop_scope(lower_t *L)
{
    if (L->scope_depth > 0) {
        L->nsyms = L->scope_stack[--L->scope_depth];
        L->ntypedefs = L->tdst[L->scope_depth];
        ccomp(L, L->cexp_stack[L->scope_depth]);
    }
}

static void add_sym(lower_t *L, const char *name, uint32_t ref,
                    uint32_t type, int is_alloca)
{
    if (L->nsyms >= MAX_SYMS) return;
    sym_t *s = &L->syms[L->nsyms++];
    ncpy(s->name, sizeof(s->name), name);
    s->ref = ref;
    s->type = type;
    s->is_alloca = is_alloca;
    s->lam = 0;
    s->wfrg = L->wpend;
    L->wpend = 0;
}

static sym_t *find_sym(lower_t *L, const char *name)
{
    for (int i = L->nsyms - 1; i >= 0; i--)
        if (strcmp(L->syms[i].name, name) == 0)
            return &L->syms[i];
    return NULL;
}

/* ---- Lookups ---- */

static void qtxt(const lower_t *L, uint32_t nd, char *out, size_t oz)
{
    size_t used = 0;

    out[0] = 0;
    for (uint32_t c = ND(L, nd)->first_child; c; c = ND(L, c)->next_sibling) {
        char part[64];
        if (ND(L, c)->type != AST_IDENT) continue;
        get_text(L, c, part, sizeof(part));
        if (used && used + 3 < oz) {
            out[used++] = ':';
            out[used++] = ':';
            out[used] = 0;
        }
        for (size_t k = 0; part[k] && used + 1 < oz; k++)
            out[used++] = part[k];
        if (used < oz) out[used] = 0;
    }
}

static int find_enum(lower_t *L, const char *name, int64_t *val)
{
    for (int i = 0; i < L->nenums; i++)
        if (strcmp(L->enums[i].name, name) == 0) {
            *val = L->enums[i].value;
            return 1;
        }
    return 0;
}

static int aggr(const lower_t *L, uint32_t t)
{
    if (t >= L->M->num_types) return 0;
    return L->M->types[t].kind == BIR_TYPE_STRUCT
        || L->M->types[t].kind == BIR_TYPE_ARRAY
        || L->M->types[t].kind == BIR_TYPE_VECTOR;
}

static const char *sname(const lower_t *L, uint32_t ty)
{
    for (int i = 0; i < L->nstructs; i++)
        if (L->structs[i].bir_type == ty) return L->structs[i].name;
    return NULL;
}

static int find_typedef(lower_t *L, const char *name, uint32_t *type)
{
    for (int i = L->ntypedefs - 1; i >= 0; i--)
        if (strcmp(L->typedefs[i].name, name) == 0) {
            if (L->typedefs[i].bad) {
                L->rtbad = L->typedefs[i].bad;
                ncpy(L->rtbadn, sizeof(L->rtbadn), L->typedefs[i].badn);
            }
            *type = L->typedefs[i].bir_type;
            return 1;
        }
    return 0;
}

static int find_binding(lower_t *L, const char *name, uint32_t *type)
{
    for (int i = L->nbindings - 1; i >= L->nbbase; i--)
        if (L->bindings[i].is_type && !L->bindings[i].is_pack
            && !L->bindings[i].is_fn
            && strcmp(L->bindings[i].name, name) == 0) {
            *type = L->bindings[i].type;
            return 1;
        }
    return 0;
}

static int find_binding_int(lower_t *L, const char *name, int64_t *val)
{
    for (int i = L->nbindings - 1; i >= L->nbbase; i--)
        if (!L->bindings[i].is_type && !L->bindings[i].is_pack
            && !L->bindings[i].is_fn
            && strcmp(L->bindings[i].name, name) == 0) {
            *val = L->bindings[i].ival;
            return 1;
        }
    return 0;
}

static const char *fbfn(const lower_t *L, const char *name)
{
    for (int i = L->nbindings - 1; i >= L->nbbase; i--)
        if (L->bindings[i].is_fn
            && strcmp(L->bindings[i].name, name) == 0)
            return L->bindings[i].fnm;
    return NULL;
}

static void pk_nm(char *out, size_t n, const char *base, int i)
{
    snprintf(out, n, "%s#%d", base, i);
}

static const fpack_t *find_fpk(const lower_t *L, const char *name)
{
    for (int i = 0; i < L->nfpk; i++)
        if (strcmp(L->fpk[i].name, name) == 0) return &L->fpk[i];
    return NULL;
}

static const binding_t *find_tpk(const lower_t *L, const char *name)
{
    for (int i = L->nbindings - 1; i >= L->nbbase; i--)
        if (L->bindings[i].is_pack
            && strcmp(L->bindings[i].name, name) == 0)
            return &L->bindings[i];
    return NULL;
}

static int pk_cnt(const lower_t *L, const char *name)
{
    const fpack_t *f = find_fpk(L, name);
    if (f) return f->n;
    const binding_t *b = find_tpk(L, name);
    if (b) return b->npk;
    return -1;
}

static void pk_rw(const lower_t *L, char *name, size_t n)
{
    for (int i = L->npkact - 1; i >= 0; i--)
        if (strcmp(L->pkact[i].name, name) == 0) {
            char tmp[128];
            pk_nm(tmp, sizeof tmp, name, L->pkact[i].idx);
            snprintf(name, n, "%s", tmp);
            return;
        }
}

static template_def_t *find_template(lower_t *L, const char *name)
{
    for (int i = 0; i < L->ntemplates; i++)
        if (strcmp(L->templates[i].name, name) == 0)
            return &L->templates[i];
    return NULL;
}

static template_def_t *ftmpl(lower_t *L, const char *name, int na);

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
    case TOK_BANG: sym = "!"; break;
    case TOK_TILDE: sym = "~"; break;
    default: sym = "?"; break;
    }
    snprintf(out, (size_t)outsz, "operator%s", sym);
}

/* ---- Instruction Emission ---- */

static uint32_t emit(lower_t *L, uint16_t op, uint32_t type,
                     uint8_t nops, uint8_t subop)
{
    if (L->M->num_insts >= BIR_MAX_INSTS) {
        bir_pfull(L->M, BIR_P_INSTS);
        return 0;
    }
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
    /* emit answers 0 on a full pool, and inst 0 is real. Don't write it. */
    if (inst >= L->M->num_insts) return;
    if (slot < 0 || slot >= BIR_OPERANDS_INLINE) return;
    L->M->insts[inst].operands[slot] = val;
}

static uint32_t natal(const lower_t *L, uint32_t t)
{
    uint32_t e = t, a = bir_balg(L->M, t, 8), g = BIR_BSZ_DEEP;

    while (g-- && e < L->M->num_types
           && (L->M->types[e].kind == BIR_TYPE_ARRAY
               || L->M->types[e].kind == BIR_TYPE_VECTOR))
        e = L->M->types[e].inner;
    for (int i = 0; i < L->nstructs; i++)
        if (L->structs[i].bir_type == e && L->structs[i].algn > a)
            a = L->structs[i].algn;
    return a ? a : 1u;
}

static uint8_t alg2(uint32_t a)
{
    uint8_t k = 0;

    while (a > 1u && k < 31) { a >>= 1; k++; }
    return k;
}

static uint32_t emalc(lower_t *L, uint16_t op, uint32_t pt, uint32_t want)
{
    uint32_t a = 1u;

    if (pt < L->M->num_types && L->M->types[pt].kind == BIR_TYPE_PTR)
        a = natal(L, L->M->types[pt].inner);
    if (want > a) a = want;
    return emit(L, op, pt, 0, alg2(a));
}

/* ---- Block Management ---- */

static uint32_t new_block(lower_t *L, const char *name)
{
    if (L->M->num_blocks >= BIR_MAX_BLOCKS) {
        bir_pfull(L->M, BIR_P_BLOCKS);
        return 0;
    }
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

static int cival(lower_t *L, uint32_t node, int64_t *out);

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

enum { WM_TYN = 16 };

static const struct {
    const char *nm;
    const char *ab;
    uint8_t     bits;
    uint8_t     cls;
} wmtys[WM_TYN] = {
    { "half",          "f16",  16, 1 },
    { "__half",        "f16",  16, 1 },
    { "_Float16",      "f16",  16, 1 },
    { "__nv_bfloat16", "bf16", 16, 2 },
    { "nv_bfloat16",   "bf16", 16, 2 },
    { "__bfloat16",    "bf16", 16, 2 },
    { "tf32",          "tf32", 32, 1 },
    { "double",        "f64",  64, 1 },
    { "float",         "f32",  32, 1 },
    { "signed char",   "s8",    8, 0 },
    { "char",          "s8",    8, 0 },
    { "unsigned char", "u8",    8, 0 },
    { "int",           "s32",  32, 0 },
    { "u4",            "u4",   32, 0 },
    { "s4",            "s4",   32, 0 },
    { "b1",            "b1",   32, 0 }
};

static int wmfty(const char *nm)
{
    for (int i = 0; i < WM_TYN; i++)
        if (strcmp(wmtys[i].nm, nm) == 0) return i;
    return -1;
}

static const char *wmbas(const ast_node_t *n)
{
    switch (n->d.btype.kind) {
    case TYPE_FLOAT:    return "float";
    case TYPE_DOUBLE:   return "double";
    case TYPE_CHAR:     return n->d.btype.is_unsigned ? "unsigned char"
                                                      : "signed char";
    case TYPE_INT:      return n->d.btype.is_unsigned ? "unsigned int" : "int";
    case TYPE_UNSIGNED: return "unsigned int";
    case TYPE_SIGNED:   return "int";
    default:            return "";
    }
}

static void wmtxt(lower_t *L, uint32_t nd, char *out, size_t oz)
{
    KA_GUARD(g, 8);

    out[0] = 0;
    while (g-- && nd) {
        uint32_t c, last = 0;
        const ast_node_t *n = ND(L, nd);

        if (n->type == AST_TYPE_SPEC && n->d.btype.kind != TYPE_NAME) {
            ncpy(out, oz, wmbas(n));
            return;
        }
        for (c = n->first_child; c; c = ND(L, c)->next_sibling)
            if (ND(L, c)->type == AST_IDENT || ND(L, c)->type == AST_SCOPE_RES)
                last = c;
        if (!last) { get_text(L, nd, out, (int)oz); return; }
        if (ND(L, last)->type == AST_IDENT) {
            get_text(L, last, out, (int)oz);
            return;
        }
        nd = last;
    }
}

static int wmfrag(lower_t *L, uint32_t node)
{
    uint32_t c, last = 0;
    int hasta = 0;
    char t[64];

    for (c = ND(L, node)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_IDENT) last = c;
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) hasta = 1;
    }
    if (!last || !hasta) return 0;
    get_text(L, last, t, sizeof t);
    return strcmp(t, "fragment") == 0;
}

static uint32_t wmbad(lower_t *L, uint16_t e, const char *nm)
{
    L->rtbad = e;
    ncpy(L->rtbadn, sizeof(L->rtbadn), nm);
    return bir_type_int(L->M, 32);
}

static int wmsub(const char *ab)
{
    return strcmp(ab, "u4") == 0 || strcmp(ab, "s4") == 0
        || strcmp(ab, "b1") == 0;
}

static int wmrole(const char *use)
{
    if (strcmp(use, "matrix_a") == 0) return (int)BIR_WM_A;
    if (strcmp(use, "matrix_b") == 0) return (int)BIR_WM_B;
    if (strcmp(use, "accumulator") == 0) return (int)BIR_WM_C;
    return -1;
}

static int wmdim(lower_t *L, uint32_t nd, const char *tx, unsigned long *out)
{
    int64_t v = 0;
    char *e;

    if (nd && cival(L, nd, &v) && v > 0 && v <= 512) {
        *out = (unsigned long)v;
        return 1;
    }
    *out = strtoul(tx, &e, 10);
    return tx[0] != 0 && *e == 0 && *out > 0 && *out <= 512;
}

static int wmshp(lower_t *L, char (*a)[64], const uint32_t *an,
                 char *shp, size_t sz)
{
    unsigned long d[3];

    for (int i = 0; i < 3; i++) {
        if (!wmdim(L, an[i + 1], a[i + 1], &d[i])) {
            (void)wmbad(L, BC_E900, a[i + 1]);
            return 0;
        }
    }
    if (snprintf(shp, sz, "m%lun%luk%lu", d[0], d[1], d[2]) < 0) {
        (void)wmbad(L, BC_E900, a[1]);
        return 0;
    }
    return 1;
}

static uint32_t wmdesc(lower_t *L, uint32_t node)
{
    char a[6][64], shp[40], q[96];
    uint32_t c, ta = 0, et, an[6];
    int na = 0, role, lay = 0, ti, row, nreg, cnt;

    for (c = ND(L, node)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) ta = c;
    if (!ta) return wmbad(L, BC_E902, "a fragment with no template arguments");
    for (c = ND(L, ta)->first_child; c && na < 6; c = ND(L, c)->next_sibling) {
        an[na] = c;
        wmtxt(L, c, a[na++], sizeof a[0]);
    }
    if (na < 5) return wmbad(L, BC_E902, "a fragment short of arguments");

    role = wmrole(a[0]);
    if (role < 0) return wmbad(L, BC_E902, a[0]);
    if (!wmshp(L, a, an, shp, sizeof shp)) return bir_type_int(L->M, 32);
    ti = wmfty(a[4]);
    if (ti < 0) return wmbad(L, BC_E901, a[4]);

    if (role == (int)BIR_WM_C) {
        if (na > 5 && a[5][0] && strcmp(a[5], "void") != 0)
            return wmbad(L, BC_E912, a[5]);
        row = bir_wmrow(shp, NULL, wmtys[ti].ab);
    } else {
        if (na < 6) return wmbad(L, BC_E903, "a fragment with no layout");
        if (strcmp(a[5], "row_major") == 0) lay = 0;
        else if (strcmp(a[5], "col_major") == 0) lay = 1;
        else return wmbad(L, BC_E903, a[5]);
        row = bir_wmrow(shp, wmtys[ti].ab, NULL);
    }
    if (row < 0) {
        if (snprintf(q, sizeof q, "%s and %s", shp, wmtys[ti].ab) < 0) q[0] = 0;
        return wmbad(L, BC_E904, q);
    }
    if (wmsub(wmtys[ti].ab)
     && ((role == (int)BIR_WM_A && lay != 0)
      || (role == (int)BIR_WM_B && lay != 1)))
        return wmbad(L, BC_E909, wmtys[ti].ab);

    nreg = (int)bir_wmn((uint32_t)row, (unsigned)role);
    cnt = nreg * ((strcmp(bir_wmma[row].abt, "f64") == 0) ? 8 : 4)
        / (wmtys[ti].bits / 8);
    et = (wmtys[ti].cls == 2) ? bir_type_bfloat(L->M)
       : (wmtys[ti].cls == 1) ? bir_type_float(L->M, wmtys[ti].bits)
       : bir_type_int(L->M, wmtys[ti].bits);
    L->wpend = 1 + (row | (role << 5) | (lay << 7));
    return bir_type_array(L->M, et, (uint32_t)cnt);
}

enum { VK_INT, VK_FLT, VK_BF };

typedef struct {
    const char *name;
    uint8_t     kind;
    uint8_t     bits;
    uint8_t     lanes;
} vecty_t;

static const vecty_t vtbl[] = {
    {"char1",VK_INT,8,1},{"char2",VK_INT,8,2},{"char3",VK_INT,8,3},{"char4",VK_INT,8,4},
    {"uchar1",VK_INT,8,1},{"uchar2",VK_INT,8,2},{"uchar3",VK_INT,8,3},{"uchar4",VK_INT,8,4},
    {"short1",VK_INT,16,1},{"short2",VK_INT,16,2},{"short3",VK_INT,16,3},{"short4",VK_INT,16,4},
    {"ushort1",VK_INT,16,1},{"ushort2",VK_INT,16,2},{"ushort3",VK_INT,16,3},{"ushort4",VK_INT,16,4},
    {"int1",VK_INT,32,1},{"int2",VK_INT,32,2},{"int3",VK_INT,32,3},{"int4",VK_INT,32,4},
    {"uint1",VK_INT,32,1},{"uint2",VK_INT,32,2},{"uint3",VK_INT,32,3},{"uint4",VK_INT,32,4},
    {"long1",VK_INT,64,1},{"long2",VK_INT,64,2},{"long3",VK_INT,64,3},{"long4",VK_INT,64,4},
    {"ulong1",VK_INT,64,1},{"ulong2",VK_INT,64,2},{"ulong3",VK_INT,64,3},{"ulong4",VK_INT,64,4},
    {"longlong1",VK_INT,64,1},{"longlong2",VK_INT,64,2},
    {"longlong3",VK_INT,64,3},{"longlong4",VK_INT,64,4},
    {"ulonglong1",VK_INT,64,1},{"ulonglong2",VK_INT,64,2},
    {"ulonglong3",VK_INT,64,3},{"ulonglong4",VK_INT,64,4},
    {"float1",VK_FLT,32,1},{"float2",VK_FLT,32,2},{"float3",VK_FLT,32,3},{"float4",VK_FLT,32,4},
    {"double1",VK_FLT,64,1},{"double2",VK_FLT,64,2},
    {"double3",VK_FLT,64,3},{"double4",VK_FLT,64,4},
    {"half2",VK_FLT,16,2},{"__half2",VK_FLT,16,2},
    {"nv_bfloat162",VK_BF,16,2},{"__nv_bfloat162",VK_BF,16,2},
    {"__hip_bfloat162",VK_BF,16,2},
    {"dim3",VK_INT,32,3},
};

static const vecty_t *vfind(const char *name)
{
    for (size_t i = 0; i < sizeof vtbl / sizeof vtbl[0]; i++)
        if (strcmp(name, vtbl[i].name) == 0) return &vtbl[i];
    return NULL;
}

static uint32_t valgn(const vecty_t *v)
{
    uint32_t e = (uint32_t)v->bits / 8u;
    uint32_t a = (v->lanes == 3) ? e : e * (uint32_t)v->lanes;
    return a > 16u ? 16u : a;
}

static uint32_t vnty(lower_t *L, int vk, int bits, int lanes)
{
    uint32_t et, f[4], ty;

    if (lanes < 1 || lanes > 4) return 0;
    for (int i = 0; i < L->nvty; i++)
        if (L->vty[i].kind == (uint8_t)vk && L->vty[i].bits == (uint16_t)bits
            && L->vty[i].lanes == (uint8_t)lanes) return L->vty[i].ty;
    if (L->nvty >= MAX_VTY) return 0;

    switch (vk) {
    case VK_FLT: et = bir_type_float(L->M, bits); break;
    case VK_BF:  et = bir_type_bfloat(L->M);      break;
    default:     et = bir_type_int(L->M, bits);   break;
    }
    if (!et) return 0;
    for (int i = 0; i < 4; i++) f[i] = et;

    ty = bir_sfwd(L->M);
    if (!ty || !bir_sfin(L->M, ty, f, lanes)) return 0;
    L->vty[L->nvty].kind  = (uint8_t)vk;
    L->vty[L->nvty].bits  = (uint16_t)bits;
    L->vty[L->nvty].lanes = (uint8_t)lanes;
    L->vty[L->nvty].ty    = ty;
    L->nvty++;
    return ty;
}

static struct_def_t *vreg(lower_t *L, const vecty_t *v)
{
    static const char *const xyzw[] = {"x", "y", "z", "w"};
    uint32_t et;
    struct_def_t *sd;

    for (int si = 0; si < L->nstructs; si++)
        if (strcmp(L->structs[si].name, v->name) == 0)
            return &L->structs[si];

    if (L->nstructs >= MAX_STRUCTS) return NULL;

    switch (v->kind) {
    case VK_FLT: et = bir_type_float(L->M, v->bits); break;
    case VK_BF:  et = bir_type_bfloat(L->M);         break;
    default:     et = bir_type_int(L->M, v->bits);   break;
    }
    sd = &L->structs[L->nstructs];
    memset(sd, 0, sizeof(*sd));
    ncpy(sd->name, sizeof(sd->name), v->name);
    sd->num_fields = v->lanes;
    sd->algn = (uint8_t)valgn(v);
    sd->bir_type = vnty(L, v->kind, v->bits, v->lanes);
    if (!sd->bir_type) return NULL;
    for (int fi = 0; fi < v->lanes; fi++) {
        sd->field_types[fi] = et;
        ncpy(sd->field_names[fi], sizeof(sd->field_names[0]), xyzw[fi]);
    }
    L->nstructs++;
    return sd;
}

static const struct { const char *name; uint16_t bits; } itbl[] = {
    {"int8_t",8},{"uint8_t",8},{"int_least8_t",8},{"uint_least8_t",8},
    {"int16_t",16},{"uint16_t",16},{"int_least16_t",16},{"uint_least16_t",16},
    {"int32_t",32},{"uint32_t",32},{"int_least32_t",32},{"uint_least32_t",32},
    {"int64_t",64},{"uint64_t",64},{"int_least64_t",64},{"uint_least64_t",64},
    {"int_fast8_t",8},{"uint_fast8_t",8},
    {"int_fast16_t",32},{"uint_fast16_t",32},
    {"int_fast32_t",32},{"uint_fast32_t",32},
    {"int_fast64_t",64},{"uint_fast64_t",64},
    {"intmax_t",64},{"uintmax_t",64},
    {"ptrdiff_t",64},{"ssize_t",64},{"off_t",64},
};

static int ifind(const char *name)
{
    for (size_t i = 0; i < sizeof itbl / sizeof itbl[0]; i++)
        if (strcmp(name, itbl[i].name) == 0) return (int)itbl[i].bits;
    return 0;
}

static int cget(lower_t *L, uint32_t node, const char *nm, uint32_t ta,
                char *out, size_t oz);

static int cslot(const lower_t *L, const char *nm)
{
    for (int i = L->nstructs - 1; i >= 0; i--)
        if (strcmp(L->structs[i].name, nm) == 0) return i;
    return -1;
}

static int sresv(lower_t *L, const char *nm)
{
    struct_def_t *sd;
    uint32_t bt;

    for (int i = 0; i < L->nstructs; i++)
        if (strcmp(L->structs[i].name, nm) == 0) return i;
    if (L->nstructs >= MAX_STRUCTS) return -1;
    bt = bir_sfwd(L->M);
    if (!bt) return -1;
    sd = &L->structs[L->nstructs];
    memset(sd, 0, sizeof(*sd));
    ncpy(sd->name, sizeof(sd->name), nm);
    sd->bir_type = bt;
    sd->fwd = 1;
    return L->nstructs++;
}

static int sfslot(const lower_t *L, const char *nm)
{
    for (int i = 0; i < L->nstructs; i++)
        if (L->structs[i].fwd && strcmp(L->structs[i].name, nm) == 0)
            return i;
    return -1;
}

static uint32_t dcty(lower_t *L, uint32_t node);
static void cndnm(const lower_t *L, uint32_t node, char *buf, int sz);

static int tpdep(const lower_t *L, uint32_t n)
{
    uint16_t q = ND(L, n)->qualifiers;
    return ((q & QUAL_PTR1) ? 1 : 0) + ((q & QUAL_PTR2) ? 2 : 0);
}

static int altry(lower_t *L, const char *nm, uint32_t ta, uint32_t *out);

static int cgknd(const char *n)
{
    if (strcmp(n, "grid_group") == 0)      return 2;
    if (strcmp(n, "thread_block") == 0)    return 1;
    if (strcmp(n, "thread_group") == 0)    return 1;
    if (strcmp(n, "coalesced_group") == 0) return 1;
    return 0;
}

static int cgadd(lower_t *L, const char *nm)
{
    for (int i = 0; i < L->ncgv; i++)
        if (strcmp(L->cgnv[i], nm) == 0) return 1;
    if (L->ncgv >= MAX_CGV) return 0;
    ncpy(L->cgnv[L->ncgv], sizeof(L->cgnv[0]), nm);
    L->ncgv++;
    return 1;
}

static int cgisg(const lower_t *L, const char *nm)
{
    for (int i = 0; i < L->ncgv; i++)
        if (strcmp(L->cgnv[i], nm) == 0) return 1;
    return 0;
}

static uint32_t resolve_type(lower_t *L, uint32_t node, int ptr_depth,
                             uint16_t cuda)
{
    uint32_t base;
    L->rtbad = 0;
    L->rtbadn[0] = '\0';
    L->rtalgn = 0;
    L->rtunk = 0;
    L->rtcg = 0;
    L->wpend = 0;
    L->rtsnm[0] = '\0';
    if (!node) return bir_type_int(L->M, 32);
    const ast_node_t *n = ND(L, node);

    if (n->type != AST_TYPE_SPEC)
        return bir_type_int(L->M, 32);

    int kind = n->d.btype.kind;

    if (kind == TYPE_DECLTYPE) {
        base = dcty(L, n->first_child);
        if (!base) {
            L->rtbad = BC_E360;
            cndnm(L, n->first_child, L->rtbadn, (int)sizeof(L->rtbadn));
            base = bir_type_int(L->M, 32);
            L->rtunk = 1;
        }
    } else if (kind == TYPE_NAME) {
        char name[128] = {0};
        uint32_t nc = n->first_child;
        if (nc) get_text(L, nc, name, sizeof(name));

        if (wmfrag(L, node))
            return wmdesc(L, node);

        if (nc && strcmp(name, "std") == 0) {
            char qn[128] = {0};
            uint32_t qc = ND(L, nc)->next_sibling;
            while (qc && ND(L, qc)->type != AST_IDENT)
                qc = ND(L, qc)->next_sibling;
            if (qc) get_text(L, qc, qn, sizeof(qn));
            L->rtbad = BC_E134;
            ncpy(L->rtbadn, sizeof(L->rtbadn), qn);
            return bir_type_int(L->M, 32);
        }
        for (uint32_t tc = nc; tc; tc = ND(L, tc)->next_sibling) {
            if (ND(L, tc)->type == AST_IDENT && tc != nc)
                get_text(L, tc, name, sizeof(name));
            if (ND(L, tc)->type == AST_TEMPLATE_ARGS) {
                char mg[128];
                int si = cget(L, node, name, tc, mg, sizeof(mg))
                       ? cslot(L, mg) : -1;
                L->rtbad = 0;
                L->rtunk = 0;
                L->rtalgn = 0;
                if (si >= 0) {
                    L->rtalgn = L->structs[si].algn;
                    ncpy(L->rtsnm, sizeof(L->rtsnm), mg);
                    if (L->structs[si].bad) {
                        L->rtbad = L->structs[si].bad;
                        ncpy(L->rtbadn, sizeof(L->rtbadn),
                             L->structs[si].badn);
                    }
                    return L->structs[si].bir_type;
                }
                if (altry(L, name, tc, &base)) return base;
                L->rtunk = 0;
                L->rtalgn = 0;
                L->rtbad = BC_E133;
                ncpy(L->rtbadn, sizeof(L->rtbadn), name);
                return bir_type_int(L->M, 32);
            }
        }
        if (nc) get_text(L, nc, name, sizeof(name));
        int cgroot = nc && strcmp(name, "cooperative_groups") == 0
                        && ND(L, nc)->next_sibling;
        for (uint32_t tc = nc; tc; tc = ND(L, tc)->next_sibling)
            if (ND(L, tc)->type == AST_IDENT && tc != nc)
                get_text(L, tc, name, sizeof(name));
        L->rtcg = cgknd(name);
        if (L->rtcg || cgroot) return bir_type_int(L->M, 32);

        uint32_t bound;
        if (find_binding(L, name, &bound))
            base = bound;
        else if (find_typedef(L, name, &bound))
            base = bound;
        else if (strcmp(name, "size_t") == 0 || strcmp(name, "uintptr_t") == 0
                 || strcmp(name, "intptr_t") == 0)
            base = bir_type_int(L->M, 64);
        else if (ifind(name))
            base = bir_type_int(L->M, ifind(name));
        else if (strcmp(name, "__half") == 0 || strcmp(name, "half") == 0
                 || strcmp(name, "_Float16") == 0)
            base = bir_type_float(L->M, 16);
        else if (strcmp(name, "__nv_bfloat16") == 0
                 || strcmp(name, "nv_bfloat16") == 0
                 || strcmp(name, "__bfloat16") == 0
                 || strcmp(name, "__hip_bfloat16") == 0)
            base = bir_type_bfloat(L->M);
        else {
            const vecty_t *v = vfind(name);
            int vfound = 0;
            if (v) {
                struct_def_t *sd = vreg(L, v);
                if (!sd) {
                    lower_error(L, node, BC_E138, MAX_STRUCTS);
                    return bir_type_int(L->M, 32);
                }
                base = sd->bir_type;
                L->rtalgn = sd->algn;
                vfound = 1;
            }
            if (!vfound) {
                /* Check user-defined structs (C++ style: Vec2 v; without 'struct') */
                for (int si = 0; si < L->nstructs; si++) {
                    if (strcmp(L->structs[si].name, name) == 0) {
                        if (L->structs[si].bad) {
                            L->rtbad = L->structs[si].bad;
                            ncpy(L->rtbadn, sizeof(L->rtbadn),
                                 L->structs[si].badn);
                        }
                        base = L->structs[si].bir_type;
                        L->rtalgn = L->structs[si].algn;
                        vfound = 1;
                        break;
                    }
                }
                if (!vfound) {
                    L->rtbad = BC_E155;
                    ncpy(L->rtbadn, sizeof(L->rtbadn), name);
                    base = bir_type_int(L->M, 32);
                    base = bir_type_int(L->M, 32);
                    L->rtunk = 1;
                }
            }
        }
    } else if (kind == TYPE_STRUCT || kind == TYPE_UNION) {
        char name[128] = {0};
        uint32_t nc = n->first_child;
        int hit = -1;
        if (nc) get_text(L, nc, name, sizeof(name));
        base = bir_type_int(L->M, 32); /* fallback */
        for (int i = 0; i < L->nstructs; i++) {
            if (strcmp(L->structs[i].name, name) == 0) { hit = i; break; }
        }
        if (hit < 0 && name[0] && nc && ND(L, nc)->type == AST_IDENT
            && strlen(name) < sizeof(L->structs[0].name))
            hit = sresv(L, name);
        if (hit >= 0) {
            if (L->structs[hit].bad) {
                L->rtbad = L->structs[hit].bad;
                ncpy(L->rtbadn, sizeof(L->rtbadn), L->structs[hit].badn);
            }
            base = L->structs[hit].bir_type;
            L->rtalgn = L->structs[hit].algn;
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

static uint32_t rtype(lower_t *L, uint32_t node, int ptr_depth, uint16_t cuda)
{
    uint32_t t = resolve_type(L, node, ptr_depth, cuda);
    if (L->rtbad) {
        lower_error(L, node, (bc_eid_t)L->rtbad, L->rtbadn);
        L->rtbad = 0;
    }
    return t;
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

static uint32_t vlane(lower_t *L, uint32_t val, int lane)
{
    uint32_t vt = ref_type(L, val);
    uint32_t st = vt, base = val;
    uint8_t as = BIR_AS_PRIVATE;
    uint32_t et, fpt, ci, gep, ld;

    if (vt >= L->M->num_types) return BIR_VAL_NONE;
    if (L->M->types[vt].kind == BIR_TYPE_PTR) {
        st = L->M->types[vt].inner;
        as = L->M->types[vt].addrspace;
    }
    if (st >= L->M->num_types) return BIR_VAL_NONE;
    if (L->M->types[st].kind != BIR_TYPE_STRUCT) return BIR_VAL_NONE;
    if (lane < 0 || lane >= (int)L->M->types[st].num_fields) return BIR_VAL_NONE;
    if (L->M->types[st].count + (uint32_t)lane >= L->M->num_type_fields)
        return BIR_VAL_NONE;
    et = L->M->type_fields[L->M->types[st].count + (uint32_t)lane];

    if (L->M->types[vt].kind != BIR_TYPE_PTR) {
        uint32_t pt = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
        uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
        uint32_t sv = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, sv, 0, val);
        set_op(L, sv, 1, BIR_MAKE_VAL(al));
        base = BIR_MAKE_VAL(al);
    }

    fpt = bir_type_ptr(L->M, et, as);
    ci  = BIR_MAKE_CONST(bir_const_int(L->M, bir_type_int(L->M, 32), lane));
    gep = emit(L, BIR_GEP, fpt, 2, 0);
    set_op(L, gep, 0, base);
    set_op(L, gep, 1, ci);
    ld = emit(L, BIR_LOAD, et, 1, 0);
    set_op(L, ld, 0, BIR_MAKE_VAL(gep));
    return BIR_MAKE_VAL(ld);
}

static uint32_t vmake(lower_t *L, const struct_def_t *sd,
                      const uint32_t *vals, int n)
{
    uint32_t st = sd->bir_type;
    uint32_t pt = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
    uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
    uint32_t ld;

    for (int i = 0; i < n && i < sd->num_fields; i++) {
        uint32_t fpt = bir_type_ptr(L->M, sd->field_types[i], BIR_AS_PRIVATE);
        uint32_t ci  = BIR_MAKE_CONST(bir_const_int(L->M,
                           bir_type_int(L->M, 32), i));
        uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
        uint32_t sv;
        set_op(L, gep, 0, BIR_MAKE_VAL(al));
        set_op(L, gep, 1, ci);
        sv = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, sv, 0, vals[i]);
        set_op(L, sv, 1, BIR_MAKE_VAL(gep));
    }
    ld = emit(L, BIR_LOAD, st, 1, 0);
    set_op(L, ld, 0, BIR_MAKE_VAL(al));
    return BIR_MAKE_VAL(ld);
}

enum { PK_F2, PK_FS, PK_FB, PK_HS, PK_HB, PK_L2, PK_LP };

static int hplan(lower_t *L, int md, int l0, int l1,
                 uint32_t a0, uint32_t a1, uint32_t *lv)
{
    switch (md) {
    case PK_F2:
        lv[0] = vlane(L, a0, 0);
        lv[1] = vlane(L, a0, 1);
        break;
    case PK_FS: case PK_HS:
        lv[0] = a0;
        lv[1] = a1;
        break;
    case PK_FB: case PK_HB:
        lv[0] = a0;
        lv[1] = a0;
        break;
    case PK_L2:
        lv[0] = vlane(L, a0, l0);
        lv[1] = vlane(L, a1, l1);
        break;
    default:
        lv[0] = vlane(L, a0, l0);
        lv[1] = vlane(L, a0, l1);
        break;
    }
    return lv[0] != BIR_VAL_NONE && lv[1] != BIR_VAL_NONE;
}

static const vecty_t *hpvt(const lower_t *L, uint32_t et)
{
    if (et >= L->M->num_types) return NULL;
    if (L->M->types[et].kind == BIR_TYPE_BFLOAT) return vfind("nv_bfloat162");
    if (L->M->types[et].kind == BIR_TYPE_FLOAT && L->M->types[et].width == 16)
        return vfind("half2");
    return NULL;
}

static int is_float_type(const lower_t *L, uint32_t t)
{
    if (t >= L->M->num_types) return 0;
    uint8_t k = L->M->types[t].kind;
    return k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT;
}

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

static int is_i1(const lower_t *L, uint32_t t)
{
    return t < L->M->num_types
        && L->M->types[t].kind == BIR_TYPE_INT
        && L->M->types[t].width == 1;
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

static int scalr(const lower_t *L, uint32_t t)
{
    uint8_t k;

    if (!t || t >= L->M->num_types) return 0;
    k = L->M->types[t].kind;
    return k == BIR_TYPE_INT || k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT;
}

static uint32_t stfit(lower_t *L, uint32_t val, uint32_t dt, uint32_t src_n)
{
    uint32_t vt;

    if (val == BIR_VAL_NONE) return val;
    vt = ref_type(L, val);
    if (vt == dt || !scalr(L, vt) || !scalr(L, dt)) return val;
    return coerce_to(L, val, dt, is_i1(L, vt) || node_is_unsigned(L, src_n));
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

static uint32_t skbty(lower_t *L, int kind)
{
    switch (kind) {
    case STYPE_VOID:    return bir_type_void(L->M);
    case STYPE_BOOL:    return bir_type_int(L->M, 1);
    case STYPE_CHAR:
    case STYPE_SCHAR:
    case STYPE_UCHAR:   return bir_type_int(L->M, 8);
    case STYPE_SHORT:
    case STYPE_USHORT:  return bir_type_int(L->M, 16);
    case STYPE_INT:
    case STYPE_UINT:
    case STYPE_ENUM:    return bir_type_int(L->M, 32);
    case STYPE_LONG:
    case STYPE_ULONG:
    case STYPE_LLONG:
    case STYPE_ULLONG:  return bir_type_int(L->M, 64);
    case STYPE_FLOAT:   return bir_type_float(L->M, 32);
    case STYPE_DOUBLE:
    case STYPE_LDOUBLE: return bir_type_float(L->M, 64);
    case STYPE_HALF:    return bir_type_float(L->M, 16);
    case STYPE_BF16:    return bir_type_bfloat(L->M);
    default:            return 0;
    }
}

static int skvk(int kind, int *bits)
{
    switch (kind) {
    case STYPE_CHAR:
    case STYPE_SCHAR:
    case STYPE_UCHAR:   *bits = 8;  return VK_INT;
    case STYPE_SHORT:
    case STYPE_USHORT:  *bits = 16; return VK_INT;
    case STYPE_INT:
    case STYPE_UINT:    *bits = 32; return VK_INT;
    case STYPE_LONG:
    case STYPE_ULONG:
    case STYPE_LLONG:
    case STYPE_ULLONG:  *bits = 64; return VK_INT;
    case STYPE_HALF:    *bits = 16; return VK_FLT;
    case STYPE_FLOAT:   *bits = 32; return VK_FLT;
    case STYPE_DOUBLE:
    case STYPE_LDOUBLE: *bits = 64; return VK_FLT;
    case STYPE_BF16:    *bits = 16; return VK_BF;
    default:            return -1;
    }
}

static uint32_t vecbt(lower_t *L, int kind, int lanes)
{
    int bits = 0, vk = skvk(kind, &bits), n = 0;

    if (vk < 0 || lanes < 1 || lanes > 4) return 0;
    for (size_t i = 0; i < sizeof vtbl / sizeof vtbl[0]; i++)
        if (vtbl[i].kind == vk && vtbl[i].bits == bits
            && vtbl[i].lanes == lanes) n++;
    if (n != 1) return 0;

    return vnty(L, vk, bits, lanes);
}

static uint32_t sstb(const lower_t *L, uint32_t sx)
{
    const char *nm;
    uint32_t hit = 0;
    int seen = 0;

    if (!L->sema || sx >= L->sema->num_structs) return 0;
    nm = L->sema->structs[sx].name;
    if (!nm[0]) return 0;
    for (int i = 0; i < L->nstructs; i++) {
        const char *sn = L->structs[i].name;
        const char *sep = strchr(sn, CI_SEP);
        size_t n = sep ? (size_t)(sep - sn) : strlen(sn);
        if (strlen(nm) != n || strncmp(sn, nm, n) != 0) continue;
        if (seen && L->structs[i].bir_type != hit) return 0;
        hit = L->structs[i].bir_type;
        seen = 1;
    }
    return hit;
}

static uint32_t sbty(lower_t *L, uint32_t ast_node)
{
    if (!L->sema || !ast_node || ast_node >= BC_MAX_NODES) return 0;
    uint32_t st = L->sema->node_types[ast_node];
    int dep = 0;
    while (dep < 4 && st && st < L->sema->num_types
           && L->sema->types[st].kind == STYPE_PTR) {
        st = L->sema->types[st].inner;
        dep++;
    }
    if (!st || st >= L->sema->num_types) return 0;

    uint32_t b;
    if (L->sema->types[st].kind == STYPE_STRUCT) {
        b = sstb(L, L->sema->types[st].extra);
        if (!b) return 0;
    } else if (L->sema->types[st].kind == STYPE_VECTOR) {
        uint32_t in = L->sema->types[st].inner;

        if (in >= L->sema->num_types) return 0;
        b = vecbt(L, L->sema->types[in].kind, L->sema->types[st].width);
        if (!b) return 0;
    } else {
        b = skbty(L, L->sema->types[st].kind);
        if (!b && L->sema->types[st].kind != STYPE_VOID) return 0;
    }
    while (dep-- > 0) b = bir_type_ptr(L->M, b, BIR_AS_GLOBAL);
    return b;
}

static int tdeduc(lower_t *L, uint32_t tnode, uint32_t cal,
                  const uint32_t *av, int na,
                  binding_t *nb, int *ntpo, uint32_t *funco);
static int alist(const lower_t *L, uint32_t first, uint32_t *av, int max);
static int tsym(lower_t *L, char *cname, size_t cz, uint32_t cal,
                uint32_t arg);
static void tscan(lower_t *L, uint32_t node);
static void scan_launches(lower_t *L, uint32_t node);
static void tkeyb(const binding_t *nb, int ntp, const char *cname,
                  char *key, size_t keyz);
static int tfind(const lower_t *L, const char *key, char *sym, size_t symz);
static template_def_t *tmpick(lower_t *L, const char *name,
                              const uint32_t *av, int na);
static void tkfn(lower_t *L, const char *cname, uint32_t func_n,
                 char *key, size_t keyz);
static int qtsym(lower_t *L, char *cnm, size_t cz, uint32_t cal);
static int tinst(lower_t *L, uint32_t node, const char *cname,
                 const binding_t *nb, int ntp, uint32_t func_n);
static int cfold(lower_t *L, uint32_t node, cval_t *out);
static uint32_t pdfn(const lower_t *L, uint32_t p);
static int cfrun(lower_t *L, uint32_t st, cval_t *out, int *don);
static uint32_t dcfld(lower_t *L, uint32_t st, const char *fn);
static int sdfnd(lower_t *L, const char *cls, const char *nm, sdm_t **out);
static int tparm(lower_t *L, uint32_t tnode, binding_t *nb);
static uint32_t tebad(lower_t *L, uint32_t cal);
static int hasta(const lower_t *L, uint32_t cal);
static void tsval(lower_t *L, uint32_t id);
static int tfidx(const lower_t *L, const char *key);
static int tswa(lower_t *L, uint32_t node, uint32_t *lab);
static int ctres(lower_t *L, uint32_t id, const char *nm, binding_t *nb,
                 int *ntpo, uint32_t *fnc, int *ixo);
static int sfind(const lower_t *L, uint32_t st);
static int cagil(lower_t *L, uint32_t il, int si, cval_t *out);

static int abnd(lower_t *L, uint32_t ta, binding_t *nb, int ntp)
{
    int i = 0;

    for (uint32_t a = ND(L, ta)->first_child; a; a = ND(L, a)->next_sibling) {
        int64_t v;
        if (i >= ntp || nb[i].is_pack) return 0;
        if (ND(L, a)->type == AST_TYPE_SPEC) {
            if (!nb[i].is_type) return 0;
            if (ND(L, a)->qualifiers & (QUAL_DECOR | QUAL_REF | QUAL_RREF))
                return 0;
            nb[i].type = resolve_type(L, a, tpdep(L, a), 0);
            if (L->rtbad || L->rtunk || !nb[i].type) return 0;
        } else {
            if (nb[i].is_type || !cival(L, a, &v)) return 0;
            nb[i].ival = v;
        }
        i++;
    }
    for (; i < ntp; i++) {
        int64_t v;
        if (!nb[i].dflt) return 0;
        if (nb[i].is_type) {
            if (ND(L, nb[i].dflt)->type != AST_TYPE_SPEC) return 0;
            nb[i].type = resolve_type(L, nb[i].dflt, tpdep(L, nb[i].dflt), 0);
            if (L->rtbad || L->rtunk || !nb[i].type) return 0;
        } else {
            if (!cival(L, nb[i].dflt, &v)) return 0;
            nb[i].ival = v;
        }
    }
    return 1;
}

static int altry(lower_t *L, const char *nm, uint32_t ta, uint32_t *out)
{
    const alia_t *al = NULL;
    binding_t *nb;
    int ntp, ok, onb = L->nbindings, obs = L->nbbase;

    for (int i = 0; i < L->nalia; i++)
        if (strcmp(L->alia[i].name, nm) == 0) { al = &L->alia[i]; break; }
    if (!al || !ta || L->adep >= MAX_TDEP) return 0;

    nb = L->anb[L->adep];
    L->adep++;
    ntp = tparm(L, al->tdcl, nb);
    ok = ntp > 0 && ntp <= MAX_BIND - onb && abnd(L, ta, nb, ntp);
    if (ok) {
        uint32_t ty = ND(L, al->use)->first_child;
        ty = ty ? ND(L, ty)->next_sibling : 0;
        for (int i = 0; i < ntp; i++) L->bindings[L->nbindings++] = nb[i];
        L->nbbase = onb;
        if (ty && ND(L, ty)->type == AST_TYPE_SPEC)
            *out = resolve_type(L, ty, ND(L, al->use)->d.oper.flags,
                                ND(L, al->use)->cuda_flags);
        else
            ok = 0;
        if (ok && (L->rtbad || L->rtunk)) ok = 0;
        L->nbindings = onb;
        L->nbbase = obs;
    }
    L->adep--;
    return ok;
}

static int areg(lower_t *L, uint32_t tdcl)
{
    uint32_t use = 0, nm;
    char n[64];

    for (uint32_t c = ND(L, tdcl)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_USING) { use = c; break; }
    if (!use) return 1;
    nm = ND(L, use)->first_child;
    if (!nm || ND(L, nm)->type != AST_IDENT) return 1;
    if (!ND(L, nm)->next_sibling) return 1;
    get_text(L, nm, n, sizeof(n));
    for (int i = 0; i < L->nalia; i++)
        if (strcmp(L->alia[i].name, n) == 0) return 1;
    if (L->nalia >= MAX_ALIAS) return 0;
    L->alia[L->nalia].tdcl = tdcl;
    L->alia[L->nalia].use = use;
    ncpy(L->alia[L->nalia].name, sizeof(L->alia[0].name), n);
    L->nalia++;
    return 1;
}

static int pk_len(lower_t *L, uint32_t pat, char nms[][64], int *nn,
                  int quiet);
static uint32_t find_func_body(const lower_t *L, uint32_t func_def);
static uint32_t arg_type(lower_t *L, uint32_t arg);
static int ccoll(lower_t *L, uint32_t node, uint32_t name_n,
                 uint32_t init_n, const char *qual, int strict);
static int cqfnd(lower_t *L, uint32_t node, char *qn, size_t qz, cexp_t **ce);
static int collect_params(const lower_t *L, uint32_t func_def,
                          uint32_t *out, int max);

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


static int tgadd(lower_t *L, const char *nm, int64_t v)
{
    if (L->ntags >= MAX_TAGS) return 0;
    ncpy(L->tags[L->ntags].name, sizeof(L->tags[0].name), nm);
    L->tags[L->ntags].val = v;
    L->ntags++;
    return 1;
}

static int tgfnd(const lower_t *L, const char *nm, int64_t *v)
{
    for (int i = L->ntags - 1; i >= 0; i--)
        if (strcmp(L->tags[i].name, nm) == 0) {
            *v = L->tags[i].val;
            return 1;
        }
    return 0;
}

static int cfind(lower_t *L, const char *name, cexp_t **out)
{
    for (int i = L->ncexp - 1; i >= 0; i--)
        if (strcmp(L->cexps[i].name, name) == 0) {
            *out = &L->cexps[i];
            return 1;
        }
    return 0;
}

static uint16_t caga(lower_t *L)
{
    if (L->ncag >= MAX_CAGG) return 0;
    memset(&L->cags[L->ncag], 0, sizeof(L->cags[0]));
    L->cags[L->ncag].sdi = -1;
    L->ncag++;
    return (uint16_t)L->ncag;
}

static cagg_t *cagp(lower_t *L, const cval_t *v)
{
    return v->agg && (int)v->agg <= L->ncag ? &L->cags[v->agg - 1] : NULL;
}

static void cagz(lower_t *L, int n0, cval_t *v)
{
    if (!v->agg) {
        L->ncag = n0;
        return;
    }
    if ((int)v->agg - 1 > n0) {
        L->cags[n0] = L->cags[v->agg - 1];
        v->agg = (uint16_t)(n0 + 1);
    }
    L->ncag = n0 + 1;
}

static int cagno(lower_t *L, bc_eid_t eid, const char *nm)
{
    L->sdbad = (uint16_t)eid;
    ncpy(L->sdbnm, sizeof L->sdbnm, nm);
    return 0;
}

static void cagfv(const cagg_t *g, int i, cval_t *out)
{
    memset(out, 0, sizeof(*out));
    out->ival = g->f[i].ival;
    out->fval = g->f[i].fval;
    out->fnd  = g->f[i].fnd;
    out->fsy  = g->f[i].fsy;
    out->isf  = g->f[i].isf;
}

static int cagfi(const lower_t *L, const cagg_t *g, const char *nm)
{
    const struct_def_t *sd;

    if (g->sdi < 0 || g->sdi >= L->nstructs) return -1;
    sd = &L->structs[g->sdi];
    for (int i = 0; i < g->nf && i < sd->num_fields; i++)
        if (strcmp(sd->field_names[i], nm) == 0) return i;
    return -1;
}

static int cadd(lower_t *L, const char *name, const cval_t *v,
                uint32_t type, int bad)
{
    if (L->ncexp >= MAX_CEXPS) return 0;
    cexp_t *c = &L->cexps[L->ncexp++];
    ncpy(c->name, sizeof(c->name), name);
    c->v    = *v;
    c->type = type;
    c->bad  = (uint8_t)(bad != 0);
    c->hag  = 0;
    if (v->agg) {
        const cagg_t *g = cagp(L, v);

        if (!g) c->bad = 1;
        else {
            c->ag  = *g;
            c->hag = 1;
        }
    }
    c->v.agg = 0;
    return 1;
}

static int cxld(lower_t *L, const cexp_t *ce, cval_t *out)
{
    *out = ce->v;
    out->agg = 0;
    if (!ce->hag) return 1;
    out->agg = caga(L);
    if (!out->agg) return 0;
    L->cags[out->agg - 1] = ce->ag;
    return 1;
}

static int cqnm(lower_t *L, uint32_t node, char *out, size_t osz)
{
    char part[64];
    uint32_t nd = node;
    size_t used = 0;

    out[0] = 0;
    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        uint32_t l = ND(L, nd)->first_child;
        uint32_t r = l ? ND(L, l)->next_sibling : 0;
        size_t k;

        if (ND(L, nd)->type != AST_SCOPE_RES || !l || !r) return 0;
        if (ND(L, l)->type != AST_IDENT) return 0;
        get_text(L, l, part, sizeof(part));
        k = strlen(part);
        if (used + k + 3 >= osz) return 0;
        if (used) { memcpy(out + used, "::", 2); used += 2; }
        memcpy(out + used, part, k);
        used += k;
        out[used] = 0;

        if (ND(L, r)->type == AST_IDENT) {
            get_text(L, r, part, sizeof(part));
            k = strlen(part);
            if (used + k + 3 >= osz) return 0;
            memcpy(out + used, "::", 2);
            memcpy(out + used + 2, part, k);
            out[used + 2 + k] = 0;
            return 1;
        }
        nd = r;
    }
    return 0;
}

static int csfx(lower_t *L, const char *full, cexp_t **out)
{
    const char *p = full;

    KA_GUARD(g, CE_NEST);
    while (g--) {
        if (cfind(L, p, out)) return 1;
        p = strstr(p, "::");
        if (!p) return 0;
        p += 2;
    }
    return 0;
}

static int cname(lower_t *L, const char *nm, cval_t *out)
{
    cexp_t *ce = NULL;
    int64_t v;

    memset(out, 0, sizeof(*out));
    if (tgfnd(L, nm, &v)) {
        out->ival = v;
        return 1;
    }
    if (cfind(L, nm, &ce)) {
        if (ce->bad) return 0;
        return cxld(L, ce, out);
    }
    if (L->scls[0]) {
        sdm_t *sm = NULL;
        if (sdfnd(L, L->scls, nm, &sm) == 1) {
            *out = sm->v;
            out->agg = 0;
            return 1;
        }
    }
    if (find_enum(L, nm, &v) || find_binding_int(L, nm, &v)) {
        out->ival = v;
        return 1;
    }
    if (strcmp(nm, "warpSize") == 0 && L->sema) {
        out->ival = L->sema->warp_size;
        return 1;
    }
    return 0;
}

static int ccvt(lower_t *L, cval_t *v, uint32_t t)
{
    if (v->agg) {
        const cagg_t *g = cagp(L, v);

        return g && g->sdi >= 0 && sfind(L, t) == g->sdi;
    }
    if (v->fnd) return t < L->M->num_types
                    && L->M->types[t].kind == BIR_TYPE_PTR;
    if (t >= L->M->num_types) return 0;

    if (is_float_type(L, t)) {
        if (!v->isf) {
            v->fval = (double)v->ival;
            v->isf  = 1;
        }
        if (L->M->types[t].width == 32)
            v->fval = (double)(float)v->fval;
        v->ival = 0;
        return 1;
    }
    if (L->M->types[t].kind != BIR_TYPE_INT) return 0;
    if (v->isf) {
        if (!(v->fval > -9.2233720368547758e18)
         || !(v->fval <  9.2233720368547758e18))
            return 0;
        v->ival = (int64_t)v->fval;
        v->isf  = 0;
        v->fval = 0.0;
    }
    {
        unsigned w = L->M->types[t].width;
        if (w == 0 || w > 64) return 0;
        if (w == 1) {
            v->ival = v->ival != 0;
            return 1;
        }
        if (w < 64) {
            int64_t  lo = -((int64_t)1 << (w - 1));
            uint64_t hi = ((uint64_t)1 << w) - 1u;
            if (v->ival < lo) return 0;
            if (v->ival > 0 && (uint64_t)v->ival > hi) return 0;
        }
    }
    return 1;
}

static int cun(int op, cval_t *v)
{
    if (v->fnd || v->agg) return 0;
    switch (op) {
    case TOK_PLUS:
        return 1;
    case TOK_MINUS:
        if (v->isf) v->fval = -v->fval;
        else        v->ival = (int64_t)(0u - (uint64_t)v->ival);
        return 1;
    case TOK_TILDE:
        if (v->isf) return 0;
        v->ival = (int64_t)~(uint64_t)v->ival;
        return 1;
    case TOK_BANG:
        v->ival = (v->isf ? v->fval == 0.0 : v->ival == 0) ? 1 : 0;
        v->isf  = 0;
        v->fval = 0.0;
        return 1;
    default:
        return 0;
    }
}

static void cbool(cval_t *l, int t)
{
    l->ival = t ? 1 : 0;
    l->isf  = 0;
    l->fval = 0.0;
}

static int cbin(int op, cval_t *l, const cval_t *r)
{
    int      f;
    double   x, y;
    uint64_t a, b;

    if (l->fnd || r->fnd || l->agg || r->agg) return 0;
    f = l->isf || r->isf;
    x = l->isf ? l->fval : (double)l->ival;
    y = r->isf ? r->fval : (double)r->ival;
    a = (uint64_t)l->ival;
    b = (uint64_t)r->ival;

    switch (op) {
    case TOK_PLUS:
        if (f) { l->fval = x + y; l->isf = 1; }
        else     l->ival = (int64_t)(a + b);
        return 1;
    case TOK_MINUS:
        if (f) { l->fval = x - y; l->isf = 1; }
        else     l->ival = (int64_t)(a - b);
        return 1;
    case TOK_STAR:
        if (f) { l->fval = x * y; l->isf = 1; }
        else     l->ival = (int64_t)(a * b);
        return 1;
    case TOK_SLASH:
        if (f) {
            if (y == 0.0) return 0;
            l->fval = x / y;
            l->isf  = 1;
            return 1;
        }
        if (r->ival == 0) return 0;
        if (l->ival == INT64_MIN && r->ival == -1) return 0;
        l->ival = l->ival / r->ival;
        return 1;
    case TOK_PERCENT:
        if (f || r->ival == 0) return 0;
        if (l->ival == INT64_MIN && r->ival == -1) return 0;
        l->ival = l->ival % r->ival;
        return 1;
    case TOK_AMP:
        if (f) return 0;
        l->ival = (int64_t)(a & b);
        return 1;
    case TOK_PIPE:
        if (f) return 0;
        l->ival = (int64_t)(a | b);
        return 1;
    case TOK_CARET:
        if (f) return 0;
        l->ival = (int64_t)(a ^ b);
        return 1;
    case TOK_SHL:
        if (f || r->ival < 0 || r->ival > 63) return 0;
        l->ival = (int64_t)(a << (unsigned)r->ival);
        return 1;
    case TOK_SHR:
        if (f || r->ival < 0 || r->ival > 63) return 0;
        l->ival = l->ival >> (unsigned)r->ival;
        return 1;
    case TOK_EQ:   cbool(l, f ? x == y : l->ival == r->ival); return 1;
    case TOK_NE:   cbool(l, f ? x != y : l->ival != r->ival); return 1;
    case TOK_LT:   cbool(l, f ? x <  y : l->ival <  r->ival); return 1;
    case TOK_GT:   cbool(l, f ? x >  y : l->ival >  r->ival); return 1;
    case TOK_LE:   cbool(l, f ? x <= y : l->ival <= r->ival); return 1;
    case TOK_GE:   cbool(l, f ? x >= y : l->ival >= r->ival); return 1;
    case TOK_LAND: cbool(l, (x != 0.0) && (y != 0.0)); return 1;
    case TOK_LOR:  cbool(l, (x != 0.0) || (y != 0.0)); return 1;
    default:
        return 0;
    }
}

static uint32_t targs(const lower_t *L, uint32_t node)
{
    uint32_t nd = node;

    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        const ast_node_t *p = ND(L, nd);
        uint32_t l, c;

        if (p->type == AST_IDENT) {
            c = p->first_child;
            return (c && ND(L, c)->type == AST_TEMPLATE_ARGS) ? c : 0;
        }
        if (p->type != AST_SCOPE_RES) return 0;
        l = p->first_child;
        if (!l) return 0;
        if (ND(L, l)->type == AST_IDENT) {
            c = ND(L, l)->first_child;
            if (c && ND(L, c)->type == AST_TEMPLATE_ARGS) return c;
        }
        nd = ND(L, l)->next_sibling;
    }
    return 0;
}

static int tamb(const lower_t *L, uint32_t ty)
{
    int n = 0;

    for (int i = 0; i < L->nstructs; i++)
        if (L->structs[i].bir_type == ty) n++;
    return n;
}

static int bguess(const lower_t *L, uint32_t node)
{
    char nm[64];
    uint32_t nc;

    if (!node || ND(L, node)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, node)->d.btype.kind != TYPE_NAME) return 0;
    nc = ND(L, node)->first_child;
    if (!nc || ND(L, nc)->type != AST_IDENT) return 0;
    get_text(L, nc, nm, sizeof nm);
    for (int i = L->nbbase; i < L->nbindings; i++)
        if (L->bindings[i].is_type && !L->bindings[i].is_pack
            && L->bindings[i].guess
            && strcmp(L->bindings[i].name, nm) == 0) return 1;
    return 0;
}

static int idty(const lower_t *L, uint32_t node, uint32_t *ty, int *gs)
{
    char nm[64];

    if (!node || ND(L, node)->type != AST_IDENT) return 0;
    if (ND(L, node)->first_child) return 0;
    get_text(L, node, nm, sizeof nm);
    for (int i = L->nbindings - 1; i >= L->nbbase; i--) {
        const binding_t *b = &L->bindings[i];

        if (!b->is_type || b->is_pack || b->is_fn) continue;
        if (strcmp(b->name, nm) != 0) continue;
        *ty = b->type;
        *gs = b->guess;
        return 1;
    }
    return 0;
}

static int tnorm(lower_t *L, uint32_t n, uint32_t *ty, uint16_t *qm)
{
    const uint16_t km = QUAL_CONST | QUAL_VOLATILE | QUAL_REF | QUAL_RREF
                      | QUAL_DECOR;
    int gs = 0;

    if (!n) return 0;
    if (idty(L, n, ty, &gs)) {
        *qm = 0;
        return !gs && *ty != 0;
    }
    if (ND(L, n)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, n)->qualifiers & QUAL_DECOR) return 0;
    if (bguess(L, n)) return 0;
    *ty = resolve_type(L, n, tpdep(L, n), 0);
    if (L->rtbad || L->rtunk) return 0;
    *qm = (uint16_t)(ND(L, n)->qualifiers & km);
    return 1;
}

static int tsame(lower_t *L, uint32_t ta, uint32_t tb, int *out)
{
    uint32_t ra = 0, rb = 0;
    uint16_t qa = 0, qb = 0;

    if (!tnorm(L, ta, &ra, &qa)) return 0;
    if (!tnorm(L, tb, &rb, &qb)) return 0;

    if (ra != rb || qa != qb) {
        *out = 0;
        return 1;
    }
    if (ra < L->M->num_types && L->M->types[ra].kind == BIR_TYPE_STRUCT
        && tamb(L, ra) > 1) return 0;
    *out = 1;
    return 1;
}

static int trait(lower_t *L, uint32_t node, cval_t *out)
{
    char nm[CE_NAMEZ];
    const char *q;
    uint32_t ta, a, b;
    int same = 0;

    if (ND(L, node)->type == AST_SCOPE_RES) {
        if (!cqnm(L, node, nm, sizeof nm)) return 0;
    } else if (ND(L, node)->type == AST_IDENT) {
        get_text(L, node, nm, sizeof nm);
    } else {
        return 0;
    }
    q = nm;
    if (strncmp(q, "std::", 5) == 0) q += 5;
    if (strcmp(q, "is_same_v") != 0 && strcmp(q, "is_same::value") != 0)
        return 0;

    ta = targs(L, node);
    if (!ta) return 0;
    a = ND(L, ta)->first_child;
    b = a ? ND(L, a)->next_sibling : 0;
    if (!b || ND(L, b)->next_sibling) return 0;
    if (!tsame(L, a, b, &same)) return 0;

    memset(out, 0, sizeof(*out));
    out->ival = same;
    return 1;
}

static int tgval(lower_t *L, uint32_t node, int64_t *out)
{
    uint32_t nd = node, ts = 0, il = 0, ta = 0, id = 0, a;
    char nm[64];
    cval_t v;

    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        int t = ND(L, nd)->type;
        uint32_t l;
        if (t == AST_PAREN) { nd = ND(L, nd)->first_child; continue; }
        if (t != AST_SCOPE_RES) break;
        l = ND(L, nd)->first_child;
        if (!l || ND(L, l)->type != AST_IDENT || !text_eq(L, l, "std"))
            return 0;
        nd = ND(L, l)->next_sibling;
    }
    if (!nd || ND(L, nd)->type != AST_CAST) return 0;
    if (ND(L, nd)->d.oper.flags) return 0;
    ts = ND(L, nd)->first_child;
    if (!ts || ND(L, ts)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, ts)->d.btype.kind != TYPE_NAME) return 0;
    il = ND(L, ts)->next_sibling;
    if (il && (ND(L, il)->type != AST_INIT_LIST || ND(L, il)->first_child))
        return 0;

    for (uint32_t c = ND(L, ts)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_IDENT) id = c;
        else if (ND(L, c)->type == AST_TEMPLATE_ARGS) ta = c;
    }
    if (!id) return 0;
    get_text(L, id, nm, sizeof nm);
    if (strcmp(nm, "true_type") == 0)  { *out = 1; return 1; }
    if (strcmp(nm, "false_type") == 0) { *out = 0; return 1; }
    if (!ta) return 0;
    a = ND(L, ta)->first_child;
    if (strcmp(nm, "bool_constant") == 0) {
        if (!a || !cfold(L, a, &v)) return 0;
        *out = v.isf ? (v.fval != 0.0) : (v.ival != 0);
        return 1;
    }
    if (strcmp(nm, "integral_constant") != 0) return 0;
    a = a ? ND(L, a)->next_sibling : 0;
    if (!a || !cfold(L, a, &v)) return 0;
    *out = v.isf ? (int64_t)v.fval : v.ival;
    return 1;
}

static int dcval(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t l = ND(L, node)->first_child;
    uint32_t r = l ? ND(L, l)->next_sibling : 0;
    uint32_t op;
    char nm[64];
    int64_t v;

    if (!l || !r) return 0;
    if (ND(L, l)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, l)->d.btype.kind != TYPE_DECLTYPE) return 0;
    if (ND(L, r)->type != AST_IDENT || !text_eq(L, r, "value")) return 0;

    op = ND(L, l)->first_child;
    KA_GUARD(g, CE_NEST);
    while (op && g-- && ND(L, op)->type == AST_PAREN)
        op = ND(L, op)->first_child;
    if (!op) return 0;
    memset(out, 0, sizeof(*out));
    if (tgval(L, op, &v)) { out->ival = v; return 1; }
    if (ND(L, op)->type != AST_IDENT) return 0;
    get_text(L, op, nm, sizeof nm);
    if (!tgfnd(L, nm, &v)) return 0;
    out->ival = v;
    return 1;
}

static int csize(lower_t *L, uint32_t node, int64_t *out)
{
    const ast_node_t *n = ND(L, node);
    uint32_t inner = n->first_child;
    uint32_t ty = 0, sz;

    if (!inner) return 0;
    if (ND(L, inner)->type == AST_TYPE_SPEC) {
        ty = resolve_type(L, inner, n->d.oper.flags, 0);
        if (L->rtbad || L->rtunk) return 0;
    } else if (ND(L, inner)->type == AST_IDENT && !n->d.oper.flags) {
        char nm[CE_NAMEZ];
        const sym_t *s;

        get_text(L, inner, nm, sizeof nm);
        s = find_sym(L, nm);
        if (s) ty = s->type;
        else if (!find_binding(L, nm, &ty) && !find_typedef(L, nm, &ty)) {
            uint32_t gi = bir_gsym(L->M, nm, L->tu);

            if (gi == BIR_SYM_NONE) return 0;
            ty = L->M->globals[gi].type;
        }
    } else if (!n->d.oper.flags) {
        ty = arg_type(L, inner);
        if (!ty) return 0;
    } else {
        return 0;
    }
    sz = bir_bsz(L->M, ty, 8);
    if (!sz) return 0;
    *out = (int64_t)sz;
    return 1;
}

static void cndnm(const lower_t *L, uint32_t node, char *buf, int sz)
{
    uint32_t nd = node, off = 0;
    int have = 0, i = 0, dep = 0;
    const char *s;

    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        const ast_node_t *p = ND(L, nd);

        if (p->type == AST_IDENT || p->type == AST_INT_LIT
            || p->type == AST_FLOAT_LIT || p->type == AST_CHAR_LIT) {
            off  = p->d.text.offset;
            have = 1;
            break;
        }
        nd = p->first_child;
    }
    if (!have || off >= BC_ANON_BASE) {
        ncpy(buf, (size_t)sz, "this expression");
        return;
    }
    s = L->src + off;
    while (i < sz - 1 && s[i]) {
        char c = s[i];

        if (c == '(') dep++;
        else if (c == ')' && dep-- == 0) break;
        buf[i] = (c == '\n' || c == '\t' || c == '\r') ? ' ' : c;
        i++;
    }
    while (i > 0 && buf[i - 1] == ' ') i--;
    buf[i] = '\0';
    if (!i) ncpy(buf, (size_t)sz, "this expression");
}

static void cxnm(const lower_t *L, uint32_t node, char *out, int oz)
{
    uint32_t nd = node, off = 0;
    int have = 0, i = 0, dep = 0;
    const char *sp;

    out[0] = '\0';
    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        const ast_node_t *p = ND(L, nd);

        if (p->type == AST_IDENT || p->type == AST_INT_LIT
            || p->type == AST_FLOAT_LIT || p->type == AST_CHAR_LIT) {
            off  = p->d.text.offset;
            have = 1;
            break;
        }
        nd = p->first_child;
    }
    if (!have || off >= BC_ANON_BASE) return;

    sp = L->src + off;
    while (i < oz - 1 && sp[i] && sp[i] != '\n') {
        char c = sp[i];

        if (c == '(' || c == '[') dep++;
        else if ((c == ')' || c == ']') && dep-- == 0) break;
        else if ((c == ',' || c == ';' || c == '>') && dep == 0) break;
        out[i] = c;
        i++;
    }
    while (i > 0 && out[i - 1] == ' ') i--;
    out[i] = '\0';
}

static void abnm(const lower_t *L, const char *nm, uint32_t bnd,
                 char *out, int oz)
{
    char ex[48];

    cxnm(L, bnd, ex, (int)sizeof ex);
    if (!ex[0] || snprintf(out, (size_t)oz, "%s[%s]", nm, ex) < 0)
        ncpy(out, (size_t)oz, nm);
}

static int cpks(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t id = ND(L, node)->first_child;
    char nm[64];
    int cnt;

    if (!id) return 0;
    get_text(L, id, nm, sizeof nm);
    cnt = pk_cnt(L, nm);
    if (cnt < 0) return 0;
    out->ival = cnt;
    return 1;
}

static uint32_t cffnd(lower_t *L, const char *nm, int na)
{
    uint32_t hit = 0;
    int seen = 0;

    for (int i = L->ncfn - 1; i >= 0; i--) {
        uint32_t pv[CF_PARM];
        int np, nrq = 0;

        if (strcmp(L->cfns[i].name, nm) != 0) continue;
        np = collect_params(L, L->cfns[i].ast, pv, CF_PARM);
        if (np < 0 || na < 0) continue;
        for (int k = 0; k < np; k++)
            if (!(ND(L, pv[k])->qualifiers & QUAL_PDEF)) nrq++;
        if (na < nrq || na > np) continue;
        if (!hit) hit = L->cfns[i].ast;
        seen++;
    }
    return seen == 1 ? hit : 0;
}

static int cargn(const lower_t *L, uint32_t first)
{
    int n = 0;

    KA_GUARD(g, BC_MAX_ARGS);
    for (uint32_t a = first; a && g--; a = ND(L, a)->next_sibling) n++;
    return n;
}

static uint32_t cfret(const lower_t *L, uint32_t fd)
{
    uint32_t bd = find_func_body(L, fd), st;

    if (!bd) return 0;
    st = ND(L, bd)->first_child;
    if (!st || ND(L, st)->next_sibling) return 0;
    if (ND(L, st)->type != AST_RETURN) return 0;
    return ND(L, st)->first_child;
}

static int cfblk(lower_t *L, uint32_t st, cval_t *out, int *don)
{
    int svc = L->ncexp, ok = 1;
    uint32_t c;

    KA_GUARD(g, CF_BODY);
    for (c = ND(L, st)->first_child; c && !*don && !L->cbrk && ok && g--;
         c = ND(L, c)->next_sibling) {
        L->sdep++;
        ok = cfrun(L, c, out, don);
        L->sdep--;
    }
    ccomp(L, svc);
    return ok && (!c || *don || L->cbrk);
}

static int cfsw(lower_t *L, uint32_t st, cval_t *out, int *don)
{
    uint32_t bd = child_at(L, st, 1), lab = 0, dfl = 0, c;
    cval_t cv, lv;
    int ok = 1;

    if (!bd || !cfold(L, child_at(L, st, 0), &cv) || cv.isf || cv.agg)
        return 0;
    for (c = ND(L, bd)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_DEFAULT) {
            if (!dfl) dfl = c;
            continue;
        }
        if (ND(L, c)->type != AST_CASE) continue;
        if (!cfold(L, ND(L, c)->first_child, &lv) || lv.isf || lv.agg)
            return 0;
        if (lv.ival == cv.ival) { lab = c; break; }
    }
    if (!lab) lab = dfl;

    KA_GUARD(g, CF_BODY);
    for (c = lab; c && !*don && !L->cbrk && ok && g--;
         c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_CASE || ND(L, c)->type == AST_DEFAULT)
            continue;
        L->sdep++;
        ok = cfrun(L, c, out, don);
        L->sdep--;
    }
    L->cbrk = 0;
    return ok;
}

static int cfnop(int t)
{
    return t == AST_NONE || t == AST_TYPE_SPEC || t == AST_STRUCT_DEF
        || t == AST_ENUM_DEF || t == AST_TYPEDEF || t == AST_USING
        || t == AST_FUNC_DECL;
}

static int cfrun(lower_t *L, uint32_t st, cval_t *out, int *don)
{
    const ast_node_t *n;

    if (!st || L->sdep >= CF_DEEP) return 0;
    n = ND(L, st);
    if (cfnop(n->type)) return 1;
    switch (n->type) {
    case AST_RETURN:
        *don = 1;
        memset(out, 0, sizeof(*out));
        return n->first_child ? cfold(L, n->first_child, out) : 1;

    case AST_BLOCK:
        return cfblk(L, st, out, don);

    case AST_BREAK:
        L->cbrk = 1;
        return 1;

    case AST_SWITCH:
        return cfsw(L, st, out, don);

    case AST_IF: {
        uint32_t cn = child_at(L, st, 0), arm;
        cval_t cv;
        int ok;

        if (n->first_child && ND(L, n->first_child)->type == AST_VAR_DECL)
            return 0;
        if (!cfold(L, cn, &cv) || cv.agg) return 0;
        arm = (cv.isf ? cv.fval != 0.0 : cv.ival != 0)
            ? child_at(L, st, 1) : child_at(L, st, 2);
        if (!arm) return 1;
        L->sdep++;
        ok = cfrun(L, arm, out, don);
        L->sdep--;
        return ok;
    }

    case AST_VAR_DECL: {
        uint32_t nn = child_at(L, st, 1), in;

        if (!(n->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST))) return 0;
        if (n->d.oper.op != 0 || !nn || ND(L, nn)->type != AST_IDENT) return 0;
        in = ND(L, nn)->next_sibling;
        if (!in) return 0;
        return ccoll(L, st, nn, in, NULL, 1);
    }

    default:
        return 0;
    }
}

static int cfcls(const char *qn, char *out, size_t oz)
{
    const char *sep = strrchr(qn, ':');
    size_t n;

    if (!sep || sep == qn || sep[-1] != ':') return 0;
    n = (size_t)(sep - 1 - qn);
    if (!n || n >= oz) return 0;
    memcpy(out, qn, n);
    out[n] = 0;
    return 1;
}

static uint32_t cfsee(lower_t *L, uint32_t cn, char *nm, size_t nz, int *mem,
                      int na)
{
    *mem = 0;
    if (!cn) return 0;
    if (ND(L, cn)->type == AST_SCOPE_RES) {
        cexp_t *ce = NULL;

        if (cqfnd(L, cn, nm, nz, &ce) == 0 && !nm[0]) return 0;
        *mem = 1;
    } else if (ND(L, cn)->type == AST_IDENT) {
        get_text(L, cn, nm, (int)nz);
    } else {
        return 0;
    }
    return cffnd(L, nm, na);
}

static int cfbnd(lower_t *L, const uint32_t *pv, const cval_t *av, int np)
{
    char pb[64];
    uint32_t pc;

    for (int i = 0; i < np; i++) {
        pb[0] = 0;
        for (pc = ND(L, pv[i])->first_child; pc; pc = ND(L, pc)->next_sibling)
            if (ND(L, pc)->type == AST_IDENT) {
                get_text(L, pc, pb, sizeof pb);
                break;
            }
        if (!pb[0] || !cadd(L, pb, &av[i], 0, 0)) return 0;
    }
    return 1;
}

static int cfargv(lower_t *L, uint32_t a, const uint32_t *pv, int np,
                 cval_t *av)
{
    for (int i = 0; i < np; i++) {
        uint32_t d;

        if (a) {
            if (!cfold(L, a, &av[i])) return 0;
            a = ND(L, a)->next_sibling;
            continue;
        }
        d = ND(L, pv[i])->qualifiers & QUAL_PDEF ? pdfn(L, pv[i]) : 0;
        if (!d || !cfold(L, d, &av[i])) return 0;
    }
    return a ? 0 : 1;
}

static int cagmc(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t cn = ND(L, node)->first_child;
    uint32_t o  = cn ? ND(L, cn)->first_child : 0;
    uint32_t fl = o ? ND(L, o)->next_sibling : 0;
    uint32_t pv[CF_PARM], fd, bdy, rx;
    cval_t av[CF_PARM], ov;
    const struct_def_t *sd;
    cagg_t *g;
    char nm[64], qn[CE_NAMEZ], svc[64];
    int np, i, save, ok, got = 0, n0 = L->ncag;

    if (!cn || ND(L, cn)->type != AST_MEMBER) return 0;
    if (!fl || ND(L, fl)->type != AST_IDENT) return 0;
    if (ND(L, cn)->d.member.is_arrow) return 0;
    if (L->cdep >= MAX_CDEP) return 0;
    if (!cfold(L, o, &ov)) return 0;
    g = cagp(L, &ov);
    if (!g || g->sdi < 0 || g->sdi >= L->nstructs) return 0;
    sd = &L->structs[g->sdi];
    get_text(L, fl, nm, sizeof nm);
    if (!nm[0] || !sd->name[0]) return 0;
    if (snprintf(qn, sizeof qn, "%s::%s", sd->name, nm) < 0) return 0;
    fd = cffnd(L, qn, cargn(L, ND(L, cn)->next_sibling));
    if (!fd) return 0;
    rx  = cfret(L, fd);
    bdy = rx ? 0 : find_func_body(L, fd);
    if (!rx && !bdy) return 0;
    np = collect_params(L, fd, pv, CF_PARM);
    if (np < 0) return 0;

    if (!cfargv(L, ND(L, cn)->next_sibling, pv, np, av)) return 0;

    g = cagp(L, &ov);
    if (!g) return 0;
    save = L->ncexp;
    ok = 1;
    for (i = 0; i < g->nf && ok; i++) {
        cval_t fv;

        cagfv(g, i, &fv);
        ok = cadd(L, sd->field_names[i], &fv, 0, 0);
    }
    if (ok) ok = cfbnd(L, pv, av, np);
    if (!ok) {
        ccomp(L, save);
        return 0;
    }
    ncpy(svc, sizeof svc, L->scls);
    ncpy(L->scls, sizeof L->scls, sd->name);
    L->cdep++;
    ok = rx ? cfold(L, rx, out) : (cfrun(L, bdy, out, &got) && got);
    L->cdep--;
    ncpy(L->scls, sizeof L->scls, svc);
    ccomp(L, save);
    if (!ok) return 0;
    cagz(L, n0, out);
    L->sdbad = 0;
    return 1;
}

static int ccall(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t pv[CF_PARM], cn = ND(L, node)->first_child;
    cval_t av[CF_PARM];
    char nm[CE_NAMEZ], svc[64];
    uint32_t fd, rx, bdy = 0;
    int np, save, ok, mem, got = 0;

    if (L->cdep >= MAX_CDEP) return 0;
    fd = cfsee(L, cn, nm, sizeof nm, &mem,
               cargn(L, ND(L, cn)->next_sibling));
    if (!fd) return 0;
    rx = cfret(L, fd);
    if (!rx && !(bdy = find_func_body(L, fd))) return 0;
    np = collect_params(L, fd, pv, CF_PARM);
    if (np < 0) return 0;

    if (!cfargv(L, ND(L, cn)->next_sibling, pv, np, av)) return 0;

    save = L->ncexp;
    if (!cfbnd(L, pv, av, np)) {
        ccomp(L, save);
        return 0;
    }
    ncpy(svc, sizeof svc, L->scls);
    if (mem && !cfcls(nm, L->scls, sizeof L->scls)) {
        ncpy(L->scls, sizeof L->scls, svc);
        ccomp(L, save);
        return 0;
    }
    L->cdep++;
    got = 0;
    ok = rx ? cfold(L, rx, out) : (cfrun(L, bdy, out, &got) && got);
    L->cdep--;
    ncpy(L->scls, sizeof L->scls, svc);
    ccomp(L, save);
    return ok;
}

static int ctcall(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t cn = ND(L, node)->first_child;
    uint32_t av[BC_MAX_ARGS], pv[CF_PARM], func_n = 0, rx, bdy = 0;
    cval_t cav[CF_PARM];
    template_def_t *tm;
    binding_t *nb;
    char nm[CE_NAMEZ];
    int na, ntp = 0, np, ok, got = 0, save, onb, obs;

    if (!cn || ND(L, cn)->type != AST_IDENT) return 0;
    if (L->cdep >= MAX_CDEP || L->xdep >= MAX_NEST) return 0;
    get_text(L, cn, nm, sizeof nm);
    if (!nm[0] || !find_template(L, nm)) return 0;
    na = alist(L, ND(L, cn)->next_sibling, av, BC_MAX_ARGS);
    if (na < 0) return 0;
    tm = tmpick(L, nm, av, na);
    if (!tm) return 0;

    nb = L->xnb[L->xdep++];
    ok = tdeduc(L, tm->ast, cn, av, na, nb, &ntp, &func_n);
    for (int i = 0; ok && i < ntp; i++)
        if (!nb[i].is_pack && !nb[i].fixed && !nb[i].dflt) ok = 0;
    if (ok && (!func_n || !(ND(L, func_n)->qualifiers & QUAL_CONSTEXPR)))
        ok = 0;
    rx  = ok ? cfret(L, func_n) : 0;
    bdy = ok && !rx ? find_func_body(L, func_n) : 0;
    np  = ok ? collect_params(L, func_n, pv, CF_PARM) : -1;
    onb = L->nbindings;
    obs = L->nbbase;
    if (!ok || (!rx && !bdy) || np < 0 || ntp > MAX_BIND - onb
        || !cfargv(L, ND(L, cn)->next_sibling, pv, np, cav)) {
        L->xdep--;
        return 0;
    }

    for (int i = 0; i < ntp; i++) L->bindings[L->nbindings++] = nb[i];
    L->nbbase = onb;
    save = L->ncexp;
    ok = cfbnd(L, pv, cav, np);
    if (ok) {
        L->cdep++;
        ok = rx ? cfold(L, rx, out) : (cfrun(L, bdy, out, &got) && got);
        L->cdep--;
    }
    ccomp(L, save);
    L->nbbase = obs;
    L->nbindings = onb;
    L->xdep--;
    return ok;
}

static const char *tsnm(const lower_t *L, const char *nm)
{
    for (int i = L->ntypedefs - 1; i >= 0; i--)
        if (strcmp(L->typedefs[i].name, nm) == 0)
            return L->typedefs[i].snm[0] ? L->typedefs[i].snm : NULL;
    return NULL;
}

static int cqmap(const lower_t *L, const char *nm, char *out, size_t oz)
{
    const char *sep = strstr(nm, "::"), *sn;
    char head[64];
    size_t hl;

    if (!sep) return 0;
    hl = (size_t)(sep - nm);
    if (hl == 0 || hl >= sizeof(head)) return 0;
    memcpy(head, nm, hl);
    head[hl] = 0;
    sn = tsnm(L, head);
    if (!sn && strchr(head, CI_SEP)) sn = head;
    if (!sn) return 0;
    return snprintf(out, oz, "%s%s", sn, sep) > 0;
}

static int cqcl(lower_t *L, uint32_t node, char *out, size_t oz)
{
    const ast_node_t *n = ND(L, node);
    uint32_t ln = n->first_child;
    uint32_t mn = ln ? ND(L, ln)->next_sibling : 0;
    uint32_t ta = ln && ND(L, ln)->type == AST_IDENT
                ? ND(L, ln)->first_child : 0;
    char cb[128], mg[128], mb[64];

    if (n->type != AST_SCOPE_RES || !ta || !mn) return 0;
    if (ND(L, ta)->type != AST_TEMPLATE_ARGS) return 0;
    if (ND(L, mn)->type != AST_IDENT || ND(L, mn)->next_sibling) return 0;
    get_text(L, ln, cb, sizeof(cb));
    if (!cget(L, node, cb, ta, mg, sizeof(mg))) return 0;
    get_text(L, mn, mb, sizeof(mb));
    return snprintf(out, oz, "%s::%s", mg, mb) > 0;
}

static int cqfnd(lower_t *L, uint32_t node, char *qn, size_t qz, cexp_t **ce)
{
    char nm[CE_NAMEZ];

    if (cqcl(L, node, qn, qz)
        || (cqnm(L, node, nm, sizeof(nm)) && cqmap(L, nm, qn, qz)))
        return cfind(L, qn, ce) ? 1 : -1;
    if (targs(L, node)) return -1;
    if (!cqnm(L, node, qn, qz)) return 0;
    return csfx(L, qn, ce) ? 1 : 0;
}

static int cqres(lower_t *L, uint32_t node, cval_t *out)
{
    cexp_t *ce = NULL;
    char qn[CE_NAMEZ];
    const char *last;
    int r = cqfnd(L, node, qn, sizeof(qn), &ce);

    if (r < 0) return 0;
    if (r > 0) {
        if (ce->bad) return 0;
        *out = ce->v;
        return 1;
    }
    if (cname(L, qn, out)) return 1;
    last = strrchr(qn, ':');
    return cname(L, last ? last + 1 : qn, out);
}

static int sdfnd(lower_t *L, const char *cls, const char *nm, sdm_t **out)
{
    char q[SD_QUE][64];
    int qn = 1, qi = 0, opq = 0;

    ncpy(q[0], sizeof(q[0]), cls);
    while (qi < qn) {
        const char *c = q[qi++];
        int si;

        for (int i = 0; i < L->nsdm; i++)
            if (strcmp(L->sdms[i].cls, c) == 0
             && strcmp(L->sdms[i].nm, nm) == 0) {
                *out = &L->sdms[i];
                return L->sdms[i].bad ? -1 : 1;
            }
        si = cslot(L, c);
        if (si < 0) {
            if (qi > 1) opq = 1;
            continue;
        }
        if (L->structs[si].bopq) opq = 1;
        for (int b = 0; b < L->structs[si].nbase; b++) {
            if (qn >= SD_QUE) { opq = 1; break; }
            ncpy(q[qn++], sizeof(q[0]), L->structs[si].base[b]);
        }
    }
    return opq ? -2 : 0;
}

static int sdtd(const lower_t *L, const char *nm, uint32_t *ty, int *unc)
{
    for (int i = L->ntypedefs - 1; i >= 0; i--)
        if (strcmp(L->typedefs[i].name, nm) == 0) {
            *ty = L->typedefs[i].bir_type;
            *unc = L->typedefs[i].bad != 0;
            return 1;
        }
    return 0;
}

static void sdunc(lower_t *L, const char *nm)
{
    L->sdbad = BC_E960;
    ncpy(L->sdbnm, sizeof L->sdbnm, nm);
}

static int sdnrm(lower_t *L, const char *nm, char *out, size_t oz)
{
    const char *s;
    uint32_t t;
    int unc = 0;

    if (cslot(L, nm) >= 0) { ncpy(out, oz, nm); return 1; }
    s = tsnm(L, nm);
    if (s && cslot(L, s) >= 0) { ncpy(out, oz, s); return 1; }
    if (!find_binding(L, nm, &t) && !sdtd(L, nm, &t, &unc)) return 0;
    if (unc) { sdunc(L, nm); return 0; }
    s = sname(L, t);
    if (s) { ncpy(out, oz, s); return 1; }
    return 0;
}

static uint32_t sdqcl(lower_t *L, uint32_t node, char *out, size_t oz)
{
    uint32_t nd = node;
    char cb[128];

    KA_GUARD(g, CE_NEST);
    while (nd && g--) {
        uint32_t l = ND(L, nd)->first_child;
        uint32_t r = l ? ND(L, l)->next_sibling : 0;
        uint32_t ta;

        if (ND(L, nd)->type != AST_SCOPE_RES || !l || !r) return 0;
        if (ND(L, r)->type == AST_SCOPE_RES) { nd = r; continue; }
        if (ND(L, r)->type != AST_IDENT || ND(L, r)->next_sibling) return 0;
        if (ND(L, l)->type != AST_IDENT) return 0;
        get_text(L, l, cb, sizeof(cb));
        ta = ND(L, l)->first_child;
        if (ta && ND(L, ta)->type == AST_TEMPLATE_ARGS) {
            uint32_t at = 0;
            const char *sn;

            if (cget(L, node, cb, ta, out, oz)) return r;
            if (!altry(L, cb, ta, &at)) return 0;
            sn = sname(L, at);
            if (!sn) return 0;
            ncpy(out, oz, sn);
            return r;
        }
        return sdnrm(L, cb, out, oz) ? r : 0;
    }
    return 0;
}

static uint32_t sdmob(lower_t *L, uint32_t node, char *out, size_t oz)
{
    uint32_t l = ND(L, node)->first_child;
    uint32_t r = l ? ND(L, l)->next_sibling : 0;
    const char *s;
    uint32_t st;
    char on[128], fn[64];
    sym_t *sy;

    if (!l || !r || ND(L, r)->type != AST_IDENT) return 0;
    if (ND(L, l)->type != AST_IDENT || ND(L, l)->first_child) return 0;
    get_text(L, l, on, sizeof(on));
    sy = find_sym(L, on);
    if (!sy) return sdnrm(L, on, out, oz) ? r : 0;
    st = sy->type;
    if (ND(L, node)->d.member.is_arrow) st = ptr_inner(L, st);
    s = st ? sname(L, st) : NULL;
    if (!s) return 0;
    get_text(L, r, fn, sizeof(fn));
    if (dcfld(L, st, fn)) return 0;
    ncpy(out, oz, s);
    return r;
}

static void sdset(lower_t *L, int code, const char *cls, const char *nm)
{
    L->sdbad = (uint16_t)(BC_E400 + code - 1);
    if (snprintf(L->sdbnm, sizeof(L->sdbnm), "%s::%s", cls, nm) < 0)
        L->sdbnm[0] = 0;
}

static int sdres(lower_t *L, uint32_t node, cval_t *out)
{
    char cls[144], nm[64];
    sdm_t *e = NULL;
    uint32_t mn;
    int scp = ND(L, node)->type == AST_SCOPE_RES;
    int r;

    L->sdbad = 0;
    L->sdty  = 0;
    L->sdarr = 0;
    if (scp) mn = sdqcl(L, node, cls, sizeof(cls));
    else if (ND(L, node)->type == AST_MEMBER)
        mn = sdmob(L, node, cls, sizeof(cls));
    else return 0;
    if (!mn) return 0;
    get_text(L, mn, nm, sizeof(nm));
    r = sdfnd(L, cls, nm, &e);
    if (r == 1 || r == -1) {
        L->sdty  = e->type;
        L->sdarr = e->arr;
    }
    if (r == 1) {
        *out = e->v;
        return 1;
    }
    if (r == -1) sdset(L, e->bad, cls, nm);
    else if (r == -2 && scp) sdset(L, 4, cls, nm);
    return 0;
}

static int sdemt(lower_t *L, uint32_t node)
{
    if (!L->sdbad) return 0;
    lower_error(L, node, (bc_eid_t)L->sdbad, L->sdbnm);
    L->sdbad = 0;
    return 1;
}

static int vtone(const lower_t *L, const char *nm)
{
    int hit = -1, n = 0;

    for (int i = 0; i < L->nvtpl; i++)
        if (strcmp(L->vtpl[i].name, nm) == 0) {
            hit = i;
            n++;
        }
    if (n != 1 || L->vtpl[hit].spec) return -1;
    return hit;
}

static int vtfnd(const lower_t *L, const char *nm)
{
    const char *p = nm, *ls = NULL;
    int hit = vtone(L, nm);

    if (hit >= 0) return hit;
    KA_GUARD(g, CE_NEST);
    while (g--) {
        const char *nx = strstr(p, "::");

        if (!nx) break;
        p  = nx + 2;
        ls = p;
    }
    return ls ? vtone(L, ls) : -1;
}

static int vtbnd(lower_t *L, uint32_t ta, binding_t *nb, int ntp)
{
    uint32_t a;
    int i = 0;

    for (a = ND(L, ta)->first_child; a; a = ND(L, a)->next_sibling) {
        uint32_t t = 0;
        uint16_t qm = 0;
        int isty = tnorm(L, a, &t, &qm);

        if (i >= ntp) return 0;
        if (isty && qm) return 0;
        if (nb[i].is_pack) {
            cval_t cv;

            if (nb[i].npk >= MAX_PKELM) return 0;
            if (isty) {
                if (nb[i].npk && nb[i].pk_nt) return 0;
                nb[i].pk_t[nb[i].npk++] = t;
            } else {
                if (ND(L, a)->type == AST_TYPE_SPEC) return 0;
                if (nb[i].npk && !nb[i].pk_nt) return 0;
                if (!cfold(L, a, &cv) || cv.isf) return 0;
                nb[i].pk_nt = 1;
                nb[i].pk_v[nb[i].npk++] = cv.ival;
            }
            nb[i].fixed = 1;
            continue;
        }
        if (isty) {
            nb[i].is_type = 1;
            nb[i].type = t;
        } else if (ND(L, a)->type == AST_TYPE_SPEC) {
            return 0;
        } else {
            cval_t cv;

            if (!cfold(L, a, &cv) || cv.isf) return 0;
            nb[i].is_type = 0;
            nb[i].ival = cv.ival;
        }
        nb[i].fixed = 1;
        i++;
    }
    for (; i < ntp; i++)
        if (!nb[i].is_pack && !nb[i].fixed) return 0;
    return 1;
}

static int vteval(lower_t *L, uint32_t node, cval_t *out)
{
    binding_t nb[MAX_TPARM];
    char nm[CE_NAMEZ];
    uint32_t ta, ty;
    int vi, ntp, onb, obs, ok;

    if (ND(L, node)->type == AST_IDENT) {
        if (!ND(L, node)->first_child) return 0;
        get_text(L, node, nm, sizeof nm);
    } else if (!cqnm(L, node, nm, sizeof nm)) {
        return 0;
    }
    vi = vtfnd(L, nm);
    if (vi < 0) return 0;
    ta = targs(L, node);
    if (!ta || L->cdep >= MAX_CDEP) return 0;

    ntp = tparm(L, L->vtpl[vi].tdcl, nb);
    if (ntp <= 0) return 0;
    if (!vtbnd(L, ta, nb, ntp)) return 0;
    if (ntp > MAX_BIND - L->nbindings) {
        lower_error(L, node, BC_E340, L->vtpl[vi].name, MAX_BIND);
        return 0;
    }

    onb = L->nbindings;
    obs = L->nbbase;
    for (int i = 0; i < ntp; i++) L->bindings[L->nbindings++] = nb[i];
    L->nbbase = onb;
    L->cdep++;
    ok = cfold(L, L->vtpl[vi].init, out);
    if (ok) {
        ty = resolve_type(L, L->vtpl[vi].ty, 0, 0);
        ok = !L->rtbad && !L->rtunk && ccvt(L, out, ty);
    }
    L->cdep--;
    L->nbindings = onb;
    L->nbbase = obs;
    return ok;
}

static int cpkb(lower_t *L, char nms[][64], int nn, int idx)
{
    for (int k = 0; k < nn; k++) {
        const binding_t *tp;
        binding_t *t;

        if (find_fpk(L, nms[k])) {
            if (L->npkact >= MAX_FPACKS) return 0;
            ncpy(L->pkact[L->npkact].name,
                 sizeof(L->pkact[0].name), nms[k]);
            L->pkact[L->npkact].idx = idx;
            L->npkact++;
            continue;
        }
        tp = find_tpk(L, nms[k]);
        if (!tp || idx < 0 || idx >= tp->npk || idx >= MAX_PKELM) return 0;
        if (L->nbindings >= MAX_BIND) return 0;
        t = &L->bindings[L->nbindings++];
        memset(t, 0, sizeof(*t));
        ncpy(t->name, sizeof(t->name), nms[k]);
        if (tp->pk_nt) {
            t->ival = tp->pk_v[idx];
        } else {
            t->is_type = 1;
            t->type = tp->pk_t[idx];
        }
    }
    return 1;
}

static int cfopr(lower_t *L, uint32_t pat, uint32_t init, char nms[][64],
                 int nn, int len, int lft, int j, cval_t *v)
{
    int sva, svb, ok, idx;

    if (init && ((lft && j == 0) || (!lft && j == len)))
        return cfold(L, init, v);
    idx = (lft && init) ? j - 1 : j;
    if (idx < 0 || idx >= len) return 0;
    sva = L->npkact;
    svb = L->nbindings;
    ok  = cpkb(L, nms, nn, idx) && cfold(L, pat, v);
    L->npkact    = sva;
    L->nbindings = svb;
    return ok;
}

static int cshrt(int op, cval_t *v)
{
    int t;

    if (v->agg) return 0;
    t = v->isf ? v->fval != 0.0 : v->ival != 0;

    if ((op == TOK_LOR && t) || (op == TOK_LAND && !t)) {
        v->isf  = 0;
        v->fval = 0.0;
        v->ival = t;
        return 1;
    }
    return 0;
}

static int cfacc(lower_t *L, uint32_t pat, uint32_t init, char nms[][64],
                 int nn, int len, int lft, int op, int m, cval_t *acc)
{
    cval_t v;
    int j, ok;

    if (op == TOK_COMMA) {
        for (j = 0; j < m; j++)
            if (!cfopr(L, pat, init, nms, nn, len, lft, j, acc)) return 0;
        return 1;
    }
    if (lft) {
        ok = cfopr(L, pat, init, nms, nn, len, lft, 0, acc);
        for (j = 1; ok && j < m; j++) {
            if (cshrt(op, acc)) break;
            ok = cfopr(L, pat, init, nms, nn, len, lft, j, &v)
                 && cbin(op, acc, &v);
        }
        return ok;
    }
    ok = cfopr(L, pat, init, nms, nn, len, lft, m - 1, acc);
    for (j = m - 2; ok && j >= 0; j--) {
        ok = cfopr(L, pat, init, nms, nn, len, lft, j, &v);
        if (!ok) break;
        if (cshrt(op, &v)) { *acc = v; break; }
        ok = cbin(op, &v, acc);
        *acc = v;
    }
    return ok;
}

static int cfld(lower_t *L, uint32_t node, cval_t *out)
{
    const ast_node_t *n = ND(L, node);
    uint32_t a = n->first_child;
    uint32_t b = a ? ND(L, a)->next_sibling : 0;
    uint32_t pat, init = 0;
    char nms[MAX_FPACKS][64];
    char cnm[CE_NAMEZ];
    cval_t acc;
    int op = n->d.oper.op;
    int nn = 0, len, m, lft, ok;

    switch (n->d.oper.flags) {
    case FLD_UL: pat = a; lft = 1; break;
    case FLD_UR: pat = a; lft = 0; break;
    case FLD_BL: pat = b; init = a; lft = 1; break;
    case FLD_BR: pat = a; init = b; lft = 0; break;
    default: return 0;
    }
    if (!pat || L->cdep >= MAX_CDEP) return 0;

    len = pk_len(L, pat, nms, &nn, 1);
    if (len < 0) return 0;
    if (len > MAX_PKELM) {
        cndnm(L, node, cnm, sizeof cnm);
        lower_error(L, node, BC_E340, cnm, MAX_PKELM);
        return 0;
    }

    m = len + (init ? 1 : 0);
    memset(out, 0, sizeof(*out));
    if (m == 0) {
        if (op == TOK_LOR)  return 1;
        if (op == TOK_LAND) { out->ival = 1; return 1; }
        return 0;
    }

    L->cdep++;
    ok = cfacc(L, pat, init, nms, nn, len, lft, op, m, &acc);
    L->cdep--;
    if (ok) *out = acc;
    return ok;
}

static int cagil(lower_t *L, uint32_t il, int si, cval_t *out)
{
    const struct_def_t *sd;
    uint32_t e;
    uint16_t h;
    int i;

    if (si < 0 || si >= L->nstructs) return 0;
    sd = &L->structs[si];
    if (sd->uni || sd->anon || sd->bopq || sd->num_fields <= 0) return 0;
    if (sd->num_fields > MAX_CAGF) return cagno(L, BC_E820, sd->name);
    for (i = 0; i < sd->num_fields; i++)
        if (sd->fanon[i] || !sd->field_names[i][0]) return 0;
    memset(out, 0, sizeof(*out));
    h = caga(L);
    if (!h) return cagno(L, BC_E821, sd->name);
    L->cags[h - 1].sdi = si;
    L->cags[h - 1].nf  = sd->num_fields;
    i = 0;
    for (e = ND(L, il)->first_child; e; e = ND(L, e)->next_sibling) {
        cval_t fv;

        if (i >= sd->num_fields) return 0;
        if (ND(L, e)->type == AST_INIT_LIST) return 0;
        if (!cfold(L, e, &fv) || fv.agg) return 0;
        if (!ccvt(L, &fv, sd->field_types[i])) return 0;
        L->cags[h - 1].f[i].ival = fv.ival;
        L->cags[h - 1].f[i].fval = fv.fval;
        L->cags[h - 1].f[i].fnd  = fv.fnd;
        L->cags[h - 1].f[i].fsy  = fv.fsy;
        L->cags[h - 1].f[i].isf  = (uint8_t)fv.isf;
        i++;
    }
    if (i != sd->num_fields) return 0;
    out->agg = h;
    return 1;
}

static int cagct(lower_t *L, uint32_t node, cval_t *out)
{
    const ast_node_t *n = ND(L, node);
    uint32_t a = n->first_child;
    uint32_t b = a ? ND(L, a)->next_sibling : 0;
    uint32_t t;

    if (!a || !b || ND(L, b)->type != AST_INIT_LIST) return 0;
    t = resolve_type(L, a, n->d.oper.flags, 0);
    return t ? cagil(L, b, sfind(L, t), out) : 0;
}

static int cagmb(lower_t *L, uint32_t node, cval_t *out)
{
    uint32_t o = ND(L, node)->first_child;
    uint32_t fl = o ? ND(L, o)->next_sibling : 0;
    cval_t ov;
    cagg_t *g;
    char nm[64];
    int fi, n0 = L->ncag;

    if (!o || !fl || ND(L, fl)->type != AST_IDENT) return 0;
    if (ND(L, node)->d.member.is_arrow) return 0;
    if (L->cdep >= MAX_CDEP) return 0;
    L->cdep++;
    fi = cfold(L, o, &ov);
    L->cdep--;
    if (!fi) return 0;
    g = cagp(L, &ov);
    if (!g) return 0;
    get_text(L, fl, nm, sizeof nm);
    fi = nm[0] ? cagfi(L, g, nm) : -1;
    if (fi < 0) return 0;
    cagfv(g, fi, out);
    L->ncag = n0;
    L->sdbad = 0;
    return 1;
}

static int ctres(lower_t *L, uint32_t id, const char *nm, binding_t *nb,
                 int *ntpo, uint32_t *fnc, int *ixo)
{
    template_def_t *tm = find_template(L, nm);
    char key[256];
    uint32_t av[1], func_n = 0;
    int ntp = 0, ok;

    *ntpo = 0;
    *fnc  = 0;
    *ixo  = -1;
    if (!tm) return 0;
    av[0] = 0;
    tm = tmpick(L, nm, av, 0);
    if (!tm) return 0;
    ok = tdeduc(L, tm->ast, id, av, 0, nb, &ntp, &func_n);
    for (int i = 0; ok && i < ntp; i++)
        if (!nb[i].is_pack && !nb[i].fixed && !nb[i].dflt) ok = 0;
    if (!ok || !func_n) return 0;
    tkeyb(nb, ntp, nm, key, sizeof key);
    tkfn(L, nm, func_n, key, sizeof key);
    *ntpo = ntp;
    *fnc  = func_n;
    *ixo  = tfidx(L, key);
    return 1;
}

static int cfnt(lower_t *L, uint32_t node, cval_t *out)
{
    char nm[CE_NAMEZ];
    uint32_t func_n;
    int ix = -1, ntp, ok;

    get_text(L, node, nm, sizeof nm);
    if (!nm[0] || L->pdep >= MAX_NEST) return 0;
    L->pdep++;
    ok = ctres(L, node, nm, L->pnb[L->pdep - 1], &ntp, &func_n, &ix);
    L->pdep--;
    if (!ok || ix < 0) return 0;
    memset(out, 0, sizeof(*out));
    out->fnd = node;
    out->fsy = (uint16_t)(ix + 1);
    return 1;
}

static int cidn(lower_t *L, uint32_t node, cval_t *out)
{
    char nm[CE_NAMEZ];

    if (trait(L, node, out) || vteval(L, node, out)) return 1;
    get_text(L, node, nm, sizeof(nm));
    if (cname(L, nm, out)) return 1;
    if (targs(L, node)) return cfnt(L, node, out);
    if (!nm[0] || bir_fsym(L->M, nm, L->tu, -1) == BIR_SYM_NONE) return 0;
    memset(out, 0, sizeof(*out));
    out->fnd = node;
    return 1;
}

static int cleaf(lower_t *L, uint32_t node, cval_t *out)
{
    const ast_node_t *p = ND(L, node);

    memset(out, 0, sizeof(*out));

    switch (p->type) {
    case AST_INT_LIT:
        out->ival = parse_int_text(lw_tptr(L, p->d.text.offset),
                                   (int)p->d.text.len);
        return 1;

    case AST_FLOAT_LIT: {
        int f32;
        out->fval = parse_float_text(lw_tptr(L, p->d.text.offset),
                                     (int)p->d.text.len, &f32);
        out->isf  = 1;
        return 1;
    }

    case AST_BOOL_LIT:
        out->ival = p->d.ival ? 1 : 0;
        return 1;

    case AST_CHAR_LIT:
        if (p->d.text.len < 3) return 0;
        out->ival = (unsigned char)lw_tptr(L, p->d.text.offset)[1];
        return 1;

    case AST_SIZEOF:
        return csize(L, node, &out->ival);

    case AST_PACK_SIZE:
        return cpks(L, node, out);

    case AST_FOLD:
        return cfld(L, node, out);

    case AST_CALL:
        return ccall(L, node, out) || ctcall(L, node, out)
               || cagmc(L, node, out);

    case AST_IDENT:
        return cidn(L, node, out);

    case AST_SCOPE_RES:
        return trait(L, node, out) || vteval(L, node, out)
               || dcval(L, node, out) || sdres(L, node, out)
               || cqres(L, node, out);

    case AST_MEMBER:
        return sdres(L, node, out) || cagmb(L, node, out);

    default:
        return 0;
    }
}

static int cfevl(lower_t *L, uint32_t node, cval_t *out)
{
    struct { uint32_t n; uint8_t st; } wk[CF_DEPTH];
    cval_t vs[CF_DEPTH];
    int wn, vn = 0;

    if (!node) return 0;
    L->sdbad = 0;
    wk[0].n  = node;
    wk[0].st = 0;
    wn = 1;

    KA_GUARD(g, 8192);
    while (wn > 0 && g--) {
        uint32_t nd = wk[wn - 1].n;
        uint8_t  st = wk[wn - 1].st;
        const ast_node_t *p = ND(L, nd);
        int op     = p->d.oper.op;
        uint32_t a = p->first_child;
        uint32_t b = a ? ND(L, a)->next_sibling : 0;
        uint32_t c = b ? ND(L, b)->next_sibling : 0;

        switch (p->type) {
        case AST_PAREN:
            if (!a) return 0;
            wk[wn - 1].n  = a;
            wk[wn - 1].st = 0;
            break;

        case AST_TERNARY:
            if (!a || !b || !c) return 0;
            if (st == 0) {
                if (wn >= CF_DEPTH) return 0;
                wk[wn - 1].st = 1;
                wk[wn].n  = a;
                wk[wn].st = 0;
                wn++;
                break;
            }
            if (vn < 1 || vs[vn - 1].agg) return 0;
            vn--;
            wk[wn - 1].n  = (vs[vn].isf ? vs[vn].fval != 0.0
                                        : vs[vn].ival != 0) ? b : c;
            wk[wn - 1].st = 0;
            break;

        case AST_CAST:
            if (!a || !b) return 0;
            if (ND(L, b)->type == AST_INIT_LIST) {
                if (vn >= CF_DEPTH) return 0;
                if (!cagct(L, nd, &vs[vn])) return 0;
                vn++;
                wn--;
                break;
            }
            if (st == 0) {
                if (wn >= CF_DEPTH) return 0;
                wk[wn - 1].st = 1;
                wk[wn].n  = b;
                wk[wn].st = 0;
                wn++;
                break;
            }
            if (vn < 1) return 0;
            if (!ccvt(L, &vs[vn - 1],
                      resolve_type(L, a, p->d.oper.flags, 0)))
                return 0;
            wn--;
            break;

        case AST_UNARY_PREFIX:
            if (!a) return 0;
            if (st == 0) {
                if (wn >= CF_DEPTH) return 0;
                wk[wn - 1].st = 1;
                wk[wn].n  = a;
                wk[wn].st = 0;
                wn++;
                break;
            }
            if (vn < 1) return 0;
            if (!cun(op, &vs[vn - 1])) return 0;
            wn--;
            break;

        case AST_BINARY:
            if (!a || !b) return 0;
            if (st == 0) {
                if (wn + 1 >= CF_DEPTH) return 0;
                wk[wn - 1].st = 1;
                wk[wn].n  = b;
                wk[wn].st = 0;
                wn++;
                wk[wn].n  = a;
                wk[wn].st = 0;
                wn++;
                break;
            }
            if (vn < 2) return 0;
            if (!cbin(op, &vs[vn - 2], &vs[vn - 1])) return 0;
            vn--;
            wn--;
            break;

        default:
            if (vn >= CF_DEPTH) return 0;
            if (!cleaf(L, nd, &vs[vn])) return 0;
            vn++;
            wn--;
            break;
        }
    }

    if (wn > 0 || vn != 1) return 0;
    *out = vs[0];
    return 1;
}

static int cfold(lower_t *L, uint32_t node, cval_t *out)
{
    int n0 = L->ncag;

    if (!cfevl(L, node, out)) {
        L->ncag = n0;
        return 0;
    }
    cagz(L, n0, out);
    return 1;
}

static int cfsc(lower_t *L, uint32_t node, cval_t *out)
{
    int n0 = L->ncag, ok = cfold(L, node, out);

    L->ncag = n0;
    if (ok && out->agg) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    return ok;
}

static int cival(lower_t *L, uint32_t node, int64_t *out)
{
    cval_t v;

    if (!cfsc(L, node, &v)) return 0;
    if (v.isf || v.fnd) return 0;
    *out = v.ival;
    return 1;
}

static uint32_t algof(lower_t *L, uint32_t tspec)
{
    int64_t v = 0;
    uint32_t e;

    if (!tspec || tspec >= L->P->num_nodes) return 0;
    e = ND(L, tspec)->algn;
    if (!e) return 0;
    if (!cival(L, e, &v) || v <= 0 || v > LO_MAXALG || (v & (v - 1)) != 0) {
        lower_error(L, tspec, BC_E220, (int)v);
        return 0;
    }
    return (uint32_t)v;
}

static uint32_t sdemv(lower_t *L, const cval_t *v, uint32_t t)
{
    if (t >= L->M->num_types)
        t = v->isf ? bir_type_float(L->M, 32) : bir_type_int(L->M, 32);
    if (v->isf)
        return BIR_MAKE_CONST(bir_const_float(L->M, t, v->fval));
    return BIR_MAKE_CONST(bir_const_int(L->M, t, v->ival));
}

static uint32_t cemit(lower_t *L, const cexp_t *ce)
{
    return sdemv(L, &ce->v, ce->type);
}

static int ccoll(lower_t *L, uint32_t node, uint32_t name_n,
                 uint32_t init_n, const char *qual, int strict)
{
    char nm[CE_NAMEZ], qn[CE_NAMEZ];
    cval_t v;
    uint32_t t = resolve_type(L, child_at(L, node, 0),
                              ND(L, node)->d.oper.flags, 0);
    int n0 = L->ncag;
    int bad = !cfold(L, init_n, &v) || !ccvt(L, &v, t);
    int ok = 1;

    if (bad && !strict) {
        L->ncag = n0;
        return 1;
    }
    if (bad) memset(&v, 0, sizeof(v));
    get_text(L, name_n, nm, sizeof(nm));
    if (!cadd(L, nm, &v, t, bad)) ok = 0;
    if (ok && qual && qual[0]) {
        int qlen = snprintf(qn, sizeof(qn), "%s::%s", qual, nm);

        if (qlen < 0 || (size_t)qlen >= sizeof(qn)
            || !cadd(L, qn, &v, t, bad)) ok = 0;
    }
    L->ncag = n0;
    return ok;
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

static int aggr_t(const lower_t *L, uint32_t t)
{
    uint8_t k;
    if (t >= L->M->num_types) return 0;
    k = L->M->types[t].kind;
    return k == BIR_TYPE_STRUCT || k == BIR_TYPE_ARRAY;
}

static int ptr_t(const lower_t *L, uint32_t t)
{
    return t < L->M->num_types && L->M->types[t].kind == BIR_TYPE_PTR;
}

static uint32_t bin_val(lower_t *L, uint32_t node, int op,
                        uint32_t lhs, uint32_t lhs_n,
                        uint32_t rhs, uint32_t rhs_n)
{
    uint32_t lt = ref_type(L, lhs), rt = ref_type(L, rhs);

    uint32_t t32 = bir_type_int(L->M, 32);
    if (is_i1(L, lt)) { lhs = coerce_to(L, lhs, t32, 1); lt = t32; }
    if (is_i1(L, rt)) { rhs = coerce_to(L, rhs, t32, 1); rt = t32; }

    uint32_t res_t = lt;
    int lf = is_float_type(L, lt), rf = is_float_type(L, rt);
    if (lf && !rf) {
        rhs = coerce_to(L, rhs, lt, node_is_unsigned(L, rhs_n));
        res_t = lt;
    } else if (!lf && rf) {
        lhs = coerce_to(L, lhs, rt, node_is_unsigned(L, lhs_n));
        res_t = rt;
    } else if (lf && rf) {
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
    int fp  = is_float_type(L, res_t);
    int opc = bin_op_code(op, fp, node_is_unsigned(L, node));
    if (opc < 0) {
        lower_error(L, node, BC_E102);
        return lhs;
    }
    if (aggr_t(L, lt) || aggr_t(L, rt)
        || (opc != BIR_ADD && opc != BIR_SUB
            && (ptr_t(L, lt) || ptr_t(L, rt)))) {
        lower_error(L, node, BC_E102);
        return lhs;
    }
    uint32_t inst = emit(L, (uint16_t)opc, res_t, 2, 0);
    set_op(L, inst, 0, lhs);
    set_op(L, inst, 1, rhs);
    return BIR_MAKE_VAL(inst);
}

/* ---- Aggregate operands ---- */

static struct_def_t *sdbyt(lower_t *L, uint32_t t)
{
    for (int i = 0; i < L->nstructs; i++)
        if (L->structs[i].bir_type == t) return &L->structs[i];
    return NULL;
}

static int aggv(const lower_t *L, uint32_t an);

static int agglv(lower_t *L, uint32_t node)
{
    const ast_node_t *n;
    char nm[CE_NAMEZ];
    const sym_t *s;
    int g = CE_NEST;

    while (node && g-- > 0 && ND(L, node)->type == AST_PAREN)
        node = ND(L, node)->first_child;
    if (!node) return 0;
    n = ND(L, node);
    if (n->type == AST_SUBSCRIPT || n->type == AST_MEMBER) return 1;
    if (n->type == AST_UNARY_PREFIX && n->d.oper.op == TOK_STAR) return 1;
    if (n->type != AST_IDENT) return 0;
    get_text(L, node, nm, sizeof(nm));
    s = find_sym(L, nm);
    return s && s->is_alloca && s->type < L->M->num_types
        && L->M->types[s->type].kind == BIR_TYPE_STRUCT;
}

static uint32_t stof(lower_t *L, uint32_t ref, uint32_t node)
{
    uint32_t t = ref_type(L, ref);

    if (t >= L->M->num_types) return 0;
    if (L->M->types[t].kind == BIR_TYPE_STRUCT) return t;
    if (L->M->types[t].kind != BIR_TYPE_PTR
        || !(agglv(L, node) || aggv(L, node))) return 0;
    t = L->M->types[t].inner;
    return (t < L->M->num_types
            && L->M->types[t].kind == BIR_TYPE_STRUCT) ? t : 0;
}

static uint32_t sderef(lower_t *L, uint32_t ref, uint32_t st)
{
    uint32_t t = ref_type(L, ref), ld;

    if (!st || t >= L->M->num_types
        || L->M->types[t].kind != BIR_TYPE_PTR) return ref;
    ld = emit(L, BIR_LOAD, st, 1, 0);
    set_op(L, ld, 0, ref);
    return BIR_MAKE_VAL(ld);
}

static uint32_t saddr(lower_t *L, uint32_t ref, uint32_t st)
{
    uint32_t t = ref_type(L, ref), pt, al, sr;

    if (t < L->M->num_types && L->M->types[t].kind == BIR_TYPE_PTR) return ref;
    if (!st) st = t;
    if (!st || st >= L->M->num_types) return ref;
    if (!BIR_VAL_IS_CONST(ref) && BIR_VAL_INDEX(ref) < L->M->num_insts) {
        const bir_inst_t *LD = &L->M->insts[BIR_VAL_INDEX(ref)];

        if (LD->op == BIR_LOAD && LD->num_operands >= 1
            && ptr_inner(L, ref_type(L, LD->operands[0])) == st)
            return LD->operands[0];
    }
    pt = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
    al = emalc(L, BIR_ALLOCA, pt, 0);
    sr = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
    set_op(L, sr, 0, ref);
    set_op(L, sr, 1, BIR_MAKE_VAL(al));
    return BIR_MAKE_VAL(al);
}

static int vpk(const vecty_t *v)
{
    return v->bits == 16 && v->lanes == 2
        && (v->kind == VK_FLT || v->kind == VK_BF);
}

static int vaop(int tok)
{
    switch (tok) {
    case TOK_PLUS:  return BIR_FADD;
    case TOK_MINUS: return BIR_FSUB;
    case TOK_STAR:  return BIR_FMUL;
    case TOK_SLASH: return BIR_FDIV;
    default:        return -1;
    }
}

static const vecty_t *vpair(lower_t *L, uint32_t ls, uint32_t rs)
{
    const struct_def_t *sd;
    const vecty_t *v;

    if (!ls || ls != rs) return NULL;
    sd = sdbyt(L, ls);
    v  = sd ? vfind(sd->name) : NULL;
    return (v && vpk(v)) ? v : NULL;
}

static uint32_t vbin(lower_t *L, const struct_def_t *sd, uint16_t vop,
                     uint32_t a, uint32_t b)
{
    uint32_t lv[4];
    int n = sd->num_fields;

    if (n < 1 || n > 4) return BIR_VAL_NONE;
    for (int i = 0; i < n; i++) {
        uint32_t x = vlane(L, a, i), y = vlane(L, b, i), r;

        if (x == BIR_VAL_NONE || y == BIR_VAL_NONE) return BIR_VAL_NONE;
        r = emit(L, vop, sd->field_types[i], 2, 0);
        set_op(L, r, 0, x);
        set_op(L, r, 1, y);
        lv[i] = BIR_MAKE_VAL(r);
    }
    return vmake(L, sd, lv, n);
}

static uint32_t opsel(lower_t *L, const char *nm, int np,
                      uint32_t t0, uint32_t t1)
{
    char mng[BIR_SYM_MAX];
    int nm2 = L->tu != BIR_TU_EXT
           && bir_mang(nm, L->tu, mng, (int)sizeof mng) == 0;

    if (np < 1 || np > 2) return BIR_SYM_NONE;
    for (uint32_t i = 0; i < L->M->num_funcs; i++) {
        const bir_func_t *F = &L->M->funcs[i];
        const bir_type_t *T;
        const char *fn;

        if (F->name >= L->M->string_len
            || F->num_params - F->sret != (uint16_t)np)
            continue;
        fn = &L->M->strings[F->name];
        if (strcmp(fn, nm) != 0 && !(nm2 && strcmp(fn, mng) == 0)) continue;
        if (F->type >= L->M->num_types) continue;
        T = &L->M->types[F->type];
        if (T->num_fields - F->sret != (uint16_t)np) continue;
        if (T->count + (uint32_t)np > L->M->num_type_fields) continue;
        {
            uint32_t p0 = L->M->type_fields[T->count];
            uint32_t p1 = np == 2 ? L->M->type_fields[T->count + 1u] : 0;
            if ((F->refm & 1u) || aggr_t(L, ptr_inner(L, p0)))
                p0 = ptr_inner(L, p0);
            if ((F->refm & 2u) || aggr_t(L, ptr_inner(L, p1)))
                p1 = ptr_inner(L, p1);
            if (p0 != t0) continue;
            if (np == 2 && p1 != t1) continue;
        }
        return i;
    }
    return BIR_SYM_NONE;
}

static uint32_t ftpar(const lower_t *L, uint32_t fty, int i);

static uint32_t srmk(lower_t *L, uint32_t fi)
{
    const bir_type_t *T;
    uint32_t ft, pt, aty;

    if (fi >= L->M->num_funcs || !L->M->funcs[fi].sret) return BIR_VAL_NONE;
    ft = L->M->funcs[fi].type;
    if (ft >= L->M->num_types) return BIR_VAL_NONE;
    T = &L->M->types[ft];
    if (T->num_fields == 0
        || T->count > L->M->num_type_fields - T->num_fields)
        return BIR_VAL_NONE;
    pt = L->M->type_fields[T->count + T->num_fields - 1u];
    aty = (pt < L->M->num_types && L->M->types[pt].kind == BIR_TYPE_PTR)
        ? L->M->types[pt].inner : 0;
    if (!aty) return BIR_VAL_NONE;
    return BIR_MAKE_VAL(emalc(L, BIR_ALLOCA,
                bir_type_ptr(L->M, aty, BIR_AS_PRIVATE), 0));
}

static uint32_t aggop(lower_t *L, uint32_t node, int op,
                      uint32_t lhs, uint32_t ls, uint32_t rhs, uint32_t rs)
{
    const vecty_t *v = vpair(L, ls, rs);
    char onm[32];
    uint32_t fi, rt, inst;

    if (v && vaop(op) >= 0) {
        struct_def_t *sd = sdbyt(L, ls);
        uint32_t r = sd ? vbin(L, sd, (uint16_t)vaop(op), lhs, rhs)
                        : BIR_VAL_NONE;
        if (r != BIR_VAL_NONE) return r;
    }
    op_name_from_tok(op, onm, sizeof(onm));
    fi = opsel(L, onm, 2, ls ? ls : ref_type(L, lhs),
                          rs ? rs : ref_type(L, rhs));
    if (fi != BIR_SYM_NONE) {
        uint32_t rm = L->M->funcs[fi].refm;
        int pa0 = aggr_t(L, ptr_inner(L, ftpar(L, L->M->funcs[fi].type, 0)));
        int pa1 = aggr_t(L, ptr_inner(L, ftpar(L, L->M->funcs[fi].type, 1)));
        uint32_t a0 = ((rm & 1u) || pa0) ? saddr(L, lhs, ls)
                                         : sderef(L, lhs, ls);
        uint32_t a1 = ((rm & 2u) || pa1) ? saddr(L, rhs, rs)
                                         : sderef(L, rhs, rs);

        uint32_t ss = srmk(L, fi);

        rt   = L->M->types[L->M->funcs[fi].type].inner;
        inst = emit(L, BIR_CALL, rt, ss == BIR_VAL_NONE ? 3 : 4, 0);
        set_op(L, inst, 0, fi);
        set_op(L, inst, 1, a0);
        set_op(L, inst, 2, a1);
        if (ss == BIR_VAL_NONE) return BIR_MAKE_VAL(inst);
        set_op(L, inst, 3, ss);
        return ss;
    }
    lower_error(L, node, BC_E102);
    return BIR_VAL_NONE;
}

static uint32_t sdec(const char *src, uint32_t srclen,
                     char *bytes, uint32_t cap)
{
    /* Adjacent string literals are merged by the parser into a
     * single AST_STRING_LIT whose source span covers both. We
     * walk the span and decode each "..." segment, skipping
     * whitespace between them. */
    uint32_t blen = 0;
    uint32_t p = 0;

    while (p < srclen) {
        while (p < srclen && (src[p] == ' ' || src[p] == '\t' ||
                              src[p] == '\n' || src[p] == '\r')) p++;
        if (p >= srclen) break;
        if (src[p] != '"') { p++; continue; }
        p++;  /* opening quote */
        while (p < srclen && src[p] != '"' && blen + 1 < cap) {
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
    return blen;
}

static uint32_t strold(const lower_t *L, uint32_t arr, uint32_t off,
                       uint32_t len)
{
    const bir_module_t *M = L->M;
    uint32_t n = M->num_globals < 0x100u ? M->num_globals : 0x100u;

    if (off + len > M->string_len) return BIR_SYM_NONE;
    for (uint32_t i = 0; i < n; i++) {
        const bir_global_t *G = &M->globals[i];
        uint32_t ci;

        if (G->type != arr || G->tu != BIR_TU_EXT || !G->is_const) continue;
        if (G->cuda_flags != CUDA_CONSTANT) continue;
        if (!BIR_VAL_IS_CONST(G->initializer)) continue;
        ci = BIR_VAL_INDEX(G->initializer);
        if (ci >= M->num_consts) continue;
        if (M->consts[ci].kind != BIR_CONST_BYTES) continue;
        if (M->consts[ci].d.bytes.len != len) continue;
        if (memcmp(&M->strings[M->consts[ci].d.bytes.off],
                   &M->strings[off], len) != 0) continue;
        return i;
    }
    return BIR_SYM_NONE;
}

static uint32_t mkstr(lower_t *L, uint32_t node, const char *bytes,
                      uint32_t blen)
{
    uint32_t off = bir_add_string(L->M, bytes, blen);
    uint32_t i8  = bir_type_int(L->M, 8);
    uint32_t arr = bir_type_array(L->M, i8, blen + 1u);
    uint32_t ptr = bir_type_ptr(L->M, arr, BIR_AS_CONSTANT);
    bir_global_t *G;
    char gname[24];
    uint32_t gi, inst;

    gi = strold(L, arr, off, blen + 1u);
    if (gi == BIR_SYM_NONE) {
        if (L->M->num_globals >= BIR_MAX_GLOBALS) {
            lower_error(L, node, BC_E127, BIR_MAX_GLOBALS);
            return BIR_VAL_NONE;
        }
        gi = L->M->num_globals;
        L->M->num_globals++;
        G = &L->M->globals[gi];
        memset(G, 0, sizeof(*G));
        snprintf(gname, sizeof(gname), ".str.%u", gi);
        G->name = bir_add_string(L->M, gname, (uint32_t)strlen(gname));
        G->type = arr;
        G->initializer = BIR_MAKE_CONST(
            bir_const_bytes(L->M, arr, off, blen + 1u));
        G->cuda_flags = CUDA_CONSTANT;
        G->addrspace = BIR_AS_CONSTANT;
        G->is_const = 1;
        G->tu = BIR_TU_EXT;
    }

    inst = emit(L, BIR_GLOBAL_REF, ptr, 1, 0);
    set_op(L, inst, 0, gi);
    return BIR_MAKE_VAL(inst);
}

static uint32_t aggun(lower_t *L, uint32_t node, int op,
                      uint32_t val, uint32_t st)
{
    struct_def_t *sd = sdbyt(L, st);
    const vecty_t *v = sd ? vfind(sd->name) : NULL;
    char onm[32];
    uint32_t fi, rt, inst, a0;

    if (v && vpk(v) && op == TOK_MINUS) {
        uint32_t lv[4];
        int i, n = sd->num_fields;

        if (n > 4) n = 4;
        for (i = 0; i < n; i++) {
            uint32_t x = vlane(L, val, i), z, r;

            if (x == BIR_VAL_NONE) break;
            z = BIR_MAKE_CONST(bir_const_float(L->M, sd->field_types[i], 0.0));
            r = emit(L, BIR_FSUB, sd->field_types[i], 2, 0);
            set_op(L, r, 0, z);
            set_op(L, r, 1, x);
            lv[i] = BIR_MAKE_VAL(r);
        }
        if (i == n) return vmake(L, sd, lv, n);
    }
    op_name_from_tok(op, onm, sizeof(onm));
    fi = opsel(L, onm, 1, st, 0);
    if (fi != BIR_SYM_NONE) {
        uint32_t ss;

        a0   = ((L->M->funcs[fi].refm & 1u)
                || aggr_t(L, ptr_inner(L, ftpar(L, L->M->funcs[fi].type, 0))))
             ? saddr(L, val, st) : sderef(L, val, st);
        ss   = srmk(L, fi);
        rt   = L->M->types[L->M->funcs[fi].type].inner;
        inst = emit(L, BIR_CALL, rt, ss == BIR_VAL_NONE ? 2 : 3, 0);
        set_op(L, inst, 0, fi);
        set_op(L, inst, 1, a0);
        if (ss == BIR_VAL_NONE) return BIR_MAKE_VAL(inst);
        set_op(L, inst, 2, ss);
        return ss;
    }
    lower_error(L, node, BC_E103);
    return BIR_VAL_NONE;
}

static uint32_t lower_expr(lower_t *L, uint32_t node);

/* ---- Pack expansion ---- */

static int pk_scan(const lower_t *L, uint32_t node,
                   char out[][64], int max, int nfound, int depth)
{
    if (!node || depth > 64 || nfound >= max) return nfound;
    const ast_node_t *n = ND(L, node);
    if (n->type == AST_IDENT) {
        char nm[64];
        get_text(L, node, nm, sizeof(nm));
        if (pk_cnt(L, nm) >= 0) {
            for (int i = 0; i < nfound; i++)
                if (strcmp(out[i], nm) == 0) return nfound;
            snprintf(out[nfound++], 64, "%s", nm);
            return nfound;
        }
    }
    for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling)
        nfound = pk_scan(L, c, out, max, nfound, depth + 1);
    return nfound;
}

static int pk_len(lower_t *L, uint32_t pat, char nms[][64], int *nn,
                  int quiet)
{
    *nn = pk_scan(L, pat, nms, MAX_FPACKS, 0, 0);
    if (*nn == 0) return -1;
    int len = pk_cnt(L, nms[0]);
    for (int i = 1; i < *nn; i++)
        if (pk_cnt(L, nms[i]) != len) {
            if (!quiet)
                lower_error(L, pat, BC_E030,
                            "pack lengths differ in one pattern");
            return -1;
        }
    return len;
}

static uint32_t pk_at(lower_t *L, uint32_t pat, char nms[][64], int nn, int i)
{
    int sva = L->npkact, svb = L->nbindings;
    uint32_t v;

    if (!cpkb(L, nms, nn, i)) {
        L->npkact    = sva;
        L->nbindings = svb;
        lower_error(L, pat, BC_E030, "a pack element Booth cannot bind");
        return BIR_VAL_NONE;
    }
    v = lower_expr(L, pat);
    L->npkact    = sva;
    L->nbindings = svb;
    return v;
}

static int pk_exp(lower_t *L, uint32_t node, uint32_t *out, int max)
{
    uint32_t pat = ND(L, node)->first_child;
    char nms[MAX_FPACKS][64];
    int nn = 0;
    int len = pk_len(L, pat, nms, &nn, 0);
    if (len < 0) {
        lower_error(L, node, BC_E030, "pack expansion over an unbound pack");
        return -1;
    }
    if (len > max) {
        lower_error(L, node, BC_E030, "pack longer than the argument list");
        return -1;
    }
    for (int i = 0; i < len; i++)
        out[i] = pk_at(L, pat, nms, nn, i);
    return len;
}

static uint32_t lfold(lower_t *L, uint32_t node)
{
    const ast_node_t *n = ND(L, node);
    int op   = n->d.oper.op;
    int form = n->d.oper.flags;
    uint32_t a = n->first_child;
    uint32_t b = a ? ND(L, a)->next_sibling : 0;
    uint32_t pat, init = 0;
    int init_1st = 0;

    switch (form) {
    case FLD_UL: case FLD_UR: pat = a; break;
    case FLD_BL: pat = b; init = a; init_1st = 1; break;
    case FLD_BR: pat = a; init = b; break;
    default: lower_error(L, node, BC_E106); return BIR_VAL_NONE;
    }

    char nms[MAX_FPACKS][64];
    int nn = 0;
    int len = pk_len(L, pat, nms, &nn, 0);
    if (len < 0) {
        lower_error(L, node, BC_E030, "fold over an unbound pack");
        return BIR_VAL_NONE;
    }

    uint32_t t1 = bir_type_int(L->M, 1);
    if (op == TOK_LAND || op == TOK_LOR) {
        int is_and = (op == TOK_LAND);
        uint32_t pt = bir_type_ptr(L->M, t1, BIR_AS_PRIVATE);
        uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
        uint32_t seed = BIR_MAKE_CONST(bir_const_int(L->M, t1, is_and ? 0 : 1));
        uint32_t s0 = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, s0, 0, seed);
        set_op(L, s0, 1, BIR_MAKE_VAL(al));
        uint32_t end_b = new_block(L, "fold.end");
        int m = len + (init ? 1 : 0);
        for (int k = 0; k < m; k++) {
            uint32_t v;
            if (init && ((init_1st && k == 0) || (!init_1st && k == m - 1)))
                v = lower_expr(L, init);
            else
                v = pk_at(L, pat, nms, nn, init_1st && init ? k - 1 : k);
            uint32_t nxt = new_block(L, "fold.on");
            uint32_t br = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, v);
            set_op(L, br, 1, is_and ? nxt : end_b);
            set_op(L, br, 2, is_and ? end_b : nxt);
            set_op(L, br, 3, end_b);
            set_block(L, nxt);
        }
        uint32_t done = BIR_MAKE_CONST(bir_const_int(L->M, t1, is_and ? 1 : 0));
        uint32_t s1 = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, s1, 0, done);
        set_op(L, s1, 1, BIR_MAKE_VAL(al));
        uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
        set_op(L, j, 0, end_b);
        set_block(L, end_b);
        uint32_t ld = emit(L, BIR_LOAD, t1, 1, 0);
        set_op(L, ld, 0, BIR_MAKE_VAL(al));
        return BIR_MAKE_VAL(ld);
    }

    if (op == TOK_COMMA) {
        uint32_t last = BIR_VAL_NONE;
        int m = len + (init ? 1 : 0);
        for (int k = 0; k < m; k++) {
            if (init && ((init_1st && k == 0) || (!init_1st && k == m - 1)))
                last = lower_expr(L, init);
            else
                last = pk_at(L, pat, nms, nn, init_1st && init ? k - 1 : k);
        }
        return last;
    }

    if (len == 0) {
        if (init) return lower_expr(L, init);
        lower_error(L, node, BC_E030, "empty pack folded over this operator");
        return BIR_VAL_NONE;
    }
    if (len > MAX_PKELM) {
        lower_error(L, node, BC_E030, "fold longer than the pack limit");
        return BIR_VAL_NONE;
    }

    uint32_t v[MAX_PKELM];
    for (int i = 0; i < len; i++)
        v[i] = pk_at(L, pat, nms, nn, i);

    if (form == FLD_UL || form == FLD_BL) {
        uint32_t acc = init ? lower_expr(L, init) : v[0];
        uint32_t acc_n = init ? init : pat;
        for (int i = init ? 0 : 1; i < len; i++) {
            acc = bin_val(L, node, op, acc, acc_n, v[i], pat);
            acc_n = pat;
        }
        return acc;
    }
    uint32_t acc = init ? lower_expr(L, init) : v[len - 1];
    uint32_t acc_n = init ? init : pat;
    for (int i = init ? len - 1 : len - 2; i >= 0; i--) {
        acc = bin_val(L, node, op, v[i], pat, acc, acc_n);
        acc_n = pat;
    }
    return acc;
}
static uint32_t lower_lvalue(lower_t *L, uint32_t node);
static uint32_t mfld(lower_t *L, const char *fn);

static int smth(const lower_t *L, int si, const char *mn)
{
    uint32_t tn, m;
    char fn[64];

    if (si < 0 || si >= L->nstructs) return 0;
    tn = ND(L, L->structs[si].node)->first_child;
    if (!tn) return 0;
    for (m = ND(L, tn)->next_sibling; m; m = ND(L, m)->next_sibling) {
        uint32_t nn;
        if (ND(L, m)->type != AST_FUNC_DEF
            && ND(L, m)->type != AST_FUNC_DECL) continue;
        nn = child_at(L, m, 1);
        if (!nn || ND(L, nn)->type != AST_IDENT) continue;
        get_text(L, nn, fn, (int)sizeof fn);
        if (strcmp(fn, mn) != 0) continue;
        return (ND(L, m)->qualifiers & QUAL_STATIC) != 0;
    }
    return 0;
}

static int hasm(const lower_t *L, const char *fn)
{
    size_t k = strlen(fn);

    for (uint32_t i = 0; i < L->M->num_funcs; i++) {
        const char *s;
        const char *d;
        if (L->M->funcs[i].name >= L->M->string_len) continue;
        s = &L->M->strings[L->M->funcs[i].name];
        d = strrchr(s, '$');
        if (!d) continue;
        if (strncmp(d + 1, fn, k) == 0 && d[1 + k] == '\0') return 1;
    }
    return 0;
}
static void     lower_stmt(lower_t *L, uint32_t node);
static void     lower_block_stmts(lower_t *L, uint32_t node);
static void     collect_struct(lower_t *L, uint32_t node);
static void     collect_typedef(lower_t *L, uint32_t node);
static void     lower_func_body(lower_t *L, uint32_t func_def,
                                uint16_t cuda_flags, const char *name_override);

static int islv(lower_t *L, uint32_t node)
{
    const ast_node_t *n;
    char nm[128];
    const sym_t *sy;
    int g = CE_NEST;

    while (node && g-- > 0) {
        uint32_t u = unfw(L, node);
        if (u != node) { node = u; continue; }
        n = ND(L, node);
        if (n->type == AST_PAREN) { node = n->first_child; continue; }
        if (n->type == AST_CAST) {
            uint32_t t = n->first_child;
            node = t ? ND(L, t)->next_sibling : 0;
            continue;
        }
        break;
    }
    if (!node) return 0;
    n = ND(L, node);
    if (n->type == AST_SUBSCRIPT || n->type == AST_MEMBER) return 1;
    if (n->type == AST_UNARY_PREFIX) return n->d.oper.op == TOK_STAR;
    if (n->type != AST_IDENT) return 0;
    get_text(L, node, nm, sizeof(nm));
    pk_rw(L, nm, sizeof(nm));
    sy = find_sym(L, nm);
    return sy != NULL && sy->is_alloca;
}

static int rflat(const lower_t *L, uint32_t t)
{
    if (t == 0 || t >= L->M->num_types) return 0;
    return L->M->types[t].kind != BIR_TYPE_STRUCT
        && L->M->types[t].kind != BIR_TYPE_ARRAY;
}

static uint32_t rtmp(lower_t *L, uint32_t val, uint32_t want)
{
    uint32_t pt, al, sr;

    if (val == BIR_VAL_NONE) return val;
    val = coerce_to(L, val, want, 0);
    pt  = bir_type_ptr(L->M, want, BIR_AS_PRIVATE);
    al  = emalc(L, BIR_ALLOCA, pt, 0);
    sr  = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
    set_op(L, sr, 0, val);
    set_op(L, sr, 1, BIR_MAKE_VAL(al));
    return BIR_MAKE_VAL(al);
}

static uint32_t rbind(lower_t *L, uint32_t anode, uint32_t want,
                      int cst, const char *who)
{
    uint32_t a, at, ld;

    if (!want || want >= L->M->num_types) {
        lower_error(L, anode, BC_E149, who);
        return BIR_VAL_NONE;
    }
    if (islv(L, anode)) {
        a = lower_lvalue(L, anode);
        if (a == BIR_VAL_NONE) return a;
        at = ptr_inner(L, ref_type(L, a));
        if (at == want) return a;
        if (!cst || !rflat(L, at) || !rflat(L, want)) {
            lower_error(L, anode, BC_E149, who);
            return BIR_VAL_NONE;
        }
        ld = emit(L, BIR_LOAD, at, 1, 0);
        set_op(L, ld, 0, a);
        return rtmp(L, BIR_MAKE_VAL(ld), want);
    }
    if (!cst || !rflat(L, want)) {
        lower_error(L, anode, BC_E148, who);
        return BIR_VAL_NONE;
    }
    return rtmp(L, lower_expr(L, anode), want);
}

static uint32_t rlv(lower_t *L, uint32_t anode, int *lv)
{
    *lv = islv(L, anode);
    return *lv ? lower_lvalue(L, anode) : lower_expr(L, anode);
}

static uint32_t rfit(lower_t *L, uint32_t anode, uint32_t val, int lv,
                     uint32_t want, int cst, const char *who)
{
    uint32_t at, ld;

    if (val == BIR_VAL_NONE) return val;
    if (!want || want >= L->M->num_types) {
        lower_error(L, anode, BC_E149, who);
        return BIR_VAL_NONE;
    }
    if (!lv) {
        if (!cst || !rflat(L, want)) {
            lower_error(L, anode, BC_E148, who);
            return BIR_VAL_NONE;
        }
        return rtmp(L, val, want);
    }
    at = ptr_inner(L, ref_type(L, val));
    if (at == want) return val;
    if (!cst || !rflat(L, at) || !rflat(L, want)) {
        lower_error(L, anode, BC_E149, who);
        return BIR_VAL_NONE;
    }
    ld = emit(L, BIR_LOAD, at, 1, 0);
    set_op(L, ld, 0, val);
    return rtmp(L, BIR_MAKE_VAL(ld), want);
}

static uint32_t ftpar(const lower_t *L, uint32_t fty, int i)
{
    const bir_type_t *T;

    if (fty >= L->M->num_types || i < 0) return 0;
    T = &L->M->types[fty];
    if (i >= (int)T->num_fields) return 0;
    if (T->count + (uint32_t)i >= L->M->num_type_fields) return 0;
    return L->M->type_fields[T->count + (uint32_t)i];
}

static uint32_t pdfn(const lower_t *L, uint32_t p)
{
    uint32_t c = ND(L, p)->first_child, last = 0;
    while (c) { last = c; c = ND(L, c)->next_sibling; }
    return last;
}

static int fdflt(lower_t *L, const char *nm, int have, uint32_t *out, int max)
{
    for (int ti = 0; ti < L->ntlv; ti++) {
        uint32_t fd = L->tlv[ti], nn, pv[MAX_PARM];
        char fn[128];
        int np, k, n = 0;

        if (ND(L, fd)->type != AST_FUNC_DEF
            && ND(L, fd)->type != AST_FUNC_DECL) continue;
        nn = child_at(L, fd, 1);
        if (!nn || ND(L, nn)->type != AST_IDENT) continue;
        get_text(L, nn, fn, sizeof(fn));
        if (strcmp(fn, nm) != 0) continue;
        np = collect_params(L, fd, pv, MAX_PARM);
        if (np <= have || np - have > max) continue;
        for (k = have; k < np; k++) {
            uint32_t d;
            if (!(ND(L, pv[k])->qualifiers & QUAL_PDEF)) break;
            d = pdfn(L, pv[k]);
            if (!d) break;
            out[n++] = d;
        }
        if (k == np && n > 0) return n;
    }
    return 0;
}

static int fcand(lower_t *L, const char *nm, int np, uint32_t *out, int cap)
{
    char mng[BIR_SYM_MAX];
    int nm2 = L->tu != BIR_TU_EXT
           && bir_mang(nm, L->tu, mng, (int)sizeof mng) == 0;

    for (int pass = 0; pass < 2; pass++) {
        const char *want = nm2 && pass == 0 ? mng : nm;
        uint16_t wtu = (nm2 && pass == 0) ? L->tu : BIR_TU_EXT;
        int n = 0;

        if (pass == 0 && !nm2) continue;
        for (uint32_t i = 0; i < L->M->num_funcs; i++) {
            const bir_func_t *F = &L->M->funcs[i];
            if (F->tu != wtu || F->name >= L->M->string_len) continue;
            if (F->num_params - F->sret != (uint16_t)np) continue;
            if (strcmp(&L->M->strings[F->name], want) != 0) continue;
            if (n >= cap) return -1;
            out[n++] = i;
        }
        if (n > 0) return n;
    }
    return 0;
}

static int aggt(const lower_t *L, uint32_t t)
{
    uint8_t k;

    if (!t || t >= L->M->num_types) return 0;
    k = L->M->types[t].kind;
    return k == BIR_TYPE_STRUCT || k == BIR_TYPE_ARRAY;
}

static int aggv(const lower_t *L, uint32_t an)
{
    uint32_t st;

    if (!L->sema || !an || an >= BC_MAX_NODES) return 0;
    st = L->sema->node_types[an];
    if (!st || st >= L->sema->num_types) return 0;
    return L->sema->types[st].kind == STYPE_STRUCT
        || L->sema->types[st].kind == STYPE_ARRAY;
}

static int acost(const lower_t *L, uint32_t at, uint32_t pt)
{
    uint8_t ak, pk;
    int af, pf, ai, pi;

    if (at == pt) return 4;
    if (!at || at >= L->M->num_types || !pt || pt >= L->M->num_types) return 1;
    if (aggt(L, pt) && ptr_inner(L, at) == pt) return 3;
    if (ptr_inner(L, at) && ptr_inner(L, at) == ptr_inner(L, pt)) return 3;
    if (aggt(L, at) || aggt(L, pt)) return 0;
    ak = L->M->types[at].kind;
    pk = L->M->types[pt].kind;
    af = (ak == BIR_TYPE_FLOAT || ak == BIR_TYPE_BFLOAT);
    pf = (pk == BIR_TYPE_FLOAT || pk == BIR_TYPE_BFLOAT);
    ai = (ak == BIR_TYPE_INT);
    pi = (pk == BIR_TYPE_INT);
    if ((af && pf) || (ai && pi))
        return L->M->types[pt].width >= L->M->types[at].width ? 3 : 2;
    if (ai && pf) return 1;
    return 0;
}

static uint32_t fpick(lower_t *L, const uint32_t *cd, int nc,
                      const uint32_t *at, const uint32_t *an, int na, int *amb)
{
    int best = -1;
    uint32_t bi = BIR_SYM_NONE;

    *amb = 0;
    for (int i = 0; i < nc; i++) {
        uint32_t fty = L->M->funcs[cd[i]].type;
        uint32_t rm  = L->M->funcs[cd[i]].refm;
        int sc = 0;

        for (int a = 0; a < na; a++) {
            uint32_t pt = ftpar(L, fty, a);
            uint32_t ta = at[a];
            if (a < 32 && (rm & (1u << a))) pt = ptr_inner(L, pt);
            if (an && aggv(L, an[a])) {
                if (ptr_inner(L, ta)) ta = ptr_inner(L, ta);
                if (aggt(L, ptr_inner(L, pt))) pt = ptr_inner(L, pt);
            }
            sc += acost(L, ta, pt);
        }
        if (sc > best) { best = sc; bi = cd[i]; *amb = 0; }
        else if (sc == best) *amb = 1;
    }
    return bi;
}

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

static uint32_t cgdim(lower_t *L, uint16_t op)
{
    uint32_t i32 = bir_type_int(L->M, 32);
    uint32_t acc = BIR_VAL_NONE;

    for (uint8_t d = 0; d < 3; d++) {
        uint32_t v = BIR_MAKE_VAL(emit(L, op, i32, 0, d));
        uint32_t m;
        if (acc == BIR_VAL_NONE) { acc = v; continue; }
        m = emit(L, BIR_MUL, i32, 2, 0);
        set_op(L, m, 0, acc);
        set_op(L, m, 1, v);
        acc = BIR_MAKE_VAL(m);
    }
    return acc;
}

static uint32_t cgrnk(lower_t *L, uint16_t id, uint16_t dm)
{
    uint32_t i32 = bir_type_int(L->M, 32);
    uint32_t acc = BIR_MAKE_VAL(emit(L, id, i32, 0, 2));

    for (int d = 1; d >= 0; d--) {
        uint32_t sz = BIR_MAKE_VAL(emit(L, dm, i32, 0, (uint8_t)d));
        uint32_t ix = BIR_MAKE_VAL(emit(L, id, i32, 0, (uint8_t)d));
        uint32_t m = emit(L, BIR_MUL, i32, 2, 0);
        uint32_t a;
        set_op(L, m, 0, acc);
        set_op(L, m, 1, sz);
        a = emit(L, BIR_ADD, i32, 2, 0);
        set_op(L, a, 0, BIR_MAKE_VAL(m));
        set_op(L, a, 1, ix);
        acc = BIR_MAKE_VAL(a);
    }
    return acc;
}

static uint32_t cggrk(lower_t *L)
{
    uint32_t i32 = bir_type_int(L->M, 32);
    uint32_t br = cgrnk(L, BIR_BLOCK_ID, BIR_GRID_DIM);
    uint32_t nt = cgdim(L, BIR_BLOCK_DIM);
    uint32_t tr = cgrnk(L, BIR_THREAD_ID, BIR_BLOCK_DIM);
    uint32_t m = emit(L, BIR_MUL, i32, 2, 0);
    uint32_t a;

    set_op(L, m, 0, br);
    set_op(L, m, 1, nt);
    a = emit(L, BIR_ADD, i32, 2, 0);
    set_op(L, a, 0, BIR_MAKE_VAL(m));
    set_op(L, a, 1, tr);
    return BIR_MAKE_VAL(a);
}

static uint32_t cggsz(lower_t *L)
{
    uint32_t i32 = bir_type_int(L->M, 32);
    uint32_t nb = cgdim(L, BIR_GRID_DIM);
    uint32_t nt = cgdim(L, BIR_BLOCK_DIM);
    uint32_t m = emit(L, BIR_MUL, i32, 2, 0);

    set_op(L, m, 0, nb);
    set_op(L, m, 1, nt);
    return BIR_MAKE_VAL(m);
}

static int cgobj(const lower_t *L, uint32_t obj)
{
    char nm[64];

    if (!obj) return 1;
    if (ND(L, obj)->type == AST_IDENT) {
        get_text(L, obj, nm, (int)sizeof(nm));
        return cgisg(L, nm) ? 2 : 1;
    }
    if (ND(L, obj)->type == AST_CALL) {
        uint32_t cn = ND(L, obj)->first_child;
        if (cn && ND(L, cn)->type == AST_SCOPE_RES) {
            uint32_t ns = ND(L, cn)->first_child;
            uint32_t fn = ns ? ND(L, ns)->next_sibling : 0;
            if (fn && text_eq(L, fn, "this_grid")) return 2;
        }
    }
    return 1;
}

static uint32_t prntf(lower_t *L, uint32_t node, uint32_t callee_n)
{
    uint32_t args[BC_MAX_ARGS];
    uint32_t an = ND(L, callee_n)->next_sibling;
    uint32_t i32 = bir_type_int(L->M, 32);
    uint32_t inst, base;
    int nargs = 0;

    while (an && nargs < BC_MAX_ARGS) {
        if (ND(L, an)->type == AST_PACK_EXP) {
            int got = pk_exp(L, an, args + nargs, BC_MAX_ARGS - nargs);
            if (got < 0) return BIR_VAL_NONE;
            nargs += got;
        } else {
            args[nargs++] = lower_expr(L, an);
        }
        an = ND(L, an)->next_sibling;
    }
    if (an) {
        lower_error(L, node, BC_E082, "printf", BC_MAX_ARGS);
        return BIR_VAL_NONE;
    }
    if (nargs < 1) {
        lower_error(L, node, BC_E073, "printf", 1, nargs);
        return BIR_VAL_NONE;
    }

    if (nargs <= BIR_OPERANDS_INLINE) {
        inst = emit(L, BIR_PRINTF, i32, (uint8_t)nargs, 0);
        for (int i = 0; i < nargs; i++)
            set_op(L, inst, i, args[i]);
        return BIR_MAKE_VAL(inst);
    }

    base = L->M->num_extra_ops;
    if (base + (uint32_t)nargs > BIR_MAX_EXTRA_OPS) {
        bir_pfull(L->M, BIR_P_EXTRAOPS);
        return BIR_VAL_NONE;
    }
    for (int i = 0; i < nargs; i++)
        L->M->extra_operands[L->M->num_extra_ops++] = args[i];
    inst = emit(L, BIR_PRINTF, i32, BIR_OPERANDS_OVERFLOW, 0);
    set_op(L, inst, 0, base);
    set_op(L, inst, 1, (uint32_t)nargs);
    return BIR_MAKE_VAL(inst);
}

/* ---- Expression Lowering ---- */

static int lskp(int t)
{
    return t == AST_TYPE_SPEC || t == AST_SCOPE_RES || t == AST_PARAM
        || t == AST_LABEL || t == AST_GOTO || t == AST_TEMPLATE_ARGS;
}

static int lhas(char nm[][64], int n, const char *s)
{
    for (int i = 0; i < n; i++)
        if (strcmp(nm[i], s) == 0) return 1;
    return 0;
}

static int lglob(const lower_t *L, const sym_t *s, const char *nm)
{
    uint16_t op;

    if (!s->is_alloca || s->ref >= L->M->num_insts) return 0;
    op = L->M->insts[s->ref].op;
    if (op != BIR_GLOBAL_REF && op != BIR_SHARED_ALLOC) return 0;
    return bir_gsym(L->M, nm, L->tu) != BIR_SYM_NONE;
}

static int lretn(const lower_t *L, uint32_t body)
{
    uint32_t st[LM_NEST];
    int sp = 0;

    if (!body) return 0;
    st[sp++] = body;
    KA_GUARD(g, 1u << 16);
    while (sp > 0 && g--) {
        const ast_node_t *n = ND(L, st[--sp]);
        if (n->type == AST_LAMBDA) continue;
        if (n->type == AST_RETURN && n->first_child) return 1;
        for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling) {
            if (sp >= LM_NEST) return 1;
            st[sp++] = c;
        }
    }
    return 0;
}

static int lwalk(lower_t *L, uint32_t root, int excl, lam_t *m,
                 char ex[][64], int *nex)
{
    uint32_t st[LM_NEST];
    int sp = 0;

    if (!root) return 1;
    st[sp++] = root;
    KA_GUARD(g, 1u << 16);
    while (sp > 0 && g--) {
        uint32_t nd = st[--sp], sk = 0;
        const ast_node_t *n = ND(L, nd);
        char nm[64];

        if (n->type == AST_IDENT) {
            sym_t *s;
            if (excl) continue;
            get_text(L, nd, nm, sizeof nm);
            if (lhas(ex, *nex, nm)) continue;
            if (lhas(m->cnm, m->ncap, nm)) continue;
            s = find_sym(L, nm);
            if (!s || lglob(L, s, nm)) continue;
            if (m->ncap >= LM_CAP) return 0;
            ncpy(m->cnm[m->ncap], sizeof m->cnm[0], nm);
            m->ncap++;
            continue;
        }
        if (n->type == AST_MEMBER && n->first_child) {
            sk = ND(L, n->first_child)->next_sibling;
        } else if (n->type == AST_VAR_DECL) {
            for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling)
                if (ND(L, c)->type == AST_IDENT) { sk = c; break; }
            if (sk && excl) {
                if (*nex >= LM_EXCL) return 0;
                get_text(L, sk, ex[*nex], 64);
                (*nex)++;
            }
        }
        for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling) {
            if (c == sk || lskp(ND(L, c)->type)) continue;
            if (sp >= LM_NEST) return 0;
            st[sp++] = c;
        }
    }
    return 1;
}

static void lpnam(const lower_t *L, uint32_t pn, char *out, int oz)
{
    out[0] = 0;
    for (uint32_t c = ND(L, pn)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_IDENT) { get_text(L, c, out, oz); return; }
}

static int lparm(lower_t *L, uint32_t lam, lam_t *m, uint32_t c)
{
    const ast_node_t *pn = ND(L, c);
    uint32_t ts = pn->first_child;
    char base[64];
    uint32_t pb;

    if (pn->d.oper.op != 0 || m->np >= LM_PARM) {
        lower_error(L, lam, BC_E030,
                    "a lambda parameter list Booth cannot lower");
        return 0;
    }
    lpnam(L, c, base, (int)sizeof base);
    if (pn->qualifiers & QUAL_RREF) {
        lower_error(L, lam, BC_E146, base[0] ? base : "a lambda parameter");
        return 0;
    }
    if (ts && ND(L, ts)->type == AST_TYPE_SPEC
        && ND(L, ts)->d.btype.kind == TYPE_AUTO && !pn->d.oper.flags) {
        m->amsk |= 1u << m->np;
        m->gen = 1;
        pb = 0;
    } else {
        pb = rtype(L, ts, pn->d.oper.flags, pn->cuda_flags);
    }
    m->pbs[m->np] = pb;
    m->pty[m->np] = pb;
    if (pn->qualifiers & QUAL_REF) {
        if (pb) m->pty[m->np] = bir_type_ptr(L->M, pb, BIR_AS_GENERIC);
        m->rfm |= 1u << (m->np + 1);
        if (pn->qualifiers & QUAL_CONST) m->rfq |= 1u << (m->np + 1);
    }
    ncpy(m->pnm[m->np], sizeof m->pnm[0], base);
    m->np++;
    return 1;
}

static int lsig(lower_t *L, uint32_t lam, lam_t *m)
{
    const ast_node_t *ln = ND(L, lam);
    uint32_t rts = 0, body = 0;

    for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling) {
        int t = ND(L, c)->type;
        if (t == AST_BLOCK) body = c;
        else if (t == AST_TYPE_SPEC) rts = c;
    }
    if (rts) {
        m->ret = rtype(L, rts, ln->d.oper.flags >> LAM_PDSH, 0);
    } else if (lretn(L, body)) {
        lower_error(L, lam, BC_E030,
                    "a lambda whose return type Booth cannot deduce, write -> T");
        return 0;
    } else {
        m->ret = bir_type_void(L->M);
    }
    for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_PARAM && !lparm(L, lam, m, c)) return 0;
    return 1;
}

static int lcty(lower_t *L, uint32_t lam, lam_t *m)
{
    for (int i = 0; i < m->ncap; i++) {
        sym_t *s = find_sym(L, m->cnm[i]);
        int agg;
        if (!s) {
            lower_error(L, lam, BC_E261, m->cnm[i],
                        "it is not a visible local");
            return 0;
        }
        if (m->crf[i]) {
            if (!s->is_alloca) {
                lower_error(L, lam, BC_E261, m->cnm[i],
                            "its address cannot be taken");
                return 0;
            }
            m->cty[i] = ref_type(L, BIR_MAKE_VAL(s->ref));
            continue;
        }
        agg = s->type < L->M->num_types
              && (L->M->types[s->type].kind == BIR_TYPE_ARRAY
                  || L->M->types[s->type].kind == BIR_TYPE_STRUCT);
        if (agg) {
            lower_error(L, lam, BC_E261, m->cnm[i],
                        "an aggregate cannot be captured by value");
            return 0;
        }
        m->cty[i] = s->type;
    }
    return 1;
}

static int lcaps(lower_t *L, uint32_t lam, lam_t *m)
{
    const ast_node_t *ln = ND(L, lam);
    char (*ex)[64] = L->lex;
    uint32_t body = 0;
    int nex = 0, nexp;

    for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_BLOCK) body = c;

    for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling) {
        char nm[64];
        if (ND(L, c)->type != AST_IDENT) continue;
        get_text(L, c, nm, sizeof nm);
        if (lhas(m->cnm, m->ncap, nm)) continue;
        if (m->ncap >= LM_CAP) {
            lower_error(L, lam, BC_E030, "more captures than Booth lays out");
            return 0;
        }
        ncpy(m->cnm[m->ncap], sizeof m->cnm[0], nm);
        m->crf[m->ncap] = (uint8_t)((ND(L, c)->qualifiers & QUAL_REF) != 0);
        m->ncap++;
    }
    nexp = m->ncap;

    if (ln->d.oper.op) {
        for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling) {
            if (ND(L, c)->type != AST_PARAM) continue;
            if (nex >= LM_EXCL) {
                lower_error(L, lam, BC_E030,
                            "a lambda body wider than the capture scan");
                return 0;
            }
            lpnam(L, c, ex[nex], 64);
            if (ex[nex][0]) nex++;
        }
        if (!lwalk(L, body, 1, m, ex, &nex)
            || !lwalk(L, body, 0, m, ex, &nex)) {
            lower_error(L, lam, BC_E030,
                        "a lambda body wider than the capture scan");
            return 0;
        }
        for (int i = nexp; i < m->ncap; i++)
            m->crf[i] = (uint8_t)(ln->d.oper.op == TOK_AMP);
    }

    return lcty(L, lam, m);
}

static int lstor(lower_t *L, const lam_t *m, uint32_t cpt, uint32_t *aout)
{
    uint32_t al = emalc(L, BIR_ALLOCA, cpt, 0);
    uint32_t t32 = bir_type_int(L->M, 32);

    for (int i = 0; i < m->ncap; i++) {
        sym_t *s = find_sym(L, m->cnm[i]);
        uint32_t fpt, gep, ci, v, st;
        if (!s) return 0;
        fpt = bir_type_ptr(L->M, m->cty[i], BIR_AS_PRIVATE);
        ci = BIR_MAKE_CONST(bir_const_int(L->M, t32, i));
        gep = emit(L, BIR_GEP, fpt, 2, 0);
        set_op(L, gep, 0, BIR_MAKE_VAL(al));
        set_op(L, gep, 1, ci);
        if (m->crf[i] || !s->is_alloca) {
            v = m->crf[i] ? BIR_MAKE_VAL(s->ref) : s->ref;
        } else {
            uint32_t ld = emit(L, BIR_LOAD, m->cty[i], 1, 0);
            set_op(L, ld, 0, BIR_MAKE_VAL(s->ref));
            v = BIR_MAKE_VAL(ld);
        }
        st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, st, 0, v);
        set_op(L, st, 1, BIR_MAKE_VAL(gep));
    }
    *aout = al;
    return 1;
}

static int lfnew(lower_t *L, uint32_t lam, lam_t *m, int li, uint32_t *cpt)
{
    uint32_t pty[LM_PARM + 2];
    uint16_t flnk = BIR_TU_EXT;
    bir_func_t *F;
    int lsr;

    if (snprintf(m->name, sizeof m->name, "lam$%d", li) < 0) return 0;
    if (L->tu != BIR_TU_EXT) {
        char mng[BIR_SYM_MAX];
        if (bir_mang(m->name, L->tu, mng, (int)sizeof mng) != 0) {
            lower_error(L, lam, BC_E126, m->name);
            return 0;
        }
        ncpy(m->name, sizeof m->name, mng);
        flnk = L->tu;
    }
    if (L->M->num_funcs >= BIR_MAX_FUNCS) {
        bir_pfull(L->M, BIR_P_FUNCS);
        return 0;
    }

    pty[0] = bir_type_ptr(L->M, m->cst, BIR_AS_PRIVATE);
    for (int i = 0; i < m->np; i++) pty[i + 1] = m->pty[i];
    lsr = aggt(L, m->ret) ? 1 : 0;
    if (lsr) pty[m->np + 1] = bir_type_ptr(L->M, m->ret, BIR_AS_PRIVATE);

    m->fidx = L->M->num_funcs++;
    F = &L->M->funcs[m->fidx];
    memset(F, 0, sizeof *F);
    F->name = bir_add_string(L->M, m->name, (uint32_t)strlen(m->name));
    F->type = bir_type_func(L->M, lsr ? bir_type_void(L->M) : m->ret,
                            pty, m->np + 1 + lsr);
    F->tu = flnk;
    F->cuda_flags = CUDA_DEVICE;
    F->num_params = (uint16_t)(m->np + 1 + lsr);
    F->sret = (uint16_t)lsr;
    F->refm = m->rfm;
    F->refq = m->rfq;
    F->first_block = L->M->num_blocks;
    *cpt = pty[0];
    return 1;
}

static int tdsnp(lower_t *L, uint32_t lam, lam_t *m)
{
    int n = L->ntypedefs - L->ntdf;

    m->tdo = L->ntdsn;
    m->tdn = 0;
    if (n <= 0) return 1;
    if (L->ntdsn + n > MAX_TDSN) {
        lower_error(L, lam, BC_E030,
                    "more local typedefs in lambdas than Booth holds");
        return 0;
    }
    for (int i = 0; i < n; i++)
        L->tdsn[L->ntdsn + i] = L->typedefs[L->ntdf + i];
    L->ntdsn += n;
    m->tdn = n;
    return 1;
}

static int tdput(lower_t *L, const lam_t *m)
{
    if (m->tdn <= 0) return 1;
    if (L->ntypedefs + m->tdn > MAX_TYPEDEFS) {
        lower_error(L, m->ast, BC_E030, "more typedefs than Booth holds");
        return 0;
    }
    for (int i = 0; i < m->tdn; i++)
        L->typedefs[L->ntypedefs++] = L->tdsn[m->tdo + i];
    return 1;
}

static int lmake(lower_t *L, uint32_t lam, uint32_t *aout)
{
    const ast_node_t *ln = ND(L, lam);
    uint32_t cpt = 0, fld[LM_CAP];
    lam_t *m;
    int li;

    if (L->nlam >= MAX_LAMS) {
        lower_error(L, lam, BC_E030, "more lambdas than Booth lowers");
        return -1;
    }
    if (ln->d.oper.flags & LAM_BADC) {
        lower_error(L, lam, BC_E030,
                    "a lambda capture list that is not a plain list of names");
        return -1;
    }
    if (ln->d.oper.flags & LAM_RREF) {
        lower_error(L, lam, BC_E030, "a lambda returning a reference");
        return -1;
    }

    li = L->nlam;
    m = &L->lams[li];
    memset(m, 0, sizeof *m);
    m->ast = lam;

    if (!tdsnp(L, lam, m)) return -1;
    if (!lsig(L, lam, m) || !lcaps(L, lam, m)) return -1;

    fld[0] = m->ncap ? m->cty[0] : bir_type_int(L->M, 8);
    for (int i = 1; i < m->ncap; i++) fld[i] = m->cty[i];
    m->cst = bir_sfwd(L->M);
    if (!m->cst || !bir_sfin(L->M, m->cst, fld, m->ncap ? m->ncap : 1)) {
        lower_error(L, lam, BC_E030,
                    "a closure the type table has no room for");
        return -1;
    }
    cpt = bir_type_ptr(L->M, m->cst, BIR_AS_PRIVATE);
    if (m->gen) {
        m->done = 1;
        if (!lstor(L, m, cpt, aout)) return -1;
        L->nlam++;
        return li;
    }
    if (!lfnew(L, lam, m, li, &cpt)) return -1;
    if (!lstor(L, m, cpt, aout)) return -1;
    L->nlam++;
    return li;
}

static int lfind(lower_t *L, uint32_t node, uint32_t *cptr)
{
    const ast_node_t *n;
    uint32_t nd = node;

    KA_GUARD(g, 8);
    while (g-- && nd && ND(L, nd)->type == AST_PAREN)
        nd = ND(L, nd)->first_child;
    if (!nd) return -1;
    n = ND(L, nd);
    if (n->type == AST_LAMBDA) {
        uint32_t al = 0;
        int li = lmake(L, nd, &al);
        if (li < 0) return -2;
        *cptr = BIR_MAKE_VAL(al);
        return li;
    }
    if (n->type == AST_IDENT) {
        char nm[128];
        sym_t *s;
        get_text(L, nd, nm, sizeof nm);
        s = find_sym(L, nm);
        if (s && s->lam > 0 && s->is_alloca) {
            *cptr = BIR_MAKE_VAL(s->ref);
            return s->lam - 1;
        }
    }
    return -1;
}

static int lgsame(const lam_t *c, const uint32_t *pt, const int64_t *tv,
                  uint32_t tm)
{
    if (c->tmsk != tm) return 0;
    for (int i = 0; i < c->np; i++) {
        if (c->pty[i] != pt[i]) return 0;
        if ((tm & (1u << i)) && c->tval[i] != tv[i]) return 0;
    }
    return 1;
}

static int lgmake(lower_t *L, uint32_t node, int gi, const uint32_t *pt,
                  const int64_t *tv, uint32_t tm)
{
    lam_t *g = &L->lams[gi];
    lam_t *c;
    uint32_t cpt = 0;
    int li;

    for (int i = 0; i < L->nlam; i++)
        if (L->lams[i].gsrc == gi + 1 && lgsame(&L->lams[i], pt, tv, tm))
            return i;

    if (L->nlam >= MAX_LAMS) {
        lower_error(L, node, BC_E030, "more lambdas than Booth lowers");
        return -1;
    }
    li = L->nlam;
    c = &L->lams[li];
    *c = *g;
    c->gen  = 0;
    c->done = 0;
    c->gsrc = gi + 1;
    c->tmsk = tm;
    for (int i = 0; i < c->np; i++) {
        c->pty[i] = pt[i];
        c->tval[i] = tv[i];
        if (!(c->amsk & (1u << i))) continue;
        if (tm & (1u << i)) {
            c->rfm &= ~(1u << (i + 1));
            c->rfq &= ~(1u << (i + 1));
            c->pbs[i] = pt[i];
            continue;
        }
        c->pbs[i] = (c->rfm & (1u << (i + 1))) ? pt[i] : 0;
        if (c->rfm & (1u << (i + 1)))
            c->pty[i] = bir_type_ptr(L->M, pt[i], BIR_AS_GENERIC);
    }
    if (!lfnew(L, c->ast, c, li, &cpt)) return -1;
    L->nlam++;
    return li;
}

static int lgdeduc(lower_t *L, uint32_t node, lam_t *m, uint32_t arg,
                   uint32_t *an, uint32_t *av, uint32_t *at, int64_t *tv,
                   uint32_t *tmo, const char *who)
{
    uint32_t a = arg, tm = 0;
    int na = 0;

    while (a && na < LM_PARM) {
        int64_t v;
        an[na] = a;
        tv[na] = 0;
        av[na] = BIR_VAL_NONE;
        if ((m->amsk & (1u << na)) && tgval(L, a, &v)) {
            tm |= 1u << na;
            tv[na] = v;
            at[na] = bir_type_int(L->M, 8);
            av[na] = BIR_MAKE_CONST(bir_const_int(L->M, at[na], 0));
        } else if (m->rfm & (1u << (na + 1))) {
            at[na] = (m->amsk & (1u << na)) ? dcty(L, a) : m->pbs[na];
            if (!at[na]) {
                lower_error(L, node, BC_E260,
                            m->pnm[na][0] ? m->pnm[na] : "auto");
                return -1;
            }
        } else if (m->amsk & (1u << na)) {
            av[na] = lower_expr(L, a);
            at[na] = ref_type(L, av[na]);
            if (av[na] == BIR_VAL_NONE || !at[na]) {
                lower_error(L, node, BC_E260,
                            m->pnm[na][0] ? m->pnm[na] : "auto");
                return -1;
            }
        } else {
            at[na] = m->pty[na];
            av[na] = lower_expr(L, a);
            if (av[na] == BIR_VAL_NONE) return -1;
        }
        na++;
        a = ND(L, a)->next_sibling;
    }
    if (a || na != m->np) {
        int got = na;
        for (; a; a = ND(L, a)->next_sibling) got++;
        lower_error(L, node, BC_E073, who, m->np, got);
        return -1;
    }
    *tmo = tm;
    return na;
}

static int lgcall(lower_t *L, uint32_t node, int li, uint32_t arg,
                  uint32_t *args, const char *who)
{
    lam_t *m = &L->lams[li];
    uint32_t an[LM_PARM], av[LM_PARM], at[LM_PARM], tm = 0;
    int64_t tv[LM_PARM];
    int na, ni;

    na = lgdeduc(L, node, m, arg, an, av, at, tv, &tm, who);
    if (na < 0) return -1;
    ni = lgmake(L, node, li, at, tv, tm);
    if (ni < 0) return -1;
    m = &L->lams[ni];
    for (int i = 0; i < na; i++) {
        if (av[i] != BIR_VAL_NONE)
            args[i] = stfit(L, av[i], m->pty[i], an[i]);
        else
            args[i] = rbind(L, an[i], m->pbs[i],
                            (int)((m->rfq >> (i + 1)) & 1u), who);
        if (args[i] == BIR_VAL_NONE) return -1;
    }
    return ni;
}

static uint32_t lcall(lower_t *L, uint32_t fidx, uint32_t cptr,
                      const uint32_t *args, int na, uint32_t rt)
{
    uint32_t es, inst, tot;

    if (2 + na <= BIR_OPERANDS_INLINE) {
        inst = emit(L, BIR_CALL, rt, (uint8_t)(2 + na), 0);
        set_op(L, inst, 0, fidx);
        set_op(L, inst, 1, cptr);
        for (int i = 0; i < na; i++) set_op(L, inst, 2 + i, args[i]);
        return BIR_MAKE_VAL(inst);
    }
    es = L->M->num_extra_ops;
    if (es + 2u + (uint32_t)na > BIR_MAX_EXTRA_OPS) {
        bir_pfull(L->M, BIR_P_EXTRAOPS);
        return BIR_VAL_NONE;
    }
    L->M->extra_operands[L->M->num_extra_ops++] = fidx;
    L->M->extra_operands[L->M->num_extra_ops++] = cptr;
    for (int i = 0; i < na; i++)
        L->M->extra_operands[L->M->num_extra_ops++] = args[i];
    tot = L->M->num_extra_ops - es;
    inst = emit(L, BIR_CALL, rt, BIR_OPERANDS_OVERFLOW, 0);
    set_op(L, inst, 0, es);
    set_op(L, inst, 1, tot);
    return BIR_MAKE_VAL(inst);
}

static uint32_t lemit(lower_t *L, uint32_t node, int li, uint32_t cptr,
                      uint32_t arg)
{
    lam_t *m = &L->lams[li];
    uint32_t args[LM_PARM], cn = ND(L, node)->first_child;
    char nm[128];
    int na = 0;

    ncpy(nm, sizeof nm, "the lambda");
    while (cn && ND(L, cn)->type == AST_PAREN) cn = ND(L, cn)->first_child;
    if (cn && ND(L, cn)->type == AST_IDENT) get_text(L, cn, nm, sizeof nm);

    if (m->gen) {
        int ni = lgcall(L, node, li, arg, args, nm);
        if (ni < 0) return BIR_VAL_NONE;
        m = &L->lams[ni];
        na = m->np;
        arg = 0;
    }

    while (arg && na < LM_PARM) {
        if (m->rfm & (1u << (na + 1)))
            args[na] = rbind(L, arg, m->pbs[na],
                             (int)((m->rfq >> (na + 1)) & 1u), nm);
        else
            args[na] = stfit(L, lower_expr(L, arg), m->pty[na], arg);
        if (args[na] == BIR_VAL_NONE) return BIR_VAL_NONE;
        na++;
        arg = ND(L, arg)->next_sibling;
    }
    if (arg || na != m->np) {
        int got = na;
        for (uint32_t a = arg; a; a = ND(L, a)->next_sibling) got++;
        lower_error(L, node, BC_E073, nm, m->np, got);
        return BIR_VAL_NONE;
    }
    uint32_t lsl = srmk(L, m->fidx);
    uint32_t crt = lsl == BIR_VAL_NONE ? m->ret : bir_type_void(L->M);
    uint32_t ci;

    if (lsl != BIR_VAL_NONE) {
        if (na >= LM_PARM) return BIR_VAL_NONE;
        args[na++] = lsl;
    }
    ci = lcall(L, m->fidx, cptr, args, na, crt);
    if (ci == BIR_VAL_NONE) return BIR_VAL_NONE;
    return lsl != BIR_VAL_NONE ? lsl : ci;
}

static void lprol(lower_t *L, const lam_t *m, uint32_t cpt)
{
    uint32_t pin[LM_PARM], t32 = bir_type_int(L->M, 32);
    uint32_t cp = emit(L, BIR_PARAM, cpt, 0, 0);

    for (int i = 0; i < m->np; i++)
        pin[i] = emit(L, BIR_PARAM, m->pty[i], 0, (uint8_t)(i + 1));
    if (m->fidx < L->M->num_funcs && L->M->funcs[m->fidx].sret)
        emit(L, BIR_PARAM, bir_type_ptr(L->M, m->ret, BIR_AS_PRIVATE), 0,
             (uint8_t)(m->np + 1));
    for (int i = 0; i < m->ncap; i++) {
        uint32_t fpt = bir_type_ptr(L->M, m->cty[i], BIR_AS_PRIVATE);
        uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M, t32, i));
        uint32_t gep = emit(L, BIR_GEP, fpt, 2, 0);
        set_op(L, gep, 0, BIR_MAKE_VAL(cp));
        set_op(L, gep, 1, ci);
        if (m->crf[i]) {
            uint32_t ld = emit(L, BIR_LOAD, m->cty[i], 1, 0);
            set_op(L, ld, 0, BIR_MAKE_VAL(gep));
            add_sym(L, m->cnm[i], ld, ptr_inner(L, m->cty[i]), 1);
        } else {
            add_sym(L, m->cnm[i], gep, m->cty[i], 1);
        }
    }
    for (int i = 0; i < m->np; i++) {
        uint32_t pt, al, st;
        if (!m->pnm[i][0] || (m->tmsk & (1u << i))) continue;
        if (m->rfm & (1u << (i + 1))) {
            add_sym(L, m->pnm[i], pin[i], m->pbs[i], 1);
            continue;
        }
        pt = bir_type_ptr(L->M, m->pty[i], BIR_AS_PRIVATE);
        al = emalc(L, BIR_ALLOCA, pt, 0);
        st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, st, 0, BIR_MAKE_VAL(pin[i]));
        set_op(L, st, 1, BIR_MAKE_VAL(al));
        add_sym(L, m->pnm[i], al, m->pty[i], 1);
    }
}

static void lambody(lower_t *L, int li)
{
    lam_t *m = &L->lams[li];
    const ast_node_t *ln = ND(L, m->ast);
    bir_func_t *F = &L->M->funcs[m->fidx];
    uint32_t body = 0, entry;
    uint32_t cpt = bir_type_ptr(L->M, m->cst, BIR_AS_PRIVATE);
    int tsv;

    for (uint32_t c = ln->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_BLOCK) body = c;

    ncpy(L->fnnm, sizeof L->fnnm, m->name);
    L->cur_func = m->fidx;
    L->cur_node = m->ast;
    L->nlabels = 0;
    F->first_block = L->M->num_blocks;
    F->num_blocks = 0;
    entry = new_block(L, "entry");
    set_block(L, entry);
    L->base_inst = L->M->num_insts;

    push_scope(L);
    (void)tdput(L, m);
    lprol(L, m, cpt);
    tsv = L->ntags;
    for (int i = 0; i < m->np; i++)
        if ((m->tmsk & (1u << i)) && !tgadd(L, m->pnm[i], m->tval[i]))
            lower_error(L, m->ast, BC_E030,
                        "more lambda tag bindings than Booth holds");
    if (body) {
        push_scope(L);
        lower_block_stmts(L, body);
        pop_scope(L);
    }
    L->ntags = tsv;
    if (!block_terminated(L)) {
        if (is_void_type(L, m->ret))
            emit(L, BIR_RET, bir_type_void(L->M), 0, 0);
        else
            emit(L, BIR_UNREACHABLE, bir_type_void(L->M), 0, 0);
    }
    pop_scope(L);
    F->total_insts = L->M->num_insts - L->base_inst;
}

static void ldrain(lower_t *L)
{
    if (L->ldrn) return;
    L->ldrn = 1;
    KA_GUARD(g, MAX_LAMS + 1u);
    while (g--) {
        int li = -1;
        for (int i = 0; i < L->nlam; i++)
            if (!L->lams[i].done) { li = i; break; }
        if (li < 0) break;
        L->lams[li].done = 1;
        lambody(L, li);
    }
    L->ldrn = 0;
}


static uint32_t vki(lower_t *L, int64_t k)
{
    return BIR_MAKE_CONST(bir_const_int(L->M, bir_type_int(L->M, 32), k));
}

static uint32_t wbin(lower_t *L, uint16_t op, uint32_t a, uint32_t b)
{
    uint32_t r = emit(L, op, bir_type_int(L->M, 32), 2, 0);
    set_op(L, r, 0, a);
    set_op(L, r, 1, b);
    return BIR_MAKE_VAL(r);
}

static uint32_t vsel(lower_t *L, uint32_t c, uint32_t t, uint32_t f)
{
    uint32_t r = emit(L, BIR_SELECT, bir_type_int(L->M, 32), 3, 0);
    set_op(L, r, 0, c);
    set_op(L, r, 1, t);
    set_op(L, r, 2, f);
    return BIR_MAKE_VAL(r);
}

static uint32_t vcmp(lower_t *L, uint8_t pr, uint32_t a, uint32_t b)
{
    uint32_t r = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2, pr);
    set_op(L, r, 0, a);
    set_op(L, r, 1, b);
    return BIR_MAKE_VAL(r);
}

static uint32_t fkf(lower_t *L, uint32_t t, double k)
{
    return BIR_MAKE_CONST(bir_const_float(L->M, t, k));
}

static uint32_t fbin(lower_t *L, uint16_t op, uint32_t t, uint32_t a,
                     uint32_t b)
{
    uint32_t r = emit(L, op, t, 2, 0);
    set_op(L, r, 0, a);
    set_op(L, r, 1, b);
    return BIR_MAKE_VAL(r);
}

static uint32_t vbyte(lower_t *L, uint32_t v, int i, int sgn)
{
    uint32_t sh = wbin(L, BIR_SHL, v, vki(L, 24 - 8 * i));
    return wbin(L, sgn ? BIR_ASHR : BIR_LSHR, sh, vki(L, 24));
}

static uint32_t vclmp(lower_t *L, uint32_t v, int64_t lo, int64_t hi)
{
    uint32_t c = vcmp(L, BIR_ICMP_SLT, v, vki(L, lo));
    v = vsel(L, c, vki(L, lo), v);
    c = vcmp(L, BIR_ICMP_SGT, v, vki(L, hi));
    return vsel(L, c, vki(L, hi), v);
}

static uint32_t vpack(lower_t *L, const uint32_t *b)
{
    uint32_t acc = vki(L, 0);
    for (int i = 0; i < 4; i++) {
        uint32_t m = wbin(L, BIR_AND, b[i], vki(L, 255));
        acc = wbin(L, BIR_OR, acc, wbin(L, BIR_SHL, m, vki(L, 8 * i)));
    }
    return acc;
}

static uint32_t vprmt(lower_t *L, uint32_t x, uint32_t y, uint32_t s)
{
    uint32_t b[4];
    for (int i = 0; i < 4; i++) {
        uint32_t nb  = wbin(L, BIR_AND, wbin(L, BIR_LSHR, s, vki(L, 4 * i)),
                            vki(L, 15));
        uint32_t idx = wbin(L, BIR_AND, nb, vki(L, 7));
        uint32_t src = vsel(L, vcmp(L, BIR_ICMP_ULT, idx, vki(L, 4)), x, y);
        uint32_t byt = wbin(L, BIR_AND,
                            wbin(L, BIR_LSHR, src,
                                 wbin(L, BIR_SHL,
                                      wbin(L, BIR_AND, idx, vki(L, 3)),
                                      vki(L, 3))),
                            vki(L, 255));
        uint32_t rep = vsel(L, vcmp(L, BIR_ICMP_UGE, byt, vki(L, 128)),
                            vki(L, 255), vki(L, 0));
        b[i] = vsel(L, vcmp(L, BIR_ICMP_NE, wbin(L, BIR_AND, nb, vki(L, 8)),
                            vki(L, 0)), rep, byt);
    }
    return vpack(L, b);
}

enum { VW_WRAP, VW_SSAT, VW_USAT, VW_EQ, VW_NE };

static uint32_t vsimd(lower_t *L, int mode, int sub, uint32_t a, uint32_t b)
{
    uint32_t o[4];
    int sgn = mode == VW_SSAT;
    for (int i = 0; i < 4; i++) {
        uint32_t x = vbyte(L, a, i, sgn);
        uint32_t y = vbyte(L, b, i, sgn);
        switch (mode) {
        case VW_EQ:
        case VW_NE:
            o[i] = vsel(L, vcmp(L, mode == VW_EQ ? BIR_ICMP_EQ : BIR_ICMP_NE,
                                x, y), vki(L, 255), vki(L, 0));
            break;
        default: {
            uint32_t r = wbin(L, sub ? BIR_SUB : BIR_ADD, x, y);
            if (mode == VW_SSAT)      r = vclmp(L, r, -128, 127);
            else if (mode == VW_USAT) r = vclmp(L, r, 0, 255);
            o[i] = r;
            break;
        }
        }
    }
    return vpack(L, o);
}

static uint32_t mtmp(lower_t *L, uint32_t node)
{
    uint32_t rv, rt, rp, al, sv;

    if (ND(L, node)->type != AST_CALL) return lower_lvalue(L, node);
    rv = lower_expr(L, node);
    if (rv == BIR_VAL_NONE) return BIR_VAL_NONE;
    rt = ref_type(L, rv);
    if (rt < L->M->num_types && L->M->types[rt].kind == BIR_TYPE_PTR)
        return rv;
    rp = bir_type_ptr(L->M, rt, BIR_AS_PRIVATE);
    al = emalc(L, BIR_ALLOCA, rp, 0);
    sv = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
    set_op(L, sv, 0, rv);
    set_op(L, sv, 1, BIR_MAKE_VAL(al));
    return BIR_MAKE_VAL(al);
}

#define AC_DEEP 8
#define AC_LEAF 512

static int acscl(const lower_t *L, uint32_t t)
{
    uint8_t k;

    if (t >= L->M->num_types) return 0;
    k = L->M->types[t].kind;
    return k == BIR_TYPE_INT || k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT
        || k == BIR_TYPE_PTR || k == BIR_TYPE_VECTOR;
}

static uint32_t acgep(lower_t *L, uint32_t base, uint32_t ft, uint8_t as,
                      int idx)
{
    uint32_t pt = bir_type_ptr(L->M, ft, as);
    uint32_t g  = emit(L, BIR_GEP, pt, 2, 0);

    set_op(L, g, 0, base);
    set_op(L, g, 1, BIR_MAKE_CONST(bir_const_int(L->M,
                                   bir_type_int(L->M, 32), idx)));
    return BIR_MAKE_VAL(g);
}

static int acwlk(lower_t *L, uint32_t dst, uint32_t src, uint32_t ty,
                 uint8_t das, uint8_t sas, int deep, int *nl)
{
    const bir_type_t *T;
    uint32_t i, n, fs, best = 0, bsz = 0;

    if (ty >= L->M->num_types || deep > AC_DEEP || *nl >= AC_LEAF) return 0;
    T = &L->M->types[ty];

    if (acscl(L, ty)) {
        uint32_t ld = emit(L, BIR_LOAD, ty, 1, 0);
        uint32_t st;

        set_op(L, ld, 0, src);
        st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, st, 0, BIR_MAKE_VAL(ld));
        set_op(L, st, 1, dst);
        (*nl)++;
        return 1;
    }
    if (T->kind == BIR_TYPE_ARRAY) {
        uint32_t et = T->inner;

        for (i = 0; i < T->count; i++)
            if (!acwlk(L, acgep(L, dst, et, das, (int)i),
                       acgep(L, src, et, sas, (int)i),
                       et, das, sas, deep + 1, nl)) return 0;
        return 1;
    }
    if (T->kind != BIR_TYPE_STRUCT) return 0;
    n  = T->num_fields;
    fs = T->count;
    if (n == 0) return 1;
    if (fs > L->M->num_type_fields - n) return 0;
    if (T->uni) {
        for (i = 0; i < n; i++) {
            uint32_t z = bir_bsz(L->M, L->M->type_fields[fs + i], 8);
            if (z > bsz) { bsz = z; best = i; }
        }
        n = best + 1u;
        i = best;
    } else {
        i = 0;
    }
    for (; i < n; i++) {
        uint32_t ft = L->M->type_fields[fs + i];

        if (!acwlk(L, acgep(L, dst, ft, das, (int)i),
                   acgep(L, src, ft, sas, (int)i),
                   ft, das, sas, deep + 1, nl)) return 0;
    }
    return 1;
}

static int acpy(lower_t *L, uint32_t node, uint32_t dst, uint32_t src,
                uint32_t ty)
{
    uint32_t dt = ref_type(L, dst), st = ref_type(L, src);
    char nm[80];
    int nl = 0;

    if (dst == BIR_VAL_NONE || src == BIR_VAL_NONE) return 0;
    if (!is_ptr_type(L, dt) || !is_ptr_type(L, st)) return 0;
    if (acwlk(L, dst, src, ty, L->M->types[dt].addrspace,
              L->M->types[st].addrspace, 0, &nl))
        return 1;
    if (bir_type_str(L->M, ty, nm, (int)sizeof nm) < 0) nm[0] = '\0';
    lower_error(L, node, BC_E700, nm, AC_LEAF, AC_DEEP);
    return 0;
}

static uint32_t apld(const lower_t *L, uint32_t ref, uint32_t ty)
{
    uint32_t i;

    if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref)) return BIR_VAL_NONE;
    i = BIR_VAL_INDEX(ref);
    if (i >= L->M->num_insts || L->M->insts[i].op != BIR_LOAD
        || L->M->insts[i].num_operands < 1) return BIR_VAL_NONE;
    return ptr_inner(L, ref_type(L, L->M->insts[i].operands[0])) == ty
         ? L->M->insts[i].operands[0] : BIR_VAL_NONE;
}

static uint32_t apsrc(lower_t *L, uint32_t node, uint32_t ref, uint32_t ty)
{
    uint32_t rt = ref_type(L, ref), p;
    char nm[80];

    if (ref == BIR_VAL_NONE) return ref;
    if (is_ptr_type(L, rt)) return ref;
    p = apld(L, ref, ty);
    if (p != BIR_VAL_NONE) return p;
    if (bir_type_str(L->M, ty, nm, (int)sizeof nm) < 0) nm[0] = '\0';
    lower_error(L, node, BC_E701, nm);
    return BIR_VAL_NONE;
}

static int pnam(lower_t *L, uint32_t node, const char *nm)
{
    char b[CE_NAMEZ];
    int g = CE_NEST;

    while (node && g-- > 0) {
        uint16_t k = ND(L, node)->type;

        if (k != AST_PAREN && k != AST_MEMBER && k != AST_SUBSCRIPT) break;
        node = ND(L, node)->first_child;
    }
    if (!node || ND(L, node)->type != AST_IDENT) return 0;
    get_text(L, node, b, sizeof b);
    return strcmp(b, nm) == 0;
}

static int pmut(lower_t *L, uint32_t node, const char *nm, int depth)
{
    const ast_node_t *n;
    uint32_t c;

    if (!node) return 0;
    if (depth > 48) return 1;
    n = ND(L, node);
    switch (n->type) {
    case AST_BINARY:
        if ((n->d.oper.op == TOK_ASSIGN || is_compound_assign(n->d.oper.op))
            && pnam(L, n->first_child, nm)) return 1;
        break;
    case AST_UNARY_PREFIX:
    case AST_UNARY_POSTFIX:
        if ((n->d.oper.op == TOK_AMP || n->d.oper.op == TOK_INC
             || n->d.oper.op == TOK_DEC) && pnam(L, n->first_child, nm))
            return 1;
        break;
    case AST_CALL:
    case AST_LAUNCH:
    case AST_NEW:
        for (c = n->first_child; c; c = ND(L, c)->next_sibling)
            if (pnam(L, c, nm)) return 1;
        break;
    case AST_ASM:
        return 1;
    default:
        break;
    }
    for (c = n->first_child; c; c = ND(L, c)->next_sibling)
        if (pmut(L, c, nm, depth + 1)) return 1;
    return 0;
}

static uint32_t srslot(const lower_t *L, uint32_t fi)
{
    const bir_func_t *F;
    const bir_block_t *B;
    uint32_t j;

    if (fi >= L->M->num_funcs) return BIR_VAL_NONE;
    F = &L->M->funcs[fi];
    if (!F->sret || F->num_params == 0
        || F->first_block >= L->M->num_blocks) return BIR_VAL_NONE;
    B = &L->M->blocks[F->first_block];
    for (j = 0; j < B->num_insts && B->first_inst + j < L->M->num_insts; j++) {
        const bir_inst_t *I = &L->M->insts[B->first_inst + j];

        if (I->op == BIR_PARAM && I->subop == (uint8_t)(F->num_params - 1u))
            return BIR_MAKE_VAL(B->first_inst + j);
    }
    return BIR_VAL_NONE;
}

#define WM_ROW(w)  (int)((((unsigned)(w)) - 1u) & 31u)
#define WM_ROLE(w) (int)(((((unsigned)(w)) - 1u) >> 5) & 3u)
#define WM_LAY(w)  (int)(((((unsigned)(w)) - 1u) >> 7) & 1u)

static sym_t *wmsym(lower_t *L, uint32_t nd)
{
    KA_GUARD(g, 8);
    char t[128];

    while (g-- && nd) {
        int k = ND(L, nd)->type;
        if (k == AST_IDENT) {
            sym_t *s;
            get_text(L, nd, t, sizeof t);
            s = find_sym(L, t);
            return (s && s->wfrg) ? s : NULL;
        }
        if (k != AST_PAREN && k != AST_SUBSCRIPT) return NULL;
        nd = ND(L, nd)->first_child;
    }
    return NULL;
}

static int wmqual(lower_t *L, uint32_t cal)
{
    KA_GUARD(g, 8);
    char t[64];
    uint32_t nd = cal;

    while (g-- && nd && ND(L, nd)->type == AST_SCOPE_RES) {
        uint32_t c = ND(L, nd)->first_child;
        if (c && ND(L, c)->type == AST_IDENT) {
            get_text(L, c, t, sizeof t);
            if (strcmp(t, "wmma") == 0) return 1;
        }
        nd = c ? ND(L, c)->next_sibling : 0;
    }
    return 0;
}

static int wmlay(lower_t *L, uint32_t nd, char *nm, size_t nz)
{
    wmtxt(L, nd, nm, nz);
    if (strcmp(nm, "mem_row_major") == 0) return 0;
    if (strcmp(nm, "mem_col_major") == 0) return 1;
    return -1;
}

static uint32_t wmstr(lower_t *L, uint32_t nd)
{
    return coerce_to(L, lower_expr(L, nd), bir_type_int(L->M, 32), 0);
}

static uint32_t wmld(lower_t *L, uint32_t node, const uint32_t *ag, int nag)
{
    sym_t *s = wmsym(L, ag[0]);
    char nm[64];
    int row, role, lay;
    uint32_t r, o[3];

    if (!s) {
        lower_error(L, node, BC_E905, "load_matrix_sync");
        return BIR_VAL_NONE;
    }
    row = WM_ROW(s->wfrg);
    role = WM_ROLE(s->wfrg);
    lay = WM_LAY(s->wfrg);
    if (role == (int)BIR_WM_C) {
        if (nag != 4) {
            lower_error(L, node, BC_E906, "load_matrix_sync");
            return BIR_VAL_NONE;
        }
        lay = wmlay(L, ag[3], nm, sizeof nm);
        if (lay < 0) {
            lower_error(L, node, BC_E908, nm);
            return BIR_VAL_NONE;
        }
    } else if (nag != 3) {
        lower_error(L, node, BC_E906, "load_matrix_sync");
        return BIR_VAL_NONE;
    }
    o[0] = lower_lvalue(L, ag[0]);
    o[1] = lower_expr(L, ag[1]);
    o[2] = wmstr(L, ag[2]);
    r = emit(L, BIR_WLD, bir_type_void(L->M), 3,
             (uint8_t)(row | (lay << 5) | (role << 6)));
    for (uint8_t k = 0; k < 3; k++) set_op(L, r, k, o[k]);
    return BIR_MAKE_VAL(r);
}

static uint32_t wmst(lower_t *L, uint32_t node, const uint32_t *ag, int nag)
{
    sym_t *s = wmsym(L, ag[1]);
    char nm[64];
    int row, lay;
    uint32_t r, o[3];

    if (!s || WM_ROLE(s->wfrg) != (int)BIR_WM_C) {
        lower_error(L, node, BC_E905, "store_matrix_sync");
        return BIR_VAL_NONE;
    }
    if (nag != 4) {
        lower_error(L, node, BC_E906, "store_matrix_sync");
        return BIR_VAL_NONE;
    }
    lay = wmlay(L, ag[3], nm, sizeof nm);
    if (lay < 0) {
        lower_error(L, node, BC_E908, nm);
        return BIR_VAL_NONE;
    }
    row = WM_ROW(s->wfrg);
    o[0] = lower_expr(L, ag[0]);
    o[1] = lower_lvalue(L, ag[1]);
    o[2] = wmstr(L, ag[2]);
    r = emit(L, BIR_WST, bir_type_void(L->M), 3,
             (uint8_t)(row | (lay << 5)));
    for (uint8_t k = 0; k < 3; k++) set_op(L, r, k, o[k]);
    return BIR_MAKE_VAL(r);
}

static int wmroles(const sym_t *sd, const sym_t *sa, const sym_t *sb,
                   const sym_t *sc)
{
    return WM_ROLE(sa->wfrg) == (int)BIR_WM_A
        && WM_ROLE(sb->wfrg) == (int)BIR_WM_B
        && WM_ROLE(sc->wfrg) == (int)BIR_WM_C
        && WM_ROLE(sd->wfrg) == (int)BIR_WM_C
        && WM_ROW(sb->wfrg) == WM_ROW(sa->wfrg)
        && WM_ROW(sd->wfrg) == WM_ROW(sc->wfrg);
}

static uint32_t wmmm(lower_t *L, uint32_t node, const uint32_t *ag, int bm)
{
    sym_t *sd = wmsym(L, ag[0]), *sa = wmsym(L, ag[1]);
    sym_t *sb = wmsym(L, ag[2]), *sc = wmsym(L, ag[3]);
    const char *fn = bm ? "bmma_sync" : "mma_sync";
    char q[96], nm[64];
    int ra, rc, row, bop = 0;
    uint32_t r, o[4];

    if (!sd || !sa || !sb || !sc) {
        lower_error(L, node, BC_E905, fn);
        return BIR_VAL_NONE;
    }
    ra = WM_ROW(sa->wfrg);
    rc = WM_ROW(sc->wfrg);
    if (bm) {
        wmtxt(L, ag[4], nm, sizeof nm);
        if (strcmp(nm, "bmmaBitOpXOR") == 0) bop = 0;
        else if (strcmp(nm, "bmmaBitOpAND") == 0) bop = 1;
        else {
            lower_error(L, node, BC_E911, nm);
            return BIR_VAL_NONE;
        }
    }
    row = bir_wmrow(bir_wmma[ra].shp, bir_wmma[ra].abt, bir_wmma[rc].act);
    if (row < 0 || !wmroles(sd, sa, sb, sc)
     || (bm && strcmp(bir_wmma[ra].abt, "b1") != 0)) {
        if (snprintf(q, sizeof q, "%s, %s and %s", bir_wmma[ra].shp,
                     bir_wmma[ra].abt, bir_wmma[rc].act) < 0) q[0] = 0;
        lower_error(L, node, BC_E907, q);
        return BIR_VAL_NONE;
    }
    for (int k = 0; k < 4; k++) o[k] = lower_lvalue(L, ag[k]);
    r = emit(L, BIR_WMMA, bir_type_void(L->M), 4,
             (uint8_t)(row | (WM_LAY(sa->wfrg) << 5)
                           | (WM_LAY(sb->wfrg) << 6) | (bop << 7)));
    for (uint8_t k = 0; k < 4; k++) set_op(L, r, k, o[k]);
    return BIR_MAKE_VAL(r);
}

static uint32_t wmfl(lower_t *L, uint32_t node, const uint32_t *ag)
{
    sym_t *s = wmsym(L, ag[0]);
    uint32_t at, et, pt, base, v, cnt, i, r = 0;

    if (!s) {
        lower_error(L, node, BC_E905, "fill_fragment");
        return BIR_VAL_NONE;
    }
    at = s->type;
    if (at >= L->M->num_types || L->M->types[at].kind != BIR_TYPE_ARRAY) {
        lower_error(L, node, BC_E905, "fill_fragment");
        return BIR_VAL_NONE;
    }
    et = L->M->types[at].inner;
    cnt = L->M->types[at].count;
    pt = bir_type_ptr(L->M, et, BIR_AS_PRIVATE);
    base = lower_lvalue(L, ag[0]);
    v = coerce_to(L, lower_expr(L, ag[1]), et, 0);
    for (i = 0; i < cnt && i < 32u; i++) {
        uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                          bir_type_int(L->M, 32), (int64_t)i));
        uint32_t g = emit(L, BIR_GEP, pt, 2, 0);
        set_op(L, g, 0, base);
        set_op(L, g, 1, ci);
        r = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
        set_op(L, r, 0, v);
        set_op(L, r, 1, BIR_MAKE_VAL(g));
    }
    return r ? BIR_MAKE_VAL(r) : BIR_VAL_NONE;
}

static int wmkind(const char *nm)
{
    if (strcmp(nm, "load_matrix_sync") == 0) return 1;
    if (strcmp(nm, "store_matrix_sync") == 0) return 2;
    if (strcmp(nm, "mma_sync") == 0) return 3;
    if (strcmp(nm, "bmma_sync") == 0) return 4;
    if (strcmp(nm, "fill_fragment") == 0) return 5;
    return 0;
}

static int wmcall(lower_t *L, uint32_t node, uint32_t cal, const char *nm,
                  uint32_t *out)
{
    uint32_t ag[6], a;
    int nag = 0, kind = wmkind(nm), need, any = 0;

    if (!kind) return 0;
    for (a = ND(L, cal)->next_sibling; a && nag < 6; a = ND(L, a)->next_sibling)
        ag[nag++] = a;
    for (int k = 0; k < nag; k++)
        if (wmsym(L, ag[k]) != NULL) any = 1;
    if (!any && !wmqual(L, cal)) return 0;

    need = (kind == 5) ? 2 : (kind == 1) ? 3 : (kind == 4) ? 5 : 4;
    if (nag < need || nag > ((kind == 1) ? 4 : need)) {
        lower_error(L, node, BC_E906, nm);
        *out = BIR_VAL_NONE;
        return 1;
    }
    switch (kind) {
    case 1:  *out = wmld(L, node, ag, nag); break;
    case 2:  *out = wmst(L, node, ag, nag); break;
    case 3:  *out = wmmm(L, node, ag, 0); break;
    case 4:  *out = wmmm(L, node, ag, 1); break;
    default: *out = wmfl(L, node, ag); break;
    }
    return 1;
}

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
        pk_rw(L, name, sizeof(name));

        /* Builtin constant: warpSize (HIP) */
        if (strcmp(name, "warpSize") == 0 && L->sema) {
            int wave_size = L->sema->warp_size;
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
        if (!s && tgfnd(L, name, &bval))
            return BIR_MAKE_CONST(bir_const_int(L->M,
                bir_type_int(L->M, 32), bval));
        if (!s) {
            cexp_t *ce = NULL;
            sdm_t *sm = NULL;
            if (L->scls[0] && sdfnd(L, L->scls, name, &sm) == 1)
                return sdemv(L, &sm->v, sm->type);
            if (cfind(L, name, &ce)) {
                if (ce->bad) {
                    lower_error(L, node, BC_E128, name);
                    return BIR_VAL_NONE;
                }
                return cemit(L, ce);
            }
            /* Check file-scope globals (__shared__, __device__, __constant__) */
            uint32_t gi = bir_gsym(L->M, name, L->tu);
            if (gi != BIR_SYM_NONE) {
                bir_global_t *G = &L->M->globals[gi];
                int adrspc = G->addrspace;
                uint32_t ptr_t = bir_type_ptr(L->M, G->type, adrspc);
                if (G->cuda_flags & CUDA_SHARED) {
                    uint32_t sa = emalc(L, BIR_SHARED_ALLOC, ptr_t, 0);
                    add_sym(L, name, sa, G->type, 1);
                } else {
                    uint32_t gr = emit(L, BIR_GLOBAL_REF, ptr_t, 1, 0);
                    set_op(L, gr, 0, gi);
                    add_sym(L, name, gr, G->type, 1);
                }
                s = find_sym(L, name);
            }
        }
        if (!s) {
            uint32_t fi, fp = mfld(L, name);
            if (fp != BIR_VAL_NONE) {
                uint32_t ft = ptr_inner(L, ref_type(L, fp)), ld;
                if (ft < L->M->num_types
                    && (L->M->types[ft].kind == BIR_TYPE_ARRAY
                     || L->M->types[ft].kind == BIR_TYPE_STRUCT))
                    return fp;
                ld = emit(L, BIR_LOAD, ft, 1, 0);
                set_op(L, ld, 0, fp);
                return BIR_MAKE_VAL(ld);
            }
            if (strcmp(name, "__func__") == 0
                || strcmp(name, "__FUNCTION__") == 0
                || strcmp(name, "__PRETTY_FUNCTION__") == 0)
                return mkstr(L, node, L->fnnm,
                             (uint32_t)strlen(L->fnnm));
            if (n->first_child
                && ND(L, n->first_child)->type == AST_TEMPLATE_ARGS
                && tsym(L, name, sizeof(name), node, 0) <= 0) {
                uint32_t bd = tebad(L, node);

                if (bd) {
                    char en[96];

                    cxnm(L, bd, en, (int)sizeof en);
                    lower_error(L, node, BC_E720, en);
                } else {
                    char tn[96];

                    get_text(L, node, tn, sizeof(tn));
                    lower_error(L, node, BC_E721, tn);
                }
                return BIR_VAL_NONE;
            }
            fi = bir_fsym(L->M, name, L->tu, -1);
            if (fi != BIR_SYM_NONE) {
                uint32_t pt = bir_type_ptr(L->M, L->M->funcs[fi].type,
                                           BIR_AS_GENERIC);
                uint32_t r = emit(L, BIR_FNREF, pt, 1, 0);
                set_op(L, r, 0, fi);
                return BIR_MAKE_VAL(r);
            }
            lower_error(L, node, BC_E101);
            return BIR_VAL_NONE;
        }
        if (s->lam > 0) {
            lower_error(L, node, BC_E170, "a closure used as a value");
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

    case AST_SCOPE_RES: {
        char qn[CE_NAMEZ];
        cexp_t *ce = NULL;
        cval_t v;
        int r;

        if (sdres(L, node, &v)) return sdemv(L, &v, L->sdty);
        r = cqfnd(L, node, qn, sizeof(qn), &ce);

        if (r > 0) {
            if (ce->bad) {
                lower_error(L, node, BC_E128, qn);
                return BIR_VAL_NONE;
            }
            return cemit(L, ce);
        }
        if (cfsc(L, node, &v)) return sdemv(L, &v, 0);
        if (!sdemt(L, node)) lower_error(L, node, BC_E106);
        return BIR_VAL_NONE;
    }

    case AST_PAREN:
        return lower_expr(L, n->first_child);

    case AST_MEMBER: {
        uint32_t result;
        cval_t mv;
        if (try_cuda_builtin(L, node, &result))
            return result;
        if (sdres(L, node, &mv)) return sdemv(L, &mv, L->sdty);
        if (sdemt(L, node)) return BIR_VAL_NONE;

        /* Struct member: get address, then load */
        uint32_t addr = lower_lvalue(L, node);
        if (addr == BIR_VAL_NONE) return BIR_VAL_NONE;
        uint32_t pt = ref_type(L, addr);
        uint32_t et = ptr_inner(L, pt);
        if (!et) et = bir_type_int(L->M, 32);
        if (et < L->M->num_types
            && L->M->types[et].kind == BIR_TYPE_ARRAY)
            return addr;
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
                        uint32_t val = stfit(L, lower_expr(L, el),
                                             sd->field_types[fi], el);
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

            uint32_t vt = ref_type(L, val);
            uint32_t dt = ptr_inner(L, ref_type(L, ptr));
            if (!is_ptr_type(L, vt) && aggt(L, dt)) {
                uint32_t sp = apld(L, val, dt);

                if (sp != BIR_VAL_NONE) {
                    val = sp;
                    vt  = ref_type(L, val);
                }
            }
            if (is_ptr_type(L, vt) && aggt(L, dt)
                && acpy(L, node, ptr, val, dt)) {
                uint32_t ald = emit(L, BIR_LOAD, dt, 1, 0);
                set_op(L, ald, 0, ptr);
                return BIR_MAKE_VAL(ald);
            }

            uint32_t t_void = bir_type_void(L->M);
            uint32_t st;
            val = stfit(L, val, dt, rhs_n);
            st = emit(L, BIR_STORE, t_void, 2, 0);
            set_op(L, st, 0, val);
            set_op(L, st, 1, ptr);
            return val;
        }

        /* Compound assignment (+=, *=, etc.) */
        if (is_compound_assign(op)) {
            uint32_t ptr = lower_lvalue(L, lhs_n);
            uint32_t pt  = ref_type(L, ptr);
            uint32_t et  = ptr_inner(L, pt);
            int base = compound_base(op);
            if (!et) et  = bir_type_int(L->M, 32);

            if (et < L->M->num_types
                && L->M->types[et].kind == BIR_TYPE_STRUCT) {
                struct_def_t *sd = sdbyt(L, et);
                uint32_t rv  = lower_expr(L, rhs_n);
                uint32_t rvs = stof(L, rv, rhs_n);
                uint32_t acc = BIR_VAL_NONE, sst;

                if (vpair(L, et, rvs) && sd && vaop(base) >= 0)
                    acc = vbin(L, sd, (uint16_t)vaop(base), ptr, rv);
                if (acc == BIR_VAL_NONE) {
                    lower_error(L, node, BC_E102);
                    return BIR_VAL_NONE;
                }
                sst = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                set_op(L, sst, 0, acc);
                set_op(L, sst, 1, ptr);
                return acc;
            }

            uint32_t old = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, old, 0, ptr);
            uint32_t rhs = lower_expr(L, rhs_n);

            if (base != TOK_SHL && base != TOK_SHR)
                rhs = stfit(L, rhs, et, rhs_n);
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
            uint32_t ls  = stof(L, lhs, lhs_n);
            uint32_t rs  = stof(L, rhs, rhs_n);
            int fp;

            if (ls || rs)
                return aggop(L, node, op, lhs, ls, rhs, rs);

            fp = is_float_type(L, lt);
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
            uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
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
            uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
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
            uint32_t rt  = ref_type(L, rhs);
            uint32_t ls  = stof(L, lhs, lhs_n);
            uint32_t rs  = stof(L, rhs, rhs_n);

            if (ls || rs)
                return aggop(L, node, op, lhs, ls, rhs, rs);

            if (op == TOK_PLUS && is_ptr_type(L, lt) && is_ptr_type(L, rt)) {
                lower_error(L, node, BC_E102);
                return BIR_VAL_NONE;
            }

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

            return bin_val(L, node, op, lhs, lhs_n, rhs, rhs_n);
        }
    }

    case AST_UNARY_PREFIX: {
        int op = n->d.oper.op;
        uint32_t operand = n->first_child;

        if (op == TOK_PLUS)
            return lower_expr(L, operand);

        if (op == TOK_MINUS || op == TOK_BANG || op == TOK_TILDE) {
            uint32_t val = lower_expr(L, operand);
            uint32_t t   = ref_type(L, val);
            uint32_t st  = stof(L, val, operand);
            uint32_t z, inst;

            if (st) return aggun(L, node, op, val, st);
            if (op == TOK_MINUS) {
                if (is_float_type(L, t)) {
                    z = BIR_MAKE_CONST(bir_const_float(L->M, t, 0.0));
                    inst = emit(L, BIR_FSUB, t, 2, 0);
                } else {
                    z = BIR_MAKE_CONST(bir_const_int(L->M, t, 0));
                    inst = emit(L, BIR_SUB, t, 2, 0);
                }
                set_op(L, inst, 0, z);
                set_op(L, inst, 1, val);
                return BIR_MAKE_VAL(inst);
            }
            if (op == TOK_BANG) {
                z    = BIR_MAKE_CONST(bir_const_int(L->M, t, 0));
                inst = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2,
                            BIR_ICMP_EQ);
            } else {
                z    = BIR_MAKE_CONST(bir_const_int(L->M, t, -1));
                inst = emit(L, BIR_XOR, t, 2, 0);
            }
            set_op(L, inst, 0, val);
            set_op(L, inst, 1, z);
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

            if (et < L->M->num_types
                && L->M->types[et].kind == BIR_TYPE_STRUCT) {
                lower_error(L, node, BC_E103);
                return BIR_VAL_NONE;
            }

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

            if (et < L->M->num_types
                && L->M->types[et].kind == BIR_TYPE_STRUCT) {
                lower_error(L, node, BC_E104);
                return BIR_VAL_NONE;
            }

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
        uint32_t cptr = 0;
        uint32_t fwn = unfw(L, node);
        int cli;

        if (fwn != node) return lower_expr(L, fwn);
        {
            char fwm[32];
            if (fwstd(L, callee_n, fwm, (int)sizeof fwm)) {
                lower_error(L, node, BC_E421, fwm);
                return BIR_VAL_NONE;
            }
        }
        cli = lfind(L, callee_n, &cptr);

        if (cli == -2) return BIR_VAL_NONE;
        if (cli >= 0)
            return lemit(L, node, cli, cptr, ND(L, callee_n)->next_sibling);

        /* ---- Cooperative groups: phantom types meet real hardware ---- */
        if (ND(L, callee_n)->type == AST_SCOPE_RES) {
            uint32_t ns_n = ND(L, callee_n)->first_child;
            uint32_t fn_n = ns_n ? ND(L, ns_n)->next_sibling : 0;
            if (ns_n && fn_n && (text_eq(L, fn_n, "this_thread_block")
                              || text_eq(L, fn_n, "this_grid"))) {
                uint32_t i32 = bir_type_int(L->M, 32);
                return BIR_MAKE_CONST(bir_const_int(L->M, i32, 0));
            }
        }
        if (ND(L, callee_n)->type == AST_MEMBER) {
            uint32_t obj_n = ND(L, callee_n)->first_child;
            uint32_t fld_n = obj_n ? ND(L, obj_n)->next_sibling : 0;
            char cgn[128];
            cgn[0] = 0;
            if (fld_n && ND(L, fld_n)->type == AST_IDENT)
                get_text(L, fld_n, cgn, sizeof(cgn));
            if (fld_n && ND(L, fld_n)->type == AST_IDENT && !hasm(L, cgn)) {
                int gk = cgobj(L, obj_n);
                if (text_eq(L, fld_n, "sync")) {
                    if (gk == 2)
                        return BIR_MAKE_VAL(emit(L, BIR_GRIDBAR,
                                                 bir_type_void(L->M), 0, 0));
                    /* .sync() → barrier. All threads, please stop and wait politely. */
                    uint32_t inst = emit(L, BIR_BARRIER, bir_type_void(L->M), 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
                if (text_eq(L, fld_n, "thread_rank")) {
                    if (gk == 2) return cggrk(L);
                    /* .thread_rank() → threadIdx.x. A thread by any other name. */
                    uint32_t i32 = bir_type_int(L->M, 32);
                    uint32_t inst = emit(L, BIR_THREAD_ID, i32, 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
                if (text_eq(L, fld_n, "size")
                    || text_eq(L, fld_n, "num_threads")) {
                    if (gk == 2) return cggsz(L);
                    /* .size() → blockDim.x. How many of us are there? */
                    uint32_t i32 = bir_type_int(L->M, 32);
                    uint32_t inst = emit(L, BIR_BLOCK_DIM, i32, 0, 0);
                    return BIR_MAKE_VAL(inst);
                }
                if (gk == 2 && text_eq(L, fld_n, "block_rank"))
                    return cgrnk(L, BIR_BLOCK_ID, BIR_GRID_DIM);
                if (gk == 2 && text_eq(L, fld_n, "num_blocks"))
                    return cgdim(L, BIR_GRID_DIM);
            }
        }

        char cname[128];
        int fbc = 0;
        uint32_t self = BIR_VAL_NONE;
        uint32_t hdst = L->srdst, hdty = L->srdty;
        L->srdst = BIR_VAL_NONE;
        L->srdty = 0;
        get_text(L, callee_n, cname, sizeof(cname));
        if (ND(L, callee_n)->type == AST_MEMBER) {
            uint32_t obj_n = ND(L, callee_n)->first_child;
            uint32_t fld_n = obj_n ? ND(L, obj_n)->next_sibling : 0;
            char mn[64], full[144];
            mn[0] = 0;
            if (fld_n && ND(L, fld_n)->type == AST_IDENT)
                get_text(L, fld_n, mn, sizeof(mn));
            if (mn[0] && hasm(L, mn)) {
                uint32_t op, st;
                int si;
                op = ND(L, callee_n)->d.member.is_arrow
                   ? lower_expr(L, obj_n) : mtmp(L, obj_n);
                if (op == BIR_VAL_NONE) return BIR_VAL_NONE;
                st = ptr_inner(L, ref_type(L, op));
                si = sfind(L, st);
                if (si < 0) {
                    lower_error(L, node, BC_E105, mn);
                    return BIR_VAL_NONE;
                }
                if (snprintf(full, sizeof(full), "%s$%s",
                             L->structs[si].name, mn) < 0) {
                    lower_error(L, node, BC_E105, mn);
                    return BIR_VAL_NONE;
                }
                ncpy(cname, sizeof(cname), full);
                if (!smth(L, si, mn)) self = op;
            }
        }
        {
            const char *fb = fbfn(L, cname);
            cexp_t *ce = NULL;
            char cfn[128];

            if (!fb && cfind(L, cname, &ce) && !ce->bad && ce->v.fnd) {
                if (ce->v.fsy && (int)ce->v.fsy <= L->ninst)
                    fb = L->inst[ce->v.fsy - 1].sym;
                else {
                    get_text(L, ce->v.fnd, cfn, sizeof(cfn));
                    if (cfn[0]) fb = cfn;
                }
            }
            if (fb) {
                ncpy(cname, sizeof(cname), fb);
                fbc = bir_fsym(L->M, cname, L->tu, -1) != BIR_SYM_NONE;
            }
        }
        if (ND(L, callee_n)->type == AST_SCOPE_RES) {
            char qn[144], cd[144];
            char *sep;
            qtxt(L, callee_n, qn, sizeof(qn));
            sep = strstr(qn, "::");
            if (sep) {
                uint32_t bt = 0;
                const char *base, *tail = sep + 2, *lst;
                *sep = 0;
                base = tsnm(L, qn);
                if (!base && find_binding(L, qn, &bt)) {
                    const char *sn = sname(L, bt);
                    base = sn && strchr(sn, CI_SEP) && tamb(L, bt) > 1
                         ? NULL : sn;
                } else if (!base) {
                    base = qn;
                }
                lst = strrchr(tail, ':');
                if (lst) tail = lst + 1;
                if (base && snprintf(cd, sizeof(cd), "%s$%s",
                                     base, sep + 2) > 0
                         && bir_fsym(L->M, cd, L->tu, -1) != BIR_SYM_NONE)
                    ncpy(cname, sizeof(cname), cd);
                else if (!cname[0])
                    ncpy(cname, sizeof(cname), tail);
            }
        }

        /* ---- CUDA builtin functions ---- */

        {
            uint32_t wrv = BIR_VAL_NONE;
            if (wmcall(L, node, callee_n, cname, &wrv)) return wrv;
        }


        /* Barriers: zero args, void return */
        if (strcmp(cname, "__syncthreads") == 0) {
            uint32_t inst = emit(L, BIR_BARRIER, bir_type_void(L->M), 0, 0);
            return BIR_MAKE_VAL(inst);
        }
        {
            static const struct { const char *n; uint8_t sc; } ftab[] = {
                {"__threadfence_block",  0},
                {"__threadfence",        1},
                {"__threadfence_system", 2}
            };
            for (int fi = 0; fi < (int)(sizeof ftab / sizeof ftab[0]); fi++) {
                if (strcmp(cname, ftab[fi].n) != 0) continue;
                return BIR_MAKE_VAL(emit(L, BIR_FENCE, bir_type_void(L->M),
                                         0, ftab[fi].sc));
            }
        }

        if (strcmp(cname, "__nanosleep") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0, inst;
            if (an == 0) {
                lower_error(L, node, BC_E073, "__nanosleep", 1, 0);
                return BIR_VAL_NONE;
            }
            a0 = lower_expr(L, an);
            if (a0 == BIR_VAL_NONE) return BIR_VAL_NONE;
            a0 = stfit(L, a0, bir_type_int(L->M, 32), an);
            inst = emit(L, BIR_NANOSLP, bir_type_void(L->M), 1, 0);
            set_op(L, inst, 0, a0);
            return BIR_MAKE_VAL(inst);
        }

        {
            static const struct { const char *n; uint8_t rd; } stab[] = {
                {"__syncthreads_or",    0},
                {"__syncthreads_and",   1},
                {"__syncthreads_count", 2}
            };
            for (int si = 0; si < (int)(sizeof stab / sizeof stab[0]); si++) {
                uint32_t an, a0, rt, inst;
                if (strcmp(cname, stab[si].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                if (an == 0) {
                    lower_error(L, node, BC_E073, stab[si].n, 1, 0);
                    return BIR_VAL_NONE;
                }
                a0 = lower_expr(L, an);
                if (a0 == BIR_VAL_NONE) return BIR_VAL_NONE;
                a0 = stfit(L, a0, bir_type_int(L->M, 32), an);
                rt = (stab[si].rd == 2) ? bir_type_int(L->M, 32)
                                        : bir_type_int(L->M, 1);
                inst = emit(L, BIR_BARRED, rt, 1, stab[si].rd);
                set_op(L, inst, 0, a0);
                return BIR_MAKE_VAL(inst);
            }
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
            for (int bi = 0; bi < (int)(sizeof atab / sizeof atab[0]); bi++) {
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

        /* __umulhi / __umul64hi: high half of a 32x32 or 64x64 unsigned
           product. Keystone for ZK field arithmetic — Montgomery multiply is
           mul-lo + mul-hi + add-with-carry. */
        if (strcmp(cname, "__umulhi") == 0 ||
            strcmp(cname, "__umul64hi") == 0) {
            int w = (cname[6] == '6') ? 64 : 32;   /* __umul64hi vs __umulhi */
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            an = ND(L, an)->next_sibling;
            uint32_t a1 = lower_expr(L, an);
            uint32_t rt = bir_type_int(L->M, w);
            uint32_t inst = emit(L, BIR_UMULHI, rt, 2, 0);
            set_op(L, inst, 0, a0);
            set_op(L, inst, 1, a1);
            return BIR_MAKE_VAL(inst);
        }

        /* __popc / __clz / __ffs / __brev. The ll suffix carries no weight,
           the operand's own type picks the width. */
        {
            static const struct { const char *n; uint16_t op; int ffs; } btab[] = {
                {"__popc",   BIR_POPCOUNT, 0},
                {"__popcll", BIR_POPCOUNT, 0},
                {"__clz",    BIR_CLZ,      0},
                {"__clzll",  BIR_CLZ,      0},
                {"__ffs",    BIR_CTZ,      1},
                {"__ffsll",  BIR_CTZ,      1},
                {"__brev",   BIR_BREV,     0},
                {"__brevll", BIR_BREV,     0},
            };
            for (int bi = 0; bi < (int)(sizeof btab / sizeof btab[0]); bi++) {
                if (strcmp(cname, btab[bi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0 = lower_expr(L, an);
                uint32_t at = ref_type(L, a0);
                uint32_t t32 = bir_type_int(L->M, 32);
                uint32_t rt = (btab[bi].op == BIR_BREV) ? at : t32;
                uint32_t inst = emit(L, btab[bi].op, rt, 1, 0);
                set_op(L, inst, 0, a0);
                if (!btab[bi].ffs) return BIR_MAKE_VAL(inst);

                /* ffs is ctz+1, but zero answers 0 rather than width+1 */
                uint32_t nz = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2,
                                   BIR_ICMP_NE);
                set_op(L, nz, 0, a0);
                set_op(L, nz, 1, BIR_MAKE_CONST(bir_const_int(L->M, at, 0)));
                uint32_t inc = emit(L, BIR_ADD, t32, 2, 0);
                set_op(L, inc, 0, BIR_MAKE_VAL(inst));
                set_op(L, inc, 1, BIR_MAKE_CONST(bir_const_int(L->M, t32, 1)));
                uint32_t sel = emit(L, BIR_SELECT, t32, 3, 0);
                set_op(L, sel, 0, BIR_MAKE_VAL(nz));
                set_op(L, sel, 1, BIR_MAKE_VAL(inc));
                set_op(L, sel, 2, BIR_MAKE_CONST(bir_const_int(L->M, t32, 0)));
                return BIR_MAKE_VAL(sel);
            }
        }

        /* Warp shuffle: up to 4-arg (mask, val, lane/delta, [width]) or 2..3-arg bare (val, lane/delta, [width]) */
        {
            static const struct { const char *n; uint16_t op; int sync; } stab[] = {
                {"__shfl_sync",      BIR_SHFL,     1},
                {"__shfl_up_sync",   BIR_SHFL_UP,  1},
                {"__shfl_down_sync", BIR_SHFL_DOWN,1},
                {"__shfl_xor_sync",  BIR_SHFL_XOR, 1},
                {"__shfl",           BIR_SHFL,     0},
                {"__shfl_up",        BIR_SHFL_UP,  0},
                {"__shfl_down",      BIR_SHFL_DOWN,0},
                {"__shfl_xor",       BIR_SHFL_XOR, 0},
            };
            for (int bi = 0; bi < (int)(sizeof stab / sizeof stab[0]); bi++) {
                if (strcmp(cname, stab[bi].n) != 0) continue;
                uint32_t sa[4];
                int sn = 0;
                if (!stab[bi].sync) {
                    uint32_t t32 = bir_type_int(L->M, 32);
                    sa[sn++] = BIR_MAKE_CONST(bir_const_int(L->M, t32, (int64_t)0xFFFFFFFFu));
                }
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

        {
            static const struct { const char *n; uint16_t op; int cmp; } rtab[] = {
                {"__reduce_add_sync", BIR_ADD, 0},
                {"__reduce_and_sync", BIR_AND, 0},
                {"__reduce_or_sync",  BIR_OR,  0},
                {"__reduce_xor_sync", BIR_XOR, 0},
                {"__reduce_min_sync", 0,       1},
                {"__reduce_max_sync", 0,       2},
            };
            for (int bi = 0; bi < (int)(sizeof rtab / sizeof rtab[0]); bi++) {
                if (strcmp(cname, rtab[bi].n) != 0) continue;
                uint32_t mn = ND(L, callee_n)->next_sibling;
                uint32_t vn = mn ? ND(L, mn)->next_sibling : 0;
                int ws = L->sema ? L->sema->warp_size : 32;
                int64_t mv = 0;
                if (!vn || !cival(L, mn, &mv)
                        || (uint32_t)mv != 0xFFFFFFFFu) {
                    lower_error(L, node, BC_E143, cname);
                    return BIR_VAL_NONE;
                }
                uint32_t v = lower_expr(L, vn);
                uint32_t rt = ref_type(L, v);
                uint32_t t32 = bir_type_int(L->M, 32);
                uint32_t mk = BIR_MAKE_CONST(bir_const_int(L->M, t32,
                                             (int64_t)0xFFFFFFFFu));
                int uns = node_is_unsigned(L, vn);
                if (ws < 2 || ws > 64) ws = 32;
                for (int off = ws / 2; off >= 1; off >>= 1) {
                    uint32_t sh = emit(L, BIR_SHFL_XOR, rt, 4, 0);
                    set_op(L, sh, 0, mk);
                    set_op(L, sh, 1, v);
                    set_op(L, sh, 2,
                           BIR_MAKE_CONST(bir_const_int(L->M, t32, off)));
                    set_op(L, sh, 3,
                           BIR_MAKE_CONST(bir_const_int(L->M, t32, ws)));
                    if (rtab[bi].cmp) {
                        uint8_t pr = rtab[bi].cmp == 2
                                   ? (uns ? BIR_ICMP_UGT : BIR_ICMP_SGT)
                                   : (uns ? BIR_ICMP_ULT : BIR_ICMP_SLT);
                        uint32_t c = emit(L, BIR_ICMP,
                                          bir_type_int(L->M, 1), 2, pr);
                        set_op(L, c, 0, v);
                        set_op(L, c, 1, BIR_MAKE_VAL(sh));
                        uint32_t sl = emit(L, BIR_SELECT, rt, 3, 0);
                        set_op(L, sl, 0, BIR_MAKE_VAL(c));
                        set_op(L, sl, 1, v);
                        set_op(L, sl, 2, BIR_MAKE_VAL(sh));
                        v = BIR_MAKE_VAL(sl);
                    } else {
                        uint32_t r = emit(L, rtab[bi].op, rt, 2, 0);
                        set_op(L, r, 0, v);
                        set_op(L, r, 1, BIR_MAKE_VAL(sh));
                        v = BIR_MAKE_VAL(r);
                    }
                }
                return v;
            }
        }

        /* Warp vote: 2-arg (mask, pred) or 1-arg bare (pred) */
        {
            static const struct { const char *n; uint16_t op; int sync; } vtab[] = {
                {"__ballot_sync", BIR_BALLOT,   1},
                {"__any_sync",    BIR_VOTE_ANY, 1},
                {"__all_sync",    BIR_VOTE_ALL, 1},
                {"__ballot",      BIR_BALLOT,   0},
                {"__any",         BIR_VOTE_ANY, 0},
                {"__all",         BIR_VOTE_ALL, 0},
            };
            for (int bi = 0; bi < (int)(sizeof vtab / sizeof vtab[0]); bi++) {
                if (strcmp(cname, vtab[bi].n) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t a0, a1;
                if (vtab[bi].sync) {
                    a0 = lower_expr(L, an);
                    an = ND(L, an)->next_sibling;
                    a1 = lower_expr(L, an);
                } else {
                    uint32_t t32 = bir_type_int(L->M, 32);
                    a0 = BIR_MAKE_CONST(bir_const_int(L->M, t32, (int64_t)0xFFFFFFFFu));
                    a1 = lower_expr(L, an);
                }
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
            char vname[64];
            struct_def_t *vsd = NULL;
            const vecty_t *vt;

            ncpy(vname, sizeof(vname), cname + 5);
            vt = vfind(vname);
            if (vt) {
                vsd = vreg(L, vt);
                if (!vsd) {
                    lower_error(L, node, BC_E138, MAX_STRUCTS);
                    return BIR_VAL_NONE;
                }
            }
            if (vsd) {
                uint32_t st = vsd->bir_type;
                uint32_t pt = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
                uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);

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

        {
            static const struct { const char *n; int lane; int wide; } h2f[] = {
                {"__low2float", 0, 1}, {"__high2float", 1, 1},
                {"__low2half",  0, 0}, {"__high2half",  1, 0},
                {"__low2bfloat16", 0, 0}, {"__high2bfloat16", 1, 0},
            };
            for (int hi = 0; hi < (int)(sizeof h2f / sizeof h2f[0]); hi++) {
                uint32_t an, v, lv, r;
                if (strcmp(cname, h2f[hi].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                v  = lower_expr(L, an);
                lv = vlane(L, v, h2f[hi].lane);
                if (lv == BIR_VAL_NONE) {
                    lower_error(L, node, BC_E143, cname);
                    return BIR_VAL_NONE;
                }
                if (!h2f[hi].wide) return lv;
                r = emit(L, BIR_FPEXT, bir_type_float(L->M, 32), 1, 0);
                set_op(L, r, 0, lv);
                return BIR_MAKE_VAL(r);
            }
        }
        if (strcmp(cname, "__half22float2") == 0
            || strcmp(cname, "__bfloat1622float2") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t v  = lower_expr(L, an);
            uint32_t f32 = bir_type_float(L->M, 32);
            const vecty_t *vt = vfind("float2");
            struct_def_t *sd = vt ? vreg(L, vt) : NULL;
            uint32_t lv[2];
            if (!sd) {
                lower_error(L, node, BC_E138, MAX_STRUCTS);
                return BIR_VAL_NONE;
            }
            for (int i = 0; i < 2; i++) {
                uint32_t l = vlane(L, v, i), e;
                if (l == BIR_VAL_NONE) {
                    lower_error(L, node, BC_E143, cname);
                    return BIR_VAL_NONE;
                }
                e = emit(L, BIR_FPEXT, f32, 1, 0);
                set_op(L, e, 0, l);
                lv[i] = BIR_MAKE_VAL(e);
            }
            return vmake(L, sd, lv, 2);
        }
        if (strcmp(cname, "__half2half2") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t v  = lower_expr(L, an);
            const vecty_t *vt = vfind("half2");
            struct_def_t *sd = vt ? vreg(L, vt) : NULL;
            uint32_t lv[2];
            if (!sd) {
                lower_error(L, node, BC_E138, MAX_STRUCTS);
                return BIR_VAL_NONE;
            }
            lv[0] = v; lv[1] = v;
            return vmake(L, sd, lv, 2);
        }
        {
            static const struct {
                const char *n; uint8_t md; uint8_t bf; uint8_t l0; uint8_t l1;
            } pk[] = {
                {"__float22half2_rn",     PK_F2, 0, 0, 1},
                {"__float22bfloat162_rn", PK_F2, 1, 0, 1},
                {"__floats2half2_rn",     PK_FS, 0, 0, 1},
                {"__floats2bfloat162_rn", PK_FS, 1, 0, 1},
                {"__float2half2_rn",      PK_FB, 0, 0, 0},
                {"__float2bfloat162_rn",  PK_FB, 1, 0, 0},
                {"__halves2half2",        PK_HS, 0, 0, 1},
                {"__halves2bfloat162",    PK_HS, 1, 0, 1},
                {"__bfloat162bfloat162",  PK_HB, 1, 0, 0},
                {"__lows2half2",          PK_L2, 0, 0, 0},
                {"__highs2half2",         PK_L2, 0, 1, 1},
                {"__lows2bfloat162",      PK_L2, 1, 0, 0},
                {"__highs2bfloat162",     PK_L2, 1, 1, 1},
                {"__low2half2",           PK_LP, 0, 0, 0},
                {"__high2half2",          PK_LP, 0, 1, 1},
                {"__low2bfloat162",       PK_LP, 1, 0, 0},
                {"__high2bfloat162",      PK_LP, 1, 1, 1},
                {"__lowhigh2highlow",     PK_LP, 2, 1, 0},
            };
            for (int ki = 0; ki < (int)(sizeof pk / sizeof pk[0]); ki++) {
                uint32_t an, a0, a1 = BIR_VAL_NONE, lv[2];
                const vecty_t *vt;
                struct_def_t *sd;
                int two = pk[ki].md == PK_FS || pk[ki].md == PK_HS
                       || pk[ki].md == PK_L2;

                if (strcmp(cname, pk[ki].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                a0 = lower_expr(L, an);
                if (two) {
                    an = an ? ND(L, an)->next_sibling : 0;
                    a1 = lower_expr(L, an);
                }
                if (a0 == BIR_VAL_NONE || (two && a1 == BIR_VAL_NONE)) {
                    lower_error(L, node, BC_E073, cname, two ? 2 : 1, 0);
                    return BIR_VAL_NONE;
                }
                if (!hplan(L, pk[ki].md, pk[ki].l0, pk[ki].l1, a0, a1, lv)) {
                    lower_error(L, node, BC_E940, cname);
                    return BIR_VAL_NONE;
                }
                vt = pk[ki].bf == 2 ? hpvt(L, ref_type(L, lv[0]))
                   : vfind(pk[ki].bf ? "nv_bfloat162" : "half2");
                sd = vt ? vreg(L, vt) : NULL;
                if (!sd) {
                    lower_error(L, node, BC_E940, cname);
                    return BIR_VAL_NONE;
                }
                if (pk[ki].md <= PK_FB)
                    for (int i = 0; i < 2; i++) {
                        uint32_t r = emit(L, BIR_FPTRUNC,
                                          sd->field_types[i], 1, 0);
                        set_op(L, r, 0, lv[i]);
                        lv[i] = BIR_MAKE_VAL(r);
                    }
                return vmake(L, sd, lv, 2);
            }
        }
        if (strcmp(cname, "__hmul2") == 0 || strcmp(cname, "__hadd2") == 0
            || strcmp(cname, "__hsub2") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            uint32_t a1;
            const vecty_t *vt = vfind("half2");
            struct_def_t *sd = vt ? vreg(L, vt) : NULL;
            uint16_t hop = (cname[3] == 'm') ? BIR_FMUL
                         : (cname[3] == 'a') ? BIR_FADD : BIR_FSUB;
            uint32_t lv[2];
            an = an ? ND(L, an)->next_sibling : 0;
            a1 = lower_expr(L, an);
            if (!sd) {
                lower_error(L, node, BC_E138, MAX_STRUCTS);
                return BIR_VAL_NONE;
            }
            for (int i = 0; i < 2; i++) {
                uint32_t x = vlane(L, a0, i), y = vlane(L, a1, i), r;
                if (x == BIR_VAL_NONE || y == BIR_VAL_NONE) {
                    lower_error(L, node, BC_E143, cname);
                    return BIR_VAL_NONE;
                }
                r = emit(L, hop, sd->field_types[i], 2, 0);
                set_op(L, r, 0, x);
                set_op(L, r, 1, y);
                lv[i] = BIR_MAKE_VAL(r);
            }
            return vmake(L, sd, lv, 2);
        }

        {
            static const struct {
                const char *n; int w; int uns; int rnd;
            } cvt[] = {
                {"__float2int_rn",   32, 0, 0}, {"__float2int_rz",   32, 0, 3},
                {"__float2int_ru",   32, 0, 2}, {"__float2int_rd",   32, 0, 1},
                {"__float2uint_rn",  32, 1, 0}, {"__float2uint_rz",  32, 1, 3},
                {"__float2uint_ru",  32, 1, 2}, {"__float2uint_rd",  32, 1, 1},
                {"__float2ll_rn",    64, 0, 0}, {"__float2ll_rz",    64, 0, 3},
                {"__float2ll_ru",    64, 0, 2}, {"__float2ll_rd",    64, 0, 1},
                {"__float2ull_rn",   64, 1, 0}, {"__float2ull_rz",   64, 1, 3},
                {"__float2ull_ru",   64, 1, 2}, {"__float2ull_rd",   64, 1, 1},
                {"__double2int_rn",  32, 0, 0}, {"__double2int_rz",  32, 0, 3},
                {"__double2int_ru",  32, 0, 2}, {"__double2int_rd",  32, 0, 1},
                {"__double2ll_rn",   64, 0, 0}, {"__double2ll_rz",   64, 0, 3},
                {"__double2ll_ru",   64, 0, 2}, {"__double2ll_rd",   64, 0, 1},
            };
            for (int ci = 0; ci < (int)(sizeof cvt / sizeof cvt[0]); ci++) {
                uint32_t an, v, vt, rt, r;
                if (strcmp(cname, cvt[ci].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                v = lower_expr(L, an);
                if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
                vt = ref_type(L, v);
                if (cvt[ci].rnd != 3) {
                    uint16_t ro = cvt[ci].rnd == 0 ? BIR_RNDNE
                                : cvt[ci].rnd == 1 ? BIR_FLOOR : BIR_CEIL;
                    r = emit(L, ro, vt, 1, 0);
                    set_op(L, r, 0, v);
                    v = BIR_MAKE_VAL(r);
                }
                rt = bir_type_int(L->M, (uint16_t)cvt[ci].w);
                r = emit(L, cvt[ci].uns ? BIR_FPTOUI : BIR_FPTOSI, rt, 1, 0);
                set_op(L, r, 0, v);
                return BIR_MAKE_VAL(r);
            }
        }

        {
            static const struct { const char *n; int uns; int dbl; } i2f[] = {
                {"__int2float_rn",  0, 0}, {"__uint2float_rn", 1, 0},
                {"__ll2float_rn",   0, 0}, {"__ull2float_rn",  1, 0},
                {"__int2double_rn", 0, 1}, {"__uint2double_rn",1, 1},
                {"__ll2double_rn",  0, 1}, {"__ull2double_rn", 1, 1},
            };
            for (int ci = 0; ci < (int)(sizeof i2f / sizeof i2f[0]); ci++) {
                uint32_t an, v, rt, r;
                if (strcmp(cname, i2f[ci].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                v = lower_expr(L, an);
                if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
                rt = bir_type_float(L->M, i2f[ci].dbl ? 64 : 32);
                r = emit(L, i2f[ci].uns ? BIR_UITOFP : BIR_SITOFP, rt, 1, 0);
                set_op(L, r, 0, v);
                return BIR_MAKE_VAL(r);
            }
        }

        if (strcmp(cname, "__byte_perm") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t x = lower_expr(L, an);
            uint32_t y, sl;
            an = an ? ND(L, an)->next_sibling : 0;
            y = lower_expr(L, an);
            an = an ? ND(L, an)->next_sibling : 0;
            sl = lower_expr(L, an);
            if (x == BIR_VAL_NONE || y == BIR_VAL_NONE || sl == BIR_VAL_NONE) {
                lower_error(L, node, BC_E073, cname, 3, 0);
                return BIR_VAL_NONE;
            }
            return vprmt(L, x, y, sl);
        }

        {
            static const struct { const char *n; int mode; int sub; } vtb[] = {
                {"__vadd4",   VW_WRAP, 0}, {"__vsub4",   VW_WRAP, 1},
                {"__vaddss4", VW_SSAT, 0}, {"__vsubss4", VW_SSAT, 1},
                {"__vaddus4", VW_USAT, 0}, {"__vsubus4", VW_USAT, 1},
                {"__vcmpeq4", VW_EQ,   0}, {"__vcmpne4", VW_NE,   0},
            };
            for (int vi = 0; vi < (int)(sizeof vtb / sizeof vtb[0]); vi++) {
                uint32_t an, a0, a1;
                if (strcmp(cname, vtb[vi].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                a0 = lower_expr(L, an);
                an = an ? ND(L, an)->next_sibling : 0;
                a1 = lower_expr(L, an);
                if (a0 == BIR_VAL_NONE || a1 == BIR_VAL_NONE) {
                    lower_error(L, node, BC_E073, cname, 2, 0);
                    return BIR_VAL_NONE;
                }
                return vsimd(L, vtb[vi].mode, vtb[vi].sub, a0, a1);
            }
        }

        if (strcmp(cname, "__cvta_generic_to_shared") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t v = lower_expr(L, an);
            uint32_t r;
            if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
            r = emit(L, BIR_PTRTOINT, bir_type_int(L->M, 64), 1, 0);
            set_op(L, r, 0, v);
            return BIR_MAKE_VAL(r);
        }

        if (strcmp(cname, "__builtin_assume") == 0)
            return vki(L, 0);

        {
            static const struct { const char *n; int kind; } ftb[] = {
                {"isnan", 0}, {"isinf", 1}, {"isfinite", 2},
            };
            for (int fi2 = 0; fi2 < (int)(sizeof ftb / sizeof ftb[0]); fi2++) {
                uint32_t an, v, vt, ab, lim, r;
                double mx;
                if (strcmp(cname, ftb[fi2].n) != 0) continue;
                an = ND(L, callee_n)->next_sibling;
                v = lower_expr(L, an);
                if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
                uint32_t i1 = bir_type_int(L->M, 1), fin, big;
                vt = ref_type(L, v);
                mx = L->M->types[vt].width <= 32
                   ? 3.4028234663852886e+38 : 1.7976931348623157e+308;
                ab = emit(L, BIR_FABS, vt, 1, 0);
                set_op(L, ab, 0, v);
                lim = BIR_MAKE_CONST(bir_const_float(L->M, vt, mx));
                fin = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OLE);
                set_op(L, fin, 0, BIR_MAKE_VAL(ab));
                set_op(L, fin, 1, lim);
                if (ftb[fi2].kind == 2) return BIR_MAKE_VAL(fin);
                big = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OGT);
                set_op(L, big, 0, BIR_MAKE_VAL(ab));
                set_op(L, big, 1, lim);
                if (ftb[fi2].kind == 1) return BIR_MAKE_VAL(big);
                r = vsel(L, BIR_MAKE_VAL(fin), vki(L, 0),
                         vsel(L, BIR_MAKE_VAL(big), vki(L, 0), vki(L, 1)));
                return vcmp(L, BIR_ICMP_NE, r, vki(L, 0));
            }
        }

        if (strcmp(cname, "erff") == 0 || strcmp(cname, "erf") == 0) {
            static const double ec[5] = {
                 0.254829592, -0.284496736,  1.421413741,
                -1.453152027,  1.061405429
            };
            uint32_t f32 = bir_type_float(L->M, 32);
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t v = lower_expr(L, an);
            uint32_t ab, t, sq, ex, pol, r, neg, c;
            if (v == BIR_VAL_NONE) return BIR_VAL_NONE;
            v = stfit(L, v, f32, an);
            ab = emit(L, BIR_FABS, f32, 1, 0);
            set_op(L, ab, 0, v);
            t = fbin(L, BIR_FMUL, f32,
                     BIR_MAKE_VAL(ab), fkf(L, f32, 0.3275911));
            t = fbin(L, BIR_FADD, f32, t, fkf(L, f32, 1.0));
            t = fbin(L, BIR_FDIV, f32, fkf(L, f32, 1.0), t);
            pol = fkf(L, f32, ec[4]);
            for (int k = 3; k >= 0; k--)
                pol = fbin(L, BIR_FADD, f32,
                           fkf(L, f32, ec[k]),
                           fbin(L, BIR_FMUL, f32, pol, t));
            pol = fbin(L, BIR_FMUL, f32, pol, t);
            sq = fbin(L, BIR_FMUL, f32, BIR_MAKE_VAL(ab), BIR_MAKE_VAL(ab));
            sq = fbin(L, BIR_FMUL, f32, sq, fkf(L, f32, -1.4426950408889634));
            ex = emit(L, BIR_EXP2, f32, 1, 0);
            set_op(L, ex, 0, sq);
            r = fbin(L, BIR_FSUB, f32, fkf(L, f32, 1.0),
                     fbin(L, BIR_FMUL, f32, pol, BIR_MAKE_VAL(ex)));
            neg = fbin(L, BIR_FSUB, f32, fkf(L, f32, 0.0), r);
            c = emit(L, BIR_FCMP, bir_type_int(L->M, 1), 2, BIR_FCMP_OLT);
            set_op(L, c, 0, v);
            set_op(L, c, 1, fkf(L, f32, 0.0));
            {
                uint32_t sl2 = emit(L, BIR_SELECT, f32, 3, 0);
                set_op(L, sl2, 0, BIR_MAKE_VAL(c));
                set_op(L, sl2, 1, neg);
                set_op(L, sl2, 2, r);
                return BIR_MAKE_VAL(sl2);
            }
        }

        {
            static const char *const nyi[] = {
                "__prmt", "__activemask", "__match_any_sync",
                "__funnelshift_l", "__funnelshift_r"
            };
            for (int ni = 0; ni < (int)(sizeof nyi / sizeof nyi[0]); ni++)
                if (strcmp(cname, nyi[ni]) == 0) {
                    lower_error(L, node, BC_E640, cname);
                    return BIR_VAL_NONE;
                }
        }

        if (strcmp(cname, "__dp4a") == 0) {
            uint32_t i32 = bir_type_int(L->M, 32);
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            uint32_t a1, acc;
            an = an ? ND(L, an)->next_sibling : 0;
            a1 = lower_expr(L, an);
            an = an ? ND(L, an)->next_sibling : 0;
            acc = lower_expr(L, an);
            for (int b = 0; b < 4; b++) {
                uint32_t sh = BIR_MAKE_CONST(bir_const_int(L->M, i32, 24 - 8 * b));
                uint32_t k24 = BIR_MAKE_CONST(bir_const_int(L->M, i32, 24));
                uint32_t src[2], ext[2];
                uint32_t pr, nx;
                src[0] = a0; src[1] = a1;
                for (int k = 0; k < 2; k++) {
                    uint32_t sl = emit(L, BIR_SHL, i32, 2, 0);
                    uint32_t ar;
                    set_op(L, sl, 0, src[k]);
                    set_op(L, sl, 1, sh);
                    ar = emit(L, BIR_ASHR, i32, 2, 0);
                    set_op(L, ar, 0, BIR_MAKE_VAL(sl));
                    set_op(L, ar, 1, k24);
                    ext[k] = BIR_MAKE_VAL(ar);
                }
                pr = emit(L, BIR_MUL, i32, 2, 0);
                set_op(L, pr, 0, ext[0]);
                set_op(L, pr, 1, ext[1]);
                nx = emit(L, BIR_ADD, i32, 2, 0);
                set_op(L, nx, 0, acc);
                set_op(L, nx, 1, BIR_MAKE_VAL(pr));
                acc = BIR_MAKE_VAL(nx);
            }
            return acc;
        }

        if (strcmp(cname, "__ldg") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t pv = lower_expr(L, an);
            uint32_t pt = ref_type(L, pv);
            uint32_t et = ptr_inner(L, pt), ld;
            if (!et) {
                lower_error(L, node, BC_E143, cname);
                return BIR_VAL_NONE;
            }
            ld = emit(L, BIR_LOAD, et, 1, 0);
            set_op(L, ld, 0, pv);
            return BIR_MAKE_VAL(ld);
        }

        if (strcmp(cname, "ldexpf") == 0 || strcmp(cname, "scalbnf") == 0) {
            uint32_t f32 = bir_type_float(L->M, 32);
            uint32_t i32 = bir_type_int(L->M, 32);
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t x = lower_expr(L, an);
            uint32_t e, run;
            an = an ? ND(L, an)->next_sibling : 0;
            e = coerce_to(L, lower_expr(L, an), i32, 0);
            run = x;
            for (int st = 0; st < 3; st++) {
                uint32_t lo = BIR_MAKE_CONST(bir_const_int(L->M, i32, -126));
                uint32_t hi = BIR_MAKE_CONST(bir_const_int(L->M, i32, 126));
                uint32_t bias = BIR_MAKE_CONST(bir_const_int(L->M, i32, 127));
                uint32_t sh23 = BIR_MAKE_CONST(bir_const_int(L->M, i32, 23));
                uint32_t cl, sl, ch, sh, add, shl, bc, mul, sub;
                cl = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2, BIR_ICMP_SLT);
                set_op(L, cl, 0, e); set_op(L, cl, 1, lo);
                sl = emit(L, BIR_SELECT, i32, 3, 0);
                set_op(L, sl, 0, BIR_MAKE_VAL(cl));
                set_op(L, sl, 1, lo); set_op(L, sl, 2, e);
                ch = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2, BIR_ICMP_SGT);
                set_op(L, ch, 0, BIR_MAKE_VAL(sl)); set_op(L, ch, 1, hi);
                sh = emit(L, BIR_SELECT, i32, 3, 0);
                set_op(L, sh, 0, BIR_MAKE_VAL(ch));
                set_op(L, sh, 1, hi); set_op(L, sh, 2, BIR_MAKE_VAL(sl));
                add = emit(L, BIR_ADD, i32, 2, 0);
                set_op(L, add, 0, BIR_MAKE_VAL(sh)); set_op(L, add, 1, bias);
                shl = emit(L, BIR_SHL, i32, 2, 0);
                set_op(L, shl, 0, BIR_MAKE_VAL(add)); set_op(L, shl, 1, sh23);
                bc = emit(L, BIR_BITCAST, f32, 1, 0);
                set_op(L, bc, 0, BIR_MAKE_VAL(shl));
                mul = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, mul, 0, run); set_op(L, mul, 1, BIR_MAKE_VAL(bc));
                run = BIR_MAKE_VAL(mul);
                sub = emit(L, BIR_SUB, i32, 2, 0);
                set_op(L, sub, 0, e); set_op(L, sub, 1, BIR_MAKE_VAL(sh));
                e = BIR_MAKE_VAL(sub);
            }
            return run;
        }

        if (strcmp(cname, "max") == 0 || strcmp(cname, "min") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t a0 = lower_expr(L, an);
            uint32_t a1, t0, r;
            an = an ? ND(L, an)->next_sibling : 0;
            a1 = lower_expr(L, an);
            t0 = ref_type(L, a0);
            if (is_float_type(L, t0)) {
                r = emit(L, cname[1] == 'a' ? BIR_FMAX : BIR_FMIN, t0, 2, 0);
                set_op(L, r, 0, a0);
                set_op(L, r, 1, a1);
                return BIR_MAKE_VAL(r);
            }
            {
                uint32_t t1 = ref_type(L, a1);
                uint32_t wt = t0;
                uint32_t cm;
                if (t0 < L->M->num_types && t1 < L->M->num_types
                    && L->M->types[t1].width > L->M->types[t0].width)
                    wt = t1;
                a0 = coerce_to(L, a0, wt, 0);
                a1 = coerce_to(L, a1, wt, 0);
                cm = emit(L, BIR_ICMP, bir_type_int(L->M, 1), 2,
                          cname[1] == 'a' ? BIR_ICMP_SGT : BIR_ICMP_SLT);
                set_op(L, cm, 0, a0);
                set_op(L, cm, 1, a1);
                r = emit(L, BIR_SELECT, wt, 3, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(cm));
                set_op(L, r, 1, a0);
                set_op(L, r, 2, a1);
                return BIR_MAKE_VAL(r);
            }
        }

        if (strcmp(cname, "memcpy") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t dp = lower_expr(L, an);
            uint32_t sp, nv, off = 0;
            int64_t n64;
            an = an ? ND(L, an)->next_sibling : 0;
            sp = lower_expr(L, an);
            an = an ? ND(L, an)->next_sibling : 0;
            nv = lower_expr(L, an);
            if (!BIR_VAL_IS_CONST(nv)
                || BIR_VAL_INDEX(nv) >= L->M->num_consts
                || L->M->consts[BIR_VAL_INDEX(nv)].kind != BIR_CONST_INT) {
                lower_error(L, node, BC_E143, cname);
                return BIR_VAL_NONE;
            }
            n64 = L->M->consts[BIR_VAL_INDEX(nv)].d.ival;
            if (n64 < 0 || n64 > 256) {
                lower_error(L, node, BC_E143, cname);
                return BIR_VAL_NONE;
            }
            uint8_t das = 0, sas = 0;
            {
                uint32_t dt = ref_type(L, dp), st2 = ref_type(L, sp);
                if (dt < L->M->num_types) das = L->M->types[dt].addrspace;
                if (st2 < L->M->num_types) sas = L->M->types[st2].addrspace;
            }
            while (off < (uint32_t)n64) {
                uint32_t left = (uint32_t)n64 - off;
                uint32_t w = left >= 8 ? 8 : (left >= 4 ? 4 : (left >= 2 ? 2 : 1));
                uint32_t it = bir_type_int(L->M, (int)(w * 8u));
                uint32_t spt = bir_type_ptr(L->M, it, sas);
                uint32_t dpt = bir_type_ptr(L->M, it, das);
                uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                                  bir_type_int(L->M, 32), off / w));
                uint32_t sg, dg, ld, sv;
                sg = emit(L, BIR_GEP, spt, 2, 0);
                set_op(L, sg, 0, sp); set_op(L, sg, 1, ci);
                dg = emit(L, BIR_GEP, dpt, 2, 0);
                set_op(L, dg, 0, dp); set_op(L, dg, 1, ci);
                ld = emit(L, BIR_LOAD, it, 1, 0);
                set_op(L, ld, 0, BIR_MAKE_VAL(sg));
                sv = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
                set_op(L, sv, 0, BIR_MAKE_VAL(ld));
                set_op(L, sv, 1, BIR_MAKE_VAL(dg));
                off += w;
            }
            return dp;
        }

        if (strcmp(cname, "__trap") == 0) {
            uint32_t inst = emit(L, BIR_TRAP, bir_type_void(L->M), 0, 0);
            return BIR_MAKE_VAL(inst);
        }

        if (strcmp(cname, "assert") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t cv, okb, bad, br;
            if (!an) {
                lower_error(L, node, BC_E143, cname);
                return BIR_VAL_NONE;
            }
            cv  = lower_expr(L, an);
            okb = new_block(L, "asrt.ok");
            bad = new_block(L, "asrt.bad");
            br  = emit(L, BIR_BR_COND, bir_type_void(L->M), 4, 0);
            set_op(L, br, 0, cv);
            set_op(L, br, 1, okb);
            set_op(L, br, 2, bad);
            set_op(L, br, 3, okb);
            set_block(L, bad);
            emit(L, BIR_TRAP, bir_type_void(L->M), 0, 0);
            if (!block_terminated(L)) {
                uint32_t j = emit(L, BIR_BR, bir_type_void(L->M), 1, 0);
                set_op(L, j, 0, okb);
            }
            set_block(L, okb);
            return BIR_VAL_NONE;
        }

        if (strcmp(cname, "printf") == 0)
            return prntf(L, node, callee_n);

        {
            static const char *const nyi[] = {
                "vprintf", "__brkpt",
                "__assertfail", "malloc", "free",
            };
            for (int ni = 0; ni < (int)(sizeof nyi / sizeof nyi[0]); ni++) {
                if (strcmp(cname, nyi[ni]) != 0) continue;
                lower_error(L, node, BC_E143, cname);
                return BIR_VAL_NONE;
            }
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

        if (strcmp(cname, "__isnanf") == 0 || strcmp(cname, "isnanf") == 0
            || strcmp(cname, "__isnan") == 0 || strcmp(cname, "isnan") == 0) {
            uint32_t an = ND(L, callee_n)->next_sibling;
            uint32_t v = lower_expr(L, an);
            uint32_t vt = ref_type(L, v);
            uint32_t i1 = bir_type_int(L->M, 1);
            uint32_t zf = BIR_MAKE_CONST(bir_const_float(L->M, vt, 0.0));
            uint32_t le = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OLE);
            set_op(L, le, 0, v); set_op(L, le, 1, zf);
            uint32_t gt = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OGT);
            set_op(L, gt, 0, v); set_op(L, gt, 1, zf);
            uint32_t o = emit(L, BIR_OR, i1, 2, 0);
            set_op(L, o, 0, BIR_MAKE_VAL(le)); set_op(L, o, 1, BIR_MAKE_VAL(gt));
            uint32_t r = emit(L, BIR_ICMP, i1, 2, BIR_ICMP_EQ);
            set_op(L, r, 0, BIR_MAKE_VAL(o));
            set_op(L, r, 1, BIR_MAKE_CONST(bir_const_int(L->M, i1, 0)));
            return BIR_MAKE_VAL(r);
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
                {"truncf",BIR_FTRUNC},{"trunc",BIR_FTRUNC},
                {"floor",BIR_FLOOR},{"ceil",BIR_CEIL},
                {"rintf",BIR_RNDNE},{"rint",BIR_RNDNE},
                {"nearbyintf",BIR_RNDNE},{"nearbyint",BIR_RNDNE},
            };
            for (int mi = 0; mi < (int)(sizeof mt1 / sizeof mt1[0]); mi++) {
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
                {"fmax",BIR_FMAX},{"fmin",BIR_FMIN},{"fmod",BIR_FREM},
            };
            for (int mi = 0; mi < (int)(sizeof mt2 / sizeof mt2[0]); mi++) {
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
            if (strcmp(cname, "log1pf") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t one = BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.0));
                uint32_t sm = emit(L, BIR_FADD, f32, 2, 0);
                set_op(L, sm, 0, v); set_op(L, sm, 1, one);
                uint32_t lg = emit(L, BIR_LOG2, f32, 1, 0);
                set_op(L, lg, 0, BIR_MAKE_VAL(sm));
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.6931471805599453));
                uint32_t r = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(lg)); set_op(L, r, 1, k);
                return BIR_MAKE_VAL(r);
            }
            if (strcmp(cname, "expm1f") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t k = BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.4426950408889634));
                uint32_t m = emit(L, BIR_FMUL, f32, 2, 0);
                set_op(L, m, 0, v); set_op(L, m, 1, k);
                uint32_t e = emit(L, BIR_EXP2, f32, 1, 0);
                set_op(L, e, 0, BIR_MAKE_VAL(m));
                uint32_t one = BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.0));
                uint32_t r = emit(L, BIR_FSUB, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(e)); set_op(L, r, 1, one);
                return BIR_MAKE_VAL(r);
            }
            if (strcmp(cname, "roundf") == 0 || strcmp(cname, "round") == 0) {
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t v = lower_expr(L, an);
                uint32_t i1 = bir_type_int(L->M, 1);
                uint32_t zf = BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.0));
                uint32_t t = emit(L, BIR_FTRUNC, f32, 1, 0);
                set_op(L, t, 0, v);
                uint32_t d = emit(L, BIR_FSUB, f32, 2, 0);
                set_op(L, d, 0, v); set_op(L, d, 1, BIR_MAKE_VAL(t));
                uint32_t ad = emit(L, BIR_FABS, f32, 1, 0);
                set_op(L, ad, 0, BIR_MAKE_VAL(d));
                uint32_t c = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OGE);
                set_op(L, c, 0, BIR_MAKE_VAL(ad));
                set_op(L, c, 1, BIR_MAKE_CONST(bir_const_float(L->M, f32, 0.5)));
                uint32_t ng = emit(L, BIR_FCMP, i1, 2, BIR_FCMP_OLT);
                set_op(L, ng, 0, v); set_op(L, ng, 1, zf);
                uint32_t sg = emit(L, BIR_SELECT, f32, 3, 0);
                set_op(L, sg, 0, BIR_MAKE_VAL(ng));
                set_op(L, sg, 1, BIR_MAKE_CONST(bir_const_float(L->M, f32, -1.0)));
                set_op(L, sg, 2, BIR_MAKE_CONST(bir_const_float(L->M, f32, 1.0)));
                uint32_t m = emit(L, BIR_SELECT, f32, 3, 0);
                set_op(L, m, 0, BIR_MAKE_VAL(c));
                set_op(L, m, 1, BIR_MAKE_VAL(sg));
                set_op(L, m, 2, zf);
                uint32_t r = emit(L, BIR_FADD, f32, 2, 0);
                set_op(L, r, 0, BIR_MAKE_VAL(t)); set_op(L, r, 1, BIR_MAKE_VAL(m));
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

        /* ---- Warp-collective 16x16x16 f16 matrix multiply ---- */
        if (strncmp(cname, "__builtin_mma_", 14) == 0) {
            static const char *const mma_tab[] = {
                "m16n16k16_f16", "m16n16k16_bf16",
                "m16n16k8_f16",  "m16n16k8_bf16",
            };
            const char *sfx = cname + 14;
            for (int mi = 0; mi < (int)(sizeof mma_tab / sizeof mma_tab[0]); mi++) {
                if (strcmp(sfx, mma_tab[mi]) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t ops[6];
                int na = 0;
                while (an != 0 && na < 6) {
                    ops[na++] = lower_expr(L, an);
                    an = ND(L, an)->next_sibling;
                }
                if (na != 6) return BIR_VAL_NONE;
                uint32_t r = emit(L, BIR_MMA, bir_type_void(L->M), 6,
                                  (uint8_t)mi);
                for (int k = 0; k < 6; k++) set_op(L, r, k, ops[k]);
                return BIR_MAKE_VAL(r);
            }
            return BIR_VAL_NONE;
        }

        /* ---- MFMA over per-lane fragments in memory ---- */
        if (strncmp(cname, "__builtin_mfma_", 15) == 0) {
            static const char *const mfrg_tab[] = {
                "f32_4x4x4_f16", "f32_16x16x16_f16", "f32_32x32x8_f16",
                "f32_4x4x4_bf16", "f32_16x16x16_bf16", "f32_32x32x8_bf16",
                "f32_4x4x1_f32", "f32_16x16x4_f32", "f32_32x32x2_f32",
                "i32_4x4x4_i8", "i32_16x16x16_i8", "i32_32x32x8_i8",
                "f32_16x16x32_fp8_fp8", "f32_16x16x32_fp8_bf8",
                "f32_16x16x32_bf8_fp8", "f32_16x16x32_bf8_bf8",
                "f32_32x32x16_fp8_fp8", "f32_32x32x16_fp8_bf8",
                "f32_32x32x16_bf8_fp8", "f32_32x32x16_bf8_bf8",
                "f64_4x4x4_f64", "f64_16x16x4_f64",
                "i32_16x16x32_i8", "i32_32x32x16_i8",
            };
            const char *sfx = cname + 15;
            for (int mi = 0; mi < (int)(sizeof mfrg_tab / sizeof mfrg_tab[0]); mi++) {
                if (strcmp(sfx, mfrg_tab[mi]) != 0) continue;
                uint32_t an = ND(L, callee_n)->next_sibling;
                uint32_t ops[3];
                int na = 0;
                while (an != 0 && na < 3) {
                    ops[na++] = lower_expr(L, an);
                    an = ND(L, an)->next_sibling;
                }
                if (na != 3) return BIR_VAL_NONE;
                uint32_t r = emit(L, BIR_MFRG, bir_type_void(L->M), 3,
                                  (uint8_t)mi);
                for (int k = 0; k < 3; k++) set_op(L, r, k, ops[k]);
                return BIR_MAKE_VAL(r);
            }
            return BIR_VAL_NONE;
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
            for (int mi = 0; mi < (int)(sizeof mfma_tab / sizeof mfma_tab[0]); mi++) {
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
            for (int oi = 0; oi < (int)(sizeof ockl / sizeof ockl[0]); oi++) {
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
            for (int oi = 0; oi < (int)(sizeof ou / sizeof ou[0]); oi++) {
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
            for (int oi = 0; oi < (int)(sizeof ob / sizeof ob[0]); oi++) {
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

        uint32_t args[BC_MAX_ARGS], argn[BC_MAX_ARGS], defs[BC_MAX_ARGS];
        uint32_t cand[OV_CAND];
        int nargs = 0, haspk = 0, ncand = 0, ndef = 0, di;
        uint32_t arg = ND(L, callee_n)->next_sibling;
        uint32_t rfm = 0, fi, lvm = 0;

        if (!fbc && !qtsym(L, cname, sizeof(cname), callee_n)
            && tsym(L, cname, sizeof(cname), callee_n, arg) < 0) {
            uint32_t bd = tebad(L, callee_n);
            char en[96];

            if (bd) {
                cxnm(L, bd, en, (int)sizeof en);
                lower_error(L, node, BC_E720, en);
            } else {
                get_text(L, callee_n, en, sizeof(en));
                lower_error(L, node, BC_E721, en);
            }
            return BIR_VAL_NONE;
        }

        if (self != BIR_VAL_NONE) nargs = 1;
        for (uint32_t a = arg; a; a = ND(L, a)->next_sibling) {
            if (ND(L, a)->type == AST_PACK_EXP) haspk = 1;
            nargs++;
        }
        if (!haspk) ncand = fcand(L, cname, nargs, cand, OV_CAND);
        if (!haspk && ncand == 0 && nargs < BC_MAX_ARGS) {
            ndef = fdflt(L, cname, nargs, defs, BC_MAX_ARGS - nargs);
            if (ndef > 0)
                ncand = fcand(L, cname, nargs + ndef, cand, OV_CAND);
            if (ncand <= 0) ndef = 0;
        }
        if (ncand < 0) {
            lower_error(L, node, BC_E156, cname);
            return BIR_VAL_NONE;
        }
        for (int ci = 1; ci < ncand; ci++)
            if (L->M->funcs[cand[ci]].refm != L->M->funcs[cand[0]].refm
                || L->M->funcs[cand[ci]].refq != L->M->funcs[cand[0]].refq) {
                lower_error(L, node, BC_E156, cname);
                return BIR_VAL_NONE;
            }
        fi = ncand == 1 ? cand[0] : BIR_SYM_NONE;
        if (ncand > 0) rfm = L->M->funcs[cand[0]].refm;

        nargs = 0;
        memset(argn, 0, sizeof argn);
        if (self != BIR_VAL_NONE) args[nargs++] = self;
        while (arg && nargs < BC_MAX_ARGS) {
            if (ND(L, arg)->type == AST_PACK_EXP) {
                int got = pk_exp(L, arg, args + nargs, BC_MAX_ARGS - nargs);
                if (got < 0) return BIR_VAL_NONE;
                nargs += got;
            } else if (nargs < 32 && (rfm & (1u << nargs))) {
                int lv = 0;

                argn[nargs] = arg;
                args[nargs] = rlv(L, arg, &lv);
                if (args[nargs] == BIR_VAL_NONE) return BIR_VAL_NONE;
                if (lv) lvm |= 1u << nargs;
                nargs++;
            } else {
                argn[nargs] = arg;
                args[nargs++] = lower_expr(L, arg);
            }
            arg = ND(L, arg)->next_sibling;
        }
        /* Sema rejects this first, so reaching it means the two caps have
         * drifted apart. Dropping the tail would emit a call with the wrong
         * operands and no sign anything was lost. */
        if (arg) {
            lower_error(L, node, BC_E082, "call", BC_MAX_ARGS);
            return BIR_VAL_NONE;
        }
        for (di = 0; di < ndef && nargs < BC_MAX_ARGS; di++) {
            if (nargs < 32 && (rfm & (1u << nargs))) {
                int lv = 0;

                argn[nargs] = defs[di];
                args[nargs] = rlv(L, defs[di], &lv);
                if (args[nargs] == BIR_VAL_NONE) return BIR_VAL_NONE;
                if (lv) lvm |= 1u << nargs;
                nargs++;
                continue;
            }
            argn[nargs] = defs[di];
            args[nargs++] = lower_expr(L, defs[di]);
        }

        if (fi == BIR_SYM_NONE && ncand > 1) {
            uint32_t at[BC_MAX_ARGS];
            int amb = 0;
            for (int i = 0; i < nargs; i++) at[i] = ref_type(L, args[i]);
            fi = fpick(L, cand, ncand, at, argn, nargs, &amb);
            if (amb || fi == BIR_SYM_NONE) {
                lower_error(L, node, BC_E156, cname);
                return BIR_VAL_NONE;
            }
        }
        if (fi == BIR_SYM_NONE) {
            fi = bir_fsym(L->M, cname, L->tu, nargs);
            if (fi == BIR_SYM_NONE) {
                uint32_t any = bir_fsym(L->M, cname, L->tu, -1);
                if (any == BIR_SYM_NONE) {
                    uint32_t bd = tebad(L, callee_n);
                    char en[96];

                    if (bd) {
                        cxnm(L, bd, en, (int)sizeof en);
                        lower_error(L, node, BC_E720, en);
                    } else if (hasta(L, callee_n)
                               || find_template(L, cname)) {
                        get_text(L, callee_n, en, sizeof(en));
                        lower_error(L, node, BC_E721, en);
                    } else {
                        lower_error(L, node, BC_E105, cname);
                    }
                } else
                    lower_error(L, node, BC_E073, cname,
                                (int)(L->M->funcs[any].num_params
                                      - L->M->funcs[any].sret), nargs);
                return BIR_VAL_NONE;
            }
            if (L->M->funcs[fi].refm) {
                lower_error(L, node, BC_E148, cname);
                return BIR_VAL_NONE;
            }
        }

        uint32_t ftype = L->M->funcs[fi].type;
        uint32_t ret_t = L->M->types[ftype].inner;

        for (int i = 0; i < nargs; i++) {
            if (i < 32 && (L->M->funcs[fi].refm & (1u << i))) {
                args[i] = rfit(L, argn[i] ? argn[i] : node, args[i],
                               (int)((lvm >> i) & 1u),
                               ptr_inner(L, ftpar(L, ftype, i)),
                               (int)((L->M->funcs[fi].refq >> i) & 1u), cname);
                if (args[i] == BIR_VAL_NONE) return BIR_VAL_NONE;
                continue;
            }
            {
                uint32_t pt = ftpar(L, ftype, i);
                uint32_t pin = ptr_inner(L, pt);

                if (aggt(L, pin) && ref_type(L, args[i]) == pin) {
                    args[i] = apsrc(L, argn[i] ? argn[i] : node,
                                    args[i], pin);
                    if (args[i] == BIR_VAL_NONE) return BIR_VAL_NONE;
                    continue;
                }
            }
            args[i] = stfit(L, args[i], ftpar(L, ftype, i), argn[i]);
        }

        uint32_t sslot = BIR_VAL_NONE;
        if (L->M->funcs[fi].sret) {
            uint32_t aty = ptr_inner(L, ftpar(L, ftype,
                                     (int)L->M->funcs[fi].num_params - 1));
            if (!aty || nargs >= BC_MAX_ARGS) {
                lower_error(L, node, BC_E105, cname);
                return BIR_VAL_NONE;
            }
            if (hdst != BIR_VAL_NONE && hdty == aty) {
                sslot = hdst;
                for (int i = 0; i < nargs; i++)
                    if (args[i] == hdst) sslot = BIR_VAL_NONE;
            }
            if (sslot == BIR_VAL_NONE)
                sslot = BIR_MAKE_VAL(emalc(L, BIR_ALLOCA,
                            bir_type_ptr(L->M, aty, BIR_AS_PRIVATE), 0));
            args[nargs++] = sslot;
        }

        if (1 + nargs <= BIR_OPERANDS_INLINE) {
            uint32_t inst = emit(L, BIR_CALL, ret_t, (uint8_t)(1+nargs), 0);
            set_op(L, inst, 0, fi);
            for (int i = 0; i < nargs; i++)
                set_op(L, inst, 1+i, args[i]);
            return sslot != BIR_VAL_NONE ? sslot : BIR_MAKE_VAL(inst);
        }
        /* Overflow mode: pack into extra_operands */
        {
            uint32_t extra_start = L->M->num_extra_ops;
            /* All of it or none: packing what fits drops arguments. */
            if (L->M->num_extra_ops + 1u + (uint32_t)nargs > BIR_MAX_EXTRA_OPS) {
                bir_pfull(L->M, BIR_P_EXTRAOPS);
                return BIR_VAL_NONE;
            }
            L->M->extra_operands[L->M->num_extra_ops++] = fi;
            for (int i = 0; i < nargs; i++)
                L->M->extra_operands[L->M->num_extra_ops++] = args[i];
            uint32_t total = L->M->num_extra_ops - extra_start;
            uint32_t inst = emit(L, BIR_CALL, ret_t, BIR_OPERANDS_OVERFLOW, 0);
            set_op(L, inst, 0, extra_start);
            set_op(L, inst, 1, total);
            return sslot != BIR_VAL_NONE ? sslot : BIR_MAKE_VAL(inst);
        }
    }

    case AST_CAST: {
        uint32_t type_n = n->first_child;
        uint32_t expr_n = ND(L, type_n)->next_sibling;
        int pdepth      = n->d.oper.flags;

        /* C++ aggregate init: Type{expr, ...} → alloca + field stores + load */
        if (expr_n && ND(L, expr_n)->type == AST_INIT_LIST) {
            uint32_t st = rtype(L, type_n, pdepth, 0);
            struct_def_t *sd = NULL;
            for (int si = 0; si < L->nstructs; si++) {
                if (L->structs[si].bir_type == st) { sd = &L->structs[si]; break; }
            }
            if (sd) {
                uint32_t ptr_t = bir_type_ptr(L->M, st, BIR_AS_PRIVATE);
                uint32_t alloca = emalc(L, BIR_ALLOCA, ptr_t, 0);
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

        uint32_t dst_t  = rtype(L, type_n, pdepth, 0);
        uint32_t val    = lower_expr(L, expr_n);
        uint32_t src_t  = ref_type(L, val);

        if (src_t == dst_t) return val;
        if (aggr(L, src_t) != aggr(L, dst_t)) {
            lower_error(L, node, BC_E030,
                        "conversion between a scalar and an aggregate");
            return BIR_VAL_NONE;
        }

        int sf = is_float_type(L, src_t), df = is_float_type(L, dst_t);
        int sp = is_ptr_type(L, src_t),   dp = is_ptr_type(L, dst_t);
        if ((sp && df) || (sf && dp)) {
            lower_error(L, node, BC_E440,
                        "a pointer and a floating type never convert");
            return BIR_VAL_NONE;
        }
        if (aggv(L, expr_n) && scalr(L, dst_t)) {
            lower_error(L, node, BC_E440,
                        "an aggregate has no conversion to a scalar");
            return BIR_VAL_NONE;
        }
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
        uint32_t t = bir_type_int(L->M, 64);
        int64_t sz;

        if (!csize(L, node, &sz)) {
            char snm[CE_NAMEZ], det[CE_NAMEZ + 40];

            cndnm(L, n->first_child, snm, sizeof snm);
            if (snprintf(det, sizeof det, "sizeof '%s', whose size Booth "
                         "cannot compute", snm) < 0)
                det[0] = 0;
            lower_error(L, node, BC_E030, det);
            return BIR_VAL_NONE;
        }
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
        char bytes[2048];
        uint32_t blen = sdec(L->src + ND(L, node)->d.text.offset,
                             ND(L, node)->d.text.len, bytes, sizeof bytes);

        /* Stash the bytes in the strings table and create a global
         * for them. We never deduplicate; two source-distinct
         * literals stay distinct, which avoids the pointer-identity
         * surprise dedup would create. */
        return mkstr(L, node, bytes, blen);
    }

    case AST_PACK_SIZE: {
        uint32_t id = n->first_child;
        char nm[64];
        if (id) get_text(L, id, nm, sizeof(nm)); else nm[0] = 0;
        int cnt = pk_cnt(L, nm);
        if (cnt < 0) {
            lower_error(L, node, BC_E030, "sizeof... of an unbound pack");
            return BIR_VAL_NONE;
        }
        return BIR_MAKE_CONST(bir_const_int(L->M,
            bir_type_int(L->M, 32), cnt));
    }

    case AST_FOLD:
        return lfold(L, node);

    case AST_PACK_EXP:
        lower_error(L, node, BC_E030,
                    "pack expansion outside an argument list");
        return BIR_VAL_NONE;

    case AST_LAMBDA:
        lower_error(L, node, BC_E170, "a lambda");
        return BIR_VAL_NONE;

    case AST_NEW:
        lower_error(L, node, BC_E170, "new");
        return BIR_VAL_NONE;

    case AST_DELETE:
        lower_error(L, node, BC_E170, "delete");
        return BIR_VAL_NONE;

    default:
        lower_error(L, node, BC_E106);
        return BIR_VAL_NONE;
    }
}

#define FP_DEEP 4
#define FP_WORK 16

static int sfind(const lower_t *L, uint32_t st)
{
    for (int si = 0; si < L->nstructs; si++)
        if (L->structs[si].bir_type == st) return si;
    return -1;
}

static int fpath(const lower_t *L, uint32_t st, const char *fn,
                 uint32_t *path, int *plen)
{
    struct { uint32_t ty; int dep; uint32_t pf[FP_DEEP]; } wk[FP_WORK];
    int wn = 0;

    wk[wn].ty = st; wk[wn].dep = 0; wn++;

    KA_GUARD(g, 4096);
    while (wn > 0 && g--) {
        int si, fi, d, k;
        uint32_t pf[FP_DEEP];
        uint32_t ty;

        wn--;
        ty = wk[wn].ty;
        d  = wk[wn].dep;
        for (k = 0; k < d && k < FP_DEEP; k++) pf[k] = wk[wn].pf[k];

        si = sfind(L, ty);
        if (si < 0) continue;

        for (fi = 0; fi < L->structs[si].num_fields; fi++) {
            if (strcmp(L->structs[si].field_names[fi], fn) != 0) continue;
            if (d >= FP_DEEP) return 0;
            for (k = 0; k < d; k++) path[k] = pf[k];
            path[d] = (uint32_t)fi;
            *plen = d + 1;
            return 1;
        }

        for (fi = 0; fi < L->structs[si].num_fields; fi++) {
            if (!L->structs[si].fanon[fi]) continue;
            if (d + 1 >= FP_DEEP || wn >= FP_WORK) continue;
            wk[wn].ty  = L->structs[si].field_types[fi];
            wk[wn].dep = d + 1;
            for (k = 0; k < d; k++) wk[wn].pf[k] = pf[k];
            wk[wn].pf[d] = (uint32_t)fi;
            wn++;
        }
    }
    return 0;
}

static uint32_t dcfld(lower_t *L, uint32_t st, const char *fn)
{
    uint32_t path[FP_DEEP], ty = st;
    int plen = 0;

    if (!fpath(L, st, fn, path, &plen)) return 0;
    for (int i = 0; i < plen && i < FP_DEEP; i++) {
        int si = sfind(L, ty);
        if (si < 0 || path[i] >= (uint32_t)L->structs[si].num_fields)
            return 0;
        ty = L->structs[si].field_types[path[i]];
    }
    return ty;
}

static uint32_t dcty(lower_t *L, uint32_t node)
{
    uint32_t nd = node, l, r, st = 0;
    char nm[128], on[128];
    cval_t dv;
    sym_t *s;

    KA_GUARD(g, CE_NEST);
    while (nd && g-- && ND(L, nd)->type == AST_PAREN)
        nd = ND(L, nd)->first_child;
    if (!nd) return 0;

    if (ND(L, nd)->type == AST_IDENT) {
        get_text(L, nd, nm, sizeof nm);
        s = find_sym(L, nm);
        return s ? s->type : 0;
    }
    if (ND(L, nd)->type != AST_SCOPE_RES && ND(L, nd)->type != AST_MEMBER)
        return 0;

    sdres(L, nd, &dv);
    if (L->sdty && !L->sdarr) {
        L->sdbad = 0;
        return L->sdty;
    }

    l = ND(L, nd)->first_child;
    r = l ? ND(L, l)->next_sibling : 0;
    if (!l || !r) return 0;
    if (ND(L, l)->type != AST_IDENT || ND(L, r)->type != AST_IDENT) return 0;
    get_text(L, l, on, sizeof on);
    get_text(L, r, nm, sizeof nm);

    if (ND(L, nd)->type == AST_MEMBER) {
        s = find_sym(L, on);
        if (!s) return 0;
        st = s->type;
        if (ND(L, nd)->d.member.is_arrow) st = ptr_inner(L, st);
    } else if (!find_binding(L, on, &st) && !find_typedef(L, on, &st)) {
        int si = -1;
        for (int i = 0; i < L->nstructs; i++)
            if (strcmp(L->structs[i].name, on) == 0) { si = i; break; }
        if (si < 0) return 0;
        st = L->structs[si].bir_type;
    }
    return st ? dcfld(L, st, nm) : 0;
}

/* ---- L-Value Lowering ---- */

static uint32_t mfld(lower_t *L, const char *fn)
{
    uint32_t path[FP_DEEP], cur, cst, ld;
    int plen = 0, pi;
    sym_t *s;

    if (!L->mrec) return BIR_VAL_NONE;
    s = find_sym(L, "this");
    if (!s || !s->is_alloca) return BIR_VAL_NONE;
    if (!fpath(L, L->mrec, fn, path, &plen)) return BIR_VAL_NONE;

    ld = emit(L, BIR_LOAD, s->type, 1, 0);
    set_op(L, ld, 0, BIR_MAKE_VAL(s->ref));
    cur = BIR_MAKE_VAL(ld);
    cst = L->mrec;
    for (pi = 0; pi < plen; pi++) {
        uint32_t ft, fpt, idx, gep;
        int si = sfind(L, cst);
        if (si < 0) return BIR_VAL_NONE;
        ft  = L->structs[si].field_types[path[pi]];
        fpt = bir_type_ptr(L->M, ft, BIR_AS_GENERIC);
        idx = BIR_MAKE_CONST(bir_const_int(L->M,
            bir_type_int(L->M, 32), (int64_t)path[pi]));
        gep = emit(L, BIR_GEP, fpt, 2, 0);
        set_op(L, gep, 0, cur);
        set_op(L, gep, 1, idx);
        cur = BIR_MAKE_VAL(gep);
        cst = ft;
    }
    return cur;
}

static uint32_t lower_lvalue(lower_t *L, uint32_t node)
{
    if (!node) return BIR_VAL_NONE;
    node = unfw(L, node);
    const ast_node_t *n = ND(L, node);

    switch (n->type) {
    case AST_IDENT: {
        char name[128];
        get_text(L, node, name, sizeof(name));
        pk_rw(L, name, sizeof(name));
        sym_t *s = find_sym(L, name);
        if (!s) {
            uint32_t fp = mfld(L, name);
            if (fp != BIR_VAL_NONE) return fp;
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

        if (wmsym(L, obj)) {
            char wn[128];
            get_text(L, obj, wn, sizeof(wn));
            lower_error(L, node, BC_E910, wn);
            return BIR_VAL_NONE;
        }

        uint32_t obj_ptr;
        if (n->d.member.is_arrow)
            obj_ptr = lower_expr(L, obj);
        else
            obj_ptr = mtmp(L, obj);
        if (obj_ptr == BIR_VAL_NONE) return BIR_VAL_NONE;

        uint32_t pt = ref_type(L, obj_ptr);
        uint32_t st = ptr_inner(L, pt);

        /* Preserve source address space: local alloca → private,
         * global array element → global. Don't assume private —
         * cells[i].mat needs a global load, not a scratch read. */
        uint8_t src_as = L->M->types[pt].addrspace;
        {
            uint32_t path[FP_DEEP];
            int plen = 0;
            if (fpath(L, st, fname, path, &plen)) {
                uint32_t cur = obj_ptr, cst = st;
                for (int pi = 0; pi < plen; pi++) {
                    int si = sfind(L, cst);
                    uint32_t ft, fpt, idx, gep;
                    if (si < 0) return BIR_VAL_NONE;
                    ft  = L->structs[si].field_types[path[pi]];
                    fpt = bir_type_ptr(L->M, ft, src_as);
                    idx = BIR_MAKE_CONST(bir_const_int(L->M,
                        bir_type_int(L->M, 32), (int64_t)path[pi]));
                    gep = emit(L, BIR_GEP, fpt, 2, 0);
                    set_op(L, gep, 0, cur);
                    set_op(L, gep, 1, idx);
                    cur = BIR_MAKE_VAL(gep);
                    cst = ft;
                }
                return cur;
            }
        }
        for (int si = 0; si < L->nstructs; si++) {
            if (L->structs[si].bir_type != st) continue;
            if (L->structs[si].fwd) {
                lower_error(L, node, BC_E802, L->structs[si].name);
                return BIR_VAL_NONE;
            }
            if (!L->structs[si].anon) break;
            lower_error(L, node, BC_E240, fname, L->structs[si].name);
            return BIR_VAL_NONE;
        }
        lower_error(L, node, BC_E110);
        return BIR_VAL_NONE;
    }

    case AST_PAREN:
        return lower_lvalue(L, n->first_child);

    case AST_CAST: {
        uint32_t type_n = n->first_child;
        uint32_t src_n  = type_n ? ND(L, type_n)->next_sibling : 0;
        uint32_t lv, dt, st, pt, bc;

        if (!src_n) {
            lower_error(L, node, BC_E111);
            return BIR_VAL_NONE;
        }
        lv = lower_lvalue(L, src_n);
        if (lv == BIR_VAL_NONE) return BIR_VAL_NONE;
        dt = rtype(L, type_n, n->d.oper.flags, 0);
        st = ref_type(L, lv);
        if (st < L->M->num_types && ptr_inner(L, st) == dt) return lv;
        pt = bir_type_ptr(L->M, dt,
                          st < L->M->num_types ? L->M->types[st].addrspace
                                               : BIR_AS_PRIVATE);
        bc = emit(L, BIR_BITCAST, pt, 1, 0);
        set_op(L, bc, 0, lv);
        return BIR_MAKE_VAL(bc);
    }

    default:
        lower_error(L, node, BC_E111);
        return BIR_VAL_NONE;
    }
}

/* ---- Statement Lowering ---- */

static void rlocal(lower_t *L, uint32_t node, uint32_t name_n,
                   const char *name, uint32_t elem_t)
{
    const ast_node_t *n = ND(L, node);
    uint32_t in = ND(L, name_n)->next_sibling, av, it;

    if (in && ND(L, in)->type == AST_STRUCT_DEF)
        in = ND(L, in)->next_sibling;
    if (!in || ND(L, in)->type == AST_NONE) {
        lower_error(L, node, BC_E152, name);
        return;
    }
    if (!islv(L, in)) {
        if (!(n->qualifiers & QUAL_CONST) || !rflat(L, elem_t)) {
            lower_error(L, node, BC_E148, name);
            return;
        }
        av = rtmp(L, lower_expr(L, in), elem_t);
        if (av == BIR_VAL_NONE || BIR_VAL_IS_CONST(av)) return;
        add_sym(L, name, BIR_VAL_INDEX(av), elem_t, 1);
        return;
    }
    av = lower_lvalue(L, in);
    if (av == BIR_VAL_NONE || BIR_VAL_IS_CONST(av)) return;
    it = ptr_inner(L, ref_type(L, av));
    if (it != elem_t) {
        lower_error(L, node, BC_E149, name);
        return;
    }
    add_sym(L, name, BIR_VAL_INDEX(av), elem_t, 1);
}

static uint32_t agep(lower_t *L, uint32_t base, uint32_t ety, int64_t i)
{
    uint32_t pt = bir_type_ptr(L->M, ety, BIR_AS_PRIVATE);
    uint32_t ci = BIR_MAKE_CONST(bir_const_int(L->M,
                      bir_type_int(L->M, 32), i));
    uint32_t g  = emit(L, BIR_GEP, pt, 2, 0);

    set_op(L, g, 0, base);
    set_op(L, g, 1, ci);
    return BIR_MAKE_VAL(g);
}

static void astor(lower_t *L, uint32_t val, uint32_t ptr)
{
    uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);

    set_op(L, st, 0, val);
    set_op(L, st, 1, ptr);
}

static uint32_t azero(lower_t *L, uint32_t ty)
{
    if (is_float_type(L, ty))
        return BIR_MAKE_CONST(bir_const_float(L->M, ty, 0.0));
    if (is_ptr_type(L, ty))
        return BIR_MAKE_CONST(bir_const_null(L->M, ty));
    return BIR_MAKE_CONST(bir_const_int(L->M, ty, 0));
}

static int zfill(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 int depth)
{
    const bir_type_t *t;

    if (depth > AI_NEST || ty >= L->M->num_types) {
        lower_error(L, node, BC_E661,
                    "an initialiser nested deeper than Booth follows");
        return 0;
    }
    t = &L->M->types[ty];
    if (t->kind == BIR_TYPE_ARRAY) {
        for (uint32_t i = 0; i < t->count; i++)
            if (!zfill(L, node, agep(L, base, t->inner, (int64_t)i),
                       t->inner, depth + 1)) return 0;
        return 1;
    }
    if (t->kind == BIR_TYPE_STRUCT) {
        const struct_def_t *sd;
        int si = sfind(L, ty), nf;

        if (si < 0) {
            lower_error(L, node, BC_E661,
                        "an aggregate whose fields Booth does not know");
            return 0;
        }
        sd = &L->structs[si];
        nf = sd->uni ? (sd->num_fields > 0) : sd->num_fields;
        for (int f = 0; f < nf; f++)
            if (!zfill(L, node, agep(L, base, sd->field_types[f], f),
                       sd->field_types[f], depth + 1)) return 0;
        return 1;
    }
    if (!scalr(L, ty) && !is_ptr_type(L, ty)) {
        lower_error(L, node, BC_E661,
                    "an element with no zero Booth can write");
        return 0;
    }
    if (L->nzst >= AI_STORES) {
        lower_error(L, node, BC_E661,
                    "more elements to value-initialise than Booth writes");
        return 0;
    }
    L->nzst++;
    astor(L, azero(L, ty), base);
    return 1;
}

static int ainit(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 uint32_t il, int depth, const char *nm);

static int aelid(lower_t *L, uint32_t ty, uint32_t el)
{
    uint32_t et;

    if (!aggr(L, ty) || ND(L, el)->type == AST_INIT_LIST) return 0;
    et = arg_type(L, el);
    return et && et != ty && !aggr(L, et);
}

static int awalk(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 uint32_t *cur, int depth, const char *nm);

static int amemb(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 uint32_t *cur, int depth, const char *nm)
{
    const bir_type_t *t = &L->M->types[ty];

    if (t->kind == BIR_TYPE_ARRAY) {
        for (uint32_t i = 0; i < t->count; i++)
            if (!awalk(L, node, agep(L, base, t->inner, (int64_t)i),
                       t->inner, cur, depth + 1, nm)) return 0;
        return 1;
    }
    if (t->kind == BIR_TYPE_STRUCT) {
        const struct_def_t *sd;
        int si = sfind(L, ty), nf;

        if (si < 0) {
            lower_error(L, node, BC_E661,
                        "an aggregate whose fields Booth does not know");
            return 0;
        }
        sd = &L->structs[si];
        nf = sd->uni ? (sd->num_fields > 0) : sd->num_fields;
        for (int f = 0; f < nf; f++)
            if (!awalk(L, node, agep(L, base, sd->field_types[f], f),
                       sd->field_types[f], cur, depth + 1, nm)) return 0;
        return 1;
    }
    lower_error(L, node, BC_E661, "an aggregate Booth cannot take apart");
    return 0;
}

static int awalk(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 uint32_t *cur, int depth, const char *nm)
{
    uint32_t el = *cur;

    if (depth > AI_NEST || ty >= L->M->num_types) {
        lower_error(L, node, BC_E661,
                    "an initialiser nested deeper than Booth follows");
        return 0;
    }
    if (!el) return zfill(L, node, base, ty, depth);
    if (ND(L, el)->type == AST_INIT_LIST) {
        *cur = ND(L, el)->next_sibling;
        return ainit(L, el, base, ty, el, depth + 1, nm);
    }
    if (aelid(L, ty, el)) return amemb(L, node, base, ty, cur, depth, nm);
    *cur = ND(L, el)->next_sibling;
    astor(L, stfit(L, lower_expr(L, el), ty, el), base);
    return 1;
}

static int ainit(lower_t *L, uint32_t node, uint32_t base, uint32_t ty,
                 uint32_t il, int depth, const char *nm)
{
    uint32_t cur = ND(L, il)->first_child;
    int ok;

    if (depth > AI_NEST) {
        lower_error(L, node, BC_E661,
                    "an initialiser nested deeper than Booth follows");
        return 0;
    }
    if (aggr(L, ty)) ok = amemb(L, node, base, ty, &cur, depth, nm);
    else if (cur)    ok = awalk(L, node, base, ty, &cur, depth, nm);
    else             ok = zfill(L, node, base, ty, depth);
    if (!ok) return 0;
    if (cur) {
        lower_error(L, node, BC_E660, nm);
        return 0;
    }
    return 1;
}

static uint32_t rfnty(const lower_t *L)
{
    uint32_t ft;

    if (L->cur_func >= L->M->num_funcs) return 0;
    ft = L->M->funcs[L->cur_func].type;
    if (ft >= L->M->num_types) return 0;
    return L->M->types[ft].inner;
}

static int rilst(lower_t *L, uint32_t il, uint32_t slot, uint32_t aty)
{
    uint32_t t_void = bir_type_void(L->M);
    uint32_t ptr, alc, base, ld, rt;

    if (!aty || aty >= L->M->num_types) return 0;
    if (slot != BIR_VAL_NONE) {
        if (!ainit(L, il, slot, aty, il, 0, "a return value")) return 0;
        emit(L, BIR_RET, t_void, 0, 0);
        return 1;
    }
    ptr  = bir_type_ptr(L->M, aty, BIR_AS_PRIVATE);
    alc  = emalc(L, BIR_ALLOCA, ptr, 0);
    base = BIR_MAKE_VAL(alc);
    if (!ainit(L, il, base, aty, il, 0, "a return value")) return 0;
    ld = emit(L, BIR_LOAD, aty, 1, 0);
    set_op(L, ld, 0, base);
    rt = emit(L, BIR_RET, t_void, 1, 0);
    set_op(L, rt, 0, BIR_MAKE_VAL(ld));
    return 1;
}

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

    {
        uint32_t ln = ND(L, name_n)->next_sibling;
        if (ln && ND(L, ln)->type == AST_STRUCT_DEF)
            ln = ND(L, ln)->next_sibling;
        while (ln && ND(L, ln)->type == AST_PAREN)
            ln = ND(L, ln)->first_child;
        if (ln && ND(L, ln)->type == AST_LAMBDA) {
            uint32_t al = 0;
            int li = lmake(L, ln, &al), ns = L->nsyms;
            if (li < 0) return;
            add_sym(L, name, al, L->lams[li].cst, 1);
            if (L->nsyms == ns) {
                lower_error(L, node, BC_E030, "more locals than Booth tracks");
                return;
            }
            L->syms[L->nsyms - 1].lam = li + 1;
            return;
        }
    }

    if (n->qualifiers & QUAL_PINIT) {
        lower_error(L, node, BC_E136, name);
        return;
    }

    uint32_t elem_t = rtype(L, type_n, n->d.oper.flags, n->cuda_flags);
    uint32_t want = L->rtalgn;

    if (L->rtcg == 2 && !cgadd(L, name)) {
        lower_error(L, node, BC_E030, "more grid groups than Booth tracks");
        return;
    }
    uint32_t decl = algof(L, type_n);

    if (decl > want) want = decl;

    if (n->qualifiers & QUAL_RREF) {
        lower_error(L, node, BC_E146, name);
        return;
    }
    if (n->qualifiers & QUAL_REF) {
        rlocal(L, node, name_n, name, elem_t);
        return;
    }

    if ((n->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST)) && n->d.oper.op == 0) {
        uint32_t ci = ND(L, name_n)->next_sibling;
        int n0 = L->ncag;
        cval_t cv;

        if (ci && ND(L, ci)->type == AST_STRUCT_DEF)
            ci = ND(L, ci)->next_sibling;
        if (ci && cfold(L, ci, &cv) && ccvt(L, &cv, elem_t)) {
            if (!cadd(L, name, &cv, elem_t, 0)) {
                L->ncag = n0;
                lower_error(L, node, BC_E130, MAX_CEXPS);
                return;
            }
            if ((n->qualifiers & QUAL_CONSTEXPR) && cv.fnd) {
                L->ncag = n0;
                return;
            }
        }
        L->ncag = n0;
    }

    uint32_t next = ND(L, name_n)->next_sibling;
    /* Skip embedded anonymous struct_def (sema already registered it) */
    if (next && ND(L, next)->type == AST_STRUCT_DEF)
        next = ND(L, next)->next_sibling;
    int is_array = 0;
    uint32_t dims[8];
    int ndim = 0;
    int max_dims = n->d.oper.op;  /* how many [N] the parser saw */

    while (next && ndim < 8 && ndim < max_dims) {
        int64_t aval;
        if (ND(L, next)->type == AST_INIT_LIST) break;
        if (!cival(L, next, &aval) || aval < 0 || aval > CE_MAXDIM) {
            char an[96];

            abnm(L, name, next, an, (int)sizeof an);
            if (!sdemt(L, node)) lower_error(L, node, BC_E131, an);
            return;
        }
        dims[ndim++] = (uint32_t)aval;
        is_array = 1;
        next = ND(L, next)->next_sibling;
    }

    /* ---- __shared__ variables: LDS allocation, no initializer ---- */
    if (n->cuda_flags & CUDA_SHARED) {
        if (!is_array && max_dims < 0) {
            uint32_t arr_t = bir_type_array(L->M, elem_t, 0);
            uint32_t ptr_t = bir_type_ptr(L->M, arr_t, BIR_AS_SHARED);
            uint32_t sa = emalc(L, BIR_SHARED_ALLOC, ptr_t, want);
            add_sym(L, name, sa, arr_t, 1);
            return;
        }
        if (is_array) {
            uint32_t arr_t = elem_t;
            for (int d = ndim - 1; d >= 0; d--)
                arr_t = bir_type_array(L->M, arr_t, dims[d]);
            uint32_t ptr_t = bir_type_ptr(L->M, arr_t, BIR_AS_SHARED);
            uint32_t sa = emalc(L, BIR_SHARED_ALLOC, ptr_t, want);
            add_sym(L, name, sa, arr_t, 1);
        } else {
            uint32_t ptr_t = bir_type_ptr(L->M, elem_t, BIR_AS_SHARED);
            uint32_t sa = emalc(L, BIR_SHARED_ALLOC, ptr_t, want);
            add_sym(L, name, sa, elem_t, 1);
        }
        return;
    }

    if (is_array) {
        uint32_t arr_t = elem_t;
        for (int d = ndim - 1; d >= 0; d--)
            arr_t = bir_type_array(L->M, arr_t, dims[d]);
        uint32_t ptr_t = bir_type_ptr(L->M, arr_t, BIR_AS_PRIVATE);
        uint32_t alloca = emalc(L, BIR_ALLOCA, ptr_t, want);
        add_sym(L, name, alloca, arr_t, 1);
        /* Array initializer list: int arr[3] = {1, 2, 3}; */
        if (next && ND(L, next)->type == AST_INIT_LIST) {
            L->nzst = 0;
            ainit(L, node, BIR_MAKE_VAL(alloca), arr_t, next, 0, name);
        }
        return;
    }

    /* Scalar / pointer variable */
    uint32_t ptr_t = bir_type_ptr(L->M, elem_t, BIR_AS_PRIVATE);
    uint32_t alloca = emalc(L, BIR_ALLOCA, ptr_t, want);
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
            L->nzst = 0;
            ainit(L, node, BIR_MAKE_VAL(alloca), elem_t, init_n, 0, name);
        }
    } else if (init_n && ND(L, init_n)->type != AST_NONE) {
        uint32_t init_v;
        uint32_t vt;
        int did_copy = 0;

        if (aggt(L, elem_t)) {
            L->srdst = BIR_MAKE_VAL(alloca);
            L->srdty = elem_t;
        }
        init_v = lower_expr(L, init_n);
        L->srdst = BIR_VAL_NONE;
        L->srdty = 0;
        vt = ref_type(L, init_v);
        if (init_v == BIR_MAKE_VAL(alloca)) did_copy = 1;
        else if (is_ptr_type(L, vt) && aggt(L, elem_t))
            did_copy = acpy(L, init_n, BIR_MAKE_VAL(alloca), init_v, elem_t);
        if (!did_copy) {
            uint32_t st;
            init_v = stfit(L, init_v, elem_t, init_n);
            st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, init_v);
            set_op(L, st, 1, BIR_MAKE_VAL(alloca));
        }
    }
}

static int acons(lower_t *L, uint32_t node, uint32_t opn, int isout,
                 char *md, char *cl)
{
    char c[32], why[64];
    uint32_t k = sdec(L->src + ND(L, opn)->d.text.offset,
                      ND(L, opn)->d.text.len, c, sizeof c);
    uint32_t i = 0;

    *md = ' ';
    if (i < k && (c[i] == '=' || c[i] == '+')) *md = c[i++];
    while (i < k && (c[i] == '&' || c[i] == '%')) i++;
    if (k - i == 1 && c[i] && strchr("rlfdh", c[i])) {
        *cl = c[i];
        if (isout == (*md != ' ')) return 1;
    } else if (k - i == 1 && c[i] == 'n' && !isout) {
        *cl = 'n';
        if (*md == ' ') return 1;
    }
    if (k >= sizeof c) k = sizeof c - 1;
    c[k] = '\0';
    if (snprintf(why, sizeof why, "the operand constraint %c%s%c",
                 '"', c, '"') < 0)
        why[0] = '\0';
    lower_error(L, node, BC_E221, why);
    return 0;
}

static void asmi(lower_t *L, uint8_t ai, const uint32_t *ops, int n)
{
    uint32_t inst, base;
    int i;

    if (n <= BIR_OPERANDS_INLINE) {
        inst = emit(L, BIR_INLINE_ASM, bir_type_void(L->M), (uint8_t)n, ai);
        for (i = 0; i < n; i++)
            set_op(L, inst, i, ops[i]);
        return;
    }
    base = L->M->num_extra_ops;
    if (base + (uint32_t)n > BIR_MAX_EXTRA_OPS) {
        bir_pfull(L->M, BIR_P_EXTRAOPS);
        return;
    }
    for (i = 0; i < n; i++)
        L->M->extra_operands[L->M->num_extra_ops++] = ops[i];
    inst = emit(L, BIR_INLINE_ASM, bir_type_void(L->M),
                BIR_OPERANDS_OVERFLOW, ai);
    set_op(L, inst, 0, base);
    set_op(L, inst, 1, (uint32_t)n);
}

static void lower_asm(lower_t *L, uint32_t node)
{
    char tmpl[BIR_ASM_TMPLZ], cons[2 * BIR_ASM_MAXOP + 1];
    uint32_t ops[BIR_ASM_MAXOP];
    const ast_node_t *n = ND(L, node);
    uint32_t tn = n->first_child, opn, to, co, ai;
    int nout = n->d.oper.op, ntot = n->d.oper.flags, i;
    uint8_t vol = (n->qualifiers & QUAL_VOLATILE) ? 1u : 0u;

    if (!tn || ntot < 0 || ntot > BIR_ASM_MAXOP || nout < 0 || nout > ntot)
        return;

    to = bir_add_string(L->M, tmpl,
                        sdec(L->src + ND(L, tn)->d.text.offset,
                             ND(L, tn)->d.text.len, tmpl, sizeof tmpl));

    opn = ND(L, tn)->next_sibling;
    for (i = 0; i < ntot; i++) {
        char md, cl;
        uint32_t e;

        if (!opn || ND(L, opn)->type != AST_ASM_OP) return;
        if (!acons(L, node, opn, i < nout, &md, &cl)) return;
        cons[2 * i] = md;
        cons[2 * i + 1] = cl;
        e = ND(L, opn)->first_child;
        ops[i] = (i < nout) ? lower_lvalue(L, e) : lower_expr(L, e);
        if (ops[i] == BIR_VAL_NONE) return;
        if (cl == 'n' && !BIR_VAL_IS_CONST(ops[i])) {
            lower_error(L, node, BC_E221,
                        "an \"n\" operand that is not a constant");
            return;
        }
        opn = ND(L, opn)->next_sibling;
    }
    cons[2 * ntot] = '\0';
    co = bir_add_string(L->M, cons, (uint32_t)(2 * ntot));

    ai = bir_asmd(L->M, to, co, (uint16_t)nout, (uint16_t)ntot, vol);
    if (ai == BIR_SYM_NONE) {
        lower_error(L, node, BC_E221,
                    "more asm statements than Booth holds");
        return;
    }

    asmi(L, (uint8_t)ai, ops, ntot);
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
        if (n->qualifiers & QUAL_TYPEDEF) collect_typedef(L, node);
        else lower_var_decl(L, node);
        break;

    case AST_ASM:
        lower_asm(L, node);
        break;

    case AST_RETURN: {
        uint32_t t_void = bir_type_void(L->M);
        uint32_t slot = srslot(L, L->cur_func);
        if (n->first_child
            && ND(L, n->first_child)->type == AST_INIT_LIST) {
            uint32_t aty = slot != BIR_VAL_NONE
                         ? ptr_inner(L, ref_type(L, slot)) : rfnty(L);
            if (!rilst(L, n->first_child, slot, aty)) {
                if (!aty || aty >= L->M->num_types)
                    lower_error(L, n->first_child, BC_E106);
                emit(L, BIR_RET, t_void, 0, 0);
            }
            break;
        }
        if (n->first_child && slot != BIR_VAL_NONE) {
            uint32_t aty = ptr_inner(L, ref_type(L, slot));
            uint32_t val;
            L->srdst = slot;
            L->srdty = aty;
            val = lower_expr(L, n->first_child);
            L->srdst = BIR_VAL_NONE;
            L->srdty = 0;
            if (aty && val != BIR_VAL_NONE && val != slot) {
                val = apsrc(L, n->first_child, val, aty);
                if (val != BIR_VAL_NONE)
                    (void)acpy(L, n->first_child, slot, val, aty);
            }
            emit(L, BIR_RET, t_void, 0, 0);
            break;
        }
        if (n->first_child) {
            uint32_t val  = lower_expr(L, n->first_child);
            uint32_t inst;
            if (L->cur_func < L->M->num_funcs) {
                uint32_t ft = L->M->funcs[L->cur_func].type;
                if (ft < L->M->num_types)
                    val = stfit(L, val, L->M->types[ft].inner,
                                n->first_child);
            }
            inst = emit(L, BIR_RET, t_void, 1, 0);
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

        if (n->d.oper.flags & IF_CEXP) {
            char cnm[CE_NAMEZ];
            cval_t cv;
            uint32_t arm;

            if (!cfsc(L, cond_n, &cv)) {
                cndnm(L, cond_n, cnm, sizeof cnm);
                if (!sdemt(L, node)) lower_error(L, node, BC_E158, cnm);
                return;
            }
            arm = (cv.isf ? cv.fval != 0.0 : cv.ival != 0) ? then_n : else_n;
            if (arm) {
                push_scope(L);
                lower_stmt(L, arm);
                pop_scope(L);
            }
            return;
        }

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
        /* Children: one init declarator per name, then cond, incr, body */
        uint32_t init_n = ND(L, node)->first_child;
        uint32_t cond_n = init_n, incr_n, body_n;

        push_scope(L);

        while (cond_n && ND(L, cond_n)->type == AST_VAR_DECL) {
            lower_var_decl(L, cond_n);
            cond_n = ND(L, cond_n)->next_sibling;
        }
        if (cond_n == init_n) {
            if (init_n && ND(L, init_n)->type != AST_NONE)
                lower_expr(L, init_n);
            cond_n = init_n ? ND(L, init_n)->next_sibling : 0;
        }
        incr_n = cond_n ? ND(L, cond_n)->next_sibling : 0;
        body_n = incr_n ? ND(L, incr_n)->next_sibling : 0;

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
        uint32_t lab = 0;

        if (tswa(L, node, &lab)) {
            push_scope(L);
            KA_GUARD(g, CE_STMT);
            for (uint32_t a = lab; a && g--; a = ND(L, a)->next_sibling) {
                uint32_t at = ND(L, a)->type;

                if (at == AST_BREAK) break;
                if (at != AST_CASE && at != AST_DEFAULT) lower_stmt(L, a);
                if (at == AST_RETURN || block_terminated(L)) break;
            }
            pop_scope(L);
            return;
        }

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
            /* Worst case is cond, default, then a pair per case. Flag up
               front; the pack below still truncates, but nothing reads it. */
            if (L->M->num_extra_ops + 2u + 2u * (uint32_t)ncases
                > BIR_MAX_EXTRA_OPS)
                bir_pfull(L->M, BIR_P_EXTRAOPS);
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

    case AST_RANGE_FOR:
        lower_error(L, node, BC_E135);
        break;

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
        if (ND(L, c)->type == AST_PARAM) {
            if (count >= max) return -1;
            out[count++] = c;
        }
        c = ND(L, c)->next_sibling;
    }
    return count;
}

static void lfbody(lower_t *L, uint32_t func_def,
                   uint16_t cuda_flags, const char *name_override);

static void lower_func_body(lower_t *L, uint32_t func_def,
                            uint16_t cuda_flags, const char *name_override)
{
    int tsv = L->tsc, csv = L->clow;
    int fsv = L->ntdf, ssv = L->ntdsn;

    L->tsc = 0;
    L->clow = 1;
    L->ntdf = L->ntypedefs;
    lfbody(L, func_def, cuda_flags, name_override);
    ldrain(L);
    L->ntdf = fsv;
    L->ntdsn = ssv;
    L->clow = csv;
    L->tsc = tsv;
}

static void lfbody(lower_t *L, uint32_t func_def,
                   uint16_t cuda_flags, const char *name_override)
{
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

    if (name_override && name_n) {
        char sfn[128];
        get_text(L, name_n, sfn, sizeof(sfn));
        ncpy(L->fnnm, sizeof(L->fnnm), sfn[0] ? sfn : fname);
    } else {
        ncpy(L->fnnm, sizeof(L->fnnm), fname);
    }

    uint16_t flnk = BIR_TU_EXT;
    uint16_t quals = ND(L, func_def)->qualifiers;
    if (L->tu != BIR_TU_EXT && (quals & QUAL_STATIC)) {
        char mng[BIR_SYM_MAX];
        if (bir_mang(fname, L->tu, mng, (int)sizeof fname) != 0) {
            lower_error(L, func_def, BC_E126, fname);
            return;
        }
        memcpy(fname, mng, strlen(mng) + 1u);
        flnk = L->tu;
    }

    if (L->tu != BIR_TU_EXT) {
        uint32_t dup = sym_fx(L, fname);
        if (dup != BIR_SYM_NONE) {
            if (!(quals & QUAL_INLINE) && !name_override)
                lower_error(L, func_def, BC_E126, fname);
            return;
        }
    }

    if (quals & (QUAL_REF | QUAL_RREF)) {
        lower_error(L, func_def, BC_E145, fname);
        return;
    }

    uint32_t ret_t = rtype(L, type_n, ret_ptr, 0);

    uint32_t param_nodes[MAX_PARM];
    int nparams = collect_params(L, func_def, param_nodes, MAX_PARM);
    if (nparams < 0) {
        lower_error(L, func_def, BC_E082, fname, MAX_PARM);
        return;
    }

    uint32_t param_types[MAX_PARM];
    uint32_t pbase[MAX_PARM];
    uint32_t rfm = 0, rfq = 0;
    int np = 0;
    L->nfpk = 0;
    if (L->mrec) {
        param_types[0] = bir_type_ptr(L->M, L->mrec, BIR_AS_GENERIC);
        pbase[0] = param_types[0];
        ncpy(L->pnms[0], sizeof(L->pnms[0]), "this");
        np = 1;
    }
    for (int i = 0; i < nparams && np < MAX_PARM; i++) {
        const ast_node_t *pn = ND(L, param_nodes[i]);
        if (pn->d.oper.op == PRM_VARG) continue;
        uint32_t pt_type_n = pn->first_child;
        int pdepth = pn->d.oper.flags; /* stored by parser */
        uint16_t p_cuda = pn->cuda_flags;
        int prr  = (pn->qualifiers & QUAL_RREF) != 0;
        int pref = prr || (pn->qualifiers & QUAL_REF) != 0;

        char base[64];
        base[0] = 0;
        for (uint32_t pc = pn->first_child; pc; pc = ND(L, pc)->next_sibling)
            if (ND(L, pc)->type == AST_IDENT) {
                get_text(L, pc, base, sizeof(base));
                break;
            }

        if (pn->d.oper.op == PRM_PACK) {
            char tn[64];
            if ((pn->qualifiers & (QUAL_REF | QUAL_RREF)) && base[0]) {
                lower_error(L, param_nodes[i], BC_E151, fname);
                return;
            }
            tn[0] = 0;
            if (pt_type_n && ND(L, pt_type_n)->first_child)
                get_text(L, ND(L, pt_type_n)->first_child, tn, sizeof(tn));
            const binding_t *b = find_tpk(L, tn);
            if (!b || b->pk_nt) {
                lower_error(L, param_nodes[i], BC_E030,
                            "function parameter pack with no bound types");
                return;
            }
            if (base[0] && L->nfpk < MAX_FPACKS) {
                snprintf(L->fpk[L->nfpk].name,
                         sizeof(L->fpk[0].name), "%s", base);
                L->fpk[L->nfpk].n = b->npk;
                L->nfpk++;
            }
            for (int k = 0; k < b->npk && np < MAX_PARM; k++) {
                int svb = L->nbindings;
                binding_t *t;
                if (L->nbindings >= MAX_BIND) {
                    lower_error(L, param_nodes[i], BC_E030,
                                "template nesting deeper than Booth binds");
                    return;
                }
                t = &L->bindings[L->nbindings++];
                memset(t, 0, sizeof(*t));
                snprintf(t->name, sizeof(t->name), "%s", tn);
                t->is_type = 1;
                t->type = b->pk_t[k];
                param_types[np] = rtype(L, pt_type_n, pdepth, p_cuda);
                pbase[np] = param_types[np];
                L->nbindings = svb;
                pk_nm(L->pnms[np], sizeof(L->pnms[0]), base, k);
                np++;
            }
            continue;
        }

        if (pref && (cuda_flags & CUDA_GLOBAL)) {
            lower_error(L, param_nodes[i], BC_E147, base[0] ? base : fname);
            return;
        }
        if (pref && np >= 32) {
            lower_error(L, param_nodes[i], BC_E030,
                        "reference parameter past the 32 Booth can track");
            return;
        }
        pbase[np] = rtype(L, pt_type_n, pdepth, p_cuda);
        if (L->rtcg == 2 && !pdepth && !cgadd(L, base)) {
            lower_error(L, param_nodes[i], BC_E030,
                        "more grid groups than Booth tracks");
            return;
        }
        if (pref) {
            param_types[np] = bir_type_ptr(L->M, pbase[np], BIR_AS_GENERIC);
            rfm |= 1u << np;
            if (prr || (pn->qualifiers & QUAL_CONST)) rfq |= 1u << np;
        } else if (aggt(L, pbase[np]) && !(cuda_flags & CUDA_GLOBAL)) {
            param_types[np] = bir_type_ptr(L->M, pbase[np], BIR_AS_PRIVATE);
        } else {
            param_types[np] = pbase[np];
        }
        snprintf(L->pnms[np], sizeof(L->pnms[0]), "%s", base);
        np++;
    }
    int fsret = aggt(L, ret_t) && !(cuda_flags & CUDA_GLOBAL);
    if (fsret) {
        if (np >= MAX_PARM) {
            lower_error(L, func_def, BC_E082, fname, MAX_PARM);
            return;
        }
        param_types[np] = bir_type_ptr(L->M, ret_t, BIR_AS_PRIVATE);
        pbase[np] = ret_t;
        L->pnms[np][0] = '\0';
        np++;
    }
    int nparams_x = np;

    uint32_t fn_type = bir_type_func(L->M,
                                     fsret ? bir_type_void(L->M) : ret_t,
                                     param_types, nparams_x);

    /* Create function. Bailing leaves cur_func on the previous one. */
    if (L->M->num_funcs >= BIR_MAX_FUNCS) {
        bir_pfull(L->M, BIR_P_FUNCS);
        return;
    }
    uint32_t fi = L->M->num_funcs++;
    L->cur_func = fi;

    bir_func_t *F = &L->M->funcs[fi];
    memset(F, 0, sizeof(*F));
    F->name = bir_add_string(L->M, fname, (uint32_t)strlen(fname));
    F->type = fn_type;
    F->tu = flnk;
    F->cuda_flags = cuda_flags;
    F->num_params = (uint16_t)nparams_x;
    F->first_block = L->M->num_blocks;
    F->num_blocks = 0;
    F->launch_bounds_max = ND(L, func_def)->launch_bounds_max;
    F->launch_bounds_min = ND(L, func_def)->launch_bounds_min;
    F->refm = rfm;
    F->refq = rfq;
    F->sret = (uint16_t)fsret;

    L->nlabels = 0;

    uint32_t entry = new_block(L, "entry");
    uint32_t body_n = find_func_body(L, func_def);
    set_block(L, entry);
    L->base_inst = L->M->num_insts;

    push_scope(L);
    for (int i = 0; i < nparams_x; i++) {
        uint32_t inst = emit(L, BIR_PARAM, param_types[i], 0, (uint8_t)i);
        if (L->pnms[i][0] && i < 32 && (rfm & (1u << i))) {
            add_sym(L, L->pnms[i], inst, pbase[i], 1);
        } else if (L->pnms[i][0] && aggt(L, pbase[i])
                   && param_types[i] != pbase[i]) {
            if (pmut(L, body_n, L->pnms[i], 0)) {
                uint32_t pt = bir_type_ptr(L->M, pbase[i], BIR_AS_PRIVATE);
                uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);

                (void)acpy(L, func_def, BIR_MAKE_VAL(al), BIR_MAKE_VAL(inst),
                           pbase[i]);
                add_sym(L, L->pnms[i], al, pbase[i], 1);
            } else {
                add_sym(L, L->pnms[i], inst, pbase[i], 1);
            }
        } else if (L->pnms[i][0]) {
            /* Promote all params to allocas so they're reassignable.
               mem2reg cleans up the ones that are never written. */
            uint32_t pt = bir_type_ptr(L->M, param_types[i], BIR_AS_PRIVATE);
            uint32_t al = emalc(L, BIR_ALLOCA, pt, 0);
            uint32_t st = emit(L, BIR_STORE, bir_type_void(L->M), 2, 0);
            set_op(L, st, 0, BIR_MAKE_VAL(inst));
            set_op(L, st, 1, BIR_MAKE_VAL(al));
            add_sym(L, L->pnms[i], al, param_types[i], 1);
        }
    }

    if (body_n) {
        push_scope(L);
        lower_block_stmts(L, body_n);
        pop_scope(L);
    }

    if (!block_terminated(L)) {
        if (is_void_type(L, ret_t))
            emit(L, BIR_RET, bir_type_void(L->M), 0, 0);
        else
            emit(L, BIR_UNREACHABLE, bir_type_void(L->M), 0, 0);
    }

    pop_scope(L);

    F->total_insts = L->M->num_insts - L->base_inst;
}

static void cstat(lower_t *L, uint32_t node, const char *ovr)
{
    uint32_t type_n = ND(L, node)->first_child;
    char sn[64];

    if (!type_n) return;
    sn[0] = 0;
    uint32_t nm = ND(L, type_n)->first_child;
    if (nm && ND(L, nm)->type == AST_IDENT) get_text(L, nm, sn, sizeof(sn));
    if (ovr) ncpy(sn, sizeof(sn), ovr);
    if (!sn[0]) return;

    for (uint32_t m = ND(L, type_n)->next_sibling; m;
         m = ND(L, m)->next_sibling) {
        char fn[64], full[144];
        uint32_t fnn, rt = 0;
        uint16_t cu;
        if (ND(L, m)->type != AST_FUNC_DEF) continue;
        cu = ND(L, m)->cuda_flags;
        if (!(cu & (CUDA_GLOBAL | CUDA_DEVICE))) continue;
        fnn = child_at(L, m, 1);
        if (!fnn || ND(L, fnn)->type != AST_IDENT) continue;
        get_text(L, fnn, fn, sizeof(fn));
        if (strcmp(fn, sn) == 0) continue;
        if (!(ND(L, m)->qualifiers & QUAL_STATIC)) {
            int rsi = -1;
            for (int si = 0; si < L->nstructs; si++)
                if (strcmp(L->structs[si].name, sn) == 0) { rsi = si; break; }
            if (rsi < 0) continue;
            rt = L->structs[rsi].bir_type;
        }
        if (snprintf(full, sizeof(full), "%s$%s", sn, fn) < 0) continue;
        L->mrec = rt;
        ncpy(L->scls, sizeof(L->scls), sn);
        tscan(L, m);
        lower_func_body(L, m, cu, full);
        L->scls[0] = 0;
        L->mrec = 0;
    }
}

/* ---- Declaration Collection (Pass 1) ---- */

static uint32_t farr(lower_t *L, uint32_t base, uint32_t decl,
                     uint32_t name_n, uint32_t *aft, int *bad, uint32_t *bn)
{
    uint32_t dims[LO_MAX_DIM], nd = 0, t = base, i, c;
    int pd = ND(L, decl)->d.oper.op;

    if (bn) *bn = 0;
    *aft = name_n ? ND(L, name_n)->next_sibling : 0;
    if (pd <= 0 || !name_n) return base;

    c = *aft;
    while (c && nd < (uint32_t)pd && nd < LO_MAX_DIM) {
        int64_t v;
        if (!cival(L, c, &v) || v <= 0 || v > CE_MAXDIM) {
            *bad = 1;
            if (bn) *bn = c;
            break;
        }
        dims[nd++] = (uint32_t)v;
        c = ND(L, c)->next_sibling;
    }
    if (nd < (uint32_t)pd) *bad = 1;
    *aft = c;
    if (*bad || nd == 0) return base;

    for (i = nd; i > 0; i--)
        t = bir_type_array(L->M, t, dims[i - 1]);
    return t;
}

static void salgn(lower_t *L, struct_def_t *sd, uint32_t node)
{
    uint32_t onat = 0, ocud = 0, want = 1;
    uint32_t decl = algof(L, ND(L, node)->first_child);

    if (decl > want) want = decl;

    for (int i = 0; i < sd->num_fields; i++) {
        uint32_t sz = bir_bsz(L->M, sd->field_types[i], 8);
        uint32_t na = bir_balg(L->M, sd->field_types[i], 8);
        uint32_t ca = sd->falgn[i] ? sd->falgn[i] : na;

        if (!sz || !na) return;
        if (ca > want) want = ca;
        onat = (onat + na - 1u) & ~(na - 1u);
        ocud = (ocud + ca - 1u) & ~(ca - 1u);
        if (onat != ocud) {
            lower_error(L, node, BC_E142, sd->field_names[i], (int)ca);
            return;
        }
        onat += sz;
        ocud += sz;
    }
    if (want > bir_balg(L->M, sd->bir_type, 8))
        sd->algn = (uint16_t)want;
}

static int sdreg(lower_t *L, uint32_t mem, struct_def_t *sd)
{
    uint32_t ft = ND(L, mem)->first_child;
    uint32_t nn = ft ? ND(L, ft)->next_sibling : 0;
    uint16_t q  = ND(L, mem)->qualifiers;
    cval_t v;
    sdm_t *e;

    if (!nn || ND(L, nn)->type != AST_IDENT) return 1;
    if (L->nsdm >= MAX_SDM) return 0;
    e = &L->sdms[L->nsdm++];
    memset(e, 0, sizeof(*e));
    ncpy(e->cls, sizeof(e->cls), sd->name);
    get_text(L, nn, e->nm, sizeof(e->nm));
    e->type = resolve_type(L, ft, ND(L, mem)->d.oper.flags, 0);
    e->arr  = ND(L, mem)->d.oper.op != 0;
    L->rtbad = 0;
    memset(&v, 0, sizeof(v));
    if (!(q & (QUAL_CONST | QUAL_CONSTEXPR))) e->bad = 2;
    else if (ND(L, mem)->d.oper.op != 0) e->bad = 3;
    else if (!ND(L, nn)->next_sibling) e->bad = 1;
    else if (!cfsc(L, ND(L, nn)->next_sibling, &v) || !ccvt(L, &v, e->type))
        e->bad = 3;
    else e->v = v;
    return 1;
}

static void sdbas(lower_t *L, struct_def_t *sd, uint32_t name_n)
{
    for (uint32_t b = name_n ? ND(L, name_n)->next_sibling : 0; b;
         b = ND(L, b)->next_sibling) {
        if (ND(L, b)->type != AST_BASE) continue;
        if (!ND(L, b)->d.text.len || sd->nbase >= MAX_BASE) {
            sd->bopq = 1;
            continue;
        }
        get_text(L, b, sd->base[sd->nbase], sizeof(sd->base[0]));
        sd->nbase++;
    }
}

static void collect_struct(lower_t *L, uint32_t node)
{
    if (L->nstructs >= MAX_STRUCTS) {
        lower_error(L, node, BC_E138, MAX_STRUCTS);
        return;
    }

    uint32_t type_n = ND(L, node)->first_child;
    char ovr[144], svc[64];

    ncpy(ovr, sizeof(ovr), L->covr);
    L->covr[0] = 0;
    if (!type_n) return;

    if (!ovr[0])
        for (int si = 0; si < L->nstructs; si++)
            if (L->structs[si].node == node) return;

    uint32_t name_n = ND(L, type_n)->first_child;
    char snm[64];
    struct_def_t *sd;
    int slot;

    snm[0] = 0;
    if (name_n && ND(L, name_n)->type == AST_IDENT)
        get_text(L, name_n, snm, sizeof(snm));
    if (ovr[0]) ncpy(snm, sizeof(snm), ovr);

    slot = snm[0] ? sfslot(L, snm) : -1;
    if (slot >= 0) {
        uint32_t held = L->structs[slot].bir_type;
        sd = &L->structs[slot];
        memset(sd, 0, sizeof(*sd));
        sd->bir_type = held;
    } else {
        sd = &L->structs[L->nstructs++];
        memset(sd, 0, sizeof(*sd));
        sd->bir_type = bir_sfwd(L->M);
    }
    sd->node = node;
    ncpy(sd->name, sizeof(sd->name), snm);
    sdbas(L, sd, name_n);
    ncpy(svc, sizeof(svc), L->scls);
    ncpy(L->scls, sizeof(L->scls), sd->name);

    if (ND(L, type_n)->d.btype.kind == TYPE_UNION) {
        sd->uni = 1;
        bir_umrk(L->M, sd->bir_type);
    }

    /* Collect fields: children after type_n */
    uint32_t member = ND(L, type_n)->next_sibling;
    int over = 0;
    while (member) {
        if (sd->num_fields >= MAX_FIELDS) { over = 1; break; }
        if (ND(L, member)->type == AST_FUNC_DEF
            && (ND(L, member)->qualifiers & QUAL_CONSTEXPR)) {
            uint32_t mid = child_at(L, member, 1);
            char mnm[64];

            if (mid && ND(L, mid)->type == AST_IDENT
                && L->ncfn < MAX_CFNS && sd->name[0]) {
                get_text(L, mid, mnm, sizeof mnm);
                if (snprintf(L->cfns[L->ncfn].name,
                             sizeof(L->cfns[0].name), "%s::%s",
                             sd->name, mnm) > 0) {
                    L->cfns[L->ncfn].ast = member;
                    L->ncfn++;
                }
            }
            member = ND(L, member)->next_sibling;
            continue;
        }
        if (ND(L, member)->type == AST_STRUCT_DEF) {
            uint32_t ats = ND(L, member)->first_child;
            uint32_t anm = ats ? ND(L, ats)->first_child : 0;
            if (!anm || ND(L, anm)->type != AST_IDENT) {
                int fnd = -1;
                for (int si = 0; si < L->nstructs; si++)
                    if (L->structs[si].node == member) { fnd = si; break; }
                if (fnd < 0 || L->structs[fnd].num_fields == 0) sd->anon = 1;
                else {
                    sd->field_types[sd->num_fields] =
                        L->structs[fnd].bir_type;
                    sd->falgn[sd->num_fields] = L->structs[fnd].algn;
                    sd->fanon[sd->num_fields] = 1;
                    sd->field_names[sd->num_fields][0] = '\0';
                    sd->num_fields++;
                }
            }
            member = ND(L, member)->next_sibling;
            continue;
        }
        if (ND(L, member)->type == AST_VAR_DECL) {
            uint32_t ft_n = ND(L, member)->first_child;
            uint32_t fn_n = ft_n ? ND(L, ft_n)->next_sibling : 0;

            if (ND(L, member)->qualifiers & QUAL_STATIC) {
                if (!sdreg(L, member, sd))
                    lower_error(L, member, BC_E130, MAX_SDM);
                member = ND(L, member)->next_sibling;
                continue;
            }

            if (ND(L, member)->qualifiers & (QUAL_REF | QUAL_RREF)) {
                uint32_t rn = ft_n ? ND(L, ft_n)->next_sibling : 0;
                char rnm[64];
                rnm[0] = 0;
                if (rn && ND(L, rn)->type == AST_IDENT)
                    get_text(L, rn, rnm, sizeof(rnm));
                lower_error(L, member, BC_E150, rnm[0] ? rnm : sd->name);
                member = ND(L, member)->next_sibling;
                continue;
            }

            uint32_t ft = resolve_type(L, ft_n, ND(L, member)->d.oper.flags, 0);
            if (L->rtbad && !sd->bad) {
                sd->bad = L->rtbad;
                ncpy(sd->badn, sizeof(sd->badn), L->rtbadn);
            }
            uint32_t fa = L->rtalgn;
            uint32_t fd = algof(L, ft_n);

            if (fd > fa) fa = fd;

            if (fn_n && ND(L, fn_n)->type == AST_IDENT) {
                /* Handle comma-separated fields: float x, y, z;
                   Parser puts all names as siblings within one var_decl */
                uint32_t extra = 0, abn = 0;
                int abad = 0;
                uint32_t at = farr(L, ft, member, fn_n, &extra, &abad, &abn);

                if (abad && !sd->bad) {
                    char fnm[64];

                    sd->bad = BC_E131;
                    get_text(L, fn_n, fnm, sizeof(fnm));
                    abnm(L, fnm, abn, sd->badn, (int)sizeof(sd->badn));
                }
                sd->field_types[sd->num_fields] = at;
                sd->falgn[sd->num_fields] = (uint16_t)fa;
                get_text(L, fn_n, sd->field_names[sd->num_fields],
                         sizeof(sd->field_names[0]));
                sd->num_fields++;

                /* Additional idents in the same var_decl */
                while (extra && ND(L, extra)->type == AST_IDENT) {
                    if (sd->num_fields >= MAX_FIELDS) { over = 1; break; }
                    sd->field_types[sd->num_fields] = ft;
                    sd->falgn[sd->num_fields] = (uint16_t)fa;
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
                        && sd->num_fields >= MAX_FIELDS) over = 1;
                    if (cfn && ND(L, cfn)->type == AST_IDENT
                        && sd->num_fields < MAX_FIELDS) {
                        uint32_t cft = resolve_type(L, cft_n,
                            ND(L, chain)->d.oper.flags, 0);
                        uint32_t caft = 0, cbn = 0;
                        int cbad = 0;
                        if (L->rtbad && !sd->bad) {
                            sd->bad = L->rtbad;
                            ncpy(sd->badn, sizeof(sd->badn), L->rtbadn);
                        }
                        cft = farr(L, cft, chain, cfn, &caft, &cbad, &cbn);
                        if (cbad && !sd->bad) {
                            char cnm[64];

                            sd->bad = BC_E131;
                            get_text(L, cfn, cnm, sizeof(cnm));
                            abnm(L, cnm, cbn, sd->badn, (int)sizeof(sd->badn));
                        }
                        uint32_t cfa = algof(L, cft_n);
                        if (cfa < L->rtalgn) cfa = L->rtalgn;
                        sd->field_types[sd->num_fields] = cft;
                        sd->falgn[sd->num_fields] = (uint16_t)cfa;
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

    if (over && !sd->bad) {
        sd->bad = BC_E144;
        ncpy(sd->badn, sizeof(sd->badn), sd->name);
    }

    ncpy(L->scls, sizeof(L->scls), svc);

    /* Create BIR struct type */
    if (bir_sfin(L->M, sd->bir_type, sd->field_types, sd->num_fields)) {
        salgn(L, sd, node);
    } else {
        memset(sd, 0, sizeof(*sd));
    }
}

#define SC_WORK  256
#define SC_ORDER 64

static int scoll(lower_t *L, uint32_t root)
{
    uint32_t wk[SC_WORK], ord[SC_ORDER];
    int wn = 0, no = 0, i;

    if (!root) return -1;
    wk[wn++] = root;

    KA_GUARD(g, 65536);
    while (wn > 0 && g--) {
        uint32_t n = wk[--wn], c;
        if (ND(L, n)->type == AST_STRUCT_DEF) {
            if (no >= SC_ORDER) { lower_error(L, n, BC_E138, SC_ORDER); return -1; }
            ord[no++] = n;
        }
        for (c = ND(L, n)->first_child; c; c = ND(L, c)->next_sibling) {
            if (wn >= SC_WORK) { lower_error(L, n, BC_E138, SC_WORK); return -1; }
            wk[wn++] = c;
        }
    }

    for (i = no; i > 0; i--)
        collect_struct(L, ord[i - 1]);

    for (i = 0; i < L->nstructs; i++)
        if (L->structs[i].node == root) return i;
    return -1;
}

static void collect_enum(lower_t *L, uint32_t node)
{
    uint32_t ts = ND(L, node)->first_child;
    uint32_t c = ts;
    char scope[48];

    scope[0] = 0;
    if (ts && (ND(L, ts)->qualifiers & QUAL_SCOPED) && ND(L, ts)->first_child)
        get_text(L, ND(L, ts)->first_child, scope, sizeof(scope));
    if (ts && ND(L, ts)->first_child
        && ND(L, ND(L, ts)->first_child)->type == AST_IDENT) {
        char tag[64];
        uint32_t seen;
        get_text(L, ND(L, ts)->first_child, tag, sizeof(tag));
        if (tag[0] && !find_typedef(L, tag, &seen)) {
            if (L->ntypedefs >= MAX_TYPEDEFS) {
                lower_error(L, node, BC_E140, MAX_TYPEDEFS);
                return;
            }
            typedef_def_t *et = &L->typedefs[L->ntypedefs++];
            memset(et, 0, sizeof(*et));
            ncpy(et->name, sizeof(et->name), tag);
            et->bir_type = bir_type_int(L->M, 32);
        }
        L->rtbad = 0;
        L->rtbadn[0] = 0;
    }
    /* Skip type_spec child */
    if (c) c = ND(L, c)->next_sibling;

    int64_t next_val = 0;
    while (c) {
        if (L->nenums >= MAX_ENUMS) {
            lower_error(L, node, BC_E139, MAX_ENUMS);
            return;
        }
        if (ND(L, c)->type == AST_ENUMERATOR) {
            char base[64], name[128];
            get_text(L, c, base, sizeof(base));
            if (snprintf(name, sizeof name, "%s%s%s",
                         scope, scope[0] ? "::" : "", base) < 0)
                return;

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

static void tdbad(lower_t *L, typedef_def_t *td)
{
    td->bad = L->rtbad;
    ncpy(td->badn, sizeof(td->badn), L->rtbadn);
    ncpy(td->snm, sizeof(td->snm), L->rtsnm);
}

static void collect_typedef(lower_t *L, uint32_t node)
{
    /* typedef TYPE NAME;
       In our AST this is a VAR_DECL with QUAL_TYPEDEF */
    if (!(ND(L, node)->qualifiers & QUAL_TYPEDEF)) return;
    if (L->ntypedefs >= MAX_TYPEDEFS) {
        lower_error(L, node, BC_E140, MAX_TYPEDEFS);
        return;
    }

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
                int prev_n = scoll(L, cc);
                if (prev_n < 0) prev_n = L->nstructs;
                /* Add a second entry under the typedef name that
                 * shares the underlying bir_type. The synthetic
                 * anon entry stays so resolve_type for the typedef
                 * target finds it during its own TYPE_STRUCT walk;
                 * the typedef-named entry is what later member
                 * access via "name v; v.field" looks up by name. */
                if (L->nstructs > prev_n &&
                    L->structs[prev_n].num_fields > 0 &&
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
    memset(td, 0, sizeof(*td));
    get_text(L, name_n, td->name, sizeof(td->name));
    td->bir_type = resolve_type(L, type_n, ND(L, node)->d.oper.flags, 0);
    tdbad(L, td);
}

static int ureg(lower_t *L, uint32_t node)
{
    uint32_t nm_n = ND(L, node)->first_child;
    uint32_t ty_n;
    typedef_def_t *td = NULL;
    char nm[64];

    if (!nm_n || ND(L, nm_n)->type != AST_IDENT) return 1;
    ty_n = ND(L, nm_n)->next_sibling;
    if (!ty_n || ND(L, ty_n)->type != AST_TYPE_SPEC) return 1;

    get_text(L, nm_n, nm, sizeof(nm));
    for (int i = 0; i < L->ntypedefs; i++)
        if (strcmp(L->typedefs[i].name, nm) == 0) { td = &L->typedefs[i]; break; }
    if (!td) {
        if (L->ntypedefs >= MAX_TYPEDEFS) return 0;
        td = &L->typedefs[L->ntypedefs++];
    }
    memset(td, 0, sizeof(*td));
    ncpy(td->name, sizeof(td->name), nm);
    td->bir_type = resolve_type(L, ty_n, ND(L, node)->d.oper.flags,
                                ND(L, node)->cuda_flags);
    tdbad(L, td);
    L->rtbad = 0;
    return 1;
}

static void ucoll(lower_t *L, uint32_t root)
{
    uint32_t wk[UC_NEST];
    int wn = 0;

    wk[wn++] = root;
    KA_GUARD(g, 8192);
    while (wn > 0 && g--) {
        uint32_t nd = wk[--wn];
        uint32_t nx = 0;

        for (uint32_t c = ND(L, nd)->first_child; c; c = nx) {
            uint16_t t = ND(L, c)->type;
            nx = ND(L, c)->next_sibling;
            if (t == AST_USING) {
                if (!ureg(L, c)) lower_error(L, c, BC_E140, MAX_TYPEDEFS);
                continue;
            }
            if (t == AST_TEMPLATE_DECL) {
                if (!areg(L, c))
                    lower_error(L, c, BC_E140, MAX_ALIAS);
                continue;
            }
            if (t != AST_NAMESPACE && t != AST_FUNC_DEF && t != AST_BLOCK)
                continue;
            if (wn >= UC_NEST) continue;
            wk[wn++] = c;
        }
    }
}

static void collect_global_var(lower_t *L, uint32_t node)
{
    uint16_t cuda = ND(L, node)->cuda_flags;
    if (!(cuda & (CUDA_SHARED | CUDA_CONSTANT | CUDA_DEVICE))) return;

    uint32_t type_n = child_at(L, node, 0);
    uint32_t name_n = child_at(L, node, 1);
    if (!name_n || ND(L, name_n)->type != AST_IDENT) return;

    char gname[128];
    get_text(L, name_n, gname, sizeof(gname));

    if (ND(L, node)->qualifiers & QUAL_PINIT) {
        lower_error(L, node, BC_E136, gname);
        return;
    }

    uint32_t elem_t = rtype(L, type_n, ND(L, node)->d.oper.flags, cuda);

    uint32_t next = ND(L, name_n)->next_sibling;
    int has_arr_size = 0;
    if (next && ND(L, node)->d.oper.op > 0
        && ND(L, next)->type != AST_INIT_LIST) {
        int64_t aval;
        if (cival(L, next, &aval) && aval >= 0 && aval <= CE_MAXDIM) {
            has_arr_size = 1;
            elem_t = bir_type_array(L->M, elem_t, (uint32_t)aval);
        }
    }

    uint16_t glnk = BIR_TU_EXT;
    if (L->tu != BIR_TU_EXT && (ND(L, node)->qualifiers & QUAL_STATIC)) {
        char mng[BIR_SYM_MAX];
        if (bir_mang(gname, L->tu, mng, (int)sizeof gname) != 0) {
            lower_error(L, node, BC_E126, gname);
            return;
        }
        memcpy(gname, mng, strlen(mng) + 1u);
        glnk = L->tu;
    }

    uint32_t old = sym_gx(L, gname);
    if (old != BIR_SYM_NONE) {
        if (L->M->globals[old].type != elem_t)
            lower_error(L, node, BC_E126, gname);
        return;
    }

    if (L->M->num_globals >= BIR_MAX_GLOBALS) {
        bir_pfull(L->M, BIR_P_GLOBALS);
        return;
    }
    uint32_t gi = L->M->num_globals++;
    bir_global_t *G = &L->M->globals[gi];
    memset(G, 0, sizeof(*G));
    G->name = bir_add_string(L->M, gname, (uint32_t)strlen(gname));
    G->type = elem_t;
    G->tu = glnk;
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

static void ccolc(lower_t *L, uint32_t node);

static void vtadd(lower_t *L, uint32_t node)
{
    uint32_t c, nmn, in;
    vtpl_t *v;

    for (c = ND(L, node)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_VAR_DECL) break;
    if (!c || ND(L, c)->d.oper.op != 0) return;
    if (ND(L, c)->qualifiers & QUAL_TYPEDEF) return;
    if (!(ND(L, c)->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST))) return;
    nmn = child_at(L, c, 1);
    if (!nmn || ND(L, nmn)->type != AST_IDENT) return;
    in = ND(L, nmn)->next_sibling;
    if (in && ND(L, in)->type == AST_STRUCT_DEF) in = ND(L, in)->next_sibling;
    if (!in) return;
    if (L->nvtpl >= MAX_VTPL) {
        char nm[64];

        get_text(L, nmn, nm, sizeof nm);
        lower_error(L, c, BC_E340, nm, MAX_VTPL);
        return;
    }
    v = &L->vtpl[L->nvtpl++];
    v->tdcl = node;
    v->ty   = child_at(L, c, 0);
    v->init = in;
    v->spec = ND(L, nmn)->first_child != 0;
    get_text(L, nmn, v->name, sizeof(v->name));
}

static void collect_template(lower_t *L, uint32_t node)
{
    ccolc(L, node);
    vtadd(L, node);
    if (L->ntemplates >= MAX_TEMPLATES) {
        lower_error(L, node, BC_E141, MAX_TEMPLATES);
        return;
    }

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
static uint32_t arg_type(lower_t *L, uint32_t arg)
{
    const ast_node_t *a;

    arg = unfw(L, arg);
    a = ND(L, arg);
    switch (a->type) {
    case AST_FLOAT_LIT: {
        int f32;
        parse_float_text(L->src + a->d.text.offset, (int)a->d.text.len, &f32);
        return f32 ? bir_type_float(L->M, 32) : bir_type_float(L->M, 64);
    }
    case AST_INT_LIT:  return bir_type_int(L->M, 32);
    case AST_BOOL_LIT: return bir_type_int(L->M, 1);
    case AST_CAST:     return resolve_type(L, a->first_child,
                                           a->d.oper.flags, 0);
    default:           break;
    }
    if (a->type == AST_IDENT) {
        char nm[128];
        const sym_t *sy;

        get_text(L, arg, nm, (int)sizeof nm);
        sy = find_sym(L, nm);
        if (sy && sy->type && !sy->lam) return sy->type;
    }
    if (a->type == AST_SUBSCRIPT && a->first_child && L->atdp < MAX_CDEP) {
        uint32_t bt;

        L->atdp++;
        bt = arg_type(L, a->first_child);
        L->atdp--;
        if (bt < L->M->num_types
            && (L->M->types[bt].kind == BIR_TYPE_ARRAY
                || L->M->types[bt].kind == BIR_TYPE_PTR)
            && L->M->types[bt].inner)
            return L->M->types[bt].inner;
    }
    if (a->type == AST_MEMBER && a->first_child && L->atdp < MAX_CDEP) {
        uint32_t obj = a->first_child;
        uint32_t fld = ND(L, obj)->next_sibling;
        uint32_t bt;
        char fn[64];
        int si;

        L->atdp++;
        bt = arg_type(L, obj);
        L->atdp--;
        if (a->d.member.is_arrow) bt = ptr_inner(L, bt);
        si = fld && ND(L, fld)->type == AST_IDENT ? sfind(L, bt) : -1;
        if (si >= 0) {
            get_text(L, fld, fn, (int)sizeof fn);
            for (int f = 0; f < L->structs[si].num_fields; f++)
                if (strcmp(L->structs[si].field_names[f], fn) == 0
                    && L->structs[si].field_types[f])
                    return L->structs[si].field_types[f];
        }
    }
    return sbty(L, arg);
}

static uint32_t tpeel(const lower_t *L, uint32_t t, int dep)
{
    while (dep-- > 0) {
        if (!t || t >= L->M->num_types
            || L->M->types[t].kind != BIR_TYPE_PTR)
            return 0;
        t = L->M->types[t].inner;
    }
    return t;
}

static int tfarg(lower_t *L, uint32_t a, const char *vn,
                 char *out, size_t oz)
{
    template_def_t *tm = find_template(L, vn);
    binding_t *nb2;
    char key[256];
    int ntp = 0, ok;
    uint32_t fn2 = 0;

    if (!tm || L->xdep >= MAX_NEST) return 0;
    nb2 = L->xnb[L->xdep];
    L->xdep++;
    ok = tdeduc(L, tm->ast, a, NULL, 0, nb2, &ntp, &fn2);
    if (ok) {
        tkeyb(nb2, ntp, vn, key, sizeof key);
        tkfn(L, vn, fn2, key, sizeof key);
        if (!tfind(L, key, out, oz)) {
            ok = L->tsc && tinst(L, 0, vn, nb2, ntp, fn2) >= 0
                        && tfind(L, key, out, oz);
        }
    }
    L->xdep--;
    return ok;
}

static int tesc(lower_t *L, uint32_t a, binding_t *nb, int i)
{
    char qn[128];
    cval_t sv;
    int64_t v = 0;

    qtxt(L, a, qn, sizeof(qn));
    if (qn[0] && find_enum(L, qn, &v)) nb[i].ival = v;
    else if (sdres(L, a, &sv) && !sv.isf) nb[i].ival = sv.ival;
    else return 0;
    nb[i].is_type = 0;
    return 1;
}

static int tenam(lower_t *L, uint32_t a, binding_t *nb, int i)
{
    const ast_node_t *an = ND(L, a);
    char vn[128], fs[128];
    uint32_t vi = a, bt = 0;
    cexp_t *ce = NULL;
    const char *fw;
    int64_t v = 0;

    if (an->type == AST_SCOPE_RES) {
        vi = an->first_child;
        while (vi && ND(L, vi)->next_sibling)
            vi = ND(L, vi)->next_sibling;
        if (!vi || ND(L, vi)->type != AST_IDENT) return 0;
        if (tesc(L, a, nb, i)) return 1;
    }
    get_text(L, vi, vn, sizeof(vn));
    fw = fbfn(L, vn);
    if (an->type == AST_IDENT && targs(L, a)
        && tfarg(L, a, vn, fs, sizeof fs)) {
        nb[i].is_type = 0;
        nb[i].is_fn = 1;
        ncpy(nb[i].fnm, sizeof(nb[i].fnm), fs);
    } else if (tgfnd(L, vn, &v)) {
        nb[i].is_type = 0;
        nb[i].ival = v;
    } else if (fw) {
        nb[i].is_type = 0;
        nb[i].is_fn = 1;
        ncpy(nb[i].fnm, sizeof(nb[i].fnm), fw);
    } else if (find_binding_int(L, vn, &v)) {
        nb[i].is_type = 0;
        nb[i].ival = v;
    } else if (find_binding(L, vn, &bt)) {
        nb[i].is_type = 1;
        nb[i].type = bt;
    } else if (find_enum(L, vn, &v)) {
        nb[i].is_type = 0;
        nb[i].ival = v;
    } else if (cfind(L, vn, &ce) && !ce->bad && !ce->v.isf) {
        nb[i].is_type = 0;
        nb[i].ival = ce->v.ival;
    } else if (find_template(L, vn)
               || bir_fsym(L->M, vn, L->tu, -1) != BIR_SYM_NONE) {
        nb[i].is_type = 0;
        nb[i].is_fn = 1;
        ncpy(nb[i].fnm, sizeof(nb[i].fnm), vn);
    } else {
        return 0;
    }
    return 1;
}

static int tpka(lower_t *L, uint32_t a, binding_t *b)
{
    int64_t pv;

    if (b->npk >= MAX_PKELM) return 0;
    if (ND(L, a)->type == AST_TYPE_SPEC) {
        if (b->npk && b->pk_nt) return 0;
        b->pk_t[b->npk++] = rtype(L, a, tpdep(L, a), 0);
        return !L->rtunk;
    }
    if (b->npk && !b->pk_nt) return 0;
    if (!cival(L, a, &pv)) return 0;
    b->pk_nt = 1;
    b->pk_v[b->npk++] = pv;
    return 1;
}

static int tpkx(lower_t *L, uint32_t pat, binding_t *b)
{
    char nms[MAX_FPACKS][64];
    int nn = 0, len, ok = 1;

    if (!pat) return 0;
    len = pk_len(L, pat, nms, &nn, 1);
    if (len < 0 || len > MAX_PKELM) return 0;
    for (int j = 0; j < len && ok; j++) {
        int sva = L->npkact, svb = L->nbindings;
        ok = cpkb(L, nms, nn, j) && tpka(L, pat, b);
        L->npkact    = sva;
        L->nbindings = svb;
    }
    return ok;
}

static int texpl(lower_t *L, uint32_t cal, binding_t *nb, int ntp)
{
    uint32_t ta = 0;
    if (!cal) return 1;
    for (uint32_t c = ND(L, cal)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) { ta = c; break; }
    if (!ta) return 1;

    int i = 0;
    for (uint32_t a = ND(L, ta)->first_child; a && i < ntp;
         a = ND(L, a)->next_sibling) {
        const ast_node_t *an = ND(L, a);
        if (nb[i].is_pack) {
            if (an->type == AST_PACK_EXP) {
                if (!tpkx(L, an->first_child, &nb[i])) return 0;
            } else if (!tpka(L, a, &nb[i])) {
                return 0;
            }
            nb[i].fixed = 1;
            continue;
        }
        switch (an->type) {
        case AST_TYPE_SPEC:
            nb[i].is_type = 1;
            nb[i].type = rtype(L, a, tpdep(L, a), 0);
            if (L->rtunk) return 0;
            break;
        case AST_INT_LIT:
            nb[i].is_type = 0;
            nb[i].ival = parse_int_text(L->src + an->d.text.offset,
                                        (int)an->d.text.len);
            break;
        case AST_BOOL_LIT:
            nb[i].is_type = 0;
            nb[i].ival = an->d.ival ? 1 : 0;
            break;
        case AST_IDENT:
        case AST_SCOPE_RES:
            if (!tenam(L, a, nb, i)) return 0;
            break;
        default: {
            int64_t cv;
            if (!cival(L, a, &cv)) return 0;
            nb[i].is_type = 0;
            nb[i].ival = cv;
            break;
        }
        }
        nb[i].fixed = 1;
        i++;
    }
    return 1;
}

static int tetyp(lower_t *L, uint32_t a)
{
    char nm[64];
    uint32_t t;

    get_text(L, a, nm, sizeof(nm));
    if (find_typedef(L, nm, &t) || find_binding(L, nm, &t)) return 1;
    for (int i = 0; i < L->nstructs; i++)
        if (strcmp(L->structs[i].name, nm) == 0) return 1;
    for (int i = 0; i < L->nclss; i++)
        if (strcmp(L->clss[i].name, nm) == 0) return 1;
    return 0;
}

static int hasta(const lower_t *L, uint32_t cal)
{
    if (!cal) return 0;
    for (uint32_t c = ND(L, cal)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) return 1;
    return 0;
}

static uint32_t tebad(lower_t *L, uint32_t cal)
{
    uint32_t ta = 0;

    if (!cal) return 0;
    for (uint32_t c = ND(L, cal)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) { ta = c; break; }
    if (!ta) return 0;

    for (uint32_t a = ND(L, ta)->first_child; a; a = ND(L, a)->next_sibling) {
        int64_t v;

        if (ND(L, a)->type == AST_PACK_EXP) continue;
        if (ND(L, a)->type == AST_TYPE_SPEC) {
            uint32_t t = rtype(L, a, tpdep(L, a), 0);

            if (!t || L->rtbad || L->rtunk) return a;
            continue;
        }
        if (!cival(L, a, &v)) {
            if (ND(L, a)->type == AST_SCOPE_RES) continue;
            if (ND(L, a)->type != AST_IDENT || !tetyp(L, a)) return a;
        }
    }
    return 0;
}

static int alist(const lower_t *L, uint32_t first, uint32_t *av, int max)
{
    int n = 0;
    for (uint32_t a = first; a; a = ND(L, a)->next_sibling) {
        if (n >= max) return -1;
        av[n++] = a;
    }
    return n;
}

static int tparm(lower_t *L, uint32_t tnode, binding_t *nb)
{
    int ntp = 0;

    for (uint32_t tp = ND(L, tnode)->first_child;
         tp && ND(L, tp)->type == AST_TEMPLATE_PARAM && ntp < MAX_TPARM;
         tp = ND(L, tp)->next_sibling) {
        int fl = ND(L, tp)->d.oper.flags;
        memset(&nb[ntp], 0, sizeof(nb[0]));
        nb[ntp].is_type = !(fl & TP_NTYP);
        if (fl & TP_PACK) {
            nb[ntp].is_pack = 1;
            nb[ntp].is_type = 0;
        }
        for (uint32_t tpc = ND(L, tp)->first_child; tpc;
             tpc = ND(L, tpc)->next_sibling)
            if (ND(L, tpc)->type == AST_IDENT) {
                get_text(L, tpc, nb[ntp].name, sizeof(nb[0].name));
                nb[ntp].dflt = ND(L, tpc)->next_sibling;
                break;
            }
        ntp++;
    }
    return ntp;
}

static int isqp(const lower_t *L, uint32_t ts, char *out, int oz)
{
    uint32_t c, pe, ta = 0;
    char nm[64];

    nm[0] = 0;
    if (!ts || ND(L, ts)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, ts)->d.btype.kind != TYPE_NAME) return 0;
    for (c = ND(L, ts)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_IDENT) get_text(L, c, nm, sizeof nm);
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) { ta = c; break; }
    }
    if (!ta || strcmp(nm, "index_sequence") != 0) return 0;
    pe = ND(L, ta)->first_child;
    if (!pe || ND(L, pe)->next_sibling) return 0;
    if (ND(L, pe)->type != AST_PACK_EXP) return 0;
    pe = ND(L, pe)->first_child;
    if (!pe || ND(L, pe)->type != AST_IDENT) return 0;
    get_text(L, pe, out, oz);
    return 1;
}

static int isqa(lower_t *L, uint32_t a, int64_t *v, int max)
{
    uint32_t ts = a, c, ta = 0;
    char nm[64];
    int n = 0;

    nm[0] = 0;
    if (ND(L, ts)->type == AST_SCOPE_RES) {
        ts = ND(L, ts)->first_child;
        while (ts && ND(L, ts)->type != AST_CAST)
            ts = ND(L, ts)->next_sibling;
    }
    if (!ts || ND(L, ts)->type != AST_CAST) return -1;
    ts = ND(L, ts)->first_child;
    if (!ts || ND(L, ts)->type != AST_TYPE_SPEC) return -1;
    for (c = ND(L, ts)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_IDENT) get_text(L, c, nm, sizeof nm);
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) { ta = c; break; }
    }
    if (!ta) return -1;
    if (strcmp(nm, "make_index_sequence") == 0) {
        uint32_t e = ND(L, ta)->first_child;
        int64_t cnt;
        if (!e || ND(L, e)->next_sibling) return -1;
        if (!cival(L, e, &cnt) || cnt < 0 || cnt > max) return -1;
        for (int64_t i = 0; i < cnt; i++) v[n++] = i;
        return n;
    }
    if (strcmp(nm, "index_sequence") != 0) return -1;
    for (c = ND(L, ta)->first_child; c; c = ND(L, c)->next_sibling) {
        int64_t x;
        if (n >= max || !cival(L, c, &x)) return -1;
        v[n++] = x;
    }
    return n;
}

static int isqb(lower_t *L, uint32_t ts, uint32_t a, binding_t *nb, int ntp)
{
    int64_t sv[MAX_PKELM];
    char sqn[64];
    int nv, hit = 0;

    if (!isqp(L, ts, sqn, (int)sizeof sqn)) return -1;
    nv = isqa(L, a, sv, MAX_PKELM);
    if (nv < 0) return 0;
    for (int b = 0; b < ntp; b++) {
        if (!nb[b].is_pack || strcmp(nb[b].name, sqn) != 0) continue;
        hit = 1;
        if (nb[b].fixed) break;
        nb[b].pk_nt = 1;
        nb[b].npk = 0;
        for (int k = 0; k < nv; k++) nb[b].pk_v[nb[b].npk++] = sv[k];
        nb[b].fixed = 1;
        break;
    }
    return hit;
}

static int tdopn(const binding_t *nb, int ntp, const char *tname)
{
    for (int b = 0; b < ntp; b++)
        if (nb[b].is_type && !nb[b].fixed && nb[b].type == 0
            && strcmp(nb[b].name, tname) == 0)
            return 1;
    return 0;
}

static void tdty(lower_t *L, binding_t *nb, int ntp, const char *tname,
                 uint32_t at)
{
    for (int b = 0; b < ntp; b++) {
        if (!nb[b].is_type || nb[b].fixed
            || strcmp(nb[b].name, tname) != 0 || nb[b].type != 0)
            continue;
        nb[b].type  = at ? at : bir_type_float(L->M, 32);
        nb[b].guess = !at;
    }
}

static int tdpk(lower_t *L, const ast_node_t *pn, const char *tname,
                binding_t *nb, int ntp, const uint32_t *av, int na, int ai)
{
    int b = -1;
    for (int k = 0; k < ntp; k++)
        if (nb[k].is_pack && strcmp(nb[k].name, tname) == 0) b = k;
    if (b < 0) return 0;
    if (nb[b].fixed) {
        ai += nb[b].npk;
        if (ai > na) return 0;
        if (nb[b].pk_nt) return ai >= na;
    }

    while (ai < na && nb[b].npk < MAX_PKELM) {
        uint32_t at = tpeel(L, arg_type(L, av[ai]), pn->d.oper.flags);
        if (!at) return 0;
        nb[b].pk_t[nb[b].npk++] = at;
        ai++;
    }
    return ai >= na;
}

static int tdflt(lower_t *L, binding_t *nb, int ntp)
{
    for (int i = 0; i < ntp; i++) {
        if (nb[i].is_pack || nb[i].fixed || nb[i].is_fn) continue;
        if (nb[i].is_type) {
            if (nb[i].type == 0 && nb[i].dflt)
                nb[i].type = rtype(L, nb[i].dflt, 0, 0);
            if (nb[i].type == 0)
                nb[i].type = bir_type_float(L->M, 32);
            continue;
        }
        if (!nb[i].dflt) return 0;
        if (!cival(L, nb[i].dflt, &nb[i].ival)) return 0;
    }
    return 1;
}

static void cscn(lower_t *L, uint32_t node)
{
    const ast_node_t *n = ND(L, node);
    uint32_t nm, ta;
    char cb[128], mg[128];

    if (n->type != AST_IDENT && n->type != AST_TYPE_SPEC) return;
    nm = n->type == AST_IDENT ? node : n->first_child;
    ta = n->type == AST_IDENT ? n->first_child
       : (nm ? ND(L, nm)->next_sibling : 0);
    while (ta && ND(L, ta)->type != AST_TEMPLATE_ARGS) {
        if (ND(L, ta)->type == AST_IDENT) nm = ta;
        ta = ND(L, ta)->next_sibling;
    }
    if (!ta || !nm || ND(L, nm)->type != AST_IDENT) return;
    get_text(L, nm, cb, sizeof(cb));
    for (int i = 0; i < L->nclss; i++)
        if (strcmp(L->clss[i].name, cb) == 0) {
            cget(L, node, cb, ta, mg, sizeof(mg));
            return;
        }
    {
        uint32_t at = 0;
        (void)altry(L, cb, ta, &at);
    }
}

static void ccolc(lower_t *L, uint32_t node)
{
    uint32_t sd = 0, ts = 0, nm;
    cltd_t *cd;

    for (uint32_t c = ND(L, node)->first_child; c; c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_STRUCT_DEF) {
            sd = c;
            ts = ND(L, c)->first_child;
            break;
        }
        if (ND(L, c)->type == AST_TYPE_SPEC) { ts = c; break; }
    }
    if (!ts || ND(L, ts)->type != AST_TYPE_SPEC) return;
    nm = ND(L, ts)->first_child;
    if (!nm || ND(L, nm)->type != AST_IDENT) return;
    if (L->nclss >= MAX_CLST) {
        lower_error(L, node, BC_E030, "more class templates than Booth holds");
        return;
    }
    cd = &L->clss[L->nclss++];
    memset(cd, 0, sizeof(*cd));
    cd->ast = node;
    cd->sdef = sd;
    get_text(L, nm, cd->name, sizeof(cd->name));
    for (uint32_t c = ND(L, nm)->next_sibling; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_TEMPLATE_ARGS) { cd->spec = c; break; }
}

static int cbeq(const binding_t *a, const binding_t *b)
{
    if (a->is_pack || b->is_pack) return 0;
    if (a->is_fn != b->is_fn) return 0;
    if (a->is_fn) return strcmp(a->fnm, b->fnm) == 0;
    if (a->is_type != b->is_type) return 0;
    if (a->is_type) return a->type != 0 && a->type == b->type;
    return a->ival == b->ival;
}

static int carg1(lower_t *L, uint32_t a, binding_t *b)
{
    const ast_node_t *an = ND(L, a);
    int64_t cv;

    memset(b, 0, sizeof(*b));
    switch (an->type) {
    case AST_TYPE_SPEC:
        b->is_type = 1;
        b->guess = bguess(L, a);
        b->type = resolve_type(L, a, tpdep(L, a), 0);
        if (L->rtbad || L->rtunk) { L->rtbad = 0; L->rtunk = 0; return 0; }
        return b->type != 0;
    case AST_INT_LIT:
        b->ival = parse_int_text(L->src + an->d.text.offset,
                                 (int)an->d.text.len);
        return 1;
    case AST_BOOL_LIT:
        b->ival = an->d.ival ? 1 : 0;
        return 1;
    case AST_IDENT:
    case AST_SCOPE_RES:
        return tenam(L, a, b, 0);
    default:
        if (!cival(L, a, &cv)) return 0;
        b->ival = cv;
        return 1;
    }
}

static int cpnm(const lower_t *L, uint32_t a, char *out, int oz)
{
    const ast_node_t *an = ND(L, a);
    uint32_t c;

    if (an->type == AST_IDENT) {
        get_text(L, a, out, oz);
        return 1;
    }
    if (an->type != AST_TYPE_SPEC) return 0;
    if (an->d.btype.kind != TYPE_NAME) return 0;
    if (an->qualifiers & (QUAL_DECOR | QUAL_REF | QUAL_RREF
                          | QUAL_PTR1 | QUAL_PTR2)) return 0;
    c = an->first_child;
    if (!c || ND(L, c)->type != AST_IDENT || ND(L, c)->next_sibling) return 0;
    get_text(L, c, out, oz);
    return 1;
}

static int cbnd(binding_t *nb, int b, const binding_t *av)
{
    binding_t sv = *av;

    if (nb[b].is_pack) return 0;
    if (nb[b].is_type && !sv.is_type) return 0;
    if (!nb[b].is_type && sv.is_type) return 0;
    ncpy(sv.name, sizeof(sv.name), nb[b].name);
    sv.dflt = nb[b].dflt;
    sv.fixed = 1;
    if (nb[b].fixed && !cbeq(&nb[b], &sv)) return 0;
    nb[b] = sv;
    return 1;
}

static int cmat(lower_t *L, const cltd_t *cd, const binding_t *ab, int na,
                binding_t *nb, int *ntpo, uint8_t *cvec)
{
    int ntp = tparm(L, cd->ast, nb), i;
    char pn[64];

    *ntpo = ntp;
    for (i = 0; i < na; i++) cvec[i] = 0;
    if (!cd->spec) {
        if (na > ntp) return 0;
        for (i = 0; i < na; i++)
            if (!cbnd(nb, i, &ab[i])) return 0;
        return tdflt(L, nb, ntp);
    }
    i = 0;
    for (uint32_t pa = ND(L, cd->spec)->first_child; pa;
         pa = ND(L, pa)->next_sibling) {
        binding_t pv;
        int b = -1;

        if (i >= na) return 0;
        if (cpnm(L, pa, pn, sizeof(pn)))
            for (int k = 0; k < ntp; k++)
                if (strcmp(nb[k].name, pn) == 0) { b = k; break; }
        if (b >= 0) {
            if (!cbnd(nb, b, &ab[i])) return 0;
        } else {
            cvec[i] = 1;
            if (ab[i].guess || !carg1(L, pa, &pv)) return 0;
            if (!cbeq(&pv, &ab[i])) return 0;
        }
        i++;
    }
    if (i != na) return 0;
    for (int k = 0; k < ntp; k++)
        if (!nb[k].fixed) return 0;
    return 1;
}

static int cdom(const uint8_t *a, const uint8_t *b, int na)
{
    int ge = 1, gt = 0;

    for (int k = 0; k < na; k++) {
        if (a[k] < b[k]) ge = 0;
        if (a[k] > b[k]) gt = 1;
    }
    return ge && gt;
}

static int csel(lower_t *L, const char *nm, const binding_t *ab, int na,
                binding_t *nb, int *ntpo)
{
    int cand[MAX_CLST], nc = 0, best = -1, part = 0, dep = L->cidep;
    uint8_t (*cv)[MAX_CARG] = L->ccv[dep];
    binding_t *tb = L->cnb[dep];
    int nt = 0;

    for (int i = 0; i < L->nclss && nc < MAX_CLST; i++) {
        if (strcmp(L->clss[i].name, nm) != 0 || !L->clss[i].sdef) continue;
        if (!cmat(L, &L->clss[i], ab, na, tb, &nt, cv[nc])) continue;
        if (L->clss[i].spec) part = 1;
        cand[nc++] = i;
    }
    if (part)
        for (int i = nc - 1; i >= 0; i--)
            if (!L->clss[cand[i]].spec) {
                for (int k = i; k < nc - 1; k++) {
                    cand[k] = cand[k + 1];
                    memcpy(cv[k], cv[k + 1], MAX_CARG);
                }
                nc--;
            }
    if (nc == 0) return -1;
    if (nc == 1) best = 0;
    for (int i = 0; nc > 1 && i < nc; i++) {
        int dom = 1;
        for (int j = 0; j < nc && dom; j++)
            if (i != j && !cdom(cv[i], cv[j], na)) dom = 0;
        if (dom) { if (best >= 0) return -2; best = i; }
    }
    if (best < 0) return -2;
    if (!cmat(L, &L->clss[cand[best]], ab, na, nb, ntpo, cv[best])) return -1;
    return cand[best];
}

static int cdfl(lower_t *L, const char *nm, binding_t *ab, int na)
{
    binding_t *tb = L->cnb[L->cidep];
    int ntp, i;

    for (i = 0; i < L->nclss; i++)
        if (!L->clss[i].spec && strcmp(L->clss[i].name, nm) == 0) break;
    if (i >= L->nclss) return na;
    ntp = tparm(L, L->clss[i].ast, tb);
    while (na < ntp && na < MAX_CARG) {
        if (!tb[na].dflt || !carg1(L, tb[na].dflt, &ab[na])) break;
        na++;
    }
    return na;
}

static int cinst(lower_t *L, uint32_t node, const char *nm, int ci,
                 const binding_t *ab, int na,
                 const binding_t *nb, int ntp, uint32_t sdef,
                 char *out, size_t oz)
{
    char key[256], nk[96], mang[128];
    int nsame = 0, onb, obs, ocx;
    size_t cl = strlen(nm);

    if (snprintf(nk, sizeof(nk), "%s#%d", nm, ci) < 0) return 0;
    tkeyb(ab, na, nk, key, sizeof(key));
    for (int i = 0; i < L->ncins; i++) {
        if (strcmp(L->cins[i].key, key) == 0) {
            ncpy(out, oz, L->cins[i].sym);
            return 1;
        }
        if (strncmp(L->cins[i].sym, nm, cl) == 0
            && L->cins[i].sym[cl] == CI_SEP)
            nsame++;
    }
    if (L->clow) return 0;
    onb = L->nbindings;
    obs = L->nbbase;
    if (L->ncins >= MAX_CINS || ntp > MAX_BIND - onb) {
        lower_error(L, node, BC_E030,
                    "more class instantiations than Booth holds");
        return 0;
    }
    if (snprintf(mang, sizeof(mang), "%s$%d", nm, nsame) < 0) return 0;
    if (snprintf(L->cins[L->ncins].key, sizeof(L->cins[0].key),
                 "%s", key) < 0) return 0;
    if (snprintf(L->cins[L->ncins].sym, sizeof(L->cins[0].sym),
                 "%s", mang) < 0) return 0;
    L->ncins++;
    ncpy(out, oz, mang);

    for (int i = 0; i < ntp; i++) L->bindings[L->nbindings++] = nb[i];
    L->nbbase = onb;
    ocx = L->ncexp;
    ncpy(L->covr, sizeof(L->covr), mang);
    collect_struct(L, sdef);
    L->covr[0] = 0;
    cstat(L, sdef, mang);
    ccomp(L, ocx);
    L->nbbase = obs;
    L->nbindings = onb;
    return 1;
}

static int cget(lower_t *L, uint32_t node, const char *nm, uint32_t ta,
                char *out, size_t oz)
{
    binding_t *ab;
    int na = 0, ntp = 0, sel, dep = L->cidep;

    if (dep + 1 >= MAX_CNST || !ta) return 0;
    ab = L->cab[dep];
    L->cidep = dep + 1;
    for (uint32_t a = ND(L, ta)->first_child; a; a = ND(L, a)->next_sibling) {
        if (na >= MAX_CARG || !carg1(L, a, &ab[na])) {
            L->cidep = dep;
            return 0;
        }
        na++;
    }
    na = cdfl(L, nm, ab, na);
    if (na == 0) {
        L->cidep = dep;
        return 0;
    }
    sel = csel(L, nm, ab, na, L->cnb[dep], &ntp);
    if (sel == -2) lower_error(L, node, BC_E320, nm);
    if (sel >= 0)
        sel = cinst(L, node, nm, sel, ab, na, L->cnb[dep], ntp,
                    L->clss[sel].sdef, out, oz);
    else
        sel = 0;
    L->cidep = dep;
    return sel;
}

static int ckey(const lower_t *L, uint32_t at, char *out, size_t oz)
{
    const char *sn = sname(L, at);

    if (!sn) return 0;
    for (int i = 0; i < L->ncins; i++)
        if (strcmp(L->cins[i].sym, sn) == 0) {
            ncpy(out, oz, L->cins[i].key);
            return 1;
        }
    return 0;
}

static int cfld1(const char *key, int k, int *ist, int64_t *v)
{
    const char *p = key;

    for (int i = 0; i <= k; i++) {
        p = strchr(p, '|');
        if (!p) return 0;
        p++;
    }
    if (*p == 't') *ist = 1;
    else if (*p == 'n') *ist = 0;
    else return 0;
    *v = (int64_t)strtoll(p + 1, NULL, 10);
    return 1;
}

static uint32_t targn(const lower_t *L, uint32_t ts)
{
    uint32_t nmn, ta;

    if (!ts || ND(L, ts)->type != AST_TYPE_SPEC) return 0;
    if (ND(L, ts)->d.btype.kind != TYPE_NAME) return 0;
    nmn = ND(L, ts)->first_child;
    if (!nmn || ND(L, nmn)->type != AST_IDENT) return 0;
    ta = ND(L, nmn)->next_sibling;
    while (ta && ND(L, ta)->type != AST_TEMPLATE_ARGS)
        ta = ND(L, ta)->next_sibling;
    return ta;
}

static int tdbnd(lower_t *L, uint32_t a, binding_t *nb, int ntp,
                 int ist, int64_t v)
{
    char an[64];
    uint32_t id = a;

    if (ND(L, a)->type == AST_TYPE_SPEC) {
        if (ND(L, a)->d.btype.kind != TYPE_NAME) return 0;
        id = ND(L, a)->first_child;
        if (!id || ND(L, id)->type != AST_IDENT) return 0;
        if (ND(L, id)->next_sibling) return 0;
    } else if (ND(L, a)->type != AST_IDENT || ND(L, a)->first_child) {
        return 0;
    }
    get_text(L, id, an, (int)sizeof an);
    for (int b = 0; b < ntp; b++) {
        if (nb[b].is_pack || nb[b].is_fn) continue;
        if (strcmp(nb[b].name, an) != 0) continue;
        if (nb[b].is_type != ist) return -1;
        if (ist) {
            if (nb[b].type && nb[b].type != (uint32_t)v) return -1;
            nb[b].type  = (uint32_t)v;
            nb[b].guess = 0;
        } else {
            if (nb[b].fixed && nb[b].ival != v) return -1;
            nb[b].ival = v;
        }
        nb[b].fixed = 1;
        return 1;
    }
    return 0;
}

static int tdcls(lower_t *L, const ast_node_t *pn, binding_t *nb, int ntp,
                 uint32_t arg)
{
    uint32_t ta = targn(L, pn->first_child), a;
    char key[256], cn[64];
    uint32_t at;
    size_t cl;
    int k = 0;

    if (!ta) return 0;
    get_text(L, ND(L, pn->first_child)->first_child, cn, (int)sizeof cn);
    at = tpeel(L, arg_type(L, arg), pn->d.oper.flags);
    if (!at || !ckey(L, at, key, sizeof key)) return 0;
    cl = strlen(cn);
    if (strncmp(key, cn, cl) != 0 || key[cl] != '#') return -1;
    for (a = ND(L, ta)->first_child; a; a = ND(L, a)->next_sibling, k++) {
        int ist = 0, r;
        int64_t v = 0, cv = 0;

        if (!cfld1(key, k, &ist, &v)) return -1;
        r = tdbnd(L, a, nb, ntp, ist, v);
        if (r < 0) return -1;
        if (r > 0) continue;
        if (ist) {
            if (ND(L, a)->type != AST_TYPE_SPEC) return -1;
            if (resolve_type(L, a, 0, 0) != (uint32_t)v) return -1;
        } else {
            if (!cival(L, a, &cv) || cv != v) return -1;
        }
    }
    return k > 0;
}

static int tdpar(lower_t *L, uint32_t pnn, binding_t *nb, int ntp,
                 const uint32_t *av, int na, int *aio)
{
    const ast_node_t *pn = ND(L, pnn);
    uint32_t ts = pn->first_child;
    int ai = *aio;
    char tname[64];

    tname[0] = 0;
    if (ts && ND(L, ts)->type == AST_TYPE_SPEC
           && ND(L, ts)->d.btype.kind == TYPE_NAME
           && ND(L, ts)->first_child)
        get_text(L, ND(L, ts)->first_child, tname, sizeof(tname));

    if (pn->d.oper.op == PRM_PACK)
        return tdpk(L, pn, tname, nb, ntp, av, na, ai) ? -1 : 0;

    if (ai < na) {
        int sq = isqb(L, ts, av[ai], nb, ntp);

        if (sq == 0) return 0;
        if (sq > 0) { *aio = ai + 1; return 1; }
    }
    if (ai < na && targn(L, ts)) {
        int r = tdcls(L, pn, nb, ntp, av[ai]);

        if (r < 0) return 0;
        if (r > 0) { *aio = ai + 1; return 1; }
    }
    if (tname[0] && ai < na) {
        uint32_t at = tpeel(L, arg_type(L, av[ai]), pn->d.oper.flags);

        if (at && at < L->M->num_types
            && L->M->types[at].kind == BIR_TYPE_ARRAY
            && !(pn->qualifiers & (QUAL_REF | QUAL_RREF))
            && tdopn(nb, ntp, tname))
            return 0;
        tdty(L, nb, ntp, tname, at);
    }
    if (ai < na) *aio = ai + 1;
    return 1;
}

static int tdeduc(lower_t *L, uint32_t tnode, uint32_t cal,
                  const uint32_t *av, int na,
                  binding_t *nb, int *ntpo, uint32_t *funco)
{
    int ntp = tparm(L, tnode, nb);
    int ai = 0;
    *ntpo = ntp;

    uint32_t func_n = 0;
    for (uint32_t fc = ND(L, tnode)->first_child; fc;
         fc = ND(L, fc)->next_sibling)
        if (ND(L, fc)->type == AST_FUNC_DEF) { func_n = fc; break; }
    if (!func_n) return 0;
    *funco = func_n;

    if (!texpl(L, cal, nb, ntp)) return 0;

    uint32_t fpn[MAX_PARM];
    int nfp = collect_params(L, func_n, fpn, MAX_PARM);
    if (nfp < 0) return 0;

    for (int i = 0; i < nfp; i++) {
        int r = tdpar(L, fpn[i], nb, ntp, av, na, &ai);

        if (r <= 0) return r == 0 ? 0 : 1;
    }

    for (int i = 0; i < ntp; i++)
        if (nb[i].is_type && !nb[i].fixed && nb[i].type == 0) {
            nb[i].type  = bir_type_float(L->M, 32);
            nb[i].guess = 1;
        }
    return tdflt(L, nb, ntp);
}

static void tkeyb(const binding_t *nb, int ntp, const char *cname,
                  char *key, size_t keyz)
{
    int kl = snprintf(key, keyz, "%s", cname);
    for (int i = 0; i < ntp && kl > 0 && kl < (int)keyz - 32; i++) {
        if (nb[i].is_pack) {
            kl += snprintf(key + kl, keyz - (size_t)kl, "|p%d%c",
                           nb[i].npk, nb[i].pk_nt ? 'n' : 't');
            for (int k = 0; k < nb[i].npk && kl < (int)keyz - 24; k++) {
                if (nb[i].pk_nt)
                    kl += snprintf(key + kl, keyz - (size_t)kl,
                                   ":%lld", (long long)nb[i].pk_v[k]);
                else
                    kl += snprintf(key + kl, keyz - (size_t)kl,
                                   ":%u", nb[i].pk_t[k]);
            }
        } else if (nb[i].is_fn) {
            kl += snprintf(key + kl, keyz - (size_t)kl, "|f%s", nb[i].fnm);
        } else if (nb[i].is_type) {
            kl += snprintf(key + kl, keyz - (size_t)kl, "|t%u", nb[i].type);
        } else {
            kl += snprintf(key + kl, keyz - (size_t)kl,
                           "|n%lld", (long long)nb[i].ival);
        }
    }
}

static int tfidx(const lower_t *L, const char *key)
{
    for (int i = 0; i < L->ninst; i++)
        if (strcmp(L->inst[i].key, key) == 0) return i;
    return -1;
}

static int tfind(const lower_t *L, const char *key, char *sym, size_t symz)
{
    int i = tfidx(L, key);

    if (i < 0) return 0;
    ncpy(sym, symz, L->inst[i].sym);
    return 1;
}

static int tisin(const lower_t *L, const char *sym);

static int tmang(lower_t *L, const char *cname, int slot,
                 char *out, size_t oz)
{
    char base[112];

    ncpy(base, sizeof(base), cname);
    KA_GUARD(g, MAX_INSTS);
    for (;;) {
        uint32_t f;

        if (slot == 0) ncpy(out, oz, base);
        else if (snprintf(out, oz, "%s$%d", base, slot) < 0) return 0;
        f = bir_fsym(L->M, out, L->tu, -1);
        if (!tisin(L, out)
            && (f == BIR_SYM_NONE || L->M->funcs[f].tu != L->tu)) return 1;
        if (!g--) return 0;
        slot++;
    }
}

static int tinst(lower_t *L, uint32_t node, const char *cname,
                 const binding_t *nb, int ntp, uint32_t func_n)
{
    char key[256];
    tkeyb(nb, ntp, cname, key, sizeof(key));
    tkfn(L, cname, func_n, key, sizeof(key));

    int nsame = 0;
    size_t cl = strlen(cname);
    for (int i = 0; i < L->ninst; i++) {
        if (strcmp(L->inst[i].key, key) == 0) return 0;
        if (strncmp(L->inst[i].key, cname, cl) == 0
            && L->inst[i].key[cl] == '|')
            nsame++;
    }
    if (L->ninst >= MAX_INSTS) {
        lower_error(L, node, BC_E030, "too many template instantiations");
        return -1;
    }

    char mangled[128];

    if (!tmang(L, cname, nsame, mangled, sizeof(mangled))) return -1;
    if (snprintf(L->inst[L->ninst].key, sizeof(L->inst[0].key), "%s", key) < 0
        || snprintf(L->inst[L->ninst].sym, sizeof(L->inst[0].sym), "%s",
                    mangled) < 0) return -1;
    L->ninst++;

    if (bir_fsym(L->M, mangled, L->tu, -1) != BIR_SYM_NONE) return 0;

    int old_nb = L->nbindings, old_bs = L->nbbase;
    if (ntp > MAX_BIND - old_nb) {
        lower_error(L, node, BC_E030,
                    "template nesting deeper than Booth binds");
        return -1;
    }
    for (int i = 0; i < ntp; i++)
        L->bindings[L->nbindings++] = nb[i];
    L->nbbase = old_nb;

    if (L->tdep < MAX_TDEP) {
        L->tdep++;
        tscan(L, func_n);
        L->tdep--;
    }

    uint16_t cuda = ND(L, func_n)->cuda_flags;
    uint32_t mrs = L->mrec;
    L->mrec = 0;
    lower_func_body(L, func_n, cuda, mangled);
    L->mrec = mrs;
    L->nbbase = old_bs;
    L->nbindings = old_nb;
    return 1;
}

static int tisin(const lower_t *L, const char *sym)
{
    for (int i = 0; i < L->ninst; i++)
        if (strcmp(L->inst[i].sym, sym) == 0) return 1;
    return 0;
}

static int qtsym(lower_t *L, char *cnm, size_t cz, uint32_t cal)
{
    uint32_t l, r, ta;
    char tn[64], mn[64], mg[128];

    if (!cal || ND(L, cal)->type != AST_SCOPE_RES) return 0;
    l = ND(L, cal)->first_child;
    r = l ? ND(L, l)->next_sibling : 0;
    if (!l || !r) return 0;
    if (ND(L, l)->type != AST_IDENT || ND(L, r)->type != AST_IDENT) return 0;
    if (ND(L, r)->next_sibling) return 0;
    ta = ND(L, l)->first_child;
    if (!ta || ND(L, ta)->type != AST_TEMPLATE_ARGS) return 0;
    get_text(L, l, tn, sizeof(tn));
    if (!cget(L, cal, tn, ta, mg, sizeof(mg))) return 0;
    get_text(L, r, mn, sizeof(mn));
    return snprintf(cnm, cz, "%s$%s", mg, mn) > 0;
}

static int tguess(const binding_t *nb, int ntp)
{
    for (int i = 0; i < ntp; i++)
        if (nb[i].guess) return 1;
    return 0;
}

static int tsym(lower_t *L, char *cname, size_t cz, uint32_t cal,
                uint32_t arg)
{
    template_def_t *tm = find_template(L, cname);
    if (!tm) return 0;

    uint32_t av[BC_MAX_ARGS];
    int na = alist(L, arg, av, BC_MAX_ARGS);
    binding_t *nb = L->tnb;
    int ntp = 0;
    uint32_t func_n = 0;
    char key[256], sym[128];

    if (na >= 0) tm = tmpick(L, cname, av, na);
    if (!tm) return 0;
    if (na >= 0 && tdeduc(L, tm->ast, cal, av, na, nb, &ntp, &func_n)
        && !tguess(nb, ntp)) {
        tkeyb(nb, ntp, cname, key, sizeof(key));
        tkfn(L, cname, func_n, key, sizeof(key));
        if (tfind(L, key, sym, sizeof(sym))) {
            ncpy(cname, cz, sym);
            return 1;
        }
    }
    return tisin(L, cname) ? -1 : 0;
}

static void vscn(lower_t *L, uint32_t vd)
{
    const ast_node_t *n = ND(L, vd);
    uint32_t nn = child_at(L, vd, 1);
    char vn[128];
    uint32_t t;

    if (n->qualifiers & QUAL_TYPEDEF) return;
    if (!nn || ND(L, nn)->type != AST_IDENT) return;
    get_text(L, nn, vn, (int)sizeof vn);
    if (!vn[0]) return;
    t = rtype(L, n->first_child, n->d.oper.flags, n->cuda_flags);
    if (t) {
        uint32_t aft = 0;
        int bad = 0;

        t = farr(L, t, vd, nn, &aft, &bad, NULL);
        if (bad) return;
        add_sym(L, vn, BIR_VAL_NONE, t, 0);
    }
}

static void pscn(lower_t *L, uint32_t fd)
{
    uint32_t pv[MAX_PARM];
    int np = collect_params(L, fd, pv, MAX_PARM);

    for (int i = 0; i < np; i++) {
        const ast_node_t *pn = ND(L, pv[i]);
        char pnm[64];
        uint32_t t;

        if (pn->d.oper.op == PRM_VARG || pn->d.oper.op == PRM_PACK) continue;
        pnm[0] = 0;
        for (uint32_t pc = pn->first_child; pc; pc = ND(L, pc)->next_sibling)
            if (ND(L, pc)->type == AST_IDENT) {
                get_text(L, pc, pnm, (int)sizeof pnm);
                break;
            }
        if (!pnm[0] || find_sym(L, pnm)) continue;
        t = rtype(L, pn->first_child, pn->d.oper.flags, pn->cuda_flags);
        if (t) add_sym(L, pnm, BIR_VAL_NONE, t, 0);
    }
}

static void tscall(lower_t *L, uint32_t node)
{
    uint32_t cn = ND(L, node)->first_child;
    template_def_t *tm;
    char cname[128];
    const char *fb;
    uint32_t av[BC_MAX_ARGS], func_n = 0;
    int na, ntp = 0;

    if (!cn || ND(L, cn)->type != AST_IDENT) return;
    get_text(L, cn, cname, sizeof(cname));
    fb = fbfn(L, cname);
    if (fb) ncpy(cname, sizeof(cname), fb);
    tm = fb && bir_fsym(L->M, cname, L->tu, -1) != BIR_SYM_NONE
       ? NULL : find_template(L, cname);
    if (!tm) return;
    na = alist(L, ND(L, cn)->next_sibling, av, BC_MAX_ARGS);
    if (na < 0 || L->xdep >= MAX_NEST) return;
    tm = tmpick(L, cname, av, na);
    if (!tm) return;
    {
        binding_t *nb = L->xnb[L->xdep++];

        if (tdeduc(L, tm->ast, cn, av, na, nb, &ntp, &func_n))
            tinst(L, node, cname, nb, ntp, func_n);
        L->xdep--;
    }
}

static int cxarm(lower_t *L, uint32_t node, uint32_t *arm)
{
    char nm[sizeof L->sdbnm];
    uint16_t sv = L->sdbad;
    cval_t cv;
    int ok;

    ncpy(nm, sizeof nm, L->sdbnm);
    ok = cfsc(L, child_at(L, node, 0), &cv);
    L->sdbad = sv;
    ncpy(L->sdbnm, sizeof L->sdbnm, nm);
    if (!ok || cv.agg) return 0;
    *arm = (cv.isf ? cv.fval != 0.0 : cv.ival != 0)
         ? child_at(L, node, 1) : child_at(L, node, 2);
    return 1;
}

static int tswa(lower_t *L, uint32_t node, uint32_t *lab)
{
    uint32_t bd = child_at(L, node, 1), c, dfl = 0, hit = 0;
    char nm[sizeof L->sdbnm];
    uint16_t sv = L->sdbad;
    cval_t cv, lv;
    int ok;

    *lab = 0;
    if (!bd) return 0;
    ncpy(nm, sizeof nm, L->sdbnm);
    ok = cfsc(L, child_at(L, node, 0), &cv) && !cv.isf && !cv.fnd;
    KA_GUARD(g, CE_STMT);
    for (c = ND(L, bd)->first_child; ok && c && g--;
         c = ND(L, c)->next_sibling) {
        if (ND(L, c)->type == AST_DEFAULT) {
            if (!dfl) dfl = c;
            continue;
        }
        if (ND(L, c)->type != AST_CASE) continue;
        if (!cfsc(L, ND(L, c)->first_child, &lv) || lv.isf || lv.fnd) {
            ok = 0;
            break;
        }
        if (lv.ival == cv.ival) {
            hit = c;
            break;
        }
    }
    L->sdbad = sv;
    ncpy(L->sdbnm, sizeof L->sdbnm, nm);
    if (!ok) return 0;
    *lab = hit ? hit : dfl;
    return 1;
}

static void tsval(lower_t *L, uint32_t id)
{
    template_def_t *tm;
    char cname[128];
    const char *fb;
    uint32_t av[1], func_n = 0;
    int ntp = 0;

    if (!ND(L, id)->first_child
        || ND(L, ND(L, id)->first_child)->type != AST_TEMPLATE_ARGS) return;
    get_text(L, id, cname, sizeof(cname));
    fb = fbfn(L, cname);
    if (fb) ncpy(cname, sizeof(cname), fb);
    tm = fb && bir_fsym(L->M, cname, L->tu, -1) != BIR_SYM_NONE
       ? NULL : find_template(L, cname);
    if (!tm || L->xdep >= MAX_NEST) return;

    av[0] = 0;
    {
        binding_t *nb = L->xnb[L->xdep++];

        if (tdeduc(L, tm->ast, id, av, 0, nb, &ntp, &func_n)) {
            int all = 1;

            for (int i = 0; i < ntp; i++)
                if (!nb[i].is_pack && !nb[i].fixed && !nb[i].dflt) all = 0;
            if (all) tinst(L, id, cname, nb, ntp, func_n);
        }
        L->xdep--;
    }

}

static void tcxd(lower_t *L, uint32_t node)
{
    const ast_node_t *n = ND(L, node);
    uint32_t nn, in;

    if (n->type != AST_VAR_DECL || n->d.oper.op != 0
        || (n->qualifiers & QUAL_TYPEDEF)
        || !(n->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST))) return;

    nn = child_at(L, node, 1);
    in = nn && ND(L, nn)->type == AST_IDENT ? ND(L, nn)->next_sibling : 0;
    if (in && ND(L, in)->type == AST_STRUCT_DEF) in = ND(L, in)->next_sibling;
    if (in && !ccoll(L, node, nn, in, NULL, 0))
        lower_error(L, node, BC_E130, MAX_CEXPS);
}

static int tswk(lower_t *L, uint32_t node)
{
    uint32_t lab = 0;

    if (!tswa(L, node, &lab)) return 0;
    tscan(L, child_at(L, node, 0));
    KA_GUARD(g, CE_STMT);
    for (uint32_t c = lab; c && g--; c = ND(L, c)->next_sibling) {
        uint32_t ct = ND(L, c)->type;

        if (ct == AST_BREAK) break;
        if (ct != AST_CASE && ct != AST_DEFAULT) tscan(L, c);
        if (ct == AST_RETURN) break;
    }
    return 1;
}

static void tscan(lower_t *L, uint32_t node)
{
    if (!node) return;
    const ast_node_t *n = ND(L, node);
    int csv = n->type == AST_BLOCK ? L->ncexp : -1;
    int psv = n->type == AST_FUNC_DEF || n->type == AST_BLOCK
            ? L->nsyms : -1;

    int tsv = psv >= 0 ? L->ntypedefs : -1;

    if (n->type == AST_IF && (n->d.oper.flags & IF_CEXP)) {
        uint32_t arm = 0;

        if (cxarm(L, node, &arm)) {
            tscan(L, child_at(L, node, 0));
            if (arm) tscan(L, arm);
            return;
        }
    }

    if (n->type == AST_FUNC_DEF) pscn(L, node);

    tcxd(L, node);

    cscn(L, node);

    if (n->type == AST_CALL) tscall(L, node);

    if (n->type == AST_SWITCH && tswk(L, node)) return;

    {
        uint32_t cal = n->type == AST_CALL ? n->first_child : 0;
        int tvok = n->type != AST_SCOPE_RES && n->type != AST_TYPE_SPEC
                && n->type != AST_MEMBER && n->type != AST_TEMPLATE_ARGS;

        for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling) {
            if (tvok && c != cal && ND(L, c)->type == AST_IDENT)
                tsval(L, c);
            tscan(L, c);
        }
    }

    if (n->type == AST_VAR_DECL) {
        if (n->qualifiers & QUAL_TYPEDEF) {
            if (n->d.oper.op == 0) collect_typedef(L, node);
        } else {
            vscn(L, node);
        }
    }
    if (psv >= 0) L->nsyms = psv;
    if (tsv >= 0) L->ntypedefs = tsv;
    if (csv >= 0) ccomp(L, csv);
}

static uint32_t fwlk(const lower_t *L, uint32_t nd)
{
    nd = unfw(L, nd);
    if (L->nfwf <= 0 || !nd || ND(L, nd)->type != AST_IDENT) return nd;
    const fwfr_t *f = &L->fwf[L->nfwf - 1];
    char nm[64];
    get_text(L, nd, nm, sizeof(nm));
    for (int i = 0; i < f->nb; i++)
        if (strcmp(f->b[i].name, nm) == 0) return f->b[i].arg;
    return nd;
}

static int fwmen(const lower_t *L, uint32_t nd, int depth)
{
    const fwfr_t *f = &L->fwf[L->nfwf - 1];
    if (!nd || depth > 8 || L->nfwf <= 0 || !f->pn[0]) return 0;
    if (ND(L, nd)->type == AST_IDENT) {
        char nm[64];
        get_text(L, nd, nm, sizeof(nm));
        if (strcmp(nm, f->pn) == 0) return 1;
    }
    for (uint32_t c = ND(L, nd)->first_child; c; c = ND(L, c)->next_sibling)
        if (fwmen(L, c, depth + 1)) return 1;
    return 0;
}

static int spkx(lower_t *L, uint32_t node, uint32_t *av, int n, int max)
{
    uint32_t pat = ND(L, node)->first_child, t0;
    char nms[MAX_FPACKS][64];
    int nn = 0, len;

    if (!pat) return -1;
    len = pk_len(L, pat, nms, &nn, 1);
    if (len < 0 || n + len > max) return -1;
    t0 = arg_type(L, pat);
    for (int j = 0; j < len; j++) {
        int sva = L->npkact, svb = L->nbindings;
        uint32_t tj = cpkb(L, nms, nn, j) ? arg_type(L, pat) : 0;
        L->npkact    = sva;
        L->nbindings = svb;
        if (!t0 || tj != t0) {
            L->pkbad = 1;
            return -1;
        }
        av[n + j] = pat;
    }
    return len;
}

static int subav(lower_t *L, uint32_t first, uint32_t *av, int max)
{
    int n = 0;
    for (uint32_t c = first; c; c = ND(L, c)->next_sibling) {
        if (L->nfwf > 0 && ND(L, c)->type == AST_PACK_EXP
            && fwmen(L, ND(L, c)->first_child, 0)) {
            const fwfr_t *f = &L->fwf[L->nfwf - 1];
            for (int k = 0; k < f->npk; k++) {
                if (n >= max) return -1;
                av[n++] = f->pk[k];
            }
            continue;
        }
        if (ND(L, c)->type == AST_PACK_EXP) {
            int got = spkx(L, c, av, n, max);
            if (got < 0) return -1;
            n += got;
            continue;
        }
        if (n >= max) return -1;
        av[n++] = fwlk(L, c);
    }
    return n;
}

static int lav(lower_t *L, uint32_t node, uint32_t *av, int max)
{
    int ncfg = ND(L, node)->d.oper.op;
    if (ncfg < 2 || ncfg > 4) ncfg = 2;

    uint32_t c = ND(L, node)->first_child;
    for (int i = 0; i <= ncfg && c; i++) c = ND(L, c)->next_sibling;
    return subav(L, c, av, max);
}

static uint32_t tfunc(const lower_t *L, const template_def_t *tm)
{
    for (uint32_t fc = ND(L, tm->ast)->first_child; fc;
         fc = ND(L, fc)->next_sibling)
        if (ND(L, fc)->type == AST_FUNC_DEF) return fc;
    return 0;
}

static template_def_t *ftmpl(lower_t *L, const char *name, int na)
{
    template_def_t *first = NULL;

    for (int i = 0; i < L->ntemplates; i++) {
        template_def_t *t = &L->templates[i];
        uint32_t fn, pv[MAX_PARM];
        int np, nrq = 0, vg = 0;

        if (strcmp(t->name, name) != 0) continue;
        if (!first) first = t;
        if (na < 0) return t;
        fn = tfunc(L, t);
        if (!fn) continue;
        np = collect_params(L, fn, pv, MAX_PARM);
        if (np < 0) continue;
        for (int k = 0; k < np; k++) {
            if (ND(L, pv[k])->d.oper.op) { vg = 1; continue; }
            if (!(ND(L, pv[k])->qualifiers & QUAL_PDEF)) nrq++;
        }
        if (vg) np--;
        if (na >= nrq && (vg || na <= np)) return t;
    }
    return first;
}

static int tmamb(const lower_t *L, const char *name)
{
    int n = 0;

    for (int i = 0; i < L->ntemplates; i++)
        if (strcmp(L->templates[i].name, name) == 0) n++;
    return n > 1;
}

static int tmnms(lower_t *L, uint32_t tnode, char tb[][64], int max)
{
    int n = 0;

    for (uint32_t tp = ND(L, tnode)->first_child;
         tp && ND(L, tp)->type == AST_TEMPLATE_PARAM && n < max;
         tp = ND(L, tp)->next_sibling) {
        tb[n][0] = 0;
        for (uint32_t c = ND(L, tp)->first_child; c;
             c = ND(L, c)->next_sibling)
            if (ND(L, c)->type == AST_IDENT) {
                get_text(L, c, tb[n], 64);
                break;
            }
        n++;
    }
    return n;
}

static int tmdep(lower_t *L, char tb[][64], int ntb, uint32_t ts,
                 int dep)
{
    char nm[64];

    if (!ts || dep > TM_DEEP) return 1;
    if (ND(L, ts)->type == AST_TEMPLATE_ARGS) return 1;
    if (ND(L, ts)->type == AST_IDENT) {
        get_text(L, ts, nm, sizeof(nm));
        for (int i = 0; i < ntb; i++)
            if (strcmp(tb[i], nm) == 0) return 1;
    }
    for (uint32_t c = ND(L, ts)->first_child; c; c = ND(L, c)->next_sibling)
        if (tmdep(L, tb, ntb, c, dep + 1)) return 1;
    return 0;
}

static uint32_t tmpt(lower_t *L, uint32_t pn, char tb[][64], int ntb)
{
    const ast_node_t *p = ND(L, pn);
    uint32_t t;
    int sv = L->rtunk;

    if (p->d.oper.op) return 0;
    if (tmdep(L, tb, ntb, p->first_child, 0)) return 0;
    L->rtunk = 0;
    t = rtype(L, p->first_child, p->d.oper.flags, p->cuda_flags);
    if (L->rtunk) t = 0;
    L->rtunk = sv;
    return t;
}

static int tmarg(lower_t *L, uint32_t pnn, uint32_t tnode, uint32_t arg,
                 char tb[][64], int ntb)
{
    const ast_node_t *pn = ND(L, pnn);
    uint32_t pt, at;

    if (targn(L, pn->first_child)) {
        binding_t *sb;
        int stp, r;

        if (L->pdep >= MAX_NEST) return 0;
        sb  = L->pnb[L->pdep++];
        stp = tparm(L, tnode, sb);
        r   = tdcls(L, pn, sb, stp, arg);
        L->pdep--;
        return r;
    }
    pt = tmpt(L, pnn, tb, ntb);
    if (!pt) return 0;
    at = arg_type(L, arg);
    if (!at) return 0;
    return pt == at ? 1 : -1;
}

static template_def_t *tmpick(lower_t *L, const char *name,
                              const uint32_t *av, int na)
{
    template_def_t *hit = NULL;
    char tb[MAX_TPARM][64];
    int nhit = 0, best = 0;

    if (na <= 0 || !tmamb(L, name)) return ftmpl(L, name, na);
    for (int i = 0; i < L->ntemplates; i++) {
        template_def_t *t = &L->templates[i];
        uint32_t fn, pv[MAX_PARM];
        int np, ntb, ok = 1, sure = 0;

        if (strcmp(t->name, name) != 0) continue;
        fn = tfunc(L, t);
        if (!fn) continue;
        np = collect_params(L, fn, pv, MAX_PARM);
        if (np != na) continue;
        ntb = tmnms(L, t->ast, tb, MAX_TPARM);
        for (int k = 0; k < np && ok; k++) {
            int r = tmarg(L, pv[k], t->ast, av[k], tb, ntb);

            if (r < 0) ok = 0;
            else if (r > 0) sure++;
        }
        if (!ok) continue;
        if (sure > best) {
            best = sure;
            hit  = t;
            nhit = 1;
        } else if (sure == best) {
            nhit++;
        }
    }
    return nhit == 1 && best > 0 ? hit : ftmpl(L, name, na);
}

static void tkfn(lower_t *L, const char *cname, uint32_t func_n,
                 char *key, size_t keyz)
{
    size_t kl = strlen(key);

    if (!tmamb(L, cname) || kl + 12 >= keyz) return;
    if (snprintf(key + kl, keyz - kl, "@%u", func_n) < 0) key[kl] = 0;
}

static int hasl(lower_t *L, uint32_t nd, int depth, int hop)
{
    if (!nd || depth > 64) return 0;
    if (ND(L, nd)->type == AST_LAUNCH) return 1;
    if (ND(L, nd)->type == AST_CALL && hop > 0) {
        uint32_t cn = ND(L, nd)->first_child;
        if (cn && ND(L, cn)->type == AST_IDENT) {
            char nm[128];
            get_text(L, cn, nm, sizeof(nm));
            template_def_t *tm = find_template(L, nm);
            uint32_t fn = tm ? tfunc(L, tm) : 0;
            if (fn && hasl(L, find_func_body(L, fn), 0, hop - 1)) return 1;
        }
    }
    for (uint32_t c = ND(L, nd)->first_child; c; c = ND(L, c)->next_sibling)
        if (hasl(L, c, depth + 1, hop)) return 1;
    return 0;
}

static int fwpush(lower_t *L, uint32_t func_n, const uint32_t *av, int na)
{
    uint32_t fpn[MAX_PARM];
    int nfp = collect_params(L, func_n, fpn, MAX_PARM);
    int ai = 0;
    if (nfp < 0 || L->nfwf >= MAX_FWD) return 0;

    fwfr_t *f = &L->fwf[L->nfwf];
    f->nb = 0;
    f->npk = 0;
    f->pn[0] = 0;
    L->fwovf = 0;

    for (int i = 0; i < nfp; i++) {
        const ast_node_t *pn = ND(L, fpn[i]);
        char base[64];
        base[0] = 0;
        for (uint32_t pc = pn->first_child; pc; pc = ND(L, pc)->next_sibling)
            if (ND(L, pc)->type == AST_IDENT) {
                get_text(L, pc, base, sizeof(base));
                break;
            }
        if (pn->d.oper.op == PRM_VARG) return 0;

        if (pn->d.oper.op == PRM_PACK) {
            if (!base[0]) return 0;
            ncpy(f->pn, sizeof(f->pn), base);
            while (ai < na) {
                if (f->npk >= MAX_FWPK) { L->fwovf = MAX_FWPK; return 0; }
                f->pk[f->npk++] = av[ai++];
            }
            continue;
        }
        if (ai >= na) {
            uint32_t d = pdfn(L, fpn[i]);
            if (!(pn->qualifiers & QUAL_PDEF) || !d) return 0;
            if (!base[0]) continue;
            if (f->nb >= MAX_FWB) { L->fwovf = MAX_FWB; return 0; }
            ncpy(f->b[f->nb].name, sizeof(f->b[0].name), base);
            f->b[f->nb].arg = d;
            f->nb++;
            continue;
        }
        if (!base[0]) { ai++; continue; }
        if (f->nb >= MAX_FWB) { L->fwovf = MAX_FWB; return 0; }
        ncpy(f->b[f->nb].name, sizeof(f->b[0].name), base);
        f->b[f->nb].arg = av[ai++];
        f->nb++;
    }
    if (ai < na) return 0;
    L->nfwf++;
    return 1;
}

static int slnch(lower_t *L, uint32_t node)
{
    uint32_t callee_n = fwlk(L, ND(L, node)->first_child);
    if (!callee_n || ND(L, callee_n)->type != AST_IDENT) return 0;

    char cname[128];
    get_text(L, callee_n, cname, sizeof(cname));

    if (!find_template(L, cname)) return 0;

    uint32_t av[BC_MAX_ARGS];
    int na;

    L->pkbad = 0;
    na = lav(L, node, av, BC_MAX_ARGS);
    if (na < 0) {
        if (L->pkbad) lower_error(L, node, BC_E420, cname);
        else lower_error(L, node, BC_E082, cname, BC_MAX_ARGS);
        return 1;
    }
    template_def_t *tmpl = ftmpl(L, cname, na);
    if (!tmpl) return 0;

    binding_t *nb = L->snb;
    int ntp = 0;
    uint32_t func_n = 0;

    if (!tdeduc(L, tmpl->ast, callee_n, av, na, nb, &ntp, &func_n)) {
        int pk = 0;
        for (int i = 0; i < ntp; i++)
            if (nb[i].is_pack) pk = 1;
        if (func_n)
            lower_error(L, node, BC_E030, pk
                ? "a parameter pack Booth cannot deduce at this launch"
                : "a template argument Booth cannot deduce at this launch");
        return 1;
    }

    tinst(L, node, cname, nb, ntp, func_n);
    return 1;
}

static int swrap(lower_t *L, uint32_t node)
{
    uint32_t cn = ND(L, node)->first_child;
    if (!cn || ND(L, cn)->type != AST_IDENT) return 0;

    char cname[128];
    get_text(L, cn, cname, sizeof(cname));
    if (!find_template(L, cname)) return 0;

    uint32_t av[BC_MAX_ARGS];
    int na;

    L->pkbad = 0;
    na = subav(L, ND(L, cn)->next_sibling, av, BC_MAX_ARGS);
    template_def_t *tmpl = ftmpl(L, cname, na);
    if (!tmpl) return 0;

    uint32_t func_n = tfunc(L, tmpl);
    if (!func_n) return 0;

    uint32_t body = find_func_body(L, func_n);
    if (!hasl(L, body, 0, MAX_FWD)) return 0;

    if (na < 0) {
        lower_error(L, node, L->pkbad ? BC_E420 : BC_E157, cname);
        return 1;
    }

    binding_t *nb = L->tnb;
    int ntp = 0;
    uint32_t dfn = 0;
    int have = tdeduc(L, tmpl->ast, cn, av, na, nb, &ntp, &dfn);

    int old_nb = L->nbindings, old_bs = L->nbbase;
    if (have && ntp > MAX_BIND - old_nb) {
        lower_error(L, node, BC_E157, cname);
        return 1;
    }
    for (int i = 0; have && i < ntp; i++)
        L->bindings[L->nbindings++] = nb[i];

    if (!fwpush(L, func_n, av, na)) {
        L->nbbase = old_bs;
        L->nbindings = old_nb;
        if (L->fwovf) lower_error(L, node, BC_E340, cname, L->fwovf);
        else lower_error(L, node, BC_E157, cname);
        return 1;
    }

    scan_launches(L, body);
    L->nfwf--;
    L->nbbase = old_bs;
    L->nbindings = old_nb;
    return 1;
}

static int lgen(const lower_t *L, uint32_t lam)
{
    for (uint32_t c = ND(L, lam)->first_child; c; c = ND(L, c)->next_sibling) {
        uint32_t ts;
        if (ND(L, c)->type != AST_PARAM) continue;
        ts = ND(L, c)->first_child;
        if (ts && ND(L, ts)->type == AST_TYPE_SPEC
            && ND(L, ts)->d.btype.kind == TYPE_AUTO) return 1;
    }
    return 0;
}

static uint32_t lvlam(const lower_t *L, uint32_t node)
{
    uint32_t nm_n, ln;

    if (ND(L, node)->type != AST_VAR_DECL) return 0;
    nm_n = child_at(L, node, 1);
    if (!nm_n || ND(L, nm_n)->type != AST_IDENT) return 0;
    ln = ND(L, nm_n)->next_sibling;
    if (ln && ND(L, ln)->type == AST_STRUCT_DEF) ln = ND(L, ln)->next_sibling;
    KA_GUARD(g, CE_NEST);
    while (ln && g-- && ND(L, ln)->type == AST_PAREN)
        ln = ND(L, ln)->first_child;
    return (ln && ND(L, ln)->type == AST_LAMBDA) ? ln : 0;
}

static uint32_t lbody(const lower_t *L, uint32_t lam)
{
    uint32_t body = 0;

    for (uint32_t c = ND(L, lam)->first_child; c; c = ND(L, c)->next_sibling)
        if (ND(L, c)->type == AST_BLOCK) body = c;
    return body;
}

static int sglc(lower_t *L, uint32_t node)
{
    uint32_t cn = ND(L, node)->first_child, lam = 0, a, body;
    int gi = -1, tsv = L->ntags, ok = 1;
    char nm[64];

    KA_GUARD(g, CE_NEST);
    while (cn && g-- && ND(L, cn)->type == AST_PAREN)
        cn = ND(L, cn)->first_child;
    if (!cn) return 0;

    if (ND(L, cn)->type == AST_LAMBDA) {
        lam = cn;
    } else if (ND(L, cn)->type == AST_IDENT) {
        get_text(L, cn, nm, sizeof nm);
        for (int i = L->nglm - 1; i >= 0; i--)
            if (strcmp(L->glms[i].name, nm) == 0) { gi = i; break; }
        if (gi < 0) return 0;
        if (L->glms[gi].busy) return 1;
        lam = L->glms[gi].ast;
    } else {
        return 0;
    }
    if (!lam || !lgen(L, lam)) return 0;

    a = ND(L, cn)->next_sibling;
    for (uint32_t p = ND(L, lam)->first_child; p && a;
         p = ND(L, p)->next_sibling) {
        int64_t v;
        char pn[64];
        if (ND(L, p)->type != AST_PARAM) continue;
        lpnam(L, p, pn, (int)sizeof pn);
        if (pn[0] && tgval(L, a, &v) && !tgadd(L, pn, v)) ok = 0;
        a = ND(L, a)->next_sibling;
    }
    body = lbody(L, lam);
    if (!ok)
        lower_error(L, node, BC_E030,
                    "more lambda tag bindings than Booth holds");
    else if (body) {
        if (gi >= 0) L->glms[gi].busy = 1;
        scan_launches(L, body);
        if (gi >= 0) L->glms[gi].busy = 0;
    }
    L->ntags = tsv;
    return 1;
}

static int sfld(lower_t *L, uint32_t node)
{
    const ast_node_t *n = ND(L, node);
    uint32_t a = n->first_child;
    uint32_t b = a ? ND(L, a)->next_sibling : 0;
    uint32_t pat, init = 0;
    char nms[MAX_FPACKS][64];
    int nn = 0, len;

    switch (n->d.oper.flags) {
    case FLD_UL: case FLD_UR: pat = a; break;
    case FLD_BL: pat = b; init = a; break;
    case FLD_BR: pat = a; init = b; break;
    default: return 0;
    }
    if (!pat) return 0;
    len = pk_len(L, pat, nms, &nn, 1);
    if (len < 0 || len > MAX_PKELM) return 0;

    if (init) scan_launches(L, init);
    for (int j = 0; j < len; j++) {
        int sva = L->npkact, svb = L->nbindings;
        if (cpkb(L, nms, nn, j)) scan_launches(L, pat);
        L->npkact    = sva;
        L->nbindings = svb;
    }
    return 1;
}

static void slcx(lower_t *L, uint32_t c)
{
    const ast_node_t *n = ND(L, c);
    uint32_t nm_n, in_n;

    if (n->type != AST_VAR_DECL || n->d.oper.op != 0) return;
    if (n->qualifiers & QUAL_TYPEDEF) return;
    if (!(n->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST))) return;
    if (n->cuda_flags & (CUDA_SHARED | CUDA_CONSTANT | CUDA_DEVICE)) return;

    nm_n = child_at(L, c, 1);
    if (!nm_n || ND(L, nm_n)->type != AST_IDENT) return;
    in_n = ND(L, nm_n)->next_sibling;
    if (in_n && ND(L, in_n)->type == AST_STRUCT_DEF)
        in_n = ND(L, in_n)->next_sibling;
    if (!in_n) return;

    ccoll(L, c, nm_n, in_n, NULL,
          (n->qualifiers & QUAL_CONSTEXPR) != 0);
}

static void scan_launches(lower_t *L, uint32_t node)
{
    if (!node) return;
    const ast_node_t *n = ND(L, node);

    if (n->type == AST_LAMBDA && lgen(L, node)) return;
    if (n->type == AST_FOLD && sfld(L, node)) return;
    if (n->type == AST_LAUNCH && slnch(L, node)) return;
    if (n->type == AST_CALL && sglc(L, node)) return;
    if (n->type == AST_CALL && swrap(L, node)) return;

    if (n->type == AST_BLOCK) {
        int gsv = L->nglm;
        push_scope(L);
        for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling) {
            uint32_t lm = lvlam(L, c);
            slcx(L, c);
            if (lm && lgen(L, lm) && L->nglm < MAX_GLAM) {
                get_text(L, child_at(L, c, 1), L->glms[L->nglm].name,
                         (int)sizeof(L->glms[0].name));
                L->glms[L->nglm].ast = lm;
                L->glms[L->nglm].busy = 0;
                L->nglm++;
                continue;
            }
            scan_launches(L, c);
        }
        pop_scope(L);
        L->nglm = gsv;
        return;
    }

    for (uint32_t c = n->first_child; c; c = ND(L, c)->next_sibling)
        scan_launches(L, c);
}

static void cscan(lower_t *L, uint32_t root)
{
    uint32_t wk[CE_NEST];
    char     qn[CE_NEST][64];
    int      wn = 0;

    wk[0] = root;
    qn[0][0] = 0;
    wn = 1;

    KA_GUARD(g, 4096);
    while (wn > 0 && g--) {
        char q[64];
        uint32_t nd = wk[--wn];

        ncpy(q, sizeof(q), qn[wn]);

        for (uint32_t c = ND(L, nd)->first_child; c;
             c = ND(L, c)->next_sibling) {
            const ast_node_t *n = ND(L, c);
            uint32_t nm_n, in_n;

            if (n->type == AST_NAMESPACE) {
                uint32_t id = n->first_child;
                if (wn >= CE_NEST) continue;
                wk[wn] = c;
                qn[wn][0] = 0;
                if (id && ND(L, id)->type == AST_IDENT)
                    get_text(L, id, qn[wn], sizeof(qn[0]));
                wn++;
                continue;
            }
            if (n->type != AST_VAR_DECL || n->d.oper.op != 0) continue;
            if (n->qualifiers & QUAL_TYPEDEF) continue;
            if (!(n->qualifiers & (QUAL_CONSTEXPR | QUAL_CONST))) continue;
            if (n->cuda_flags & (CUDA_SHARED | CUDA_CONSTANT | CUDA_DEVICE))
                continue;

            nm_n = child_at(L, c, 1);
            if (!nm_n || ND(L, nm_n)->type != AST_IDENT) continue;
            in_n = ND(L, nm_n)->next_sibling;
            if (in_n && ND(L, in_n)->type == AST_STRUCT_DEF)
                in_n = ND(L, in_n)->next_sibling;
            if (!in_n) continue;

            if (!ccoll(L, c, nm_n, in_n, q,
                       (n->qualifiers & QUAL_CONSTEXPR) != 0))
                lower_error(L, c, BC_E130, MAX_CEXPS);
        }
    }
}

static int cfcol(lower_t *L, uint32_t root)
{
    uint32_t wk[CE_NEST];
    int wn = 1;

    wk[0] = root;
    KA_GUARD(g, 4096);
    while (wn > 0 && g--) {
        uint32_t nd = wk[--wn];

        for (uint32_t c = ND(L, nd)->first_child; c;
             c = ND(L, c)->next_sibling) {
            const ast_node_t *n = ND(L, c);
            uint32_t id;

            if (n->type == AST_NAMESPACE) {
                if (wn < CE_NEST) wk[wn++] = c;
                continue;
            }
            if (n->type != AST_FUNC_DEF) continue;
            if (!(n->qualifiers & QUAL_CONSTEXPR)) continue;
            id = child_at(L, c, 1);
            if (!id || ND(L, id)->type != AST_IDENT) continue;
            if (L->ncfn >= MAX_CFNS) return 0;
            get_text(L, id, L->cfns[L->ncfn].name, sizeof(L->cfns[0].name));
            L->cfns[L->ncfn].ast = c;
            L->ncfn++;
        }
    }
    return 1;
}

static int tlcol(lower_t *L, uint32_t root)
{
    uint32_t st[CE_NEST];
    uint32_t c = ND(L, root)->first_child;
    int sp = 0;

    L->ntlv = 0;
    KA_GUARD(g, 1 << 17);
    while (g--) {
        if (!c) {
            if (sp == 0) return 1;
            c = ND(L, st[--sp])->next_sibling;
            continue;
        }
        if (ND(L, c)->type == AST_NAMESPACE) {
            if (sp >= CE_NEST) return 0;
            st[sp++] = c;
            c = ND(L, c)->first_child;
            continue;
        }
        if (L->ntlv >= MAX_TLV) return 0;
        L->tlv[L->ntlv++] = c;
        c = ND(L, c)->next_sibling;
    }
    return 0;
}

/* ---- Top-Level Entry Point ---- */

int bir_lower(const parser_t *P, uint32_t ast_root, bir_module_t *M,
              const sema_ctx_t *sema,
              bc_error_t *out_errs, int *out_nerrs)
{
    bir_module_init(M);
    return bir_ltu(P, ast_root, M, sema, BIR_TU_EXT, out_errs, out_nerrs);
}

int bir_ltu(const parser_t *P, uint32_t ast_root, bir_module_t *M,
            const sema_ctx_t *sema, uint16_t tu,
            bc_error_t *out_errs, int *out_nerrs)
{
    static lower_t L_storage; /* large struct — static to avoid stack overflow */
    lower_t *L = &L_storage;
    memset(L, 0, sizeof(*L));
    L->P    = P;
    L->M    = M;
    L->src  = P->src;
    L->sema = sema;
    L->tu   = tu;
    L->tsc  = 1;
    L->srdst = BIR_VAL_NONE;

    if (!cfcol(L, ast_root))
        lower_error(L, 0, BC_E030, "too many constexpr functions");

    ucoll(L, ast_root);

    if (!tlcol(L, ast_root))
        lower_error(L, 0, BC_E030, "more declarations than Booth collects");

    int csv = L->ncexp, esv = L->nerrors;

    cscan(L, ast_root);
    L->nerrors = esv;

    for (int ti = 0; ti < L->ntlv; ti++) {
        uint32_t c = L->tlv[ti];
        const ast_node_t *n = &P->nodes[c];
        switch (n->type) {
        case AST_STRUCT_DEF:
            scoll(L, c);
            break;
        case AST_ENUM_DEF:
            collect_enum(L, c);
            break;
        case AST_VAR_DECL:
            if (n->qualifiers & QUAL_TYPEDEF)
                collect_typedef(L, c);
            break;
        case AST_TEMPLATE_DECL:
            collect_template(L, c);
            break;
        default:
            break;
        }
    }

    ucoll(L, ast_root);
    L->ncexp = csv;
    cscan(L, ast_root);

    for (int ti = 0; ti < L->ntlv; ti++) {
        uint32_t c = L->tlv[ti];
        const ast_node_t *n = &P->nodes[c];
        if (n->type == AST_VAR_DECL && !(n->qualifiers & QUAL_TYPEDEF)
            && (n->cuda_flags & (CUDA_SHARED|CUDA_CONSTANT|CUDA_DEVICE)))
            collect_global_var(L, c);
        if (n->type == AST_STRUCT_DEF) cstat(L, c, NULL);
    }

    /* Pass 2: lower non-template device functions */
    for (int ti = 0; ti < L->ntlv; ti++) {
        uint32_t c = L->tlv[ti];
        const ast_node_t *n = &P->nodes[c];
        if (n->type == AST_FUNC_DEF) {
            uint16_t cuda = n->cuda_flags;
            if (cuda & (CUDA_GLOBAL | CUDA_DEVICE)) {
                tscan(L, c);
                lower_func_body(L, c, cuda, NULL);
            }
            /* Host functions are not lowered, __host__ among them;
               pass 3 below scans them for launches */
        }
    }

    for (uint32_t tc = P->nodes[ast_root].first_child; tc;
         tc = P->nodes[tc].next_sibling)
        if (P->nodes[tc].type == AST_ASM)
            lower_error(L, tc, BC_E221, "asm outside a function");

    /* Pass 3: scan host functions for kernel launches → instantiate templates */
    for (int ti = 0; ti < L->ntlv; ti++) {
        uint32_t c = L->tlv[ti];
        const ast_node_t *n = &P->nodes[c];
        if (n->type == AST_FUNC_DEF && !(n->cuda_flags & (CUDA_GLOBAL|CUDA_DEVICE))) {
            /* Host function — scan for launches */
            uint32_t body = find_func_body(L, c);
            if (body) scan_launches(L, body);
        }
    }

    if (M->pool_full) {
        if (out_nerrs) *out_nerrs = 0;
        (void)bir_pchk(M, "lowering");
        return BC_ERR_LOWER;
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
