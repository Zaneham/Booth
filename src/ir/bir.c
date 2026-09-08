#include "bir.h"
#include "backend.h"
#include <string.h>
#include <stdio.h>

/* ---- Name Tables ---- */

/* Every opcode gets a name. Even the ones that probably shouldn't exist. */
static const char *op_names[BIR_OP_COUNT] = {
    [BIR_ADD]           = "add",
    [BIR_SUB]           = "sub",
    [BIR_MUL]           = "mul",
    [BIR_UMULHI]        = "umulhi",
    [BIR_POPCOUNT]      = "popcount",
    [BIR_CTZ]           = "ctz",
    [BIR_CLZ]           = "clz",
    [BIR_BREV]          = "brev",
    [BIR_SDIV]          = "sdiv",
    [BIR_UDIV]          = "udiv",
    [BIR_SREM]          = "srem",
    [BIR_UREM]          = "urem",
    [BIR_FADD]          = "fadd",
    [BIR_FSUB]          = "fsub",
    [BIR_FMUL]          = "fmul",
    [BIR_FDIV]          = "fdiv",
    [BIR_FREM]          = "frem",

    [BIR_AND]           = "and",
    [BIR_OR]            = "or",
    [BIR_XOR]           = "xor",
    [BIR_SHL]           = "shl",
    [BIR_LSHR]          = "lshr",
    [BIR_ASHR]          = "ashr",

    [BIR_ICMP]          = "icmp",
    [BIR_FCMP]          = "fcmp",

    [BIR_TRUNC]         = "trunc",
    [BIR_ZEXT]          = "zext",
    [BIR_SEXT]          = "sext",
    [BIR_FPTRUNC]       = "fptrunc",
    [BIR_FPEXT]         = "fpext",
    [BIR_FPTOSI]        = "fptosi",
    [BIR_FPTOUI]        = "fptoui",
    [BIR_SITOFP]        = "sitofp",
    [BIR_UITOFP]        = "uitofp",
    [BIR_PTRTOINT]      = "ptrtoint",
    [BIR_INTTOPTR]      = "inttoptr",
    [BIR_BITCAST]       = "bitcast",

    [BIR_ALLOCA]        = "alloca",
    [BIR_SHARED_ALLOC]  = "shared_alloc",
    [BIR_GLOBAL_REF]    = "global_ref",
    [BIR_LOAD]          = "load",
    [BIR_STORE]         = "store",
    [BIR_GEP]           = "gep",

    [BIR_BR]            = "br",
    [BIR_BR_COND]       = "br_cond",
    [BIR_SWITCH]        = "switch",
    [BIR_RET]           = "ret",
    [BIR_UNREACHABLE]   = "unreachable",

    [BIR_PHI]           = "phi",
    [BIR_PARAM]         = "param",

    [BIR_THREAD_ID]     = "thread_id",
    [BIR_BLOCK_ID]      = "block_id",
    [BIR_BLOCK_DIM]     = "block_dim",
    [BIR_GRID_DIM]      = "grid_dim",

    [BIR_BARRIER]       = "barrier",
    [BIR_BARRIER_GROUP] = "barrier_group",

    [BIR_ATOMIC_ADD]    = "atomic_add",
    [BIR_ATOMIC_SUB]    = "atomic_sub",
    [BIR_ATOMIC_AND]    = "atomic_and",
    [BIR_ATOMIC_OR]     = "atomic_or",
    [BIR_ATOMIC_XOR]    = "atomic_xor",
    [BIR_ATOMIC_MIN]    = "atomic_min",
    [BIR_ATOMIC_MAX]    = "atomic_max",
    [BIR_ATOMIC_XCHG]   = "atomic_xchg",
    [BIR_ATOMIC_CAS]    = "atomic_cas",
    [BIR_ATOMIC_LOAD]   = "atomic_load",
    [BIR_ATOMIC_STORE]  = "atomic_store",

    [BIR_SHFL]          = "shfl",
    [BIR_SHFL_UP]       = "shfl_up",
    [BIR_SHFL_DOWN]     = "shfl_down",
    [BIR_SHFL_XOR]      = "shfl_xor",
    [BIR_BALLOT]        = "ballot",
    [BIR_VOTE_ANY]      = "vote_any",
    [BIR_VOTE_ALL]      = "vote_all",

    [BIR_SQRT]          = "sqrt",
    [BIR_RSQ]           = "rsq",
    [BIR_RCP]           = "rcp",
    [BIR_EXP2]          = "exp2",
    [BIR_LOG2]          = "log2",
    [BIR_SIN]           = "sin",
    [BIR_COS]           = "cos",
    [BIR_FABS]          = "fabs",
    [BIR_FLOOR]         = "floor",
    [BIR_CEIL]          = "ceil",
    [BIR_FTRUNC]        = "ftrunc",
    [BIR_RNDNE]         = "rndne",
    [BIR_FMAX]          = "fmax",
    [BIR_FMIN]          = "fmin",

    [BIR_MFMA]          = "mfma",
    [BIR_MMA]           = "mma",
    [BIR_MFRG]          = "mfrg",
    [BIR_WLD]           = "wmma_load",
    [BIR_WST]           = "wmma_store",
    [BIR_WMMA]          = "wmma_mma",

    [BIR_CALL]          = "call",
    [BIR_SELECT]        = "select",
    [BIR_INLINE_ASM]    = "inline_asm",

    [BIR_TRAP]          = "trap",
    [BIR_FNREF]         = "fnref",
    [BIR_PRINTF]        = "printf",

    [BIR_FENCE]         = "fence",
    [BIR_GRIDBAR]       = "gridbar",
    [BIR_NANOSLP]       = "nanosleep",
    [BIR_BARRED]        = "barrier_red",
};

static const char *cmp_names[BIR_CMP_COUNT] = {
    [BIR_ICMP_EQ]  = "eq",  [BIR_ICMP_NE]  = "ne",
    [BIR_ICMP_SLT] = "slt", [BIR_ICMP_SLE] = "sle",
    [BIR_ICMP_SGT] = "sgt", [BIR_ICMP_SGE] = "sge",
    [BIR_ICMP_ULT] = "ult", [BIR_ICMP_ULE] = "ule",
    [BIR_ICMP_UGT] = "ugt", [BIR_ICMP_UGE] = "uge",

    [BIR_FCMP_OEQ] = "oeq", [BIR_FCMP_ONE] = "one",
    [BIR_FCMP_OLT] = "olt", [BIR_FCMP_OLE] = "ole",
    [BIR_FCMP_OGT] = "ogt", [BIR_FCMP_OGE] = "oge",
    [BIR_FCMP_UEQ] = "ueq", [BIR_FCMP_UNE] = "une",
    [BIR_FCMP_ULT] = "ult", [BIR_FCMP_ULE] = "ule",
    [BIR_FCMP_UGT] = "ugt", [BIR_FCMP_UGE] = "uge",
    [BIR_FCMP_ORD] = "ord", [BIR_FCMP_UNO] = "uno",
};

static const char *addrspace_names[BIR_AS_COUNT] = {
    [BIR_AS_PRIVATE]  = "private",
    [BIR_AS_SHARED]   = "shared",
    [BIR_AS_GLOBAL]   = "global",
    [BIR_AS_CONSTANT] = "constant",
    [BIR_AS_GENERIC]  = "generic",
};

static const char *type_kind_names[BIR_TYPE_KIND_COUNT] = {
    [BIR_TYPE_VOID]   = "void",
    [BIR_TYPE_INT]    = "int",
    [BIR_TYPE_FLOAT]  = "float",
    [BIR_TYPE_BFLOAT] = "bfloat",
    [BIR_TYPE_PTR]    = "ptr",
    [BIR_TYPE_VECTOR] = "vector",
    [BIR_TYPE_STRUCT] = "struct",
    [BIR_TYPE_ARRAY]  = "array",
    [BIR_TYPE_FUNC]   = "func",
};

static const char *order_names[BIR_ORDER_COUNT] = {
    [BIR_ORDER_RELAXED] = "relaxed",
    [BIR_ORDER_ACQUIRE] = "acquire",
    [BIR_ORDER_RELEASE] = "release",
    [BIR_ORDER_ACQ_REL] = "acq_rel",
    [BIR_ORDER_SEQ_CST] = "seq_cst",
};

const bir_wmma_t bir_wmma[BIR_WM_NROW] = {
    { "m16n16k16", "f16", "f16", 8, 8, 4 },         /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m16n16k16", "f16", "f32", 8, 8, 8 },         /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n32k16", "f16", "f16", 8, 8, 4 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n32k16", "f16", "f32", 8, 8, 8 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m32n8k16", "f16", "f16", 8, 8, 4 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m32n8k16", "f16", "f32", 8, 8, 8 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m16n16k16", "bf16", "f32", 4, 4, 8 },        /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n32k16", "bf16", "f32", 2, 8, 8 },         /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m32n8k16", "bf16", "f32", 8, 2, 8 },         /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m16n16k8", "tf32", "f32", 4, 4, 8 },         /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m16n16k16", "s8", "s32", 2, 2, 8 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n32k16", "s8", "s32", 1, 4, 8 },           /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m32n8k16", "s8", "s32", 4, 1, 8 },           /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m16n16k16", "u8", "s32", 2, 2, 8 },          /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n32k16", "u8", "s32", 1, 4, 8 },           /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m32n8k16", "u8", "s32", 4, 1, 8 },           /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n8k32", "s4", "s32", 1, 1, 2 },            /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n8k32", "u4", "s32", 1, 1, 2 },            /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n8k128", "b1", "s32", 1, 1, 2 },           /* per PTX ISA 9.2 9.7.14.4.1 */
    { "m8n8k4", "f64", "f64", 1, 1, 2 },            /* per PTX ISA 9.2 9.7.14.4.1 */
};

int bir_wmrow(const char *shp, const char *abt, const char *act)
{
    for (int i = 0; i < BIR_WM_NROW; i++) {
        const bir_wmma_t *W = &bir_wmma[i];
        if (shp && strcmp(W->shp, shp) != 0) continue;
        if (abt && strcmp(W->abt, abt) != 0) continue;
        if (act && strcmp(W->act, act) != 0) continue;
        return i;
    }
    return -1;
}

uint8_t bir_wmn(uint32_t row, unsigned role)
{
    const bir_wmma_t *W = &bir_wmma[row % BIR_WM_NROW];
    if (role == BIR_WM_A) return W->na;
    if (role == BIR_WM_B) return W->nb;
    return W->nc;
}

const char *bir_op_name(int op)
{
    if (op >= 0 && op < BIR_OP_COUNT && op_names[op])
        return op_names[op];
    return "???";
}

const char *bir_type_kind_name(int kind)
{
    if (kind >= 0 && kind < BIR_TYPE_KIND_COUNT)
        return type_kind_names[kind];
    return "???";
}

const char *bir_cmp_name(int pred)
{
    if (pred >= 0 && pred < BIR_CMP_COUNT)
        return cmp_names[pred];
    return "???";
}

const char *bir_addrspace_name(int as)
{
    if (as >= 0 && as < BIR_AS_COUNT)
        return addrspace_names[as];
    return "???";
}

const char *bir_order_name(int ord)
{
    if (ord >= 0 && ord < BIR_ORDER_COUNT)
        return order_names[ord];
    return "???";
}

/* ---- Pool overflow ---- */

/* Fixed order, so the report reads the same whichever pool filled first. */
static const struct { uint32_t bit; const char *name; uint32_t cap; }
pool_tab[] = {
    { BIR_P_TYPES,    "type",           BIR_MAX_TYPES       },
    { BIR_P_TFIELDS,  "type field",     BIR_MAX_TYPE_FIELDS },
    { BIR_P_STRINGS,  "string table",   BIR_MAX_STRINGS     },
    { BIR_P_CONSTS,   "constant",       BIR_MAX_CONSTS      },
    { BIR_P_INSTS,    "instruction",    BIR_MAX_INSTS       },
    { BIR_P_BLOCKS,   "block",          BIR_MAX_BLOCKS      },
    { BIR_P_FUNCS,    "function",       BIR_MAX_FUNCS       },
    { BIR_P_GLOBALS,  "global",         BIR_MAX_GLOBALS     },
    { BIR_P_EXTRAOPS, "extra operand",  BIR_MAX_EXTRA_OPS   },
    { BIR_P_PHIS,     "mem2reg phi",    0u                  },
};

void bir_pfull(bir_module_t *M, uint32_t bit)
{
    if (M != NULL) M->pool_full |= bit;
}

int bir_pchk(const bir_module_t *M, const char *phase)
{
    if (M == NULL || M->pool_full == 0u) return BC_OK;

    for (uint32_t i = 0; i < sizeof(pool_tab) / sizeof(pool_tab[0]); i++) {
        if (!(M->pool_full & pool_tab[i].bit)) continue;
        if (pool_tab[i].cap != 0u)
            fprintf(stderr, "E120: BIR %s pool exhausted during %s "
                    "(capacity %u). Raise the matching BIR_MAX_* and "
                    "rebuild.\n", pool_tab[i].name, phase, pool_tab[i].cap);
        else
            fprintf(stderr, "E120: BIR %s pool exhausted during %s.\n",
                    pool_tab[i].name, phase);
    }
    return BC_ERR_OVERFLOW;
}

/* ---- Module Init ---- */

void bir_module_init(bir_module_t *M)
{
    memset(M, 0, sizeof(*M));
    /* Reserve type 0 as void.  ptr_inner() returns types[t].inner,
       and callers treat 0 as "no element type".  If some other type
       (e.g. f32) happened to land at index 0, the sentinel check
       would misfire.  Pinning void at 0 prevents the collision. */
    bir_type_void(M);
}

/* ---- Type Interning ---- */

/* Simple types: compared field-by-field. No indirection needed. */
static int type_eq_simple(const bir_type_t *a, const bir_type_t *b)
{
    return a->kind == b->kind
        && a->addrspace == b->addrspace
        && a->width == b->width
        && a->inner == b->inner
        && a->count == b->count
        && a->num_fields == b->num_fields;
}

static uint32_t intern_type(bir_module_t *M, const bir_type_t *t)
{
    uint32_t guard = M->num_types;
    for (uint32_t i = 0; i < M->num_types && guard > 0; i++, guard--) {
        if (type_eq_simple(&M->types[i], t))
            return i;
    }
    if (M->num_types >= BIR_MAX_TYPES) {
        bir_pfull(M, BIR_P_TYPES);
        return 0;
    }
    uint32_t idx = M->num_types++;
    M->types[idx] = *t;
    return idx;
}

/* Compound types (struct, func): must compare actual field type indices. */
static uint32_t intern_compound(bir_module_t *M, uint8_t kind,
                                uint32_t inner, const uint32_t *fields,
                                int nfields)
{
    uint32_t guard = M->num_types;
    for (uint32_t i = 0; i < M->num_types && guard > 0; i++, guard--) {
        bir_type_t *t = &M->types[i];
        if (t->kind != kind || t->inner != inner || t->uni != 0
            || t->num_fields != (uint16_t)nfields)
            continue;
        int match = 1;
        for (int j = 0; j < nfields && match; j++) {
            if (M->type_fields[t->count + (uint32_t)j] != fields[j])
                match = 0;
        }
        if (match) return i;
    }
    if (M->num_type_fields + (uint32_t)nfields > BIR_MAX_TYPE_FIELDS) {
        bir_pfull(M, BIR_P_TFIELDS);
        return 0;
    }
    if (M->num_types >= BIR_MAX_TYPES) {
        bir_pfull(M, BIR_P_TYPES);
        return 0;
    }

    uint32_t start = M->num_type_fields;
    for (int i = 0; i < nfields; i++)
        M->type_fields[M->num_type_fields++] = fields[i];

    uint32_t idx = M->num_types++;
    bir_type_t *nt = &M->types[idx];
    memset(nt, 0, sizeof(*nt));
    nt->kind = kind;
    nt->inner = inner;
    nt->count = start;
    nt->num_fields = (uint16_t)nfields;
    return idx;
}

uint32_t bir_type_void(bir_module_t *M)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_VOID;
    return intern_type(M, &t);
}

uint32_t bir_type_int(bir_module_t *M, int width_bits)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_INT;
    t.width = (uint16_t)width_bits;
    return intern_type(M, &t);
}

uint32_t bir_type_float(bir_module_t *M, int width_bits)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_FLOAT;
    t.width = (uint16_t)width_bits;
    return intern_type(M, &t);
}

uint32_t bir_type_bfloat(bir_module_t *M)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_BFLOAT;
    t.width = 16;
    return intern_type(M, &t);
}

uint32_t bir_type_ptr(bir_module_t *M, uint32_t pointee, int addrspace)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_PTR;
    t.addrspace = (uint8_t)addrspace;
    t.inner = pointee;
    return intern_type(M, &t);
}

uint32_t bir_type_array(bir_module_t *M, uint32_t elem, uint32_t count)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_ARRAY;
    t.inner = elem;
    t.count = count;
    return intern_type(M, &t);
}

uint32_t bir_type_vector(bir_module_t *M, uint32_t elem, uint32_t count)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_VECTOR;
    t.inner = elem;
    t.width = (uint16_t)count;
    return intern_type(M, &t);
}

uint32_t bir_type_struct(bir_module_t *M, const uint32_t *fields, int nfields)
{
    return intern_compound(M, BIR_TYPE_STRUCT, 0, fields, nfields);
}

uint32_t bir_sfwd(bir_module_t *M)
{
    uint32_t idx;

    if (M->num_types >= BIR_MAX_TYPES) {
        bir_pfull(M, BIR_P_TYPES);
        return 0;
    }
    idx = M->num_types++;
    memset(&M->types[idx], 0, sizeof(M->types[idx]));
    M->types[idx].kind = BIR_TYPE_STRUCT;
    return idx;
}

int bir_sfin(bir_module_t *M, uint32_t ty, const uint32_t *fields, int nfields)
{
    int i;

    if (ty >= M->num_types || M->types[ty].kind != BIR_TYPE_STRUCT) return 0;
    if (nfields < 0 || nfields > 0xFFFF) return 0;
    if (M->num_type_fields + (uint32_t)nfields > BIR_MAX_TYPE_FIELDS) {
        bir_pfull(M, BIR_P_TFIELDS);
        return 0;
    }

    M->types[ty].count = M->num_type_fields;
    for (i = 0; i < nfields; i++)
        M->type_fields[M->num_type_fields++] = fields[i];
    M->types[ty].num_fields = (uint16_t)nfields;
    return 1;
}

uint32_t bir_type_func(bir_module_t *M, uint32_t ret,
                       const uint32_t *params, int nparams)
{
    return intern_compound(M, BIR_TYPE_FUNC, ret, params, nparams);
}

int bir_umrk(bir_module_t *M, uint32_t ty)
{
    if (ty >= M->num_types || M->types[ty].kind != BIR_TYPE_STRUCT) return 0;
    M->types[ty].uni = 1;
    return 1;
}

/* ---- Type Sizes ---- */

static uint32_t bsz_al(uint32_t sz, uint32_t psz)
{
    uint32_t a = 1;
    if (sz > psz) sz = psz;
    while (a * 2u <= sz) a *= 2u;
    return a;
}

static uint32_t bsz_up(uint32_t x, uint32_t a)
{
    return (x + a - 1u) & ~(a - 1u);
}

uint32_t bir_bsz(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    struct { uint32_t ty, mul, fld, sum, alg; } fr[BIR_BSZ_DEEP];
    uint32_t sp = 1, guard = 4u * BIR_MAX_TYPE_FIELDS;

    fr[0].ty = ty; fr[0].mul = 1; fr[0].fld = 0; fr[0].sum = 0; fr[0].alg = 1;

    while (guard--) {
        uint32_t i = sp - 1, w, va, val;
        const bir_type_t *T;

        if (fr[i].ty >= M->num_types) return 0;
        T = &M->types[fr[i].ty];

        if (T->kind == BIR_TYPE_ARRAY || T->kind == BIR_TYPE_VECTOR) {
            uint32_t n = (T->kind == BIR_TYPE_ARRAY) ? T->count
                                                     : (uint32_t)T->width;
            if (n && fr[i].mul > 0xFFFFFFFFu / n) return 0;
            fr[i].mul *= n; fr[i].ty = T->inner; continue;
        }

        if (T->kind == BIR_TYPE_STRUCT) {
            if (fr[i].fld < (uint32_t)T->num_fields) {
                uint32_t f = T->count + fr[i].fld++;
                if (f >= M->num_type_fields || sp >= BIR_BSZ_DEEP) return 0;
                fr[sp].ty = M->type_fields[f];
                fr[sp].mul = 1; fr[sp].fld = 0; fr[sp].sum = 0; fr[sp].alg = 1;
                sp++;
                continue;
            }
            w  = bsz_up(fr[i].sum, fr[i].alg);
            va = fr[i].alg;
        } else {
            switch (T->kind) {
            case BIR_TYPE_INT:
            case BIR_TYPE_FLOAT:
            case BIR_TYPE_BFLOAT: w = ((uint32_t)T->width + 7u) / 8u; break;
            case BIR_TYPE_PTR:    w = psz; break;
            default:              return 0;
            }
            va = bsz_al(w, psz);
        }
        if (!w) return 0;

        if (fr[i].mul > 0xFFFFFFFFu / w) return 0;
        val = fr[i].mul * w;
        if (--sp == 0) return val;

        if (va > fr[sp - 1].alg) fr[sp - 1].alg = va;
        if (M->types[fr[sp - 1].ty].uni) {
            if (val > fr[sp - 1].sum) fr[sp - 1].sum = val;
            continue;
        }
        if (bsz_up(fr[sp - 1].sum, va) > 0xFFFFFFFFu - val) return 0;
        fr[sp - 1].sum = bsz_up(fr[sp - 1].sum, va) + val;
    }
    return 0;
}

uint32_t bir_balg(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    uint32_t st[BIR_BSZ_DEEP], sp = 0, guard = 4u * BIR_MAX_TYPE_FIELDS;
    uint32_t best = 1;

    st[sp++] = ty;
    while (sp && guard--) {
        uint32_t t = st[--sp], w;
        const bir_type_t *T;

        if (t >= M->num_types) return 0;
        T = &M->types[t];

        if (T->kind == BIR_TYPE_ARRAY || T->kind == BIR_TYPE_VECTOR) {
            if (sp >= BIR_BSZ_DEEP) return 0;
            st[sp++] = T->inner;
            continue;
        }
        if (T->kind == BIR_TYPE_STRUCT) {
            for (uint16_t f = 0; f < T->num_fields; f++) {
                if (T->count + f >= M->num_type_fields) return 0;
                if (sp >= BIR_BSZ_DEEP) return 0;
                st[sp++] = M->type_fields[T->count + f];
            }
            continue;
        }
        switch (T->kind) {
        case BIR_TYPE_INT:
        case BIR_TYPE_FLOAT:
        case BIR_TYPE_BFLOAT: w = ((uint32_t)T->width + 7u) / 8u; break;
        case BIR_TYPE_PTR:    w = psz; break;
        default:              return 0;
        }
        w = bsz_al(w, psz);
        if (w > best) best = w;
    }
    return sp ? 0 : best;
}

uint32_t bir_gsz(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    if (ty < M->num_types && M->types[ty].kind == BIR_TYPE_PTR)
        return bir_bsz(M, M->types[ty].inner, psz);
    return bir_bsz(M, ty, psz);
}

uint32_t bir_gstr(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    if (ty < M->num_types && M->types[ty].kind == BIR_TYPE_PTR) {
        uint32_t in = M->types[ty].inner;
        if (in < M->num_types && M->types[in].kind == BIR_TYPE_ARRAY &&
            M->types[in].count == 0)
            return bir_bsz(M, M->types[in].inner, psz);
    }
    return bir_gsz(M, ty, psz);
}

static uint32_t bvty(const bir_module_t *M, uint32_t v)
{
    uint32_t i = BIR_VAL_INDEX(v);

    if (v == BIR_VAL_NONE) return M->num_types;
    if (BIR_VAL_IS_CONST(v))
        return i < M->num_consts ? M->consts[i].type : M->num_types;
    return i < M->num_insts ? M->insts[i].type : M->num_types;
}

int bir_fgep(const bir_module_t *M, const bir_inst_t *I,
             uint32_t psz, uint32_t *off)
{
    uint32_t bt, st, iv, fs, k, o = 0;
    int64_t fi;

    if (!M || !I || !off) return 0;
    if (I->op != BIR_GEP || I->num_operands < 2) return 0;

    bt = bvty(M, I->operands[0]);
    if (bt >= M->num_types || M->types[bt].kind != BIR_TYPE_PTR) return 0;
    st = M->types[bt].inner;
    if (st >= M->num_types || M->types[st].kind != BIR_TYPE_STRUCT) return 0;

    iv = I->operands[1];
    if (iv == BIR_VAL_NONE || !BIR_VAL_IS_CONST(iv)) return 0;
    if (BIR_VAL_INDEX(iv) >= M->num_consts) return 0;
    if (M->consts[BIR_VAL_INDEX(iv)].kind != BIR_CONST_INT) return 0;
    fi = M->consts[BIR_VAL_INDEX(iv)].d.ival;
    if (fi < 0 || (uint32_t)fi >= M->types[st].num_fields) return 0;

    fs = M->types[st].count;
    if (fs > M->num_type_fields - M->types[st].num_fields) return 0;

    if (I->type >= M->num_types || M->types[I->type].kind != BIR_TYPE_PTR
        || M->types[I->type].inner != M->type_fields[fs + (uint32_t)fi])
        return 0;

    if (M->types[st].uni) { *off = 0; return 1; }

    for (k = 0; k <= (uint32_t)fi; k++) {
        uint32_t ft = M->type_fields[fs + k];
        uint32_t a = bir_balg(M, ft, psz), s;

        if (!a) return 0;
        o = (o + a - 1u) & ~(a - 1u);
        if (k == (uint32_t)fi) break;
        s = bir_bsz(M, ft, psz);
        if (!s) return 0;
        if (o > 0xFFFFFFFFu - s) return 0;
        o += s;
    }

    *off = o;
    return 1;
}

/* ---- String Table ---- */

uint32_t bir_add_string(bir_module_t *M, const char *s, uint32_t len)
{
    /* Offset 0 is a live string, not a sentinel. */
    if (M->string_len + len + 1 > BIR_MAX_STRINGS) {
        bir_pfull(M, BIR_P_STRINGS);
        return 0;
    }
    uint32_t offset = M->string_len;
    memcpy(&M->strings[offset], s, len);
    M->strings[offset + len] = '\0';
    M->string_len += len + 1;
    return offset;
}

uint32_t bir_oper(const bir_module_t *M, const bir_inst_t *I, uint32_t j)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t s = I->operands[0], c = I->operands[1];
        if (j >= c || s + j >= M->num_extra_ops) return BIR_VAL_NONE;
        return M->extra_operands[s + j];
    }
    if (j >= I->num_operands || j >= BIR_OPERANDS_INLINE) return BIR_VAL_NONE;
    return I->operands[j];
}

uint32_t bir_asmd(bir_module_t *M, uint32_t tmpl, uint32_t cons,
                  uint16_t nout, uint16_t nops, uint8_t vol)
{
    uint32_t i;

    if (tmpl >= M->string_len || cons >= M->string_len) return BIR_SYM_NONE;
    for (i = 0; i < M->num_asms; i++) {
        const bir_asm_t *A = &M->asms[i];
        if (A->nout == nout && A->nops == nops && A->vol == vol
            && strcmp(&M->strings[A->tmpl], &M->strings[tmpl]) == 0
            && strcmp(&M->strings[A->cons], &M->strings[cons]) == 0)
            return i;
    }
    if (M->num_asms >= BIR_MAX_ASMS) {
        bir_pfull(M, BIR_P_ASMS);
        return BIR_SYM_NONE;
    }
    i = M->num_asms++;
    M->asms[i].tmpl = tmpl;
    M->asms[i].cons = cons;
    M->asms[i].nout = nout;
    M->asms[i].nops = nops;
    M->asms[i].vol  = vol;
    M->asms[i].pad[0] = M->asms[i].pad[1] = M->asms[i].pad[2] = 0;
    return i;
}

/* ---- Constants ---- */

/* Nothing is pinned at const 0 the way void is at type 0, so a refusal
   here is indistinguishable from a real index. Hence the bit. */

uint32_t bir_const_int(bir_module_t *M, uint32_t type, int64_t val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_INT
            && M->consts[i].type == type
            && M->consts[i].d.ival == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_INT;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.ival = val;
    return idx;
}

/* A constant whose value is a sequence of bytes interned in the
 * module's strings table. Used for string literals; could be used
 * for other byte-array constants later. The offset and length name
 * the slice of M->strings; we do not deduplicate the bytes because
 * deduplicating literals can be surprising (two source-distinct
 * literals collapsing into one pointer comparison) and we are not
 * doing the work to confirm that surprise is welcome. */

uint32_t bir_const_bytes(bir_module_t *M, uint32_t type,
                         uint32_t off, uint32_t len)
{
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_BYTES;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.bytes.off = off;
    M->consts[idx].d.bytes.len = len;
    return idx;
}

int bir_global_is_bytes(const bir_module_t *M, uint32_t gi)
{
    if (gi >= M->num_globals) return 0;
    uint32_t init = M->globals[gi].initializer;
    if (init == BIR_VAL_NONE) return 0;
    if (!BIR_VAL_IS_CONST(init)) return 0;
    uint32_t ci = BIR_VAL_INDEX(init);
    if (ci >= M->num_consts) return 0;
    return M->consts[ci].kind == BIR_CONST_BYTES;
}

int bir_mang(const char *name, uint16_t tu, char *out, int size)
{
    int n = snprintf(out, (size_t)size, "%s__%u", name, (unsigned)tu);
    if (n < 0 || n >= size) { out[0] = '\0'; return 1; }
    return 0;
}

uint32_t bir_fsym(const bir_module_t *M, const char *name, uint16_t tu,
                  int nargs)
{
    char mng[BIR_SYM_MAX];

    for (int pass = 0; pass < 2; pass++) {
        const char *want = name;
        uint16_t wtu = BIR_TU_EXT;

        if (pass == 0) {
            if (tu == BIR_TU_EXT) continue;
            if (bir_mang(name, tu, mng, (int)sizeof mng) != 0) continue;
            want = mng;
            wtu = tu;
        }
        for (uint32_t i = 0; i < M->num_funcs; i++) {
            const bir_func_t *F = &M->funcs[i];
            if (F->tu != wtu || F->name >= M->string_len) continue;
            if (strcmp(&M->strings[F->name], want) != 0) continue;
            if (nargs < 0
                || F->num_params - F->sret == (uint16_t)nargs) return i;
        }
    }
    return BIR_SYM_NONE;
}

uint32_t bir_gsym(const bir_module_t *M, const char *name, uint16_t tu)
{
    char mng[BIR_SYM_MAX];

    if (tu != BIR_TU_EXT && bir_mang(name, tu, mng, (int)sizeof mng) == 0) {
        for (uint32_t i = 0; i < M->num_globals; i++) {
            const bir_global_t *G = &M->globals[i];
            if (G->tu != tu || G->name >= M->string_len) continue;
            if (strcmp(&M->strings[G->name], mng) == 0) return i;
        }
    }
    for (uint32_t i = 0; i < M->num_globals; i++) {
        const bir_global_t *G = &M->globals[i];
        if (G->tu != BIR_TU_EXT || G->name >= M->string_len) continue;
        if (strcmp(&M->strings[G->name], name) == 0) return i;
    }
    return BIR_SYM_NONE;
}

uint32_t bir_const_float(bir_module_t *M, uint32_t type, double val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_FLOAT
            && M->consts[i].type == type
            && M->consts[i].d.fval == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_FLOAT;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.fval = val;
    return idx;
}

uint32_t bir_const_null(bir_module_t *M, uint32_t type)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_NULL && M->consts[i].type == type)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_NULL;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.ival = 0;
    return idx;
}

static const char *vfnm(const bir_module_t *M, uint32_t fi)
{
    if (fi >= M->num_funcs) return "<unknown>";
    if (M->funcs[fi].name >= M->string_len) return "<anon>";
    return &M->strings[M->funcs[fi].name];
}

static uint32_t vfret(const bir_module_t *M, uint32_t fi)
{
    uint32_t ft;

    if (fi >= M->num_funcs) return M->num_types;
    ft = M->funcs[fi].type;
    if (ft >= M->num_types || M->types[ft].kind != BIR_TYPE_FUNC)
        return M->num_types;
    return M->types[ft].inner;
}

static int vnarg(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t c = I->operands[1];
        return (c == 0u) ? -1 : (int)(c - 1u);
    }
    if (I->num_operands == 0) return -1;
    return (int)I->num_operands - 1;
}

static int vsame(const bir_module_t *M, uint32_t a, uint32_t b)
{
    if (a == b) return 1;
    if (a >= M->num_types || b >= M->num_types) return 0;
    return M->types[a].kind == BIR_TYPE_PTR
        && M->types[b].kind == BIR_TYPE_PTR;
}

static int vret1(const bir_module_t *M, uint32_t fi, const bir_inst_t *I)
{
    char want[64], got[64];
    uint32_t rt = vfret(M, fi), vt;

    if (rt >= M->num_types) return 0;
    if (M->types[rt].kind == BIR_TYPE_VOID) {
        if (I->num_operands == 0) return 0;
        (void)bir_type_str(M, rt, want, (int)sizeof want);
        (void)bir_type_str(M, bvty(M, I->operands[0]), got, (int)sizeof got);
        return be_fail(BC_E702, vfnm(M, fi), want, got);
    }
    if (I->num_operands == 0) {
        (void)bir_type_str(M, rt, want, (int)sizeof want);
        return be_fail(BC_E702, vfnm(M, fi), want, "nothing");
    }
    vt = bvty(M, I->operands[0]);
    if (vsame(M, vt, rt)) return 0;
    (void)bir_type_str(M, rt, want, (int)sizeof want);
    (void)bir_type_str(M, vt, got, (int)sizeof got);
    return be_fail(BC_E702, vfnm(M, fi), want, got);
}

static int vcal1(const bir_module_t *M, uint32_t fi, const bir_inst_t *I)
{
    char want[64], got[64];
    uint32_t ce = bir_oper(M, I, 0), rt;
    int na = vnarg(I), bad = 0;

    if (ce >= M->num_funcs) return 0;
    rt = vfret(M, ce);
    if (rt < M->num_types && !vsame(M, I->type, rt)) {
        (void)bir_type_str(M, I->type, got, (int)sizeof got);
        (void)bir_type_str(M, rt, want, (int)sizeof want);
        bad = be_fail(BC_E703, vfnm(M, ce), got, vfnm(M, fi), want);
    }
    if (na >= 0 && (uint32_t)na != M->funcs[ce].num_params)
        bad = be_fail(BC_E704, vfnm(M, ce), na,
                      (int)M->funcs[ce].num_params);
    return bad;
}

#define VCH_VAL  0
#define VCH_BLK  1
#define VCH_FUN  2
#define VCH_GLB  3

static uint32_t vflo[BIR_MAX_FUNCS];
static uint32_t vfhi[BIR_MAX_FUNCS];
static uint16_t vpc[BIR_FUNC_MAX_BLOCKS];
static uint32_t vps[BIR_FUNC_MAX_BLOCKS];

static int vslot(uint32_t op, uint32_t j)
{
    if (op == BIR_BR)      return (j == 0u) ? VCH_BLK : VCH_VAL;
    if (op == BIR_BR_COND) return (j == 0u) ? VCH_VAL : VCH_BLK;
    if (op == BIR_SWITCH) {
        if (j == 0u) return VCH_VAL;
        if (j == 1u) return VCH_BLK;
        return (j & 1u) ? VCH_BLK : VCH_VAL;
    }
    if (op == BIR_PHI)  return (j & 1u) ? VCH_VAL : VCH_BLK;
    if (op == BIR_CALL || op == BIR_FNREF)
        return (j == 0u) ? VCH_FUN : VCH_VAL;
    if (op == BIR_GLOBAL_REF) return (j == 0u) ? VCH_GLB : VCH_VAL;
    return VCH_VAL;
}

static const char *vkind(uint32_t op, uint32_t j)
{
    if (op == BIR_PHI)    return "phi predecessor";
    if (op == BIR_SWITCH) return (j == 1u) ? "switch default" : "switch case";
    return "branch";
}

static uint32_t vnop(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) return I->operands[1];
    return I->num_operands;
}

static uint32_t vnsuc(const bir_module_t *M, const bir_block_t *B)
{
    const bir_inst_t *I;
    uint32_t c;

    if (B->num_insts == 0u || B->first_inst >= M->num_insts) return 0u;
    if (B->num_insts - 1u > M->num_insts - 1u - B->first_inst) return 0u;
    I = &M->insts[B->first_inst + B->num_insts - 1u];
    if (I->op == BIR_BR)      return 1u;
    if (I->op == BIR_BR_COND) return 2u;
    if (I->op != BIR_SWITCH)  return 0u;
    if (I->num_operands != BIR_OPERANDS_OVERFLOW) return 1u;
    c = I->operands[1];
    if (c < 2u) return 0u;
    return 1u + (c - 2u) / 2u;
}

static uint32_t vsuc(const bir_module_t *M, const bir_block_t *B, uint32_t k)
{
    const bir_inst_t *I = &M->insts[B->first_inst + B->num_insts - 1u];

    if (I->op == BIR_BR)      return bir_oper(M, I, 0u);
    if (I->op == BIR_BR_COND) return bir_oper(M, I, 1u + k);
    if (I->num_operands != BIR_OPERANDS_OVERFLOW) return bir_oper(M, I, 1u);
    return (k == 0u) ? bir_oper(M, I, 1u) : bir_oper(M, I, 2u * k + 1u);
}

static int vsucis(const bir_module_t *M, uint32_t b, uint32_t tgt)
{
    const bir_block_t *B = &M->blocks[b];
    uint32_t ns = vnsuc(M, B), k;

    for (k = 0; k < ns; k++)
        if (vsuc(M, B, k) == tgt) return 1;
    return 0;
}

static const char *vbnm(const bir_module_t *M, uint32_t b)
{
    if (b >= M->num_blocks) return "<unknown>";
    if (M->blocks[b].name >= M->string_len) return "<anon>";
    return &M->strings[M->blocks[b].name];
}

static const char *vbown(const bir_module_t *M, uint32_t b)
{
    uint32_t fi;

    for (fi = 0; fi < M->num_funcs; fi++) {
        const bir_func_t *F = &M->funcs[fi];
        if (F->num_blocks != 0u && b >= F->first_block
         && b - F->first_block < F->num_blocks)
            return vfnm(M, fi);
    }
    return "no function";
}

static const char *vvown(const bir_module_t *M, uint32_t v)
{
    uint32_t fi;

    for (fi = 0; fi < M->num_funcs; fi++)
        if (M->funcs[fi].num_blocks != 0u && v >= vflo[fi] && v < vfhi[fi])
            return vfnm(M, fi);
    return "no function";
}

static void vfrng(const bir_module_t *M)
{
    uint32_t fi, b;

    for (fi = 0; fi < M->num_funcs; fi++) {
        const bir_func_t *F = &M->funcs[fi];
        uint32_t lo = M->num_insts, hi = 0u;
        for (b = 0; b < F->num_blocks; b++) {
            const bir_block_t *B;
            if (F->first_block + b >= M->num_blocks) break;
            B = &M->blocks[F->first_block + b];
            if (B->num_insts == 0u || B->first_inst >= M->num_insts) continue;
            if (B->first_inst < lo) lo = B->first_inst;
            if (B->first_inst + B->num_insts > hi)
                hi = B->first_inst + B->num_insts;
        }
        if (hi <= lo) { lo = 0u; hi = 0u; }
        vflo[fi] = lo;
        vfhi[fi] = hi;
    }
}

static int vovlp(const bir_module_t *M)
{
    uint32_t fi, fj;
    int bad = 0;

    for (fi = 0; fi < M->num_funcs; fi++) {
        const bir_func_t *A = &M->funcs[fi];
        int hb = 0, hi = 0;
        if (A->num_blocks == 0u) continue;
        for (fj = fi + 1u; fj < M->num_funcs && !(hb && hi); fj++) {
            const bir_func_t *C = &M->funcs[fj];
            if (C->num_blocks == 0u) continue;
            if (!hb
             && A->first_block < C->first_block + C->num_blocks
             && C->first_block < A->first_block + A->num_blocks) {
                hb = 1;
                bad = be_fail(BC_E862, vfnm(M, fi), vfnm(M, fj));
            }
            if (!hi && vfhi[fi] > vflo[fj] && vfhi[fj] > vflo[fi]) {
                hi = 1;
                bad = be_fail(BC_E863, vfnm(M, fi), vfnm(M, fj));
            }
        }
    }
    return bad;
}

typedef struct {
    uint32_t fi, blo, nb, vlo, vhi;
    int hb, hv, hp, hc, hg;
} vfs_t;

static uint32_t vstm;

static void vpred(const bir_module_t *M, uint32_t blo, uint32_t nb)
{
    uint32_t b, k;

    memset(vpc, 0, (size_t)nb * sizeof vpc[0]);
    for (b = 0; b < nb; b++) {
        const bir_block_t *B = &M->blocks[blo + b];
        uint32_t ns = vnsuc(M, B);
        vstm++;
        for (k = 0; k < ns; k++) {
            uint32_t s = vsuc(M, B, k), lp;
            if (s < blo) continue;
            lp = s - blo;
            if (lp >= nb || vps[lp] == vstm) continue;
            vps[lp] = vstm;
            if (vpc[lp] < 0xFFFFu) vpc[lp] = (uint16_t)(vpc[lp] + 1u);
        }
    }
}

static int vops1(const bir_module_t *M, vfs_t *X, const bir_inst_t *I,
                 uint32_t n)
{
    uint32_t k;
    int bad = 0;

    for (k = 0; k < n; k++) {
        uint32_t o = bir_oper(M, I, k);
        int cls = vslot(I->op, k);
        if (cls == VCH_FUN) continue;
        if (cls == VCH_GLB) {
            if (o < M->num_globals) continue;
            if (X->hg) continue;
            X->hg = 1;
            bad = be_fail(BC_E980, vfnm(M, X->fi), (unsigned)o,
                          (unsigned)M->num_globals);
            continue;
        }
        if (cls == VCH_BLK) {
            if (o >= X->blo && o - X->blo < X->nb) continue;
            if (X->hb) continue;
            X->hb = 1;
            bad = be_fail(BC_E860, vkind(I->op, k), vfnm(M, X->fi),
                          vbown(M, o));
            continue;
        }
        if (o == BIR_VAL_NONE || BIR_VAL_IS_CONST(o)) continue;
        if (o >= X->vlo && o < X->vhi) continue;
        if (X->hv) continue;
        X->hv = 1;
        bad = be_fail(BC_E861, vfnm(M, X->fi), vvown(M, o));
    }
    return bad;
}

static int vphi1(const bir_module_t *M, vfs_t *X, uint32_t b,
                 const bir_inst_t *I, uint32_t n)
{
    uint32_t k, ps = ++vstm, dist = 0u;
    int bp = 0, oor = 0, bad = 0;

    for (k = 0; k + 1u < n; k += 2u) {
        uint32_t p = bir_oper(M, I, k), lp;
        if (p < X->blo || p - X->blo >= X->nb) { oor = 1; continue; }
        lp = p - X->blo;
        if (!vsucis(M, p, X->blo + b)) bp = 1;
        if (vps[lp] == ps) bp = 1;
        else { vps[lp] = ps; dist++; }
    }
    if (bp && !X->hp) {
        X->hp = 1;
        bad = be_fail(BC_E864, vbnm(M, X->blo + b), vfnm(M, X->fi));
    }
    if (!bp && !oor && dist != vpc[b] && !X->hc) {
        X->hc = 1;
        bad = be_fail(BC_E865, vbnm(M, X->blo + b), vfnm(M, X->fi),
                      (int)dist, (int)vpc[b]);
    }
    return bad;
}

static int vfun1(const bir_module_t *M, vfs_t *X)
{
    uint32_t b, j;
    int bad = 0;

    for (b = 0; b < X->nb; b++) {
        const bir_block_t *B = &M->blocks[X->blo + b];
        uint32_t ni = B->num_insts;
        if (B->first_inst >= M->num_insts) continue;
        if (ni > M->num_insts - B->first_inst)
            ni = M->num_insts - B->first_inst;
        for (j = 0; j < ni; j++) {
            const bir_inst_t *I = &M->insts[B->first_inst + j];
            uint32_t n = vnop(I);
            if (I->op == BIR_RET && vret1(M, X->fi, I) != 0) bad = 1;
            if (I->op == BIR_CALL && vcal1(M, X->fi, I) != 0) bad = 1;
            if (vops1(M, X, I, n) != 0) bad = 1;
            if (I->op == BIR_PHI && vphi1(M, X, b, I, n) != 0) bad = 1;
        }
    }
    return bad;
}

int bir_vchk(const bir_module_t *M)
{
    uint32_t fi;
    int bad = 0;

    if (M == NULL) return BC_OK;

    vfrng(M);
    if (vovlp(M) != 0) bad = 1;
    memset(vps, 0, sizeof vps);
    vstm = 0u;

    for (fi = 0; fi < M->num_funcs; fi++) {
        const bir_func_t *F = &M->funcs[fi];
        vfs_t X;

        if (F->num_blocks == 0u || F->first_block >= M->num_blocks) continue;
        X.fi  = fi;
        X.blo = F->first_block;
        X.nb  = F->num_blocks;
        if (X.nb > M->num_blocks - X.blo) X.nb = M->num_blocks - X.blo;
        if (X.nb > BIR_FUNC_MAX_BLOCKS) X.nb = BIR_FUNC_MAX_BLOCKS;
        X.vlo = vflo[fi];
        X.vhi = vfhi[fi];
        X.hb = X.hv = X.hp = X.hc = 0;

        vpred(M, X.blo, X.nb);
        if (vfun1(M, &X) != 0) bad = 1;
    }
    return bad ? BC_ERR_VERIFY : BC_OK;
}
