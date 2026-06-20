#include "bir.h"
#include <string.h>

/* ---- Name Tables ---- */

/* Every opcode gets a name. Even the ones that probably shouldn't exist. */
static const char *op_names[BIR_OP_COUNT] = {
    [BIR_ADD]           = "add",
    [BIR_SUB]           = "sub",
    [BIR_MUL]           = "mul",
    [BIR_UMULHI]        = "umulhi",
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

    [BIR_CALL]          = "call",
    [BIR_SELECT]        = "select",
    [BIR_INLINE_ASM]    = "inline_asm",
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
    if (M->num_types >= BIR_MAX_TYPES)
        return 0;
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
        if (t->kind != kind || t->inner != inner
            || t->num_fields != (uint16_t)nfields)
            continue;
        int match = 1;
        for (int j = 0; j < nfields && match; j++) {
            if (M->type_fields[t->count + (uint32_t)j] != fields[j])
                match = 0;
        }
        if (match) return i;
    }
    if (M->num_type_fields + (uint32_t)nfields > BIR_MAX_TYPE_FIELDS)
        return 0;
    if (M->num_types >= BIR_MAX_TYPES)
        return 0;

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

uint32_t bir_type_func(bir_module_t *M, uint32_t ret,
                       const uint32_t *params, int nparams)
{
    return intern_compound(M, BIR_TYPE_FUNC, ret, params, nparams);
}

/* ---- String Table ---- */

uint32_t bir_add_string(bir_module_t *M, const char *s, uint32_t len)
{
    if (M->string_len + len + 1 > BIR_MAX_STRINGS)
        return 0;
    uint32_t offset = M->string_len;
    memcpy(&M->strings[offset], s, len);
    M->strings[offset + len] = '\0';
    M->string_len += len + 1;
    return offset;
}

/* ---- Constants ---- */

uint32_t bir_const_int(bir_module_t *M, uint32_t type, int64_t val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_INT
            && M->consts[i].type == type
            && M->consts[i].d.ival == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS)
        return 0;
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
    if (M->num_consts >= BIR_MAX_CONSTS) return 0;
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

uint32_t bir_const_float(bir_module_t *M, uint32_t type, double val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_FLOAT
            && M->consts[i].type == type
            && M->consts[i].d.fval == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS)
        return 0;
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
    if (M->num_consts >= BIR_MAX_CONSTS)
        return 0;
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_NULL;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.ival = 0;
    return idx;
}
