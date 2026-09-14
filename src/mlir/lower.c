/* MLIR to BIR. Anyone can define a dialect, so MLIR has no natural boundary
 * and this one is ours. An op off the list is named and refused, never
 * skipped, because a skipped op compiles and computes the wrong thing. */

#include <string.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_fe.h"
#include "mlir_lower.h"

#include "barracuda.h"
#include "bir.h"

/* One module at a time, and the map is 128 KB, so it does not go on a stack. */
#define LW_MAX_VALS   4096
#define LW_MAX_ERRS   16
#define LW_MAX_PARAMS 64

typedef struct {
    MLIR_ValueHandle key;
    string           nm;    /* register name, see lw_look */
    uint32_t         val;
} lw_ent_t;

typedef struct {
    bir_module_t *M;
    MLIR_Context *ctx;
    uint32_t      cur_func;
    uint32_t      cur_block;
    lw_ent_t      map[LW_MAX_VALS];
    uint32_t      nmap;
    uint32_t      nerr;
} lw_t;

static lw_t lw;

/* ---- Refusal ---- */

/* A file from an unsupported dialect would print a line per op, so past
 * LW_MAX_ERRS we stop talking and keep counting. */
static void
lw_no(MLIR_OpHandle op, const char *why)
{
    string n;

    if (lw.nerr < LW_MAX_ERRS) {
        n = op_type_to_string(MLIR_GetOpType(op));
        if (n.size == 0)
            n = MLIR_GetOpName(op);   /* unregistered ops keep theirs */
        fprintf(stderr, "E120: mlir: %.*s: %s\n", (int)n.size, n.str, why);
    } else if (lw.nerr == LW_MAX_ERRS) {
        fprintf(stderr, "E120: mlir: further refusals not printed\n");
    }
    lw.nerr++;
}

/* ---- Value map ---- */

static void
lw_bind(MLIR_ValueHandle v, uint32_t bv)
{
    /* A dropped binding surfaces later as an operand with no definition, which
     * lw_use refuses against the wrong op, so count it here too. */
    if (!v)
        return;
    if (lw.nmap >= LW_MAX_VALS) {
        lw.nerr++;
        return;
    }
    lw.map[lw.nmap].key = v;
    lw.map[lw.nmap].nm  = MLIR_GetValueRegisterName(v);
    lw.map[lw.nmap].val = bv;
    lw.nmap++;
}

/* Handle first, name second. arith.cmpi and the conversions build a second
 * value object for the same %n, so identity alone loses them. Later wins. */
static uint32_t
lw_look(MLIR_ValueHandle v)
{
    string   nm;
    uint32_t i;

    for (i = lw.nmap; i > 0; i--)
        if (lw.map[i - 1].key == v)
            return lw.map[i - 1].val;

    nm = MLIR_GetValueRegisterName(v);
    if (nm.size == 0)
        return BIR_VAL_NONE;
    for (i = lw.nmap; i > 0; i--)
        if (lw.map[i - 1].nm.size > 0 && str_eq(lw.map[i - 1].nm, nm))
            return lw.map[i - 1].val;
    return BIR_VAL_NONE;
}

/* ---- Types ---- */

/* No width accessor in the reader, so the textual form is what we have.
 * Anything unnamed here has no BIR equivalent and is not worth guessing at. */
static uint32_t
lw_type(MLIR_TypeHandle t)
{
    string s;
    char    buf[32];

    if (!t)
        return bir_type_void(lw.M);

    s = MLIR_GetTypeString(lw.ctx, t);
    if (s.size == 0 || s.size >= sizeof buf)
        return 0;
    memcpy(buf, s.str, (size_t)s.size);
    buf[s.size] = '\0';

    if (strcmp(buf, "i1") == 0)     return bir_type_int(lw.M, 1);
    if (strcmp(buf, "i8") == 0)     return bir_type_int(lw.M, 8);
    if (strcmp(buf, "i16") == 0)    return bir_type_int(lw.M, 16);
    if (strcmp(buf, "i32") == 0)    return bir_type_int(lw.M, 32);
    if (strcmp(buf, "i64") == 0)    return bir_type_int(lw.M, 64);
    if (strcmp(buf, "f16") == 0)    return bir_type_float(lw.M, 16);
    if (strcmp(buf, "f32") == 0)    return bir_type_float(lw.M, 32);
    if (strcmp(buf, "f64") == 0)    return bir_type_float(lw.M, 64);
    if (strcmp(buf, "bf16") == 0)   return bir_type_bfloat(lw.M);
    /* index is whatever the target's pointer arithmetic wants, 64 for now. */
    if (strcmp(buf, "index") == 0)  return bir_type_int(lw.M, 64);

    return 0;   /* type 0 is the sentinel, so this reads as "no idea" */
}

/* ---- Emission ---- */

static uint32_t
lw_emit(int op, uint32_t type, int subop)
{
    bir_module_t *M = lw.M;
    uint32_t      idx;
    bir_inst_t   *I;
    int           k;

    if (M->num_insts >= BIR_MAX_INSTS) {
        bir_pfull(M, BIR_P_INSTS);
        return BIR_VAL_NONE;
    }
    idx = M->num_insts++;
    I = &M->insts[idx];
    I->op = (uint16_t)op;
    I->num_operands = 0;
    I->subop = (uint8_t)subop;
    I->type = type;
    for (k = 0; k < BIR_OPERANDS_INLINE; k++)
        I->operands[k] = BIR_VAL_NONE;
    M->blocks[lw.cur_block].num_insts++;
    return BIR_MAKE_VAL(idx);
}

static void
lw_arg(uint32_t inst, uint32_t operand)
{
    bir_inst_t *I;
    uint32_t    idx;

    if (inst == BIR_VAL_NONE || BIR_VAL_IS_CONST(inst))
        return;
    idx = BIR_VAL_INDEX(inst);
    if (idx >= lw.M->num_insts)
        return;
    I = &lw.M->insts[idx];
    if (I->num_operands < BIR_OPERANDS_INLINE)
        I->operands[I->num_operands++] = operand;
}

/* Named because string offset 0 is a live string, so a nameless block is
 * labelled with whatever went into the table first. */
static uint32_t
lw_block(const char *name)
{
    bir_module_t *M = lw.M;
    bir_func_t   *F;
    uint32_t      idx;

    /* Block 0 is real, so a refusal answered with it hands back someone else's
     * block. Record it and let bir_pchk stop the compile. */
    if (M->num_blocks >= BIR_MAX_BLOCKS) {
        bir_pfull(M, BIR_P_BLOCKS);
        return 0;
    }
    idx = M->num_blocks++;
    M->blocks[idx].name = bir_add_string(M, name, (uint32_t)strlen(name));
    M->blocks[idx].first_inst = M->num_insts;
    M->blocks[idx].num_insts = 0;
    F = &M->funcs[lw.cur_func];
    if (F->num_blocks == 0)
        F->first_block = idx;
    F->num_blocks++;
    return idx;
}

/* ---- Operands ---- */

/* Unbound means we refused the op that should have defined it. */
static uint32_t
lw_use(MLIR_OpHandle op, size_t i)
{
    MLIR_ValueHandle v = MLIR_GetOpOperand(op, i);
    uint32_t         bv = lw_look(v);

    if (bv == BIR_VAL_NONE)
        lw_no(op, "operand has no lowered definition");
    return bv;
}

/* ---- arith ---- */

static int
lw_pred(string p)
{
    char buf[8];

    if (p.size == 0 || p.size >= sizeof buf)
        return -1;
    memcpy(buf, p.str, (size_t)p.size);
    buf[p.size] = '\0';

    if (strcmp(buf, "eq") == 0)  return BIR_ICMP_EQ;
    if (strcmp(buf, "ne") == 0)  return BIR_ICMP_NE;
    if (strcmp(buf, "slt") == 0) return BIR_ICMP_SLT;
    if (strcmp(buf, "sle") == 0) return BIR_ICMP_SLE;
    if (strcmp(buf, "sgt") == 0) return BIR_ICMP_SGT;
    if (strcmp(buf, "sge") == 0) return BIR_ICMP_SGE;
    if (strcmp(buf, "ult") == 0) return BIR_ICMP_ULT;
    if (strcmp(buf, "ule") == 0) return BIR_ICMP_ULE;
    if (strcmp(buf, "ugt") == 0) return BIR_ICMP_UGT;
    if (strcmp(buf, "uge") == 0) return BIR_ICMP_UGE;
    return -1;
}

static int
lw_fpred(string p)
{
    char buf[8];

    if (p.size == 0 || p.size >= sizeof buf)
        return -1;
    memcpy(buf, p.str, (size_t)p.size);
    buf[p.size] = '\0';

    if (strcmp(buf, "oeq") == 0) return BIR_FCMP_OEQ;
    if (strcmp(buf, "one") == 0) return BIR_FCMP_ONE;
    if (strcmp(buf, "olt") == 0) return BIR_FCMP_OLT;
    if (strcmp(buf, "ole") == 0) return BIR_FCMP_OLE;
    if (strcmp(buf, "ogt") == 0) return BIR_FCMP_OGT;
    if (strcmp(buf, "oge") == 0) return BIR_FCMP_OGE;
    if (strcmp(buf, "ueq") == 0) return BIR_FCMP_UEQ;
    if (strcmp(buf, "une") == 0) return BIR_FCMP_UNE;
    if (strcmp(buf, "ult") == 0) return BIR_FCMP_ULT;
    if (strcmp(buf, "ule") == 0) return BIR_FCMP_ULE;
    if (strcmp(buf, "ugt") == 0) return BIR_FCMP_UGT;
    if (strcmp(buf, "uge") == 0) return BIR_FCMP_UGE;
    if (strcmp(buf, "ord") == 0) return BIR_FCMP_ORD;
    if (strcmp(buf, "uno") == 0) return BIR_FCMP_UNO;
    return -1;
}

/* Two operands in, one result out, which covers every arith binop we take. */
static uint32_t
lw_bin(MLIR_OpHandle op, int birop, uint32_t ty)
{
    uint32_t a = lw_use(op, 0);
    uint32_t b = lw_use(op, 1);
    uint32_t r;

    if (a == BIR_VAL_NONE || b == BIR_VAL_NONE)
        return BIR_VAL_NONE;
    r = lw_emit(birop, ty, 0);
    lw_arg(r, a);
    lw_arg(r, b);
    return r;
}

static uint32_t
lw_const(MLIR_OpHandle op, uint32_t ty)
{
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, "value");
    MLIR_AttrKind        k;

    if (!a) {
        lw_no(op, "constant with no value attribute");
        return BIR_VAL_NONE;
    }
    k = MLIR_GetAttributeKind(a);

    /* The result type decides, not how the literal was written. The reader
     * builds `1 : f32` as an integer attribute, and trusting that put the bits
     * 0x1 where 1.0f belonged, quietly. */
    if (ty < lw.M->num_types && lw.M->types[ty].kind == BIR_TYPE_FLOAT) {
        if (k == MLIR_ATTR_KIND_FLOAT)
            return BIR_MAKE_CONST(bir_const_float(lw.M, ty, MLIR_GetAttributeFloat(a)));
        if (k == MLIR_ATTR_KIND_INTEGER)
            return BIR_MAKE_CONST(bir_const_float(lw.M, ty,
                                                  (double)MLIR_GetAttributeInteger(a)));
        lw_no(op, "float constant has no numeric value");
        return BIR_VAL_NONE;
    }

    if (k == MLIR_ATTR_KIND_INTEGER)
        return BIR_MAKE_CONST(bir_const_int(lw.M, ty, MLIR_GetAttributeInteger(a)));
    if (k == MLIR_ATTR_KIND_BOOL)
        return BIR_MAKE_CONST(bir_const_int(lw.M, ty, MLIR_GetAttributeBool(a) ? 1 : 0));

    /* A float literal on an integer type is not something to round silently. */
    lw_no(op, "constant value does not match its type");
    return BIR_VAL_NONE;
}

/* One operand, so not lw_bin, which would refuse on the missing second. */
static uint32_t
lw_un(MLIR_OpHandle op, int birop, uint32_t ty)
{
    uint32_t a = lw_use(op, 0);
    uint32_t r;

    if (a == BIR_VAL_NONE)
        return BIR_VAL_NONE;
    r = lw_emit(birop, ty, 0);
    lw_arg(r, a);
    return r;
}

/* ---- Operations ---- */

static void
lw_op(MLIR_OpHandle op)
{
    MLIR_OpType kind = MLIR_GetOpType(op);
    uint32_t    ty = 0;
    uint32_t    r = BIR_VAL_NONE;
    int         pred;

    if (MLIR_GetOpNumResultTypes(op) > 0) {
        ty = lw_type(MLIR_GetOpResult_type(op, 0));
        if (ty == 0) {
            lw_no(op, "result type has no BIR equivalent");
            return;
        }
    }

    switch (kind) {
    case OP_TYPE_ARITH_CONSTANT: r = lw_const(op, ty); break;

    case OP_TYPE_ARITH_ADDI:  r = lw_bin(op, BIR_ADD,  ty); break;
    case OP_TYPE_ARITH_SUBI:  r = lw_bin(op, BIR_SUB,  ty); break;
    case OP_TYPE_ARITH_MULI:  r = lw_bin(op, BIR_MUL,  ty); break;
    case OP_TYPE_ARITH_DIVSI: r = lw_bin(op, BIR_SDIV, ty); break;
    case OP_TYPE_ARITH_DIVUI: r = lw_bin(op, BIR_UDIV, ty); break;
    case OP_TYPE_ARITH_REMSI: r = lw_bin(op, BIR_SREM, ty); break;
    case OP_TYPE_ARITH_REMUI: r = lw_bin(op, BIR_UREM, ty); break;
    case OP_TYPE_ARITH_ADDF:  r = lw_bin(op, BIR_FADD, ty); break;
    case OP_TYPE_ARITH_SUBF:  r = lw_bin(op, BIR_FSUB, ty); break;
    case OP_TYPE_ARITH_MULF:  r = lw_bin(op, BIR_FMUL, ty); break;
    case OP_TYPE_ARITH_DIVF:  r = lw_bin(op, BIR_FDIV, ty); break;
    case OP_TYPE_ARITH_ANDI:  r = lw_bin(op, BIR_AND,  ty); break;
    case OP_TYPE_ARITH_ORI:   r = lw_bin(op, BIR_OR,   ty); break;
    case OP_TYPE_ARITH_XORI:  r = lw_bin(op, BIR_XOR,  ty); break;
    case OP_TYPE_ARITH_SHLI:  r = lw_bin(op, BIR_SHL,  ty); break;
    case OP_TYPE_ARITH_SHRUI: r = lw_bin(op, BIR_LSHR, ty); break;
    case OP_TYPE_ARITH_SHRSI: r = lw_bin(op, BIR_ASHR, ty); break;

    case OP_TYPE_ARITH_CMPI:
        pred = lw_pred(MLIR_GetAttributeString(MLIR_GetOpAttributeByName(op, "predicate")));
        if (pred < 0) { lw_no(op, "unknown integer compare predicate"); return; }
        r = lw_bin(op, BIR_ICMP, ty);
        if (r != BIR_VAL_NONE)
            lw.M->insts[BIR_VAL_INDEX(r)].subop = (uint8_t)pred;
        break;

    case OP_TYPE_ARITH_CMPF:
        pred = lw_fpred(MLIR_GetAttributeString(MLIR_GetOpAttributeByName(op, "predicate")));
        if (pred < 0) { lw_no(op, "unknown float compare predicate"); return; }
        r = lw_bin(op, BIR_FCMP, ty);
        if (r != BIR_VAL_NONE)
            lw.M->insts[BIR_VAL_INDEX(r)].subop = (uint8_t)pred;
        break;

    case OP_TYPE_ARITH_EXTSI:  r = lw_un(op, BIR_SEXT, ty); break;
    case OP_TYPE_ARITH_EXTUI:  r = lw_un(op, BIR_ZEXT, ty); break;
    case OP_TYPE_ARITH_TRUNCI: r = lw_un(op, BIR_TRUNC, ty); break;
    case OP_TYPE_ARITH_SITOFP: r = lw_un(op, BIR_SITOFP, ty); break;
    case OP_TYPE_ARITH_FPTOSI: r = lw_un(op, BIR_FPTOSI, ty); break;

    case OP_TYPE_RETURN:
    case OP_TYPE_FUNC_RETURN: {
        uint32_t v;

        r = lw_emit(BIR_RET, bir_type_void(lw.M), 0);
        if (MLIR_GetOpNumOperands(op) > 0) {
            v = lw_use(op, 0);
            if (v != BIR_VAL_NONE)
                lw_arg(r, v);
        }
        return;                 /* a return defines nothing */
    }

    default:
        lw_no(op, "op is outside the accepted subset");
        return;
    }

    if (r == BIR_VAL_NONE)
        return;
    if (MLIR_GetOpNumResults(op) > 0)
        lw_bind(MLIR_GetOpResult(op, 0), r);
}

/* ---- Functions ---- */

static uint32_t
lw_rety(const bir_module_t *M, const bir_func_t *F, uint32_t dflt)
{
    uint32_t b, j;

    for (b = 0; b < F->num_blocks && F->first_block + b < M->num_blocks; b++) {
        const bir_block_t *B = &M->blocks[F->first_block + b];

        for (j = 0; j < B->num_insts && B->first_inst + j < M->num_insts; j++) {
            const bir_inst_t *I = &M->insts[B->first_inst + j];
            uint32_t v;

            if (I->op != BIR_RET || I->num_operands < 1) continue;
            v = I->operands[0];
            if (v == BIR_VAL_NONE) continue;
            if (BIR_VAL_IS_CONST(v))
                return BIR_VAL_INDEX(v) < M->num_consts
                     ? M->consts[BIR_VAL_INDEX(v)].type : dflt;
            return BIR_VAL_INDEX(v) < M->num_insts
                 ? M->insts[BIR_VAL_INDEX(v)].type : dflt;
        }
    }
    return dflt;
}

static void
lw_func(MLIR_OpHandle op)
{
    bir_module_t        *M = lw.M;
    bir_func_t          *F;
    MLIR_AttributeHandle nm;
    MLIR_RegionHandle    body;
    MLIR_BlockHandle     entry;
    uint32_t             ptypes[LW_MAX_PARAMS];
    uint32_t             ret = 0;
    size_t               np, i, nb, j;
    string               s;

    if (M->num_funcs >= BIR_MAX_FUNCS) {
        bir_pfull(M, BIR_P_FUNCS);
        lw_no(op, "too many functions");
        return;
    }
    if (MLIR_GetOpNumRegions(op) == 0) {
        /* A declaration with no body. Nothing to emit and nothing wrong. */
        return;
    }

    lw.cur_func = M->num_funcs++;
    F = &M->funcs[lw.cur_func];
    memset(F, 0, sizeof *F);

    nm = MLIR_GetOpAttributeByName(op, "sym_name");
    if (nm) {
        s = MLIR_GetAttributeString(nm);
        F->name = bir_add_string(M, s.str, (uint32_t)s.size);
    }

    /* Never a kernel. MLIR marks those with a gpu.kernel attribute and the
     * reader skips the attributes clause without recording it, so there is
     * nothing to read and guessing would promote every helper. */
    F->cuda_flags = CUDA_DEVICE;

    body = MLIR_GetOpRegion(op, 0);
    if (!body || MLIR_GetRegionNumBlocks(body) == 0) {
        lw_no(op, "function body has no blocks");
        return;
    }
    entry = MLIR_GetRegionBlock(body, 0);

    /* The reader binds the header's arguments onto the entry block. */
    np = MLIR_GetBlockNumArgs(entry);
    if (np > LW_MAX_PARAMS) {
        lw_no(op, "too many parameters");
        return;
    }

    lw.cur_block = lw_block("entry");

    for (i = 0; i < np; i++) {
        MLIR_ValueHandle a = MLIR_GetBlockArg(entry, i);
        uint32_t         t = lw_type(MLIR_GetValueType(a));
        uint32_t         p;

        if (t == 0) {
            lw_no(op, "parameter type has no BIR equivalent");
            return;
        }
        ptypes[i] = t;
        p = lw_emit(BIR_PARAM, t, (int)i);
        lw_bind(a, p);
    }
    F->num_params = (uint16_t)np;

    if (MLIR_GetOpNumResultTypes(op) > 0)
        ret = lw_type(MLIR_GetOpResult_type(op, 0));
    if (ret == 0)
        ret = bir_type_void(M);
    F->type = bir_type_func(M, ret, ptypes, (int)np);

    /* Branches are not lowered yet, so more than one block is refused rather
     * than run straight through. */
    nb = MLIR_GetRegionNumBlocks(body);
    if (nb > 1) {
        lw_no(op, "multi-block function bodies need scf or cf lowering");
        return;
    }

    for (j = 0; j < MLIR_GetBlockNumOps(entry); j++)
        lw_op(MLIR_GetBlockOp(entry, j));

    /* Every block ends in a terminator, whatever the source did. */
    {
        bir_block_t *B = &M->blocks[lw.cur_block];
        if (B->num_insts == 0 ||
            M->insts[B->first_inst + B->num_insts - 1].op != BIR_RET)
            (void)lw_emit(BIR_RET, bir_type_void(M), 0);
    }
    F->type = bir_type_func(M, lw_rety(M, F, ret), ptypes, (int)np);

    for (j = 0; j < F->num_blocks; j++)
        F->total_insts += M->blocks[F->first_block + j].num_insts;
}

/* ---- Entry ---- */

int
ml_lowr(const ml_ctx_t *C, struct bir_module *Mp)
{
    bir_module_t     *M = (bir_module_t *)Mp;
    MLIR_OpHandle     root;
    MLIR_RegionHandle r;
    MLIR_BlockHandle  b;
    size_t            i;

    root = (MLIR_OpHandle)(uintptr_t)ml_root(C);
    if (!root || !M)
        return -1;

    memset(&lw, 0, sizeof lw);
    lw.M = M;
    lw.ctx = (MLIR_Context *)ml_ctx(C);

    bir_module_init(M);

    if (MLIR_GetOpType(root) != OP_TYPE_MODULE) {
        lw_no(root, "top level is not a module");
        return -1;
    }
    if (MLIR_GetOpNumRegions(root) == 0)
        return 0;               /* an empty module is a module */

    r = MLIR_GetOpRegion(root, 0);
    if (!r || MLIR_GetRegionNumBlocks(r) == 0)
        return 0;
    b = MLIR_GetRegionBlock(r, 0);

    for (i = 0; i < MLIR_GetBlockNumOps(b); i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(b, i);

        if (MLIR_GetOpType(op) == OP_TYPE_FUNC_FUNC)
            lw_func(op);
        else
            lw_no(op, "only func.func is accepted at module level");
    }

    if (lw.nerr > 0) {
        fprintf(stderr, "E120: mlir: %u op(s) refused, nothing emitted\n", lw.nerr);
        return -1;
    }
    return bir_pchk(M, "mlir lowering") == BC_OK ? 0 : -1;
}
