#include "nvidia.h"
#include <string.h>

/* BIR SSA -> PTX pseudo-MIR. Virtual registers, text output, no binary
 * encoding. The NVIDIA driver does the heavy lifting — register allocation,
 * scheduling, all the hard bits. We just emit polite suggestions in PTX
 * and hope for the best. Like posting a letter to a corporation. */

static struct {
    nv_module_t        *nv;
    const bir_module_t *bir;
    uint32_t            block_map[BIR_MAX_BLOCKS];
    uint32_t            lcl_off;   /* local (alloca) byte offset */
    uint32_t            shr_off;   /* shared memory byte offset */
    uint32_t            cur_func;  /* current nv_mfunc_t index */
} S;

/* Forward declaration — rslv needs em1 for f64 constant materialisation */
static void em1(uint16_t op, nv_opnd_t d, nv_opnd_t a, nv_opnd_t b);

/* ---- Deferred PHI Copies ----
 * PHI elimination requires inserting MOV copies into predecessor
 * blocks, before their terminators. With a flat instruction array,
 * we can't easily insert mid-stream during isel. So we record the
 * copies and insert them in a post-pass. Like filing an amendment
 * to a tax return — the government always gets its due, just late. */

#define NV_MAX_PCOPY 8192

typedef struct {
    uint32_t  pred_mblk;     /* predecessor MIR block to insert into */
    uint32_t  merge_mblk;    /* merge block where the PHI lives */
    uint16_t  mop;           /* NV_MOV_U32/F32/U64/PRED etc. */
    nv_opnd_t dst;           /* PHI result register */
    nv_opnd_t src;           /* incoming value (reg or imm) */
} nv_pcopy_t;

static nv_pcopy_t S_pcopy[NV_MAX_PCOPY];
static uint32_t   S_npc;

/* ---- Operand Constructors ---- */

static nv_opnd_t mop_none(void)
{
    nv_opnd_t o;
    memset(&o, 0, sizeof(o));
    return o;
}

static nv_opnd_t mop_reg(uint8_t rf, uint16_t rn)
{
    nv_opnd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = NV_MOP_REG;
    o.rfile = rf;
    o.reg_num = rn;
    return o;
}

static nv_opnd_t mop_imm(int32_t val)
{
    nv_opnd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = NV_MOP_IMM;
    o.imm = val;
    return o;
}


static nv_opnd_t mop_lbl(uint32_t bi)
{
    nv_opnd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = NV_MOP_LABEL;
    o.imm = (int32_t)bi;
    return o;
}

static nv_opnd_t mop_spec(int32_t id)
{
    nv_opnd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = NV_MOP_SPEC;
    o.imm = id;
    return o;
}

/* ---- Virtual Register Allocation ---- */

static uint16_t new_vreg(uint8_t rf)
{
    if (S.cur_func < S.nv->num_mfunc) {
        nv_mfunc_t *MF = &S.nv->mfuncs[S.cur_func];
        MF->rc[rf]++;
    }
    if (S.nv->rc[rf] >= 0xFFFE) return 0;
    return S.nv->rc[rf]++;
}

/* BIR type -> PTX register file */
static uint8_t bir_rfile(uint32_t type_idx)
{
    if (type_idx >= S.bir->num_types) return NV_RF_U32;
    const bir_type_t *T = &S.bir->types[type_idx];

    switch (T->kind) {
    case BIR_TYPE_INT:
        if (T->width <= 1)  return NV_RF_PRED;
        if (T->width <= 16) return NV_RF_U16;
        if (T->width <= 32) return NV_RF_U32;
        return NV_RF_U64;
    case BIR_TYPE_FLOAT:
        if (T->width <= 16) return NV_RF_F16;
        if (T->width <= 32) return NV_RF_F32;
        return NV_RF_F64;
    case BIR_TYPE_BFLOAT:
        return NV_RF_U16;   /* BF16 stored as u16 in PTX */
    case BIR_TYPE_PTR:
        return NV_RF_U64;   /* 64-bit pointers */
    default:
        return NV_RF_U32;
    }
}

/* Map a BIR instruction to a vreg, creating one if needed */
static nv_opnd_t map_val(uint32_t idx, uint32_t type_idx)
{
    if (idx >= BIR_MAX_INSTS) return mop_reg(NV_RF_U32, 0);
    if (S.nv->val_vreg[idx] != 0)
        return mop_reg(S.nv->val_rfile[idx], S.nv->val_vreg[idx]);

    uint8_t rf = bir_rfile(type_idx);
    uint16_t rn = new_vreg(rf);
    S.nv->val_vreg[idx] = rn;
    S.nv->val_rfile[idx] = rf;
    return mop_reg(rf, rn);
}

/* Resolve a BIR operand (instruction ref or constant) */
static nv_opnd_t rslv(uint32_t val)
{
    if (val == BIR_VAL_NONE) return mop_imm(0);

    if (BIR_VAL_IS_CONST(val)) {
        uint32_t ci = BIR_VAL_INDEX(val);
        if (ci >= S.bir->num_consts) return mop_imm(0);
        const bir_const_t *C = &S.bir->consts[ci];

        if (C->kind == BIR_CONST_ZERO || C->kind == BIR_CONST_NULL)
            return mop_imm(0);
        if (C->kind == BIR_CONST_INT)
            return mop_imm((int32_t)C->d.ival);
        if (C->kind == BIR_CONST_FLOAT) {
            uint8_t rf = bir_rfile(C->type);
            if (rf == NV_RF_F64) {
                /* Materialise f64 constant into a register.
                 * Can't fit 64 bits in a 32-bit imm field (JPL
                 * doesn't do struct bloat), so we emit a pseudo-op
                 * that carries the two halves and let the emitter
                 * reassemble the 0dXXXX literal. The eigenvalue
                 * thanks us for not truncating to float. */
                union { double d; uint32_t w[2]; } pun;
                pun.d = C->d.fval;
                nv_opnd_t dst = mop_reg(NV_RF_F64, new_vreg(NV_RF_F64));
                nv_opnd_t hi  = mop_imm((int32_t)pun.w[1]);
                nv_opnd_t lo  = mop_imm((int32_t)pun.w[0]);
                em1(NV_MOV_F64_LIT, dst, hi, lo);
                return dst;
            }
            union { float f; int32_t i; } pun;
            pun.f = (float)C->d.fval;
            return mop_imm(pun.i);
        }
        if (C->kind == BIR_CONST_UNDEF) return mop_imm(0);
        return mop_imm(0);
    }

    uint32_t si = BIR_VAL_INDEX(val);
    if (si >= S.bir->num_insts) return mop_imm(0);

    /* Already mapped? Return existing. Otherwise create. */
    if (S.nv->val_vreg[si] != 0)
        return mop_reg(S.nv->val_rfile[si], S.nv->val_vreg[si]);

    return map_val(si, S.bir->insts[si].type);
}

/* Resolve, but return the register file for the BIR value */
static uint8_t rslv_rf(uint32_t val)
{
    if (val == BIR_VAL_NONE) return NV_RF_U32;
    if (BIR_VAL_IS_CONST(val)) {
        uint32_t ci = BIR_VAL_INDEX(val);
        if (ci < S.bir->num_consts)
            return bir_rfile(S.bir->consts[ci].type);
        return NV_RF_U32;
    }
    uint32_t si = BIR_VAL_INDEX(val);
    if (si < S.bir->num_insts) {
        if (S.nv->val_rfile[si] != 0 || S.nv->val_vreg[si] != 0)
            return S.nv->val_rfile[si];
        return bir_rfile(S.bir->insts[si].type);
    }
    return NV_RF_U32;
}

/* ---- Helpers ---- */

static uint32_t n_ops(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW)
        return I->operands[1];
    return I->num_operands;
}

static uint32_t get_op(const bir_inst_t *I, uint32_t k)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t base = I->operands[0];
        if (base + k < S.bir->num_extra_ops)
            return S.bir->extra_operands[base + k];
        return BIR_VAL_NONE;
    }
    if (k < 6) return I->operands[k];
    return BIR_VAL_NONE;
}

/* Type size in bytes for load/store width selection.
 * Structs need the full sum-of-fields treatment — GEP strides
 * depend on it. Without this, parts[tid] computes base + tid*4
 * instead of base + tid*40 and you read someone else's neutron.
 * Which is bad nuclear physics even by Monte Carlo standards. */
static uint32_t type_bytes(uint32_t type_idx)
{
    if (type_idx >= S.bir->num_types) return 4;
    const bir_type_t *T = &S.bir->types[type_idx];
    switch (T->kind) {
    case BIR_TYPE_INT:
    case BIR_TYPE_FLOAT:
    case BIR_TYPE_BFLOAT:
        return ((uint32_t)T->width + 7u) / 8u;
    case BIR_TYPE_PTR:
        return 8;
    case BIR_TYPE_STRUCT: {
        uint32_t sz = 0;
        int guard = 64;
        for (uint16_t i = 0; i < T->num_fields && guard > 0;
             i++, guard--) {
            uint32_t fi = T->count + (uint32_t)i;
            if (fi < S.bir->num_type_fields)
                sz += type_bytes(S.bir->type_fields[fi]);
        }
        return sz ? sz : 4;
    }
    case BIR_TYPE_ARRAY: {
        uint32_t esz = type_bytes(T->inner);
        return T->count * esz;
    }
    default:
        return 4;
    }
}

/* ---- Emission ---- */

static uint32_t emit(uint16_t op, uint8_t nd, uint8_t nu,
                     const nv_opnd_t *ops, uint16_t flags)
{
    if (S.nv->num_minst >= NV_MAX_MINST) return 0;
    uint32_t idx = S.nv->num_minst++;
    nv_minst_t *I = &S.nv->minsts[idx];
    I->op = op;
    I->num_defs = nd;
    I->num_uses = nu;
    I->flags = flags;
    I->pad = 0;
    int total = nd + nu;
    if (total > NV_MAX_OPS) total = NV_MAX_OPS;
    for (int i = 0; i < total; i++)
        I->ops[i] = ops[i];
    for (int i = total; i < NV_MAX_OPS; i++)
        I->ops[i] = mop_none();
    return idx;
}

/* Convenience: 1 def, N uses */
static void em1(uint16_t op, nv_opnd_t d,
                nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t ops[3] = { d, a, b };
    emit(op, 1, 2, ops, 0);
}

static void em1u(uint16_t op, nv_opnd_t d, nv_opnd_t a)
{
    nv_opnd_t ops[2] = { d, a };
    emit(op, 1, 1, ops, 0);
}

static void em0(uint16_t op)
{
    nv_opnd_t ops[1] = { mop_none() };
    emit(op, 0, 0, ops, 0);
}

/* ---- Materialise Constants into Registers ---- */
/* PTX can use immediates inline for many ops, but loads/stores
 * and some ops need register operands. This helper moves a
 * constant into a typed register when needed. */

static nv_opnd_t mat_const(uint32_t val, uint8_t rf)
{
    nv_opnd_t src = rslv(val);
    if (src.kind == NV_MOP_REG) return src;

    /* Need to materialise into a register */
    uint16_t rn = new_vreg(rf);
    nv_opnd_t dst = mop_reg(rf, rn);

    uint16_t mov_op;
    switch (rf) {
    case NV_RF_U64:  mov_op = NV_MOV_U64;  break;
    case NV_RF_F32:  mov_op = NV_MOV_F32;  break;
    case NV_RF_F64:  mov_op = NV_MOV_F64;  break;
    case NV_RF_PRED: mov_op = NV_MOV_PRED; break;
    default:         mov_op = NV_MOV_U32;  break;
    }
    em1u(mov_op, dst, src);
    return dst;
}

/* ---- Integer Arithmetic ---- */

static void is_iadd(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t op = (rf == NV_RF_U64) ? NV_ADD_U64 : NV_ADD_U32;
    em1(op, d, a, b);
}

static void is_isub(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);
    em1((rf == NV_RF_U64) ? NV_SUB_S64 : NV_SUB_U32, d, a, b);
}

static void is_imul(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);
    em1((rf == NV_RF_U64) ? NV_MUL_LO_U64 : NV_MUL_LO_U32, d, a, b);
}

static void is_idiv(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);
    uint16_t op = (I->op == BIR_SDIV) ? NV_DIV_S32 : NV_DIV_U32;
    em1(op, d, a, b);
}

static void is_irem(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);
    uint16_t op = (I->op == BIR_SREM) ? NV_REM_S32 : NV_REM_U32;
    em1(op, d, a, b);
}

/* ---- Bitwise / Shift ---- */

static void is_bitop(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t op;
    int is64 = (rf == NV_RF_U64);
    switch (I->op) {
    case BIR_AND:  op = is64 ? NV_AND_B64 : NV_AND_B32; break;
    case BIR_OR:   op = is64 ? NV_OR_B64  : NV_OR_B32;  break;
    case BIR_XOR:  op = is64 ? NV_XOR_B64 : NV_XOR_B32; break;
    case BIR_SHL:  op = is64 ? NV_SHL_B64 : NV_SHL_B32; break;
    case BIR_LSHR: op = is64 ? NV_SHR_U64 : NV_SHR_U32; break;
    case BIR_ASHR: op = NV_SHR_S32; break;
    default: return;
    }
    em1(op, d, a, b);
}

/* ---- FP Arithmetic ---- */

static void is_fadd(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t op;
    switch (I->op) {
    case BIR_FADD: op = (rf == NV_RF_F64) ? NV_ADD_F64 : NV_ADD_F32; break;
    case BIR_FSUB: op = (rf == NV_RF_F64) ? NV_SUB_F64 : NV_SUB_F32; break;
    case BIR_FMUL: op = (rf == NV_RF_F64) ? NV_MUL_F64 : NV_MUL_F32; break;
    case BIR_FDIV: op = (rf == NV_RF_F64) ? NV_DIV_F64 : NV_DIV_F32; break;
    default: return;
    }
    em1(op, d, a, b);
}

static void is_frem(uint32_t idx, const bir_inst_t *I)
{
    /* PTX has no frem — lower to: rem = a - trunc(a/b) * b
     * For now, just emit div + mul + sub. */
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t dop = (rf == NV_RF_F64) ? NV_DIV_F64 : NV_DIV_F32;
    uint16_t mop = (rf == NV_RF_F64) ? NV_MUL_F64 : NV_MUL_F32;
    uint16_t sop = (rf == NV_RF_F64) ? NV_SUB_F64 : NV_SUB_F32;

    nv_opnd_t t1 = mop_reg(rf, new_vreg(rf));
    nv_opnd_t t2 = mop_reg(rf, new_vreg(rf));
    em1(dop, t1, a, b);     /* t1 = a / b */
    em1(mop, t2, t1, b);    /* t2 = t1 * b (trunc skipped for approx) */
    em1(sop, d, a, t2);     /* d  = a - t2 */
}

/* ---- Comparison ---- */

/* Predicate registers are 1-bit — they can't appear as operands to
 * setp.*.u32. If we're comparing a predicate (e.g. branch on icmp
 * result), materialise it into a u32 register first. The PTX assembler
 * is surprisingly strict about this. Who knew. */
static nv_opnd_t mat_pred(nv_opnd_t op)
{
    if (op.kind == NV_MOP_REG && op.rfile == NV_RF_PRED) {
        uint16_t rn = new_vreg(NV_RF_U32);
        nv_opnd_t dst = mop_reg(NV_RF_U32, rn);
        nv_opnd_t ops[4] = { dst, mop_imm(1), mop_imm(0), op };
        emit(NV_SELP_U32, 1, 3, ops, 0);
        return dst;
    }
    return op;
}

static void is_icmp(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    /* Override: comparison result is a predicate */
    S.nv->val_rfile[idx] = NV_RF_PRED;
    d.rfile = NV_RF_PRED;

    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    /* Predicates can't be setp source operands — widen to u32 */
    a = mat_pred(a);
    b = mat_pred(b);

    uint16_t op;
    switch (I->subop) {
    case BIR_ICMP_EQ:  op = NV_SETP_EQ_U32; break;
    case BIR_ICMP_NE:  op = NV_SETP_NE_U32; break;
    case BIR_ICMP_ULT: op = NV_SETP_LT_U32; break;
    case BIR_ICMP_ULE: op = NV_SETP_LE_U32; break;
    case BIR_ICMP_UGT: op = NV_SETP_GT_U32; break;
    case BIR_ICMP_UGE: op = NV_SETP_GE_U32; break;
    case BIR_ICMP_SLT: op = NV_SETP_LT_S32; break;
    case BIR_ICMP_SLE: op = NV_SETP_LE_S32; break;
    case BIR_ICMP_SGT: op = NV_SETP_GT_S32; break;
    case BIR_ICMP_SGE: op = NV_SETP_GE_S32; break;
    default:           op = NV_SETP_NE_U32; break;
    }
    em1(op, d, a, b);
}

static void is_fcmp(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    S.nv->val_rfile[idx] = NV_RF_PRED;
    d.rfile = NV_RF_PRED;

    uint8_t rf = rslv_rf(I->operands[0]);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t op;
    int is64 = (rf == NV_RF_F64);
    switch (I->subop) {
    case BIR_FCMP_OEQ: case BIR_FCMP_UEQ:
        op = is64 ? NV_SETP_EQ_F64 : NV_SETP_EQ_F32; break;
    case BIR_FCMP_ONE: case BIR_FCMP_UNE:
        op = is64 ? NV_SETP_NE_F64 : NV_SETP_NE_F32; break;
    case BIR_FCMP_OLT: case BIR_FCMP_ULT:
        op = is64 ? NV_SETP_LT_F64 : NV_SETP_LT_F32; break;
    case BIR_FCMP_OLE: case BIR_FCMP_ULE:
        op = is64 ? NV_SETP_LE_F64 : NV_SETP_LE_F32; break;
    case BIR_FCMP_OGT: case BIR_FCMP_UGT:
        op = is64 ? NV_SETP_GT_F64 : NV_SETP_GT_F32; break;
    case BIR_FCMP_OGE: case BIR_FCMP_UGE:
        op = is64 ? NV_SETP_GE_F64 : NV_SETP_GE_F32; break;
    default:
        op = is64 ? NV_SETP_NE_F64 : NV_SETP_NE_F32; break;
    }
    em1(op, d, a, b);
}

/* ---- Select ---- */

static void is_selp(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t cond = rslv(I->operands[0]);
    nv_opnd_t tv   = rslv(I->operands[1]);
    nv_opnd_t fv   = rslv(I->operands[2]);

    /* If cond is not a predicate reg, we need setp first */
    if (cond.kind != NV_MOP_REG || cond.rfile != NV_RF_PRED) {
        uint16_t prn = new_vreg(NV_RF_PRED);
        nv_opnd_t p = mop_reg(NV_RF_PRED, prn);
        em1(NV_SETP_NE_U32, p, cond, mop_imm(0));
        cond = p;
    }

    uint16_t op;
    switch (rf) {
    case NV_RF_U64:  op = NV_SELP_U64; break;
    case NV_RF_F32:  op = NV_SELP_F32; break;
    case NV_RF_F64:  op = NV_SELP_F64; break;
    default:         op = NV_SELP_U32; break;
    }

    nv_opnd_t ops[4] = { d, tv, fv, cond };
    emit(op, 1, 3, ops, 0);
}

/* ---- Conversions ---- */

static void is_cvt(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t s = rslv(I->operands[0]);

    uint16_t op;
    switch (I->op) {
    case BIR_FPTOSI: {
        uint32_t si0 = BIR_VAL_INDEX(I->operands[0]);
        uint8_t srf = (si0 < S.bir->num_insts) ? bir_rfile(S.bir->insts[si0].type) : NV_RF_F32;
        op = (srf == NV_RF_F64) ? NV_CVT_S32_F64 : NV_CVT_S32_F32;
        break;
    }
    case BIR_FPTOUI: {
        uint32_t si0 = BIR_VAL_INDEX(I->operands[0]);
        uint8_t srf = (si0 < S.bir->num_insts) ? bir_rfile(S.bir->insts[si0].type) : NV_RF_F32;
        op = (srf == NV_RF_F64) ? NV_CVT_U32_F64 : NV_CVT_U32_F32;
        break;
    }
    case BIR_SITOFP:  op = (bir_rfile(I->type) == NV_RF_F64) ? NV_CVT_F64_S32 : NV_CVT_F32_S32; break;
    case BIR_UITOFP:  op = (bir_rfile(I->type) == NV_RF_F64) ? NV_CVT_F64_U32 : NV_CVT_F32_U32; break;
    case BIR_FPTRUNC: op = NV_CVT_F32_F64; break;
    case BIR_FPEXT:   op = NV_CVT_F64_F32; break;
    case BIR_ZEXT:    op = NV_CVT_U64_U32; break;
    case BIR_SEXT:    op = NV_CVT_S64_S32; break;
    case BIR_TRUNC:   op = NV_CVT_U32_U64; break;
    case BIR_PTRTOINT:
    case BIR_INTTOPTR:
    case BIR_BITCAST:
        /* These are no-ops in PTX — just a mov */
        em1u(NV_MOV_U64, d, s);
        return;
    default:
        em1u(NV_MOV_U32, d, s);
        return;
    }
    em1u(op, d, s);
}

/* ---- Memory: Load ---- */

static void is_load(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint8_t drf = S.nv->val_rfile[idx];

    /* Get address space from pointer type */
    uint32_t ptr_val = I->operands[0];
    int as = BIR_AS_GLOBAL;
    if (ptr_val != BIR_VAL_NONE && !BIR_VAL_IS_CONST(ptr_val)) {
        uint32_t si = BIR_VAL_INDEX(ptr_val);
        if (si < S.bir->num_insts) {
            uint32_t pt = S.bir->insts[si].type;
            if (pt < S.bir->num_types &&
                S.bir->types[pt].kind == BIR_TYPE_PTR)
                as = S.bir->types[pt].addrspace;
        }
    }

    nv_opnd_t addr = mat_const(ptr_val, NV_RF_U64);

    uint16_t op;
    switch (as) {
    case BIR_AS_SHARED:
        op = (drf == NV_RF_F32) ? NV_LD_SHR_F32 : NV_LD_SHR_U32;
        break;
    case BIR_AS_PRIVATE:
        op = (drf == NV_RF_F64) ? NV_LD_LOC_F64 :
             (drf == NV_RF_F32) ? NV_LD_LOC_F32 :
             (drf == NV_RF_U64) ? NV_LD_LOC_U64 : NV_LD_LOC_U32;
        break;
    default: /* global */
        switch (drf) {
        case NV_RF_F32:  op = NV_LD_GLB_F32; break;
        case NV_RF_F64:  op = NV_LD_GLB_F64; break;
        case NV_RF_U64:  op = NV_LD_GLB_U64; break;
        case NV_RF_U16:  op = NV_LD_GLB_U16; break;
        default:         op = NV_LD_GLB_U32; break;
        }
        break;
    }

    em1u(op, d, addr);
}

/* ---- Memory: Store ---- */

static void is_store(const bir_inst_t *I)
{
    /* BIR store: ops[0]=value, ops[1]=address */
    nv_opnd_t val = rslv(I->operands[0]);
    uint8_t vrf = rslv_rf(I->operands[0]);

    uint32_t ptr_val = I->operands[1];
    int as = BIR_AS_GLOBAL;
    if (ptr_val != BIR_VAL_NONE && !BIR_VAL_IS_CONST(ptr_val)) {
        uint32_t si = BIR_VAL_INDEX(ptr_val);
        if (si < S.bir->num_insts) {
            uint32_t pt = S.bir->insts[si].type;
            if (pt < S.bir->num_types &&
                S.bir->types[pt].kind == BIR_TYPE_PTR)
                as = S.bir->types[pt].addrspace;
        }
    }

    nv_opnd_t addr = mat_const(ptr_val, NV_RF_U64);

    /* Value must be in a register for stores */
    if (val.kind != NV_MOP_REG)
        val = mat_const(I->operands[0], vrf);

    uint16_t op;
    switch (as) {
    case BIR_AS_SHARED:
        op = (vrf == NV_RF_F32) ? NV_ST_SHR_F32 : NV_ST_SHR_U32;
        break;
    case BIR_AS_PRIVATE:
        op = (vrf == NV_RF_F64) ? NV_ST_LOC_F64 :
             (vrf == NV_RF_F32) ? NV_ST_LOC_F32 :
             (vrf == NV_RF_U64) ? NV_ST_LOC_U64 : NV_ST_LOC_U32;
        break;
    default:
        switch (vrf) {
        case NV_RF_F32:  op = NV_ST_GLB_F32; break;
        case NV_RF_F64:  op = NV_ST_GLB_F64; break;
        case NV_RF_U64:  op = NV_ST_GLB_U64; break;
        case NV_RF_U16:  op = NV_ST_GLB_U16; break;
        default:         op = NV_ST_GLB_U32; break;
        }
        break;
    }

    /* Store: 0 defs, 2 uses (addr, value) */
    nv_opnd_t ops[2] = { addr, val };
    emit(op, 0, 2, ops, 0);
}

/* ---- Memory: Alloca ---- */

static void is_alloca(uint32_t idx, const bir_inst_t *I)
{
    /* Allocas become .local addresses. Track offset, produce a
     * mov of the offset into a u64 register. The emitter will
     * reference these as local memory. */
    nv_opnd_t d = map_val(idx, I->type);

    uint32_t sz = 4; /* default 4 bytes */
    if (I->type < S.bir->num_types) {
        const bir_type_t *T = &S.bir->types[I->type];
        if (T->kind == BIR_TYPE_PTR && T->inner < S.bir->num_types)
            sz = type_bytes(T->inner);
    }
    uint32_t off = S.lcl_off;
    S.lcl_off += sz;

    /* PTX local addresses are NOT 0-based — we need the symbolic
     * __local base. NV_LEA_LOCAL emits: mov.u64 %rd, __local+off */
    em1u(NV_LEA_LOCAL, d, mop_imm((int32_t)off));
}

/* ---- Memory: Shared Alloc ---- */

static void is_shralloc(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);

    uint32_t sz = 4;
    if (I->type < S.bir->num_types) {
        const bir_type_t *T = &S.bir->types[I->type];
        if (T->kind == BIR_TYPE_PTR && T->inner < S.bir->num_types)
            sz = type_bytes(T->inner);
    }
    uint32_t off = S.shr_off;
    S.shr_off += sz;

    em1u(NV_MOV_U64, d, mop_imm((int32_t)off));
}

/* ---- Memory: GEP ---- */

static void is_gep(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t base = rslv(I->operands[0]);

    /* Ensure base is in a register */
    if (base.kind != NV_MOP_REG)
        base = mat_const(I->operands[0], NV_RF_U64);

    uint32_t nop = n_ops(I);
    if (nop < 2) {
        em1u(NV_MOV_U64, d, base);
        return;
    }

    nv_opnd_t offset = rslv(I->operands[1]);

    /* Compute stride from pointee type */
    uint32_t stride = 4;
    if (I->type < S.bir->num_types) {
        const bir_type_t *T = &S.bir->types[I->type];
        if (T->kind == BIR_TYPE_PTR && T->inner < S.bir->num_types)
            stride = type_bytes(T->inner);
    }

    /* mad.lo.u64 %rd, index, stride, base */
    if (offset.kind == NV_MOP_REG) {
        /* Widen index to u64 if it's u32 */
        if (offset.rfile == NV_RF_U32) {
            uint16_t rn64 = new_vreg(NV_RF_U64);
            nv_opnd_t w = mop_reg(NV_RF_U64, rn64);
            em1u(NV_CVT_U64_U32, w, offset);
            offset = w;
        }
        nv_opnd_t ops[4] = { d, offset, mop_imm((int32_t)stride), base };
        emit(NV_MAD_LO_U64, 1, 3, ops, 0);
    } else {
        /* Constant offset — mul and add */
        int32_t byte_off = offset.imm * (int32_t)stride;
        nv_opnd_t off_op = mop_imm(byte_off);
        em1(NV_ADD_U64, d, base, off_op);
    }
}

/* ---- Parameters ---- */

static void is_param(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t pi = I->subop;

    uint16_t op;
    switch (rf) {
    case NV_RF_U64:  op = NV_LD_PARAM_U64; break;
    case NV_RF_F32:  op = NV_LD_PARAM_F32; break;
    case NV_RF_F64:  op = NV_LD_PARAM_F64; break;
    default:         op = NV_LD_PARAM_U32; break;
    }

    em1u(op, d, mop_imm((int32_t)pi));
}

/* ---- Thread Model ---- */

static void is_thread(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    int dim = I->subop;

    int32_t spec;
    switch (I->op) {
    case BIR_THREAD_ID:
        spec = (dim == 0) ? NV_SPEC_TID_X :
               (dim == 1) ? NV_SPEC_TID_Y : NV_SPEC_TID_Z;
        break;
    case BIR_BLOCK_ID:
        spec = (dim == 0) ? NV_SPEC_CTAID_X :
               (dim == 1) ? NV_SPEC_CTAID_Y : NV_SPEC_CTAID_Z;
        break;
    case BIR_BLOCK_DIM:
        spec = (dim == 0) ? NV_SPEC_NTID_X :
               (dim == 1) ? NV_SPEC_NTID_Y : NV_SPEC_NTID_Z;
        break;
    case BIR_GRID_DIM:
        spec = (dim == 0) ? NV_SPEC_NCTAID_X :
               (dim == 1) ? NV_SPEC_NCTAID_Y : NV_SPEC_NCTAID_Z;
        break;
    default:
        spec = NV_SPEC_TID_X;
        break;
    }

    em1u(NV_MOV_U32, d, mop_spec(spec));
}

/* ---- Control Flow ---- */

static void is_br(const bir_inst_t *I)
{
    uint32_t tgt = I->operands[0];
    uint32_t m_tgt = S.block_map[tgt];
    nv_opnd_t ops[1] = { mop_lbl(m_tgt) };
    emit(NV_BRA, 0, 1, ops, 0);
}

static void is_brcond(const bir_inst_t *I)
{
    nv_opnd_t cond = rslv(I->operands[0]);
    uint32_t true_bir  = I->operands[1];
    uint32_t false_bir = I->operands[2];

    /* If cond is not a predicate, convert to one */
    if (cond.kind != NV_MOP_REG || cond.rfile != NV_RF_PRED) {
        uint16_t prn = new_vreg(NV_RF_PRED);
        nv_opnd_t p = mop_reg(NV_RF_PRED, prn);
        em1(NV_SETP_NE_U32, p, cond, mop_imm(0));
        cond = p;
    }

    /* @%p bra $true_label */
    uint32_t m_true = S.block_map[true_bir];
    nv_opnd_t ops[2] = { cond, mop_lbl(m_true) };
    emit(NV_BRA_PRED, 0, 2, ops, 0);

    /* Fallthrough or explicit branch to false */
    uint32_t m_false = S.block_map[false_bir];
    nv_opnd_t fops[1] = { mop_lbl(m_false) };
    emit(NV_BRA, 0, 1, fops, 0);
}

/* ---- PHI Nodes ---- */

static void is_phi(uint32_t idx, const bir_inst_t *I,
                   uint32_t cur_bir_blk)
{
    /* PHI elimination via deferred predecessor copies.
     * BIR PHI operands are (block, value) pairs — block at even
     * positions, value at odd. Getting this backwards gives you
     * undefined predicate registers and a kernel that treats GPU
     * memory like a pinball machine. We learned this the hard way. */
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t nop = n_ops(I);

    uint16_t mop;
    switch (rf) {
    case NV_RF_U64:  mop = NV_MOV_U64;  break;
    case NV_RF_F32:  mop = NV_MOV_F32;  break;
    case NV_RF_F64:  mop = NV_MOV_F64;  break;
    case NV_RF_PRED: mop = NV_MOV_PRED; break;
    default:         mop = NV_MOV_U32;  break;
    }

    uint32_t merge_mblk = S.block_map[cur_bir_blk];

    int guard = 64;
    for (uint32_t k = 0; k + 1 < nop && guard > 0; k += 2, guard--) {
        uint32_t pred_bir = get_op(I, k);       /* block at even */
        uint32_t val      = get_op(I, k + 1);   /* value at odd  */

        if (pred_bir >= S.bir->num_blocks) continue;
        if (val == BIR_VAL_NONE) continue;

        uint32_t m_pred = S.block_map[pred_bir];
        /* block_map is pre-computed; m_pred may reference a block
         * not yet created during isel but valid by construction.
         * Only reject genuinely out-of-range values. */
        if (m_pred >= NV_MAX_MBLK) continue;

        nv_opnd_t src = rslv(val);

        /* Self-copy: skip */
        if (src.kind == NV_MOP_REG && src.rfile == d.rfile &&
            src.reg_num == d.reg_num)
            continue;

        /* Record deferred copy for phi_fix() */
        if (S_npc < NV_MAX_PCOPY) {
            S_pcopy[S_npc].pred_mblk  = m_pred;
            S_pcopy[S_npc].merge_mblk = merge_mblk;
            S_pcopy[S_npc].mop        = mop;
            S_pcopy[S_npc].dst        = d;
            S_pcopy[S_npc].src        = src;
            S_npc++;
        }
    }
}

/* ---- Barriers ---- */

static void is_barrier(void)
{
    em0(NV_BAR_SYNC);
}

/* ---- Atomics ---- */

static void is_atomic(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t addr = mat_const(I->operands[0], NV_RF_U64);
    nv_opnd_t val  = rslv(I->operands[1]);

    /* Materialise val if immediate */
    uint8_t vrf = rslv_rf(I->operands[1]);
    if (val.kind != NV_MOP_REG)
        val = mat_const(I->operands[1], vrf);

    uint16_t op;
    switch (I->op) {
    case BIR_ATOMIC_ADD:
        op = (vrf == NV_RF_F32) ? NV_ATOM_ADD_F32 : NV_ATOM_ADD_U32; break;
    case BIR_ATOMIC_MIN:  op = NV_ATOM_MIN_U32;  break;
    case BIR_ATOMIC_MAX:  op = NV_ATOM_MAX_U32;  break;
    case BIR_ATOMIC_AND:  op = NV_ATOM_AND_B32;  break;
    case BIR_ATOMIC_OR:   op = NV_ATOM_OR_B32;   break;
    case BIR_ATOMIC_XOR:  op = NV_ATOM_XOR_B32;  break;
    case BIR_ATOMIC_XCHG: op = NV_ATOM_XCHG_B32; break;
    case BIR_ATOMIC_CAS:  op = NV_ATOM_CAS_B32;  break;
    default:              op = NV_ATOM_ADD_U32;   break;
    }

    nv_opnd_t ops[3] = { d, addr, val };
    emit(op, 1, 2, ops, 0);
}

static void is_atm_load(uint32_t idx, const bir_inst_t *I)
{
    /* Atomic load → regular volatile load (PTX semantics) */
    is_load(idx, I);
}

static void is_atm_store(const bir_inst_t *I)
{
    /* Atomic store → regular volatile store */
    is_store(I);
}

/* ---- Warp Ops ---- */

static void is_shfl(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t val_op = I->operands[I->num_operands >= 3 ? 1 : 0];
    uint32_t lane_op = I->operands[I->num_operands >= 3 ? 2 : 1];
    nv_opnd_t val = rslv(val_op);
    nv_opnd_t lane = rslv(lane_op);

    if (val.kind != NV_MOP_REG)
        val = mat_const(val_op, NV_RF_U32);

    uint16_t op;
    switch (I->op) {
    case BIR_SHFL:      op = NV_SHFL_IDX;  break;
    case BIR_SHFL_UP:   op = NV_SHFL_UP;   break;
    case BIR_SHFL_DOWN: op = NV_SHFL_DOWN; break;
    case BIR_SHFL_XOR:  op = NV_SHFL_XOR;  break;
    default:            op = NV_SHFL_IDX;  break;
    }

    nv_opnd_t ops[3] = { d, val, lane };
    emit(op, 1, 2, ops, 0);
}

static void is_vote(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t pred_op = I->operands[I->num_operands > 1 ? 1 : 0];
    nv_opnd_t pred = rslv(pred_op);

    if (pred.kind != NV_MOP_REG || pred.rfile != NV_RF_PRED) {
        uint16_t prn = new_vreg(NV_RF_PRED);
        nv_opnd_t p = mop_reg(NV_RF_PRED, prn);
        em1(NV_SETP_NE_U32, p, pred, mop_imm(0));
        pred = p;
    }

    uint16_t op;
    switch (I->op) {
    case BIR_BALLOT:  op = NV_VOTE_BALLOT; break;
    case BIR_VOTE_ANY: op = NV_VOTE_ANY;   break;
    case BIR_VOTE_ALL: op = NV_VOTE_ALL;   break;
    default:          op = NV_VOTE_BALLOT; break;
    }

    em1u(op, d, pred);
}

/* ---- Math Builtins ---- */

static void is_math(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);

    uint16_t op;
    switch (I->op) {
    case BIR_SQRT:   op = (rf == NV_RF_F64) ? NV_SQRT_F64 : NV_SQRT_F32; break;
    case BIR_RSQ:    op = NV_RSQ_F32;   break;
    case BIR_RCP:    op = NV_RCP_F32;   break;
    case BIR_SIN:
    case BIR_COS: {
        /* BIR lowerer pre-divides by 2π (AMD turns convention).
         * PTX sin/cos want radians — multiply back.
         * Without this, sin(x/2π) instead of sin(x). Oops. */
        union { float f; int32_t i; } pn;
        pn.f = 6.2831853f; /* 2π */
        uint16_t tn = new_vreg(NV_RF_F32);
        nv_opnd_t t = mop_reg(NV_RF_F32, tn);
        em1(NV_MUL_F32, t, a, mop_imm(pn.i));
        a = t;
        op = (I->op == BIR_SIN) ? NV_SIN_F32 : NV_COS_F32;
        break;
    }
    case BIR_EXP2:   op = NV_EX2_F32; break; /* PTX ex2 is f32 only */
    case BIR_LOG2:   op = NV_LG2_F32;   break;
    case BIR_FABS:   op = (rf == NV_RF_F64) ? NV_ABS_F64 : NV_ABS_F32; break;
    case BIR_FLOOR:  op = NV_FLOOR_F32; break;
    case BIR_CEIL:   op = NV_CEIL_F32;  break;
    case BIR_FTRUNC: op = NV_TRUNC_F32; break;
    case BIR_RNDNE:  op = NV_ROUND_F32; break;
    default:         op = NV_MOV_F32;   break;
    }

    em1u(op, d, a);
}

static void is_fminmax(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);
    nv_opnd_t b = rslv(I->operands[1]);

    uint16_t op;
    switch (I->op) {
    case BIR_FMIN: op = NV_MIN_F32; break;
    case BIR_FMAX: op = NV_MAX_F32; break;
    default:       op = NV_MOV_F32; em1u(op, d, a); return;
    }
    em1(op, d, a, b);
}

/* ---- Return ---- */

static void is_ret(const bir_inst_t *I)
{
    if (I->num_operands > 0 && I->operands[0] != BIR_VAL_NONE) {
        /* Non-void return: store to return parameter (TODO) */
    }
    /* Kernels use exit, functions use ret */
    if (S.cur_func < S.nv->num_mfunc &&
        S.nv->mfuncs[S.cur_func].is_kern)
        em0(NV_EXIT);
    else
        em0(NV_RET);
}

/* ---- Per-Block Instruction Selection ---- */

static void isel_blk(uint32_t bir_bi)
{
    const bir_block_t *B = &S.bir->blocks[bir_bi];

    int guard = 65536;
    for (uint32_t ii = 0; ii < B->num_insts && guard > 0; ii++, guard--) {
        uint32_t idx = B->first_inst + ii;
        const bir_inst_t *I = &S.bir->insts[idx];

        switch (I->op) {
        /* ---- Integer Arithmetic ---- */
        case BIR_ADD:
            is_iadd(idx, I); break;
        case BIR_SUB:
            is_isub(idx, I); break;
        case BIR_MUL:
            is_imul(idx, I); break;
        case BIR_SDIV: case BIR_UDIV:
            is_idiv(idx, I); break;
        case BIR_SREM: case BIR_UREM:
            is_irem(idx, I); break;

        /* ---- Bitwise / Shift ---- */
        case BIR_AND: case BIR_OR: case BIR_XOR:
        case BIR_SHL: case BIR_LSHR: case BIR_ASHR:
            is_bitop(idx, I); break;

        /* ---- FP Arithmetic ---- */
        case BIR_FADD: case BIR_FSUB: case BIR_FMUL: case BIR_FDIV:
            is_fadd(idx, I); break;
        case BIR_FREM:
            is_frem(idx, I); break;

        /* ---- Comparison ---- */
        case BIR_ICMP:
            is_icmp(idx, I); break;
        case BIR_FCMP:
            is_fcmp(idx, I); break;

        /* ---- Select ---- */
        case BIR_SELECT:
            is_selp(idx, I); break;

        /* ---- Conversions ---- */
        case BIR_TRUNC: case BIR_ZEXT: case BIR_SEXT:
        case BIR_FPTRUNC: case BIR_FPEXT:
        case BIR_FPTOSI: case BIR_FPTOUI:
        case BIR_SITOFP: case BIR_UITOFP:
        case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST:
            is_cvt(idx, I); break;

        /* ---- Memory ---- */
        case BIR_LOAD:
            is_load(idx, I); break;
        case BIR_STORE:
            is_store(I); break;
        case BIR_ALLOCA:
            is_alloca(idx, I); break;
        case BIR_SHARED_ALLOC:
            is_shralloc(idx, I); break;
        case BIR_GEP:
            is_gep(idx, I); break;

        /* ---- Control Flow ---- */
        case BIR_BR:
            is_br(I); break;
        case BIR_BR_COND:
            is_brcond(I); break;
        case BIR_RET:
            is_ret(I); break;
        case BIR_SWITCH:
        case BIR_UNREACHABLE:
            break;

        /* ---- SSA ---- */
        case BIR_PHI:
            is_phi(idx, I, bir_bi); break;
        case BIR_PARAM:
            is_param(idx, I); break;

        /* ---- Thread Model ---- */
        case BIR_THREAD_ID: case BIR_BLOCK_ID:
        case BIR_BLOCK_DIM: case BIR_GRID_DIM:
            is_thread(idx, I); break;

        /* ---- Barriers ---- */
        case BIR_BARRIER: case BIR_BARRIER_GROUP:
            is_barrier(); break;

        /* ---- Atomics ---- */
        case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
        case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
        case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
        case BIR_ATOMIC_XCHG: case BIR_ATOMIC_CAS:
            is_atomic(idx, I); break;
        case BIR_ATOMIC_LOAD:
            is_atm_load(idx, I); break;
        case BIR_ATOMIC_STORE:
            is_atm_store(I); break;

        /* ---- Warp Ops ---- */
        case BIR_SHFL: case BIR_SHFL_UP:
        case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            is_shfl(idx, I); break;
        case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
            is_vote(idx, I); break;

        /* ---- Math Builtins ---- */
        case BIR_SQRT: case BIR_RSQ: case BIR_RCP:
        case BIR_SIN: case BIR_COS:
        case BIR_EXP2: case BIR_LOG2:
        case BIR_FABS: case BIR_FLOOR: case BIR_CEIL:
        case BIR_FTRUNC: case BIR_RNDNE:
            is_math(idx, I); break;
        case BIR_FMIN: case BIR_FMAX:
            is_fminmax(idx, I); break;

        /* ---- Stubs ---- */
        case BIR_CALL: case BIR_INLINE_ASM:
        case BIR_GLOBAL_REF: case BIR_MFMA:
            break;

        default:
            fprintf(stderr, "[NV-ISEL] WARN: unhandled BIR op %u (%s) at inst %u\n",
                    I->op, bir_op_name(I->op), idx);
            break;
        }
    }
}

/* ---- PHI Copy Insertion ----
 * The exciting part of PHI elimination: actually putting the copies
 * where they belong. Each predecessor block gets its MOV inserted
 * before the terminator. When a conditional branch targets the merge
 * block (the || short-circuit case), we can't insert on the taken
 * path without edge splitting — so we create a bridge block.
 *
 * The flat instruction array makes this feel like performing surgery
 * through a letterbox, but at least we only do it once per function. */

static void phi_ins1(uint32_t ins_pt, nv_minst_t *inst, uint32_t count)
{
    /* Shift everything from ins_pt onwards to make room */
    if (S.nv->num_minst + count > NV_MAX_MINST) return;
    memmove(&S.nv->minsts[ins_pt + count],
            &S.nv->minsts[ins_pt],
            (S.nv->num_minst - ins_pt) * sizeof(nv_minst_t));
    for (uint32_t i = 0; i < count; i++)
        S.nv->minsts[ins_pt + i] = inst[i];
    S.nv->num_minst += count;

    /* Update all blocks whose first_inst is AFTER the insertion point.
     * Blocks containing the insertion point keep their first_inst. */
    for (uint32_t bi = 0; bi < S.nv->num_mblk; bi++) {
        if (S.nv->mblks[bi].first_inst > ins_pt)
            S.nv->mblks[bi].first_inst += count;
    }
}

static nv_minst_t mk_mov(uint16_t mop, nv_opnd_t dst, nv_opnd_t src)
{
    nv_minst_t I;
    memset(&I, 0, sizeof(I));
    I.op = mop;
    I.num_defs = 1;
    I.num_uses = 1;
    I.ops[0] = dst;
    I.ops[1] = src;
    return I;
}

static nv_minst_t mk_setp(nv_opnd_t dst, nv_opnd_t a, nv_opnd_t b)
{
    nv_minst_t I;
    memset(&I, 0, sizeof(I));
    I.op = NV_SETP_NE_U32;
    I.num_defs = 1;
    I.num_uses = 2;
    I.ops[0] = dst;
    I.ops[1] = a;
    I.ops[2] = b;
    return I;
}

/* ---- Emit one PHI copy into a bridge block ----
 * Appends the copy instruction(s) to the bridge block.
 * Called BEFORE the bridge's terminating bra is emitted. */
static void brg_copy(nv_pcopy_t *pc)
{
    if (pc->mop == NV_MOV_PRED && pc->src.kind == NV_MOP_IMM) {
        nv_opnd_t ops[3] = { pc->dst, pc->src, mop_imm(0) };
        emit(NV_SETP_NE_U32, 1, 2, ops, 0);
    } else if (pc->mop == NV_MOV_PRED &&
               pc->src.kind == NV_MOP_REG &&
               pc->src.rfile == NV_RF_PRED) {
        uint16_t rn = new_vreg(NV_RF_U32);
        nv_opnd_t tmp = mop_reg(NV_RF_U32, rn);
        nv_opnd_t s_ops[4] = { tmp, mop_imm(1), mop_imm(0),
                               pc->src };
        emit(NV_SELP_U32, 1, 3, s_ops, 0);
        nv_opnd_t p_ops[3] = { pc->dst, tmp, mop_imm(0) };
        emit(NV_SETP_NE_U32, 1, 2, p_ops, 0);
    } else {
        nv_opnd_t cops[2] = { pc->dst, pc->src };
        emit(pc->mop, 1, 1, cops, 0);
    }
}

static void phi_fix(void)
{
    if (S_npc == 0) return;

    /* ---- PASS 0: Lost-copy repair ----
     * The classic SSA phi-elimination bug: if PHI_B writes vreg X
     * and PHI_A (earlier in the block) reads vreg X, the reverse
     * insertion order means B executes first and clobbers X before
     * A can read it. Fix: redirect A to read from a fresh temp,
     * and insert a save copy (temp = X) before B's write.
     * Like handing someone their own parachute before you
     * pack the shared one. */
    {
        uint32_t n_fix = 0;
        for (uint32_t i = 0; i < S_npc; i++) {
            nv_pcopy_t *ci = &S_pcopy[i];
            if (ci->src.kind != NV_MOP_REG) continue;
            for (uint32_t j = i + 1; j < S_npc; j++) {
                nv_pcopy_t *cj = &S_pcopy[j];
                if (cj->pred_mblk != ci->pred_mblk) continue;
                if (cj->dst.kind != NV_MOP_REG) continue;
                /* j later → executes first → if j.dst == i.src,
                 * j clobbers the value i needs */
                if (cj->dst.rfile == ci->src.rfile &&
                    cj->dst.reg_num == ci->src.reg_num) {
                    /* Allocate temp, rewrite i's source */
                    uint16_t tmp = new_vreg(ci->src.rfile);
                    nv_opnd_t tsrc = mop_reg(ci->src.rfile, tmp);
                    /* Insert save copy: tmp = old_src.
                     * Place it AFTER j (higher index) so it
                     * executes BEFORE j in the reverse pass. */
                    if (S_npc < NV_MAX_PCOPY) {
                        uint16_t sop;
                        switch (ci->src.rfile) {
                        case NV_RF_U64:  sop = NV_MOV_U64;  break;
                        case NV_RF_F32:  sop = NV_MOV_F32;  break;
                        case NV_RF_F64:  sop = NV_MOV_F64;  break;
                        case NV_RF_PRED: sop = NV_MOV_PRED; break;
                        default:         sop = NV_MOV_U32;  break;
                        }
                        S_pcopy[S_npc].pred_mblk  = ci->pred_mblk;
                        S_pcopy[S_npc].merge_mblk = ci->merge_mblk;
                        S_pcopy[S_npc].mop        = sop;
                        S_pcopy[S_npc].dst        = tsrc;
                        S_pcopy[S_npc].src        = ci->src;
                        S_npc++;
                    }
                    /* Rewrite victim to read from temp */
                    ci->src = tsrc;
                    n_fix++;
                    fprintf(stderr,
                        "[PHI] fixed lost copy: pred=BB%u "
                        "copy[%u].src → tmp %%%s%u "
                        "(was clobbered by copy[%u])\n",
                        ci->pred_mblk, i,
                        ci->src.rfile == NV_RF_F32 ? "f" :
                        ci->src.rfile == NV_RF_U64 ? "rd" :
                        ci->src.rfile == NV_RF_PRED ? "p" : "r",
                        tmp, j);
                    break; /* one fix per victim suffices */
                }
            }
        }
        if (n_fix > 0)
            fprintf(stderr, "[PHI] repaired %u lost copies\n", n_fix);
    }

    /* ---- PASS 1: Bridge blocks ----
     * Identify copies that need edge splitting (conditional branch
     * targets the merge block). Group by (pred, merge) and create
     * ONE bridge per group with ALL copies. Previous code created
     * separate bridges per copy, orphaning all but the last one.
     * Like building seven bypass roads and only signposting one. */

    /* Mark which copies need bridges */
    uint8_t need_brg[NV_MAX_PCOPY];
    memset(need_brg, 0, S_npc);

    for (uint32_t pi = 0; pi < S_npc; pi++) {
        nv_pcopy_t *pc = &S_pcopy[pi];
        nv_mblk_t  *PB = &S.nv->mblks[pc->pred_mblk];
        if (PB->num_insts < 2) continue;
        uint32_t tp = PB->first_inst + PB->num_insts - 1;
        uint32_t pp = tp - 1;
        nv_minst_t *prev = &S.nv->minsts[pp];
        nv_minst_t *term = &S.nv->minsts[tp];
        if (prev->op == NV_BRA_PRED && term->op == NV_BRA &&
            prev->ops[1].kind == NV_MOP_LABEL &&
            (uint32_t)prev->ops[1].imm == pc->merge_mblk) {
            need_brg[pi] = 1;
        }
    }

    /* Create one bridge per unique (pred, merge) pair */
    for (uint32_t pi = 0; pi < S_npc; pi++) {
        if (!need_brg[pi]) continue;
        nv_pcopy_t *pc = &S_pcopy[pi];

        /* Check if bridge already created for this (pred, merge) */
        int done = 0;
        for (uint32_t qi = 0; qi < pi; qi++) {
            if (need_brg[qi] &&
                S_pcopy[qi].pred_mblk == pc->pred_mblk &&
                S_pcopy[qi].merge_mblk == pc->merge_mblk) {
                done = 1; break;
            }
        }
        if (done) continue;

        /* First encounter of this (pred, merge) — create bridge */
        if (S.nv->num_mblk >= NV_MAX_MBLK) continue;

        uint32_t bridge_bi = S.nv->num_mblk++;
        nv_mblk_t *BB = &S.nv->mblks[bridge_bi];
        BB->first_inst = S.nv->num_minst;
        BB->bir_block = S.nv->mblks[pc->pred_mblk].bir_block;
        BB->num_insts = 0;

        /* Emit ALL copies for this (pred, merge) group */
        for (uint32_t qi = pi; qi < S_npc; qi++) {
            if (!need_brg[qi]) continue;
            if (S_pcopy[qi].pred_mblk != pc->pred_mblk) continue;
            if (S_pcopy[qi].merge_mblk != pc->merge_mblk) continue;
            brg_copy(&S_pcopy[qi]);
            need_brg[qi] = 2; /* mark as handled */
        }

        /* Terminating bra to merge */
        nv_opnd_t bops[1] = { mop_lbl(pc->merge_mblk) };
        emit(NV_BRA, 0, 1, bops, 0);

        BB->num_insts = S.nv->num_minst - BB->first_inst;

        /* Retarget the conditional branch */
        nv_mblk_t *PB = &S.nv->mblks[pc->pred_mblk];
        uint32_t cp = PB->first_inst + PB->num_insts - 2;
        S.nv->minsts[cp].ops[1].imm = (int32_t)bridge_bi;
    }

    /* ---- PASS 2: Standard copies (non-bridge) ----
     * Insert before terminators, processed in reverse to keep
     * insertion points valid. */
    for (int pi = (int)S_npc - 1; pi >= 0; pi--) {
        if (need_brg[pi]) continue; /* already handled in pass 1 */

        nv_pcopy_t *pc = &S_pcopy[pi];
        nv_mblk_t  *PB = &S.nv->mblks[pc->pred_mblk];
        if (PB->num_insts == 0) continue;

        uint32_t term_pos = PB->first_inst + PB->num_insts - 1;

        nv_minst_t insts[2];
        uint32_t count = 0;

        if (pc->mop == NV_MOV_PRED && pc->src.kind == NV_MOP_IMM) {
            insts[0] = mk_setp(pc->dst, pc->src, mop_imm(0));
            count = 1;
        } else if (pc->mop == NV_MOV_PRED &&
                   pc->src.kind == NV_MOP_REG &&
                   pc->src.rfile == NV_RF_PRED) {
            uint16_t rn = new_vreg(NV_RF_U32);
            nv_opnd_t tmp = mop_reg(NV_RF_U32, rn);
            memset(&insts[0], 0, sizeof(insts[0]));
            insts[0].op = NV_SELP_U32;
            insts[0].num_defs = 1;
            insts[0].num_uses = 3;
            insts[0].ops[0] = tmp;
            insts[0].ops[1] = mop_imm(1);
            insts[0].ops[2] = mop_imm(0);
            insts[0].ops[3] = pc->src;
            insts[1] = mk_setp(pc->dst, tmp, mop_imm(0));
            count = 2;
        } else {
            insts[0] = mk_mov(pc->mop, pc->dst, pc->src);
            count = 1;
        }

        phi_ins1(term_pos, insts, count);
        PB->num_insts += count;
    }

    S_npc = 0;
}

/* ---- Per-Function Setup ---- */

static int isel_func(uint32_t fi)
{
    const bir_func_t *F = &S.bir->funcs[fi];

    if (!(F->cuda_flags & (CUDA_GLOBAL | CUDA_DEVICE))) return BC_OK;
    if (S.nv->num_mfunc >= NV_MAX_MFUNC) return BC_ERR_NVIDIA;

    uint32_t mfi = S.nv->num_mfunc++;
    nv_mfunc_t *MF = &S.nv->mfuncs[mfi];

    MF->name = F->name;
    MF->first_blk = S.nv->num_mblk;
    MF->num_blks = 0;
    MF->is_kern = (F->cuda_flags & CUDA_GLOBAL) ? 1 : 0;
    MF->num_params = F->num_params;
    MF->lds_bytes = 0;
    MF->launch_max = F->launch_bounds_max;
    MF->launch_min = F->launch_bounds_min;
    MF->bir_func = fi;
    memset(MF->rc, 0, sizeof(MF->rc));
    memset(MF->params, 0, sizeof(MF->params));

    S.cur_func = mfi;
    S.lcl_off = 0;
    S.shr_off = 0;

    /* Reset per-function vreg counters */
    memset(S.nv->rc, 0, sizeof(S.nv->rc));
    /* Start at 1 so vreg 0 is sentinel */
    for (int r = 0; r < NV_RF_COUNT; r++)
        S.nv->rc[r] = 1;

    /* Build parameter descriptors from BIR */
    uint32_t first_block = F->first_block;
    if (first_block < S.bir->num_blocks) {
        const bir_block_t *entry = &S.bir->blocks[first_block];
        uint32_t pi = 0;
        int pg = 64;
        for (uint32_t ii = 0; ii < entry->num_insts && pg > 0; ii++, pg--) {
            const bir_inst_t *I = &S.bir->insts[entry->first_inst + ii];
            if (I->op != BIR_PARAM) continue;
            if (pi >= NV_MAX_PARAMS) break;
            MF->params[pi].rfile = bir_rfile(I->type);
            pi++;
        }
        MF->num_params = pi;
    }

    /* Pre-create block map */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        S.block_map[bir_bi] = S.nv->num_mblk + bi;
    }

    /* Select instructions per block */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        if (S.nv->num_mblk >= NV_MAX_MBLK) break;

        uint32_t mbi = S.nv->num_mblk;
        nv_mblk_t *MB = &S.nv->mblks[mbi];
        MB->first_inst = S.nv->num_minst;
        MB->bir_block = bir_bi;

        isel_blk(bir_bi);

        MB->num_insts = S.nv->num_minst - MB->first_inst;
        S.nv->num_mblk++;
    }

    /* Insert deferred PHI copies into predecessor blocks.
     * Must happen before register counts are recorded since
     * pred-to-pred copies allocate temporary U32 registers. */
    phi_fix();

    MF->num_blks = (uint16_t)(S.nv->num_mblk - MF->first_blk);
    MF->lds_bytes = S.shr_off;
    MF->lcl_bytes = S.lcl_off;

    /* Record per-rfile high-water marks */
    for (int r = 0; r < NV_RF_COUNT; r++)
        MF->rc[r] = S.nv->rc[r];

    return BC_OK;
}

/* ---- Public API ---- */

int nv_compile(const bir_module_t *bir, nv_module_t *nv)
{
    memset(&S, 0, sizeof(S));
    S.nv = nv;
    S.bir = bir;

    nv->bir = bir;
    nv->num_minst = 0;
    nv->num_mblk = 0;
    nv->num_mfunc = 0;
    nv->out_len = 0;

    memset(nv->rc, 0, sizeof(nv->rc));
    memset(nv->val_vreg, 0, sizeof(nv->val_vreg));
    memset(nv->val_rfile, 0, sizeof(nv->val_rfile));

    int guard = 8192;
    for (uint32_t fi = 0; fi < bir->num_funcs && guard > 0; fi++, guard--) {
        int rc = isel_func(fi);
        if (rc != BC_OK) return rc;
    }

    return BC_OK;
}
