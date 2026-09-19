#include "nvidia.h"
#include "backend.h"
#include <string.h>

/* BIR SSA -> PTX pseudo-MIR. Virtual registers, text output, no binary
 * encoding. The NVIDIA driver does the heavy lifting — register allocation,
 * scheduling, all the hard bits. We just emit polite suggestions in PTX
 * and hope for the best. Like posting a letter to a corporation. */

static struct {
    nv_module_t        *nv;
    const bir_module_t *bir;
    uint32_t            block_map[BIR_MAX_BLOCKS];
    uint32_t            blk_lst[BIR_MAX_BLOCKS];
    uint32_t            cmb;
    uint32_t            lcl_off;   /* local (alloca) byte offset */
    uint32_t            shr_off;   /* shared memory byte offset */
    uint32_t            lcl_alg;
    uint32_t            shr_alg;
    uint32_t            cur_func;  /* current nv_mfunc_t index */
    uint32_t            blk_lo;
    uint32_t            blk_hi;
    uint8_t             fneed[BIR_MAX_FUNCS];
    int                 unsz;      /* a type with no storage size was met */
    int                 had_error; /* an op we refuse to fake; fail the compile */
    int                 capr;      /* a fixed capacity already reported */
    int                 bscr;      /* a stray branch already reported */
} S;

static void nv_refuse(const char *what)
{
    (void)be_fail(BC_E541, "nvptx", what);
    S.had_error = 1;
}

/* Forward declaration — rslv needs em1 for f64 constant materialisation */
static void em1(uint16_t op, nv_opnd_t d, nv_opnd_t a, nv_opnd_t b);
static nv_opnd_t mat_pred(nv_opnd_t op);
static uint16_t movof(uint8_t rf);

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

static int pagg(uint32_t ty);
static uint32_t vtyof(uint32_t val);

/* BIR type -> PTX register file */
static uint8_t bir_rfile(uint32_t type_idx)
{
    if (type_idx >= S.bir->num_types) return NV_RF_U32;
    const bir_type_t *T = &S.bir->types[type_idx];

    switch (T->kind) {
    case BIR_TYPE_INT:
        if (T->width <= 1)  return NV_RF_U32;
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

static int psinf(nv_param_t *P, uint32_t ty)
{
    uint32_t z, a;

    P->rfile = bir_rfile(ty);
    P->size = 0u;
    P->algn = 0u;
    if (!pagg(ty)) return 1;

    z = bir_bsz(S.bir, ty, 8);
    a = bir_balg(S.bir, ty, 8);
    if (z == 0u || z > 0xFFFFu || a == 0u || a > 16u) return 0;
    P->size = (uint16_t)z;
    P->algn = (uint8_t)a;
    return 1;
}

static uint8_t def_rf(uint32_t idx, uint32_t type_idx)
{
    if (idx < S.bir->num_insts) {
        uint16_t o = S.bir->insts[idx].op;
        if (o == BIR_ICMP || o == BIR_FCMP
         || o == BIR_VOTE_ANY || o == BIR_VOTE_ALL
         || (o == BIR_BARRED && S.bir->insts[idx].subop != 2))
            return NV_RF_PRED;
    }
    return bir_rfile(type_idx);
}

/* Map a BIR instruction to a vreg, creating one if needed */
static nv_opnd_t map_val(uint32_t idx, uint32_t type_idx)
{
    if (idx >= BIR_MAX_INSTS) return mop_reg(NV_RF_U32, 0);
    if (S.nv->val_vreg[idx] != 0)
        return mop_reg(S.nv->val_rfile[idx], S.nv->val_vreg[idx]);

    uint8_t rf = def_rf(idx, type_idx);
    uint16_t rn = new_vreg(rf);
    S.nv->val_vreg[idx] = rn;
    S.nv->val_rfile[idx] = rf;
    return mop_reg(rf, rn);
}

static nv_opnd_t rslv_p(uint32_t val)
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
                 * Can't fit 64 bits in a 32-bit imm field, so we
                 * emit a pseudo-op
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

static nv_opnd_t rslv(uint32_t val)
{
    return mat_pred(rslv_p(val));
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
        uint8_t rf = (S.nv->val_rfile[si] != 0 || S.nv->val_vreg[si] != 0)
                     ? S.nv->val_rfile[si]
                     : def_rf(si, S.bir->insts[si].type);
        return rf == NV_RF_PRED ? NV_RF_U32 : rf;
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

static uint32_t type_bytes(uint32_t type_idx)
{
    return bir_bsz(S.bir, type_idx, 8);
}

static uint32_t pntsz(uint32_t ptr_val)
{
    uint32_t si, pt;
    if (ptr_val == BIR_VAL_NONE || BIR_VAL_IS_CONST(ptr_val)) return 0;
    si = BIR_VAL_INDEX(ptr_val);
    if (si >= S.bir->num_insts) return 0;
    pt = S.bir->insts[si].type;
    return bir_gstr(S.bir, pt, 8);
}

static uint32_t algnb(uint8_t subop)
{
    uint32_t k = subop & 31u;

    return (k > 12u) ? (1u << 12) : (1u << k);
}

static nv_opnd_t mapu64(uint32_t idx)
{
    uint16_t rn;

    if (idx >= BIR_MAX_INSTS) return mop_reg(NV_RF_U64, 0);
    if (S.nv->val_vreg[idx] != 0)
        return mop_reg(S.nv->val_rfile[idx], S.nv->val_vreg[idx]);
    rn = new_vreg(NV_RF_U64);
    S.nv->val_vreg[idx] = rn;
    S.nv->val_rfile[idx] = NV_RF_U64;
    return mop_reg(NV_RF_U64, rn);
}

static void unsz(const char *what, uint32_t ty)
{
    char buf[128];
    if (bir_type_str(S.bir, ty, buf, (int)sizeof buf) <= 0) buf[0] = 0;
    (void)be_fail(BC_E540, "nvptx", what, buf);
    S.unsz = 1;
}

/* ---- Emission ---- */

static void nv_cap(const char *what, unsigned lim)
{
    S.had_error = 1;
    if (S.capr) return;
    S.capr = 1;
    (void)be_fail(BC_E545, "nvptx", what, lim);
}

static uint32_t emit(uint16_t op, uint8_t nd, uint8_t nu,
                     const nv_opnd_t *ops, uint16_t flags)
{
    if (S.nv->num_minst >= NV_MAX_MINST) {
        nv_cap("machine instructions", (unsigned)NV_MAX_MINST);
        return 0;
    }
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

    em1u(movof(rf), dst, src);
    return dst;
}

static uint8_t rfwid(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U16: case NV_RF_F16:  return 16;
    case NV_RF_U64: case NV_RF_F64:  return 64;
    default:                         return 32;
    }
}

static int rfflt(uint8_t rf)
{
    return rf == NV_RF_F16 || rf == NV_RF_F32 || rf == NV_RF_F64;
}

static uint16_t cvop(uint8_t from, uint8_t to, int sgn)
{
    if (from == to) return NV_OP_COUNT;
    switch (from) {
    case NV_RF_U16:
        if (to == NV_RF_U32) return sgn ? NV_CVT_S32_S16 : NV_CVT_U32_U16;
        break;
    case NV_RF_U32: case NV_RF_B32:
        if (to == NV_RF_U16) return NV_CVT_U16_U32;
        if (to == NV_RF_U64) return sgn ? NV_CVT_S64_S32 : NV_CVT_U64_U32;
        if (to == NV_RF_F32) return sgn ? NV_CVT_F32_S32 : NV_CVT_F32_U32;
        if (to == NV_RF_F64) return sgn ? NV_CVT_F64_S32 : NV_CVT_F64_U32;
        break;
    case NV_RF_U64:
        if (to == NV_RF_U32) return NV_CVT_U32_U64;
        if (to == NV_RF_F32) return sgn ? NV_CVT_F32_S64 : NV_CVT_F32_U64;
        if (to == NV_RF_F64) return sgn ? NV_CVT_F64_S64 : NV_CVT_F64_U64;
        break;
    case NV_RF_F16:
        if (to == NV_RF_F32) return NV_CVT_F32_F16;
        if (to == NV_RF_F64) return NV_CVT_F64_F16;
        break;
    case NV_RF_F32:
        if (to == NV_RF_F16) return NV_CVT_F16_F32;
        if (to == NV_RF_F64) return NV_CVT_F64_F32;
        if (to == NV_RF_U32) return sgn ? NV_CVT_S32_F32 : NV_CVT_U32_F32;
        if (to == NV_RF_U64) return sgn ? NV_CVT_S64_F32 : NV_CVT_U64_F32;
        break;
    case NV_RF_F64:
        if (to == NV_RF_F16) return NV_CVT_F16_F64;
        if (to == NV_RF_F32) return NV_CVT_F32_F64;
        if (to == NV_RF_U32) return sgn ? NV_CVT_S32_F64 : NV_CVT_U32_F64;
        if (to == NV_RF_U64) return sgn ? NV_CVT_S64_F64 : NV_CVT_U64_F64;
        break;
    default:
        break;
    }
    return NV_OP_COUNT;
}

static uint8_t tykind(uint32_t ty)
{
    return (ty < S.bir->num_types) ? S.bir->types[ty].kind : 0;
}

static uint8_t vkind(uint32_t val)
{
    uint32_t i = BIR_VAL_INDEX(val);
    if (val == BIR_VAL_NONE) return 0;
    if (BIR_VAL_IS_CONST(val))
        return (i < S.bir->num_consts)
             ? tykind(S.bir->consts[i].type) : 0;
    return (i < S.bir->num_insts) ? tykind(S.bir->insts[i].type) : 0;
}

static const char *curfn(void)
{
    uint32_t off;
    if (S.cur_func >= S.nv->num_mfunc) return "an unnamed function";
    off = S.nv->mfuncs[S.cur_func].name;
    return (off < S.bir->string_len) ? S.bir->strings + off
                                     : "an unnamed function";
}

static const char *rfnam(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U32:  return "u32";
    case NV_RF_U64:  return "u64";
    case NV_RF_F32:  return "f32";
    case NV_RF_F64:  return "f64";
    case NV_RF_PRED: return "pred";
    case NV_RF_U16:  return "u16";
    case NV_RF_F16:  return "f16";
    case NV_RF_B32:  return "b32";
    default:         return "unknown";
    }
}

static uint8_t promo(uint8_t rf)
{
    return (rf == NV_RF_U16) ? NV_RF_U32
         : (rf == NV_RF_F16) ? NV_RF_F32 : rf;
}

static uint16_t movof(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U64:  return NV_MOV_U64;
    case NV_RF_F32:  return NV_MOV_F32;
    case NV_RF_F64:  return NV_MOV_F64;
    case NV_RF_PRED: return NV_MOV_PRED;
    case NV_RF_U16:  return NV_MOV_B16;
    case NV_RF_F16:  return NV_MOV_B16;
    case NV_RF_B32:  return NV_MOV_B32;
    default:         return NV_MOV_U32;
    }
}

static void cocpy(nv_opnd_t d, nv_opnd_t s, int sgn)
{
    uint16_t op;
    if (s.kind != NV_MOP_REG || s.rfile == d.rfile) {
        em1u(movof(d.rfile), d, s);
        return;
    }
    op = cvop(s.rfile, d.rfile, sgn);
    if (op == NV_OP_COUNT) {
        (void)be_fail(BC_E844, curfn(), rfnam(s.rfile), rfnam(d.rfile));
        S.had_error = 1;
        em1u(movof(d.rfile), d, s);
        return;
    }
    em1u(op, d, s);
}

static nv_opnd_t cofit(nv_opnd_t o, uint8_t want, int sgn)
{
    nv_opnd_t d;
    int guard;

    if (o.kind != NV_MOP_REG || o.rfile == want) return o;
    for (guard = 0; guard < 3 && o.rfile != want; guard++) {
        uint8_t step = want;
        if (cvop(o.rfile, want, sgn) == NV_OP_COUNT) {
            if (o.rfile == NV_RF_U16)      step = NV_RF_U32;
            else if (o.rfile == NV_RF_F16) step = NV_RF_F32;
            else if (want == NV_RF_U16)    step = NV_RF_U32;
            else if (want == NV_RF_F16)    step = NV_RF_F32;
            else break;
            if (step == o.rfile) break;
        }
        d = mop_reg(step, new_vreg(step));
        cocpy(d, o, sgn);
        o = d;
    }
    if (o.rfile == want) return o;
    d = mop_reg(want, new_vreg(want));
    cocpy(d, o, sgn);
    return d;
}

static nv_opnd_t cobit(nv_opnd_t o, uint8_t want)
{
    nv_opnd_t d;
    if (o.kind != NV_MOP_REG || o.rfile == want) return o;
    if (rfwid(o.rfile) != rfwid(want)) return cofit(o, want, 0);
    d = mop_reg(want, new_vreg(want));
    em1u(rfwid(want) == 64 ? NV_MOV_B64
       : rfwid(want) == 16 ? NV_MOV_B16 : NV_MOV_B32, d, o);
    return d;
}

/* ---- Integer Arithmetic ---- */

static void is_iarith(uint32_t idx, const bir_inst_t *I, uint16_t o32,
                      uint16_t o64, int sgn)
{
    uint8_t rf;
    nv_opnd_t d0 = map_val(idx, I->type);
    nv_opnd_t d, a, b;

    rf = promo(d0.rfile == NV_RF_PRED ? bir_rfile(I->type) : d0.rfile);
    d = (d0.rfile == rf) ? d0 : mop_reg(rf, new_vreg(rf));
    a = cofit(rslv(I->operands[0]), rf, sgn);
    b = cofit(rslv(I->operands[1]), rf, sgn);
    em1((rf == NV_RF_U64) ? o64 : o32, d, a, b);
    if (d0.rfile != rf) cocpy(d0, d, sgn);
}

static void is_iadd(uint32_t idx, const bir_inst_t *I)
{
    is_iarith(idx, I, NV_ADD_U32, NV_ADD_U64, 1);
}

static void is_isub(uint32_t idx, const bir_inst_t *I)
{
    is_iarith(idx, I, NV_SUB_U32, NV_SUB_S64, 1);
}

static void is_imul(uint32_t idx, const bir_inst_t *I)
{
    is_iarith(idx, I, NV_MUL_LO_U32, NV_MUL_LO_U64, 1);
}

/* High half of a wide product, the workhorse of multi-limb field arithmetic.
 * PTX has mul.hi natively at both widths, so this is a straight lowering,
 * shalalalala. */
static void is_umulhi(uint32_t idx, const bir_inst_t *I)
{
    is_iarith(idx, I, NV_MUL_HI_U32, NV_MUL_HI_U64, 0);
}

/* Native PTX at both widths, except ctz, which is brev then clz. Reversing
 * sends the trailing zeros to the top, and clz answers 32 for zero. */
static void is_bitcount(uint32_t idx, const bir_inst_t *I)
{
    int w64 = (rslv_rf(I->operands[0]) == NV_RF_U64);
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t a = rslv(I->operands[0]);

    if (I->op == BIR_POPCOUNT) {
        em1u(w64 ? NV_POPC_B64 : NV_POPC_B32, d,
             cofit(a, w64 ? NV_RF_U64 : NV_RF_U32, 0));
    } else if (I->op == BIR_CLZ) {
        em1u(w64 ? NV_CLZ_B64 : NV_CLZ_B32, d, a);
    } else if (I->op == BIR_BREV) {
        em1u(w64 ? NV_BREV_B64 : NV_BREV_B32, d, a);
    } else {
        uint8_t rf = w64 ? NV_RF_U64 : NV_RF_U32;
        nv_opnd_t t = mop_reg(rf, new_vreg(rf));
        em1u(w64 ? NV_BREV_B64 : NV_BREV_B32, t, a);
        em1u(w64 ? NV_CLZ_B64 : NV_CLZ_B32, d, t);
    }
}

static void is_idiv(uint32_t idx, const bir_inst_t *I)
{
    int sgn = (I->op == BIR_SDIV);
    is_iarith(idx, I, sgn ? NV_DIV_S32 : NV_DIV_U32,
              sgn ? NV_DIV_S64 : NV_DIV_U64, sgn);
}

static void is_irem(uint32_t idx, const bir_inst_t *I)
{
    int sgn = (I->op == BIR_SREM);
    is_iarith(idx, I, sgn ? NV_REM_S32 : NV_REM_U32,
              sgn ? NV_REM_S64 : NV_REM_U64, sgn);
}

/* ---- Bitwise / Shift ---- */

static void is_bitop(uint32_t idx, const bir_inst_t *I)
{
    uint8_t rf = bir_rfile(I->type);
    nv_opnd_t d0 = map_val(idx, I->type);
    int sgn = (I->op == BIR_ASHR);
    nv_opnd_t d, a, b;

    if (rf == NV_RF_U16) rf = NV_RF_U32;
    d = (d0.rfile == rf) ? d0 : mop_reg(rf, new_vreg(rf));
    a = (d0.rfile == rf) ? cobit(rslv(I->operands[0]), rf)
                         : cofit(rslv(I->operands[0]), rf, sgn);
    b = rslv(I->operands[1]);

    uint16_t op;
    int is64 = (rf == NV_RF_U64);
    int shft = (I->op == BIR_SHL || I->op == BIR_LSHR
             || I->op == BIR_ASHR);
    b = shft ? cofit(b, NV_RF_U32, 0)
             : ((d0.rfile == rf) ? cobit(b, rf) : cofit(b, rf, sgn));
    switch (I->op) {
    case BIR_AND:  op = is64 ? NV_AND_B64 : NV_AND_B32; break;
    case BIR_OR:   op = is64 ? NV_OR_B64  : NV_OR_B32;  break;
    case BIR_XOR:  op = is64 ? NV_XOR_B64 : NV_XOR_B32; break;
    case BIR_SHL:  op = is64 ? NV_SHL_B64 : NV_SHL_B32; break;
    case BIR_LSHR: op = is64 ? NV_SHR_U64 : NV_SHR_U32; break;
    case BIR_ASHR: op = is64 ? NV_SHR_S64 : NV_SHR_S32; break;
    default: return;
    }
    em1(op, d, a, b);
    if (d0.rfile != rf) cocpy(d0, d, 0);
}

/* ---- FP Arithmetic ---- */

static void is_fadd(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d0 = map_val(idx, I->type);
    uint8_t rf = promo(d0.rfile == NV_RF_PRED
                     ? bir_rfile(I->type) : d0.rfile);
    nv_opnd_t d = (d0.rfile == rf) ? d0 : mop_reg(rf, new_vreg(rf));
    nv_opnd_t a = cofit(rslv(I->operands[0]), rf, 1);
    nv_opnd_t b = cofit(rslv(I->operands[1]), rf, 1);

    uint16_t op;
    switch (I->op) {
    case BIR_FADD: op = (rf == NV_RF_F64) ? NV_ADD_F64 : NV_ADD_F32; break;
    case BIR_FSUB: op = (rf == NV_RF_F64) ? NV_SUB_F64 : NV_SUB_F32; break;
    case BIR_FMUL: op = (rf == NV_RF_F64) ? NV_MUL_F64 : NV_MUL_F32; break;
    case BIR_FDIV: op = (rf == NV_RF_F64) ? NV_DIV_F64 : NV_DIV_F32; break;
    default: return;
    }
    em1(op, d, a, b);
    if (d0.rfile != rf) cocpy(d0, d, 1);
}

static void is_frem(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d0 = map_val(idx, I->type);
    uint8_t rf = promo(d0.rfile == NV_RF_PRED
                     ? bir_rfile(I->type) : d0.rfile);
    nv_opnd_t d = (d0.rfile == rf) ? d0 : mop_reg(rf, new_vreg(rf));
    nv_opnd_t a = cofit(rslv(I->operands[0]), rf, 1);
    nv_opnd_t b = cofit(rslv(I->operands[1]), rf, 1);

    uint16_t dop = (rf == NV_RF_F64) ? NV_DIV_F64 : NV_DIV_F32;
    uint16_t mop = (rf == NV_RF_F64) ? NV_MUL_F64 : NV_MUL_F32;
    uint16_t sop = (rf == NV_RF_F64) ? NV_SUB_F64 : NV_SUB_F32;

    nv_opnd_t t1 = mop_reg(rf, new_vreg(rf));
    nv_opnd_t t2 = mop_reg(rf, new_vreg(rf));
    em1(dop, t1, a, b);     /* t1 = a / b */
    em1(mop, t2, t1, b);    /* t2 = t1 * b (trunc skipped for approx) */
    em1(sop, d, a, t2);     /* d  = a - t2 */
    if (d0.rfile != rf) cocpy(d0, d, 1);
}

/* ---- Comparison ---- */

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
    uint8_t r0 = rslv_rf(I->operands[0]);
    uint8_t r1 = rslv_rf(I->operands[1]);
    int sgn = (I->subop >= BIR_ICMP_SLT && I->subop <= BIR_ICMP_SGE);
    int w64 = (r0 == NV_RF_U64 || r1 == NV_RF_U64);
    uint8_t rf = w64 ? NV_RF_U64 : NV_RF_U32;
    nv_opnd_t a = cofit(rslv(I->operands[0]), rf, sgn);
    nv_opnd_t b = cofit(rslv(I->operands[1]), rf, sgn);

    uint16_t op;
    switch (I->subop) {
    case BIR_ICMP_EQ:  op = w64 ? NV_SETP_EQ_U64 : NV_SETP_EQ_U32; break;
    case BIR_ICMP_NE:  op = w64 ? NV_SETP_NE_U64 : NV_SETP_NE_U32; break;
    case BIR_ICMP_ULT: op = w64 ? NV_SETP_LT_U64 : NV_SETP_LT_U32; break;
    case BIR_ICMP_ULE: op = w64 ? NV_SETP_LE_U64 : NV_SETP_LE_U32; break;
    case BIR_ICMP_UGT: op = w64 ? NV_SETP_GT_U64 : NV_SETP_GT_U32; break;
    case BIR_ICMP_UGE: op = w64 ? NV_SETP_GE_U64 : NV_SETP_GE_U32; break;
    case BIR_ICMP_SLT: op = w64 ? NV_SETP_LT_S64 : NV_SETP_LT_S32; break;
    case BIR_ICMP_SLE: op = w64 ? NV_SETP_LE_S64 : NV_SETP_LE_S32; break;
    case BIR_ICMP_SGT: op = w64 ? NV_SETP_GT_S64 : NV_SETP_GT_S32; break;
    case BIR_ICMP_SGE: op = w64 ? NV_SETP_GE_S64 : NV_SETP_GE_S32; break;
    default:           op = w64 ? NV_SETP_NE_U64 : NV_SETP_NE_U32; break;
    }
    em1(op, d, a, b);
}

static void is_fcmp(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint8_t r0 = rslv_rf(I->operands[0]);
    uint8_t r1 = rslv_rf(I->operands[1]);
    uint8_t rf = (r0 == NV_RF_F64 || r1 == NV_RF_F64)
               ? NV_RF_F64 : NV_RF_F32;
    nv_opnd_t a = cofit(rslv(I->operands[0]), rf, 1);
    nv_opnd_t b = cofit(rslv(I->operands[1]), rf, 1);

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

static nv_opnd_t topred(nv_opnd_t c)
{
    nv_opnd_t p = mop_reg(NV_RF_PRED, new_vreg(NV_RF_PRED));
    if (c.kind == NV_MOP_REG && c.rfile == NV_RF_PRED) return c;
    if (c.kind == NV_MOP_REG && rfflt(c.rfile))
        c = cobit(c, rfwid(c.rfile) == 64 ? NV_RF_U64 : NV_RF_U32);
    if (c.kind == NV_MOP_REG && c.rfile == NV_RF_U16)
        c = cofit(c, NV_RF_U32, 0);
    em1((c.kind == NV_MOP_REG && c.rfile == NV_RF_U64)
        ? NV_SETP_NE_U64 : NV_SETP_NE_U32, p, c, mop_imm(0));
    return p;
}

static void is_selp(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d0 = map_val(idx, I->type);
    uint8_t rf = promo(d0.rfile == NV_RF_PRED
                     ? bir_rfile(I->type) : d0.rfile);
    nv_opnd_t d = (d0.rfile == rf) ? d0 : mop_reg(rf, new_vreg(rf));
    nv_opnd_t cond = rslv_p(I->operands[0]);
    nv_opnd_t tv   = cofit(rslv(I->operands[1]), rf, 1);
    nv_opnd_t fv   = cofit(rslv(I->operands[2]), rf, 1);

    if (cond.kind != NV_MOP_REG || cond.rfile != NV_RF_PRED)
        cond = topred(cond);

    uint16_t op;
    switch (rf) {
    case NV_RF_U64:  op = NV_SELP_U64; break;
    case NV_RF_F32:  op = NV_SELP_F32; break;
    case NV_RF_F64:  op = NV_SELP_F64; break;
    default:         op = NV_SELP_U32; break;
    }

    {
        nv_opnd_t ops[4] = { d, tv, fv, cond };
        emit(op, 1, 3, ops, 0);
    }
    if (d0.rfile != rf) cocpy(d0, d, 1);
}

/* ---- Conversions ---- */

static void is_f2i(nv_opnd_t d, nv_opnd_t s, uint32_t v0, int sgn)
{
    uint8_t drf = d.rfile;
    uint8_t srf = rslv_rf(v0);
    uint16_t op;
    nv_opnd_t t;

    if (vkind(v0) == BIR_TYPE_BFLOAT) {
        t = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
        em1u(NV_CVT_F32_BF16, t, s);
        s = t;
        srf = NV_RF_F32;
    }
    if (srf != NV_RF_F32 && srf != NV_RF_F64) {
        s = cofit(s, NV_RF_F32, 1);
        srf = NV_RF_F32;
    }
    if (drf == NV_RF_U64)
        op = (srf == NV_RF_F64) ? (sgn ? NV_CVT_S64_F64 : NV_CVT_U64_F64)
                                : (sgn ? NV_CVT_S64_F32 : NV_CVT_U64_F32);
    else
        op = (srf == NV_RF_F64) ? (sgn ? NV_CVT_S32_F64 : NV_CVT_U32_F64)
                                : (sgn ? NV_CVT_S32_F32 : NV_CVT_U32_F32);
    if (drf == NV_RF_U32 || drf == NV_RF_U64) { em1u(op, d, s); return; }
    t = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
    em1u(op, t, s);
    cocpy(d, t, sgn);
}

static void is_i2f(nv_opnd_t d, nv_opnd_t s, uint32_t v0, int sgn)
{
    uint8_t drf = d.rfile;
    uint8_t srf = rslv_rf(v0);
    uint16_t op;
    nv_opnd_t t;

    if (srf != NV_RF_U32 && srf != NV_RF_U64 && srf != NV_RF_B32) {
        s = cofit(s, NV_RF_U32, sgn);
        srf = NV_RF_U32;
    }
    if (drf == NV_RF_F64)
        op = (srf == NV_RF_U64) ? (sgn ? NV_CVT_F64_S64 : NV_CVT_F64_U64)
                                : (sgn ? NV_CVT_F64_S32 : NV_CVT_F64_U32);
    else
        op = (srf == NV_RF_U64) ? (sgn ? NV_CVT_F32_S64 : NV_CVT_F32_U64)
                                : (sgn ? NV_CVT_F32_S32 : NV_CVT_F32_U32);
    if (drf == NV_RF_F32 || drf == NV_RF_F64) { em1u(op, d, s); return; }
    t = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
    em1u(op, t, s);
    cocpy(d, t, 1);
}

static void is_fext(nv_opnd_t d, nv_opnd_t s, uint32_t v0, uint32_t dty)
{
    uint8_t srf = rslv_rf(v0);

    if (vkind(v0) == BIR_TYPE_BFLOAT) {
        nv_opnd_t t = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
        em1u(NV_CVT_F32_BF16, t, s);
        s = t;
        srf = NV_RF_F32;
    }
    if (tykind(dty) == BIR_TYPE_BFLOAT) {
        em1u(NV_CVT_BF16_F32, d, cofit(s, NV_RF_F32, 1));
        return;
    }
    cocpy(d, cofit(s, srf, 1), 1);
}

static void is_cast(nv_opnd_t d, nv_opnd_t s, uint32_t v0)
{
    uint8_t srf = rslv_rf(v0);

    if (s.kind != NV_MOP_REG) { em1u(movof(d.rfile), d, s); return; }
    if (rfwid(srf) == rfwid(d.rfile)) {
        em1u(srf == d.rfile ? movof(d.rfile)
           : rfwid(d.rfile) == 64 ? NV_MOV_B64
           : rfwid(d.rfile) == 16 ? NV_MOV_B16 : NV_MOV_B32, d, s);
        return;
    }
    cocpy(d, cobit(s, rfflt(d.rfile) ? d.rfile
                    : rfwid(srf) == 64 ? NV_RF_U64
                    : rfwid(srf) == 16 ? NV_RF_U16 : NV_RF_U32), 0);
}

static void is_cvt(uint32_t idx, const bir_inst_t *I)
{
    uint32_t v0 = I->operands[0];
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t s = mat_const(v0, promo(rslv_rf(v0)));

    switch (I->op) {
    case BIR_FPTOSI: case BIR_FPTOUI:
        is_f2i(d, s, v0, I->op == BIR_FPTOSI);
        return;
    case BIR_SITOFP: case BIR_UITOFP:
        is_i2f(d, s, v0, I->op == BIR_SITOFP);
        return;
    case BIR_FPTRUNC: case BIR_FPEXT:
        is_fext(d, s, v0, I->type);
        return;
    case BIR_ZEXT: case BIR_SEXT: case BIR_TRUNC:
        cocpy(d, s, I->op == BIR_SEXT);
        return;
    default:
        is_cast(d, s, v0);
        return;
    }
}

/* ---- Memory: Load ---- */

/* Get address space from pointer type */
static int pas(uint32_t ptr_val)
{
    if (ptr_val != BIR_VAL_NONE && !BIR_VAL_IS_CONST(ptr_val)) {
        uint32_t si = BIR_VAL_INDEX(ptr_val);
        if (si < S.bir->num_insts) {
            uint32_t pt = S.bir->insts[si].type;
            if (pt < S.bir->num_types &&
                S.bir->types[pt].kind == BIR_TYPE_PTR)
                return S.bir->types[pt].addrspace;
        }
    }
    return BIR_AS_GLOBAL;
}

static uint16_t ldop(int as, uint8_t drf, uint32_t psz)
{
    switch (as) {
    case BIR_AS_SHARED:
        return (psz == 1)         ? NV_LD_SHR_U8  :
               (drf == NV_RF_F16) ? NV_LD_SHR_B16 :
               (drf == NV_RF_U16) ? NV_LD_SHR_U16 :
               (drf == NV_RF_F64) ? NV_LD_SHR_F64 :
               (drf == NV_RF_U64) ? NV_LD_SHR_U64 :
               (drf == NV_RF_F32) ? NV_LD_SHR_F32 : NV_LD_SHR_U32;
    case BIR_AS_PRIVATE:
        return (psz == 1)         ? NV_LD_LOC_U8  :
               (drf == NV_RF_F16) ? NV_LD_LOC_B16 :
               (drf == NV_RF_U16) ? NV_LD_LOC_U16 :
               (drf == NV_RF_F64) ? NV_LD_LOC_F64 :
               (drf == NV_RF_F32) ? NV_LD_LOC_F32 :
               (drf == NV_RF_U64) ? NV_LD_LOC_U64 : NV_LD_LOC_U32;
    default:
        if (psz == 1) return NV_LD_GLB_U8;
        switch (drf) {
        case NV_RF_F32:  return NV_LD_GLB_F32;
        case NV_RF_F64:  return NV_LD_GLB_F64;
        case NV_RF_U64:  return NV_LD_GLB_U64;
        case NV_RF_U16:  return NV_LD_GLB_U16;
        case NV_RF_F16:  return NV_LD_GLB_B16;
        default:         return NV_LD_GLB_U32;
        }
    }
}

static uint16_t stop(int as, uint8_t vrf, uint32_t psz)
{
    switch (as) {
    case BIR_AS_SHARED:
        return (psz == 1)         ? NV_ST_SHR_U8  :
               (vrf == NV_RF_F16) ? NV_ST_SHR_B16 :
               (vrf == NV_RF_U16) ? NV_ST_SHR_U16 :
               (vrf == NV_RF_F64) ? NV_ST_SHR_F64 :
               (vrf == NV_RF_U64) ? NV_ST_SHR_U64 :
               (vrf == NV_RF_F32) ? NV_ST_SHR_F32 : NV_ST_SHR_U32;
    case BIR_AS_PRIVATE:
        return (psz == 1)         ? NV_ST_LOC_U8  :
               (vrf == NV_RF_F16) ? NV_ST_LOC_B16 :
               (vrf == NV_RF_U16) ? NV_ST_LOC_U16 :
               (vrf == NV_RF_F64) ? NV_ST_LOC_F64 :
               (vrf == NV_RF_F32) ? NV_ST_LOC_F32 :
               (vrf == NV_RF_U64) ? NV_ST_LOC_U64 : NV_ST_LOC_U32;
    default:
        if (psz == 1) return NV_ST_GLB_U8;
        switch (vrf) {
        case NV_RF_F32:  return NV_ST_GLB_F32;
        case NV_RF_F64:  return NV_ST_GLB_F64;
        case NV_RF_U64:  return NV_ST_GLB_U64;
        case NV_RF_U16:  return NV_ST_GLB_U16;
        case NV_RF_F16:  return NV_ST_GLB_B16;
        default:         return NV_ST_GLB_U32;
        }
    }
}

static int pagg(uint32_t ty)
{
    uint8_t k;

    if (ty >= S.bir->num_types) return 0;
    k = S.bir->types[ty].kind;
    return k == BIR_TYPE_STRUCT || k == BIR_TYPE_ARRAY
        || k == BIR_TYPE_VECTOR;
}

static uint32_t agea(uint32_t al, uint32_t sz)
{
    return (al < 4u && (sz & 3u) == 0u) ? 4u : al;
}

static uint32_t agstp(uint32_t off, uint32_t rem, uint32_t al)
{
    uint32_t w = (al < 8u) ? al : 8u;

    while (w > 1u && (rem < w || (off & (w - 1u)) != 0u)) w >>= 1;
    return w;
}

static uint8_t agrf(uint32_t w)
{
    return (w == 8u) ? (uint8_t)NV_RF_U64
         : (w == 2u) ? (uint8_t)NV_RF_U16 : (uint8_t)NV_RF_U32;
}

static uint32_t lslot(uint32_t sz, uint32_t a)
{
    uint32_t off = (S.lcl_off + a - 1u) & ~(a - 1u);

    if (off < S.lcl_off || off >= NV_MAX_FRAME || sz > NV_MAX_FRAME - off)
        return NV_MAX_FRAME;
    S.lcl_off = off + sz;
    if (a > S.lcl_alg) S.lcl_alg = a;
    return off;
}

static nv_opnd_t agadd(nv_opnd_t base, uint32_t off)
{
    nv_opnd_t d;

    if (off == 0u) return base;
    d = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    em1(NV_ADD_U64, d, base, mop_imm((int32_t)off));
    return d;
}

static int aghome(uint32_t idx, uint32_t ty, nv_opnd_t *out,
                  uint32_t *psz, uint32_t *pal)
{
    uint32_t sz = bir_bsz(S.bir, ty, 8);
    uint32_t al = bir_balg(S.bir, ty, 8);
    uint32_t off;

    if (sz == 0u || al == 0u) { unsz("aggregate", ty); return 0; }
    off = lslot(sz, al);
    if (off >= NV_MAX_FRAME) {
        nv_refuse("a local frame this large");
        return 0;
    }
    *out = mapu64(idx);
    *psz = sz;
    *pal = agea(al, sz);
    em1u(NV_LEA_LOCAL, *out, mop_imm((int32_t)off));
    return 1;
}

static void agcpy(nv_opnd_t dst, int das, nv_opnd_t src, int sas,
                  uint32_t sz, uint32_t al)
{
    uint32_t off = 0u, g = sz + 8u;

    while (off < sz && g--) {
        uint32_t w = agstp(off, sz - off, al);
        uint8_t rf;
        nv_opnd_t t, ops[2];

        if (w == 1u) {
            nv_refuse("an aggregate with an odd byte size");
            return;
        }
        rf = agrf(w);
        t = mop_reg(rf, new_vreg(rf));
        em1u(ldop(sas, rf, 0u), t, agadd(src, off));
        ops[0] = agadd(dst, off);
        ops[1] = t;
        emit(stop(das, rf, 0u), 0, 2, ops, 0);
        off += w;
    }
}

static void is_load(uint32_t idx, const bir_inst_t *I)
{
    uint32_t ptr_val = I->operands[0];
    int as = pas(ptr_val);
    nv_opnd_t addr = mat_const(ptr_val, NV_RF_U64);
    nv_opnd_t d;
    uint8_t drf;

    if (pagg(I->type)) {
        uint32_t sz, al;

        if (!aghome(idx, I->type, &d, &sz, &al)) return;
        agcpy(d, BIR_AS_PRIVATE, addr, as, sz, al);
        return;
    }

    d = map_val(idx, I->type);
    drf = S.nv->val_rfile[idx];
    em1u(ldop(as, drf, pntsz(ptr_val)), d, addr);
}

/* ---- Memory: Store ---- */

static void is_store(const bir_inst_t *I)
{
    /* BIR store: ops[0]=value, ops[1]=address */
    uint32_t vval = I->operands[0];
    uint32_t vty = vtyof(vval);
    uint32_t ptr_val = I->operands[1];
    int as = pas(ptr_val);
    nv_opnd_t addr = mat_const(ptr_val, NV_RF_U64);
    nv_opnd_t val;
    uint8_t vrf;

    if (pagg(vty)) {
        uint32_t sz = bir_bsz(S.bir, vty, 8);
        uint32_t al = bir_balg(S.bir, vty, 8);

        if (sz == 0u || al == 0u) { unsz("store", vty); return; }
        agcpy(addr, as, mat_const(vval, NV_RF_U64), BIR_AS_PRIVATE, sz,
              agea(al, sz));
        return;
    }

    val = rslv(vval);
    vrf = rslv_rf(vval);

    /* Value must be in a register for stores */
    if (val.kind != NV_MOP_REG)
        val = mat_const(vval, vrf);

    /* Store: 0 defs, 2 uses (addr, value) */
    nv_opnd_t ops[2] = { addr, val };
    emit(stop(as, vrf, pntsz(ptr_val)), 0, 2, ops, 0);
}

static uint8_t ccls(char c)
{
    switch (c) {
    case 'l': return NV_RF_U64;
    case 'f': return NV_RF_F32;
    case 'd': return NV_RF_F64;
    case 'h': return NV_RF_U16;
    default:  return NV_RF_U32;
    }
}

static int asmb(const bir_asm_t *A, const bir_inst_t *I, nv_asm_t *X,
                uint32_t *pv, nv_opnd_t *pr)
{
    uint32_t i;

    for (i = 0; i < A->nops; i++) {
        char md = S.bir->strings[A->cons + 2u * i];
        char cl = S.bir->strings[A->cons + 2u * i + 1u];
        uint8_t rf = ccls(cl);
        uint32_t v = bir_oper(S.bir, I, i);

        pv[i] = v;
        pr[i] = mop_none();
        if (i < A->nout) {
            nv_opnd_t d = mop_reg(rf, new_vreg(rf));
            pr[i] = mat_const(v, NV_RF_U64);
            if (md == '+')
                em1u(ldop(pas(v), rf, pntsz(v)), d, pr[i]);
            X->ops[i] = d;
            continue;
        }
        if (cl == 'n') { X->ops[i] = rslv(v); continue; }
        if (rslv(v).kind == NV_MOP_REG && rslv_rf(v) != rf) {
            nv_refuse("an inline asm operand whose register class is not "
                      "the one its constraint asks for,");
            return 0;
        }
        X->ops[i] = mat_const(v, rf);
    }
    for (i = A->nops; i < NV_MAX_ASMOP; i++)
        X->ops[i] = mop_none();
    return 1;
}

static void is_asm(const bir_inst_t *I)
{
    uint32_t pv[NV_MAX_ASMOP];
    nv_opnd_t pr[NV_MAX_ASMOP], zero[1];
    const bir_asm_t *A;
    nv_asm_t *X;
    uint32_t ai = I->subop, i;

    if (ai >= S.bir->num_asms) {
        nv_refuse("an inline asm whose descriptor went missing,");
        return;
    }
    A = &S.bir->asms[ai];
    if (A->nops > NV_MAX_ASMOP || A->nout > A->nops
        || A->cons + 2u * A->nops >= S.bir->string_len) {
        nv_refuse("an inline asm with more operands than this backend holds,");
        return;
    }
    if (S.nv->num_asm >= NV_MAX_ASM) {
        nv_refuse("more inline asm than this backend holds,");
        return;
    }
    X = &S.nv->asms[S.nv->num_asm];
    X->tmpl = A->tmpl;
    X->nops = (uint8_t)A->nops;
    X->nout = (uint8_t)A->nout;
    X->pad[0] = X->pad[1] = 0;
    if (!asmb(A, I, X, pv, pr)) return;

    zero[0] = mop_none();
    emit(NV_ASM, 0, 0, zero, (uint16_t)S.nv->num_asm);
    S.nv->num_asm++;

    for (i = 0; i < A->nout; i++) {
        nv_opnd_t st[2];
        st[0] = pr[i];
        st[1] = X->ops[i];
        emit(stop(pas(pv[i]), X->ops[i].rfile, pntsz(pv[i])), 0, 2, st, 0);
    }
}

static void is_gref(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t gi = I->num_operands ? I->operands[0] : 0u;
    if (gi >= S.bir->num_globals) {
        (void)be_fail(BC_E543, "nvptx", "global reference", (unsigned)gi);
        S.unsz = 1;
        return;
    }
    em1u(NV_LEA_GLB, d, mop_imm((int32_t)gi));
}

/* ---- Memory: Alloca ---- */

static void is_alloca(uint32_t idx, const bir_inst_t *I)
{
    /* Allocas become .local addresses. Track offset, produce a
     * mov of the offset into a u64 register. The emitter will
     * reference these as local memory. */
    nv_opnd_t d = map_val(idx, I->type);

    uint32_t sz = 0;
    if (I->type < S.bir->num_types) {
        const bir_type_t *T = &S.bir->types[I->type];
        if (T->kind == BIR_TYPE_PTR)
            sz = type_bytes(T->inner);
    }
    if (!sz) { unsz("alloca", I->type); return; }
    uint32_t a = algnb(I->subop);
    uint32_t off = (S.lcl_off + a - 1u) & ~(a - 1u);
    if (off < S.lcl_off || off >= NV_MAX_FRAME || sz > NV_MAX_FRAME - off) {
        nv_refuse("a local frame this large");
        return;
    }
    S.lcl_off = off + sz;
    if (a > S.lcl_alg) S.lcl_alg = a;

    /* PTX local addresses are NOT 0-based — we need the symbolic
     * __local base. NV_LEA_LOCAL emits: mov.u64 %rd, __local+off */
    em1u(NV_LEA_LOCAL, d, mop_imm((int32_t)off));
}

/* ---- Memory: Shared Alloc ---- */

static void is_shralloc(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);

    uint32_t sz = 0;
    uint32_t a = algnb(I->subop);
    if (I->type < S.bir->num_types) {
        const bir_type_t *T = &S.bir->types[I->type];
        if (T->kind == BIR_TYPE_PTR) {
            uint32_t in = T->inner;
            if (in < S.bir->num_types &&
                S.bir->types[in].kind == BIR_TYPE_ARRAY &&
                S.bir->types[in].count == 0) {
                if (a > S.nv->dyn_alg) S.nv->dyn_alg = a;
                em1u(NV_LEA_DSH, d, mop_imm(0));
                return;
            }
            sz = type_bytes(in);
        }
    }
    if (!sz) { unsz("shared_alloc", I->type); return; }
    uint32_t off = (S.shr_off + a - 1u) & ~(a - 1u);
    if (off < S.shr_off || off >= NV_MAX_FRAME || sz > NV_MAX_FRAME - off) {
        nv_refuse("a shared block this large");
        return;
    }
    S.shr_off = off + sz;
    if (a > S.shr_alg) S.shr_alg = a;

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
    uint32_t foff;

    if (bir_fgep(S.bir, I, 8, &foff)) {
        em1(NV_ADD_U64, d, base, mop_imm((int32_t)foff));
        return;
    }

    uint32_t stride = bir_gstr(S.bir, I->type, 8);
    if (!stride) { unsz("gep", I->type); return; }

    if (offset.kind == NV_MOP_REG) {
        offset = cofit(offset, NV_RF_U64, 1);
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
    uint32_t pi = I->subop;
    nv_opnd_t d;
    uint16_t op;

    if (pagg(I->type)) {
        uint32_t sz, al, o = 0u, g;

        if (!aghome(idx, I->type, &d, &sz, &al)) return;
        g = sz + 8u;
        while (o < sz && g--) {
            uint32_t w = agstp(o, sz - o, al);
            uint8_t crf;
            nv_opnd_t t, lops[3], sops[2];

            if (w == 1u) {
                nv_refuse("an aggregate with an odd byte size");
                return;
            }
            crf = agrf(w);
            t = mop_reg(crf, new_vreg(crf));
            lops[0] = t;
            lops[1] = mop_imm((int32_t)pi);
            lops[2] = mop_imm((int32_t)o);
            emit(w == 8u ? NV_LD_PARAM_U64
               : w == 2u ? NV_LD_PARAM_B16 : NV_LD_PARAM_U32,
                 1, 2, lops, 0);
            sops[0] = agadd(d, o);
            sops[1] = t;
            emit(stop(BIR_AS_PRIVATE, crf, 0u), 0, 2, sops, 0);
            o += w;
        }
        return;
    }

    d = map_val(idx, I->type);
    switch (rf) {
    case NV_RF_U64:  op = NV_LD_PARAM_U64; break;
    case NV_RF_F32:  op = NV_LD_PARAM_F32; break;
    case NV_RF_F64:  op = NV_LD_PARAM_F64; break;
    case NV_RF_U16:
    case NV_RF_F16:  op = NV_LD_PARAM_B16; break;
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

    if (d.rfile == NV_RF_U32) {
        em1u(NV_MOV_U32, d, mop_spec(spec));
    } else {
        nv_opnd_t t = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
        em1u(NV_MOV_U32, t, mop_spec(spec));
        cocpy(d, t, 0);
    }
}

/* ---- Control Flow ---- */

static int mblk(uint32_t bir_bi, uint32_t *out)
{
    if (bir_bi >= BIR_MAX_BLOCKS
     || bir_bi < S.blk_lo || bir_bi >= S.blk_hi) {
        S.had_error = 1;
        if (S.bscr) return 1;
        S.bscr = 1;
        nv_refuse("a branch that leaves the function it sits in, which PTX "
                  "scopes labels against,");
        return 1;
    }
    *out = S.block_map[bir_bi];
    return 0;
}

static int mblkl(uint32_t bir_bi, uint32_t *out)
{
    if (mblk(bir_bi, out) != 0) return 1;
    *out = S.blk_lst[bir_bi];
    return 0;
}

static void is_br(const bir_inst_t *I)
{
    uint32_t m_tgt;
    nv_opnd_t ops[1];
    if (mblk(I->operands[0], &m_tgt) != 0) return;
    ops[0] = mop_lbl(m_tgt);
    emit(NV_BRA, 0, 1, ops, 0);
}

static int sw_phi(uint32_t tgt, uint32_t self)
{
    if (tgt >= S.bir->num_blocks) return 0;
    const bir_block_t *B = &S.bir->blocks[tgt];
    for (uint32_t i = 0; i < B->num_insts; i++) {
        const bir_inst_t *P = &S.bir->insts[B->first_inst + i];
        if (P->op != BIR_PHI) break;
        uint32_t n = n_ops(P);
        for (uint32_t k = 0; k + 1 < n; k += 2)
            if (get_op(P, k) == self) return 1;
    }
    return 0;
}

static void is_swch(const bir_inst_t *I, uint32_t bir_bi)
{
    uint32_t n = n_ops(I);
    uint32_t dfl = get_op(I, 1);
    uint32_t k;

    if (sw_phi(dfl, bir_bi)) {
        nv_refuse("a switch whose successor takes a phi from the switch "
                  "block, which needs an edge split this backend has no "
                  "bridge for");
        return;
    }
    for (k = 2; k + 1 < n; k += 2) {
        if (sw_phi(get_op(I, k + 1), bir_bi)) {
            nv_refuse("a switch whose successor takes a phi from the switch "
                      "block, which needs an edge split this backend has no "
                      "bridge for");
            return;
        }
    }

    nv_opnd_t sel = mat_const(get_op(I, 0), NV_RF_U32);
    for (k = 2; k + 1 < n; k += 2) {
        uint32_t cval = get_op(I, k);
        uint32_t cblk = get_op(I, k + 1);
        int32_t imm;
        if (cblk >= S.bir->num_blocks) continue;
        if (BIR_VAL_IS_CONST(cval)) {
            uint32_t ci = BIR_VAL_INDEX(cval);
            imm = (ci < S.bir->num_consts)
                ? (int32_t)S.bir->consts[ci].d.ival : 0;
        } else {
            imm = (int32_t)cval;
        }
        uint16_t prn = new_vreg(NV_RF_PRED);
        nv_opnd_t p = mop_reg(NV_RF_PRED, prn);
        em1(NV_SETP_EQ_U32, p, sel, mop_imm(imm));
        uint32_t m_case;
        if (mblk(cblk, &m_case) != 0) return;
        {
            nv_opnd_t bops[2] = { p, mop_lbl(m_case) };
            emit(NV_BRA_PRED, 0, 2, bops, 0);
        }
    }
    if (dfl < S.bir->num_blocks) {
        uint32_t m_dfl;
        if (mblk(dfl, &m_dfl) != 0) return;
        {
            nv_opnd_t dops[1] = { mop_lbl(m_dfl) };
            emit(NV_BRA, 0, 1, dops, 0);
        }
    }
}

static void is_brcond(const bir_inst_t *I)
{
    nv_opnd_t cond = rslv_p(I->operands[0]);
    uint32_t true_bir  = I->operands[1];
    uint32_t false_bir = I->operands[2];

    if (cond.kind != NV_MOP_REG || cond.rfile != NV_RF_PRED)
        cond = topred(cond);

    /* @%p bra $true_label */
    uint32_t m_true, m_false;
    if (mblk(true_bir, &m_true) != 0) return;
    if (mblk(false_bir, &m_false) != 0) return;
    {
        nv_opnd_t ops[2] = { cond, mop_lbl(m_true) };
        emit(NV_BRA_PRED, 0, 2, ops, 0);
    }
    {
        nv_opnd_t fops[1] = { mop_lbl(m_false) };
        emit(NV_BRA, 0, 1, fops, 0);
    }
}

/* ---- PHI Nodes ---- */

static int aphsrc(uint32_t val, uint32_t idx, uint32_t bir_bi,
                  nv_opnd_t *src)
{
    const bir_block_t *B = &S.bir->blocks[bir_bi];
    uint32_t vi = BIR_VAL_INDEX(val);

    if (val == BIR_MAKE_VAL(idx)) return 1;
    if (BIR_VAL_IS_CONST(val) || vi >= S.bir->num_insts) {
        nv_refuse("an aggregate phi fed by a constant");
        return 2;
    }
    if (S.bir->insts[vi].op == BIR_PHI && vi >= B->first_inst
        && vi < B->first_inst + B->num_insts) {
        nv_refuse("an aggregate phi fed by another phi of its own block");
        return 2;
    }
    *src = mapu64(vi);
    return 0;
}

static void aphend(uint32_t idx, uint32_t ty, nv_opnd_t from)
{
    nv_opnd_t h;
    uint32_t sz, al;

    if (!aghome(idx, ty, &h, &sz, &al)) return;
    agcpy(h, BIR_AS_PRIVATE, from, BIR_AS_PRIVATE, sz, al);
}

static void is_phi(uint32_t idx, const bir_inst_t *I,
                   uint32_t cur_bir_blk)
{
    /* PHI elimination via deferred predecessor copies.
     * BIR PHI operands are (block, value) pairs — block at even
     * positions, value at odd. Getting this backwards gives you
     * undefined predicate registers and a kernel that treats GPU
     * memory like a pinball machine. We learned this the hard way. */
    int ag = pagg(I->type);
    uint8_t rf = ag ? (uint8_t)NV_RF_U64 : bir_rfile(I->type);
    nv_opnd_t d = ag ? mop_reg(NV_RF_U64, new_vreg(NV_RF_U64))
                     : map_val(idx, I->type);
    uint32_t nop = n_ops(I);

    uint16_t mop = movof(rf);

    uint32_t merge_mblk = S.block_map[cur_bir_blk];

    int guard = 64;
    for (uint32_t k = 0; k + 1 < nop && guard > 0; k += 2, guard--) {
        uint32_t pred_bir = get_op(I, k);       /* block at even */
        uint32_t val      = get_op(I, k + 1);   /* value at odd  */

        if (pred_bir >= S.bir->num_blocks) continue;
        if (val == BIR_VAL_NONE) continue;

        uint32_t m_pred;
        if (mblkl(pred_bir, &m_pred) != 0) return;
        if (m_pred >= NV_MAX_MBLK) continue;

        nv_opnd_t src = ag ? mop_imm(0) : rslv_p(val);
        int ar = ag ? aphsrc(val, idx, cur_bir_blk, &src) : 0;

        if (ar == 2) return;
        if (ar == 1) continue;

        /* Self-copy: skip */
        if (src.kind == NV_MOP_REG && src.rfile == d.rfile &&
            src.reg_num == d.reg_num)
            continue;

        /* Record deferred copy for phi_fix() */
        if (S_npc >= NV_MAX_PCOPY)
            nv_cap("phi copies", (unsigned)NV_MAX_PCOPY);
        if (S_npc < NV_MAX_PCOPY) {
            S_pcopy[S_npc].pred_mblk  = m_pred;
            S_pcopy[S_npc].merge_mblk = merge_mblk;
            S_pcopy[S_npc].mop        = mop;
            S_pcopy[S_npc].dst        = d;
            S_pcopy[S_npc].src        = src;
            S_npc++;
        }
    }
    if (ag) aphend(idx, I->type, d);
}

/* ---- Barriers ---- */

static void is_barrier(const bir_inst_t *I)
{
    em0(I->subop ? NV_BARWARP : NV_BAR_SYNC);
}

#define GB_NBLK 4u

static uint32_t gbcnt(uint32_t bir_bi)
{
    const bir_block_t *B;
    uint32_t n = 0;

    if (bir_bi >= S.bir->num_blocks) return 0;
    B = &S.bir->blocks[bir_bi];
    for (uint32_t ii = 0; ii < B->num_insts && ii < 65536u; ii++)
        if (S.bir->insts[B->first_inst + ii].op == BIR_GRIDBAR) n++;
    return n;
}

static uint32_t bcut(uint32_t bir_bi)
{
    nv_mblk_t *MB;

    if (S.cmb + 1u >= NV_MAX_MBLK) {
        nv_cap("machine blocks", (unsigned)NV_MAX_MBLK);
        return 0;
    }
    MB = &S.nv->mblks[S.cmb];
    MB->num_insts = S.nv->num_minst - MB->first_inst;
    S.cmb++;
    MB = &S.nv->mblks[S.cmb];
    MB->first_inst = S.nv->num_minst;
    MB->num_insts  = 0;
    MB->bir_block  = bir_bi;
    if (S.nv->num_mblk <= S.cmb) S.nv->num_mblk = S.cmb + 1u;
    return S.cmb;
}

static void gbbr(nv_opnd_t p, uint32_t t, uint32_t f)
{
    nv_opnd_t ops[2] = { p, mop_lbl(t) };
    nv_opnd_t fops[1] = { mop_lbl(f) };

    emit(NV_BRA_PRED, 0, 2, ops, 0);
    emit(NV_BRA, 0, 1, fops, 0);
}

static nv_opnd_t gbfold(uint16_t op, int32_t a, int32_t b, int32_t c)
{
    nv_opnd_t r = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
    nv_opnd_t t = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));

    em1u(NV_MOV_U32, r, mop_spec(a));
    em1u(NV_MOV_U32, t, mop_spec(b));
    em1(op, r, r, t);
    em1u(NV_MOV_U32, t, mop_spec(c));
    em1(op, r, r, t);
    return r;
}

static void gbfen(void)
{
    nv_opnd_t ops[1] = { mop_none() };
    emit(NV_MEMBAR, 0, 0, ops, NV_SCOPE_GPU);
}

static nv_opnd_t gbadd(nv_opnd_t a)
{
    nv_opnd_t d = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
    nv_opnd_t v = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
    nv_opnd_t ops[3];

    em1u(NV_MOV_U32, v, mop_imm(1));
    ops[0] = d; ops[1] = a; ops[2] = v;
    emit(NV_ATOM_ADD_U32, 1, 2, ops, NV_ASP_GLB);
    return d;
}

static void gbhd(nv_opnd_t p, uint32_t w, uint32_t l1)
{
    nv_opnd_t n;

    em0(NV_BAR_SYNC);
    n = gbfold(NV_OR_B32, NV_SPEC_TID_X, NV_SPEC_TID_Y, NV_SPEC_TID_Z);
    em1(NV_SETP_NE_U32, p, n, mop_imm(0));
    gbbr(p, w, l1);
}

static void gbarr(nv_opnd_t p, nv_opnd_t rd, nv_opnd_t rg, nv_opnd_t gen,
                  uint32_t sp, uint32_t l2)
{
    nv_opnd_t n, a;

    em1u(NV_LEA_GLB, rd, mop_imm((int32_t)S.bir->num_globals));
    em1(NV_ADD_U64, rg, rd, mop_imm(4));
    gbfen();
    em1u(NV_LD_GLB_U32, gen, rg);
    n = gbfold(NV_MUL_LO_U32, NV_SPEC_NCTAID_X, NV_SPEC_NCTAID_Y,
               NV_SPEC_NCTAID_Z);
    em1(NV_SUB_U32, n, n, mop_imm(1));
    a = gbadd(rd);
    em1(NV_SETP_NE_U32, p, a, n);
    gbbr(p, sp, l2);
}

static void gbrel(nv_opnd_t rd, nv_opnd_t rg, uint32_t w)
{
    nv_opnd_t z = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
    nv_opnd_t ops[2];
    nv_opnd_t bops[1] = { mop_lbl(w) };

    em1u(NV_MOV_U32, z, mop_imm(0));
    ops[0] = rd; ops[1] = z;
    emit(NV_ST_GLB_U32, 0, 2, ops, 0);
    gbfen();
    (void)gbadd(rg);
    emit(NV_BRA, 0, 1, bops, 0);
}

static void gbspn(nv_opnd_t p, nv_opnd_t rg, nv_opnd_t gen, uint32_t sp,
                  uint32_t w)
{
    nv_opnd_t c = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));

    em1u(NV_LD_GLB_U32, c, rg);
    em1(NV_SETP_EQ_U32, p, c, gen);
    gbbr(p, sp, w);
}

static void is_gbar(uint32_t bir_bi)
{
    uint32_t b, l1, l2, sp, w;
    nv_opnd_t p, rd, rg, gen;

    S.nv->gbar = 1;
    if (S.nv->sass == 0) { em0(NV_GBAR); return; }

    b = S.cmb; l1 = b + 1u; l2 = b + 2u; sp = b + 3u; w = b + 4u;
    p   = mop_reg(NV_RF_PRED, new_vreg(NV_RF_PRED));
    rd  = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    rg  = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    gen = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));

    gbhd(p, w, l1);
    if (bcut(bir_bi) != l1) return;
    gbarr(p, rd, rg, gen, sp, l2);
    if (bcut(bir_bi) != l2) return;
    gbrel(rd, rg, w);
    if (bcut(bir_bi) != sp) return;
    gbspn(p, rg, gen, sp, w);
    if (bcut(bir_bi) != w) return;
    em0(NV_BAR_SYNC);
    gbfen();
}

static void is_fence(const bir_inst_t *I)
{
    nv_opnd_t ops[1];
    uint16_t sc = I->subop;
    ops[0] = mop_none();
    if (sc > NV_SCOPE_SYS) sc = NV_SCOPE_SYS;
    emit(NV_MEMBAR, 0, 0, ops, sc);
}

static void is_nslp(const bir_inst_t *I)
{
    nv_opnd_t ops[1];
    ops[0] = mat_const(I->operands[0], NV_RF_U32);
    emit(NV_NANOSLP, 0, 1, ops, 0);
}

static void is_barred(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    nv_opnd_t p = rslv_p(I->operands[0]);
    uint16_t op;

    if (p.kind != NV_MOP_REG || p.rfile != NV_RF_PRED)
        p = topred(p);
    switch (I->subop) {
    case 1:  op = NV_BARRED_AND;  break;
    case 2:  op = NV_BARRED_POPC; break;
    default: op = NV_BARRED_OR;   break;
    }
    em1u(op, d, p);
}

/* ---- Atomics ---- */

static int atmw(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U32: case NV_RF_F32: case NV_RF_B32: return 32;
    case NV_RF_U64: case NV_RF_F64:                 return 64;
    default:                                        return 0;
    }
}

static int atmas(int as, uint16_t *fl)
{
    if (as == BIR_AS_SHARED)  { *fl = NV_ASP_SHR; return 1; }
    if (as == BIR_AS_GENERIC) { *fl = NV_ASP_GEN; return 1; }
    if (as == BIR_AS_GLOBAL)  { *fl = NV_ASP_GLB; return 1; }
    return 0;
}

static uint16_t atmop(uint16_t bop, uint8_t rf)
{
    int w = (atmw(rf) == 64);

    switch (bop) {
    case BIR_ATOMIC_ADD:
        return (rf == NV_RF_F32) ? NV_ATOM_ADD_F32 :
               (rf == NV_RF_F64) ? NV_ATOM_ADD_F64 :
               w ? NV_ATOM_ADD_U64 : NV_ATOM_ADD_U32;
    case BIR_ATOMIC_AND:  return w ? NV_ATOM_AND_B64  : NV_ATOM_AND_B32;
    case BIR_ATOMIC_OR:   return w ? NV_ATOM_OR_B64   : NV_ATOM_OR_B32;
    case BIR_ATOMIC_XOR:  return w ? NV_ATOM_XOR_B64  : NV_ATOM_XOR_B32;
    case BIR_ATOMIC_XCHG: return w ? NV_ATOM_XCHG_B64 : NV_ATOM_XCHG_B32;
    case BIR_ATOMIC_CAS:  return w ? NV_ATOM_CAS_B64  : NV_ATOM_CAS_B32;
    default:              return NV_OP_COUNT;
    }
}

static nv_opnd_t atmv(uint32_t v)
{
    nv_opnd_t o = rslv(v);

    if (o.kind != NV_MOP_REG)
        o = mat_const(v, rslv_rf(v));
    return o;
}

static void atmno(int eid, const char *what, unsigned n)
{
    S.had_error = 1;
    if (what) (void)be_fail(eid, "nvptx", what);
    else      (void)be_fail(eid, "nvptx", n);
}

static void is_atomic(uint32_t idx, const bir_inst_t *I)
{
    int as = pas(I->operands[0]);
    uint8_t rf = bir_rfile(I->type);
    uint16_t fl = NV_ASP_GLB, op;
    nv_opnd_t ops[4];
    uint8_t nu = 2;

    if (I->op == BIR_ATOMIC_MIN || I->op == BIR_ATOMIC_MAX) {
        atmno(BC_E560, I->op == BIR_ATOMIC_MIN ? "atomicMin" : "atomicMax", 0);
        return;
    }
    if (!atmw(rf)) {
        atmno(BC_E562, NULL, (unsigned)type_bytes(I->type));
        return;
    }
    if (!atmas(as, &fl)) {
        atmno(BC_E561, NULL, (unsigned)as);
        return;
    }
    op = atmop(I->op, rf);
    if (op == NV_OP_COUNT) {
        S.had_error = 1;
        (void)be_fail(BC_E543, "nvptx", "atomic", (unsigned)I->op);
        return;
    }

    ops[0] = map_val(idx, I->type);
    ops[1] = mat_const(I->operands[0], NV_RF_U64);
    ops[2] = atmv(I->operands[1]);
    if (op == NV_ATOM_CAS_B32 || op == NV_ATOM_CAS_B64) {
        uint32_t nv2 = get_op(I, 2);
        if (nv2 == BIR_VAL_NONE) {
            S.had_error = 1;
            (void)be_fail(BC_E543, "nvptx", "atomic", (unsigned)I->op);
            return;
        }
        ops[3] = atmv(nv2);
        nu = 3;
    }
    emit(op, 1, nu, ops, fl);
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
    uint32_t val_op = I->operands[1];
    uint32_t lane_op = I->operands[2];
    nv_opnd_t val = rslv(val_op);
    nv_opnd_t lane = rslv(lane_op);

    if (val.kind != NV_MOP_REG)
        val = mat_const(val_op, NV_RF_U32);
    val = cobit(val, NV_RF_U32);
    lane = cofit(lane, NV_RF_U32, 0);

    uint16_t op;
    switch (I->op) {
    case BIR_SHFL:      op = NV_SHFL_IDX;  break;
    case BIR_SHFL_UP:   op = NV_SHFL_UP;   break;
    case BIR_SHFL_DOWN: op = NV_SHFL_DOWN; break;
    case BIR_SHFL_XOR:  op = NV_SHFL_XOR;  break;
    default:            op = NV_SHFL_IDX;  break;
    }

    if (d.rfile == NV_RF_U32 || d.rfile == NV_RF_B32) {
        nv_opnd_t ops[3] = { d, val, lane };
        emit(op, 1, 2, ops, 0);
        return;
    }
    if (rfwid(d.rfile) > 32) {
        nv_refuse("a warp shuffle of a value wider than 32 bits");
        return;
    }
    {
        nv_opnd_t t = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
        nv_opnd_t ops[3] = { t, val, lane };
        emit(op, 1, 2, ops, 0);
        if (rfwid(d.rfile) == 32) em1u(NV_MOV_B32, d, t);
        else cocpy(d, t, 0);
    }
}

static void is_vote(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d = map_val(idx, I->type);
    uint32_t pred_op = I->operands[1];
    nv_opnd_t pred = rslv_p(pred_op);

    if (pred.kind != NV_MOP_REG || pred.rfile != NV_RF_PRED)
        pred = topred(pred);

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
    nv_opnd_t d0 = map_val(idx, I->type);
    int f64ok = (I->op == BIR_SQRT || I->op == BIR_FABS);
    uint8_t wrf = (f64ok && rf == NV_RF_F64) ? NV_RF_F64 : NV_RF_F32;
    nv_opnd_t d = (d0.rfile == wrf) ? d0 : mop_reg(wrf, new_vreg(wrf));
    /* The approx units take a register, so a folded constant argument has to
     * be moved into one first. ptxas rejects ex2.approx.f32 %f, <imm>. */
    nv_opnd_t a = cofit(mat_const(I->operands[0], wrf), wrf, 1);

    uint16_t op;
    switch (I->op) {
    case BIR_SQRT:   op = (wrf == NV_RF_F64) ? NV_SQRT_F64 : NV_SQRT_F32; break;
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
    case BIR_FABS:   op = (wrf == NV_RF_F64) ? NV_ABS_F64 : NV_ABS_F32; break;
    case BIR_FLOOR:  op = NV_FLOOR_F32; break;
    case BIR_CEIL:   op = NV_CEIL_F32;  break;
    case BIR_FTRUNC: op = NV_TRUNC_F32; break;
    case BIR_RNDNE:  op = NV_ROUND_F32; break;
    default:         op = NV_MOV_F32;   break;
    }

    em1u(op, d, a);
    if (d0.rfile != wrf) cocpy(d0, d, 1);
}

static void is_fminmax(uint32_t idx, const bir_inst_t *I)
{
    nv_opnd_t d0 = map_val(idx, I->type);
    nv_opnd_t d = (d0.rfile == NV_RF_F32)
                ? d0 : mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
    nv_opnd_t a = cofit(rslv(I->operands[0]), NV_RF_F32, 1);
    nv_opnd_t b = cofit(rslv(I->operands[1]), NV_RF_F32, 1);

    uint16_t op;
    switch (I->op) {
    case BIR_FMIN: op = NV_MIN_F32; break;
    case BIR_FMAX: op = NV_MAX_F32; break;
    default:       op = NV_MOV_F32; b = a; break;
    }
    if (op == NV_MOV_F32) em1u(op, d, a);
    else em1(op, d, a, b);
    if (d0.rfile != NV_RF_F32) cocpy(d0, d, 1);
}

/* ---- Return ---- */

static void is_ret(const bir_inst_t *I)
{
    nv_mfunc_t *MF = (S.cur_func < S.nv->num_mfunc)
                   ? &S.nv->mfuncs[S.cur_func] : NULL;

    if (MF != NULL && MF->is_kern) { em0(NV_EXIT); return; }

    if (MF != NULL && MF->hasret &&
        I->num_operands > 0 && I->operands[0] != BIR_VAL_NONE) {
        nv_opnd_t ops[1];
        ops[0] = mat_const(I->operands[0], MF->retrf);
        emit(NV_ST_RETP, 0, 1, ops, MF->retrf);
    }
    em0(NV_RET);
}

static uint32_t nopsof(const bir_inst_t *I)
{
    return (I->num_operands == BIR_OPERANDS_OVERFLOW)
         ? I->operands[1] : (uint32_t)I->num_operands;
}

static uint32_t vtyof(uint32_t val)
{
    uint32_t i = BIR_VAL_INDEX(val);
    if (val == BIR_VAL_NONE) return 0;
    if (BIR_VAL_IS_CONST(val))
        return (i < S.bir->num_consts) ? S.bir->consts[i].type : 0;
    return (i < S.bir->num_insts) ? S.bir->insts[i].type : 0;
}

static int cargs(nv_call_t *C, const bir_inst_t *I, uint32_t n,
                 const bir_type_t *FT)
{
    for (uint32_t a = 1; a < n; a++) {
        uint32_t v = bir_oper(S.bir, I, a);
        uint32_t ty = vtyof(v);
        uint8_t rf;
        if (FT != NULL && a - 1 < FT->num_fields
         && FT->count + a - 1 < S.bir->num_type_fields)
            ty = S.bir->type_fields[FT->count + a - 1];
        rf = bir_rfile(ty);
        if (rf == NV_RF_PRED) return 1;
        C->args[a - 1] = mat_const(v, rf);
        C->args[a - 1].rfile = rf;
    }
    return 0;
}

static void is_call(uint32_t idx, const bir_inst_t *I)
{
    uint32_t cf = I->operands[0];
    uint32_t n  = nopsof(I);
    const bir_type_t *FT = NULL;
    nv_call_t *C;
    nv_opnd_t ops[1];
    uint32_t ci;

    if (cf >= S.bir->num_funcs || S.bir->funcs[cf].num_blocks == 0) {
        nv_refuse("a call to a device function Booth never saw a body for,");
        return;
    }
    if (n < 1 || n - 1 > NV_MAX_CARG) {
        nv_cap("call arguments", (unsigned)NV_MAX_CARG);
        return;
    }
    if (S.nv->num_call >= NV_MAX_CALL) {
        nv_cap("call sites", (unsigned)NV_MAX_CALL);
        return;
    }

    {
        uint32_t ft = S.bir->funcs[cf].type;
        if (ft < S.bir->num_types
         && S.bir->types[ft].kind == BIR_TYPE_FUNC)
            FT = &S.bir->types[ft];
    }

    ci = S.nv->num_call++;
    C = &S.nv->calls[ci];
    memset(C, 0, sizeof(*C));
    C->fn = cf;
    C->nargs = (uint8_t)(n - 1);
    if (cargs(C, I, n, FT) != 0) {
        nv_refuse("a device call taking a bool by value, which has no "
                  "parameter form in PTX,");
        return;
    }

    if (I->type < S.bir->num_types
     && S.bir->types[I->type].kind != BIR_TYPE_VOID) {
        C->hasret = 1;
        C->retrf = def_rf(idx, I->type);
        if (C->retrf == NV_RF_PRED) C->retrf = NV_RF_U32;
        C->ret = map_val(idx, I->type);
    }

    ops[0] = mop_none();
    emit(NV_CALL, 0, 0, ops, (uint16_t)ci);
}

static uint32_t pfslot(uint32_t ty, uint32_t *alg)
{
    const bir_type_t *T = (ty < S.bir->num_types) ? &S.bir->types[ty] : NULL;
    if (T == NULL) { *alg = 0; return 0; }
    switch (T->kind) {
    case BIR_TYPE_FLOAT:
        *alg = 8; return 8;
    case BIR_TYPE_PTR:
        *alg = 8; return 8;
    case BIR_TYPE_INT:
        if (T->width > 32) { *alg = 8; return 8; }
        *alg = 4; return 4;
    default:
        *alg = 0; return 0;
    }
}

static nv_opnd_t pfval(uint32_t v, uint32_t ty, uint16_t *stop)
{
    const bir_type_t *T = (ty < S.bir->num_types) ? &S.bir->types[ty] : NULL;
    uint8_t rf = bir_rfile(ty);
    nv_opnd_t a = mat_const(v, rf);

    if (T != NULL && T->kind == BIR_TYPE_FLOAT) {
        nv_opnd_t f32, f64;
        *stop = NV_ST_LOC_F64;
        if (T->width >= 64) return a;
        f32 = a;
        if (rf != NV_RF_F32) {
            f32 = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
            em1u(NV_CVT_F32_F16, f32, a);
        }
        f64 = mop_reg(NV_RF_F64, new_vreg(NV_RF_F64));
        em1u(NV_CVT_F64_F32, f64, f32);
        return f64;
    }
    if (rf == NV_RF_U64) { *stop = NV_ST_LOC_U64; return a; }
    if (rf == NV_RF_U16) {
        nv_opnd_t w = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
        em1u(NV_CVT_U32_U16, w, a);
        *stop = NV_ST_LOC_U32;
        return w;
    }
    *stop = NV_ST_LOC_U32;
    return a;
}

static void pfpack(const bir_inst_t *I, uint32_t n, uint32_t base)
{
    uint32_t off = 0;
    uint32_t a;

    for (a = 1; a < n; a++) {
        uint32_t v = bir_oper(S.bir, I, a);
        uint32_t alg = 0;
        uint32_t sz = pfslot(vtyof(v), &alg);
        uint16_t stop = NV_ST_LOC_U32;
        nv_opnd_t adr, val, sops[2];
        off = (off + alg - 1u) & ~(alg - 1u);
        adr = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
        val = pfval(v, vtyof(v), &stop);
        em1u(NV_LEA_LOCAL, adr, mop_imm((int32_t)(base + off)));
        sops[0] = adr; sops[1] = val;
        emit(stop, 0, 2, sops, 0);
        off += sz;
    }
}

static int pfbuf(const bir_inst_t *I, uint32_t n, uint32_t *base,
                 uint32_t *bad)
{
    uint32_t off = 0;
    uint32_t a;

    for (a = 1; a < n; a++) {
        uint32_t alg = 0;
        uint32_t ty = vtyof(bir_oper(S.bir, I, a));
        uint32_t sz = pfslot(ty, &alg);
        if (sz == 0) { *bad = ty; return 1; }
        off = (off + alg - 1u) & ~(alg - 1u);
        off += sz;
    }
    off = (off + 7u) & ~7u;

    *base = (S.lcl_off + 7u) & ~7u;
    if (*base >= NV_MAX_FRAME || off > NV_MAX_FRAME - *base) return 2;
    S.lcl_off = *base + off;
    if (S.lcl_alg < 8) S.lcl_alg = 8;
    return 0;
}

static void pfcall(uint32_t idx, const bir_inst_t *I,
                   nv_opnd_t fmt, nv_opnd_t gen)
{
    uint32_t ci = S.nv->num_call++;
    nv_call_t *C = &S.nv->calls[ci];
    nv_opnd_t ops[1];

    memset(C, 0, sizeof(*C));
    C->vprt = 1;
    C->nargs = 2;
    C->args[0] = fmt;
    C->args[1] = gen;
    C->hasret = 1;
    C->retrf = NV_RF_U32;
    C->ret = map_val(idx, I->type);
    S.nv->vprt = 1;

    ops[0] = mop_none();
    emit(NV_CALL, 0, 0, ops, (uint16_t)ci);
}

static void is_prntf(uint32_t idx, const bir_inst_t *I)
{
    uint32_t n = nopsof(I);
    uint32_t fv = bir_oper(S.bir, I, 0);
    uint32_t base = 0, bad = 0;
    nv_opnd_t fmt, buf, gen;

    if (n < 1) { nv_refuse("a printf with no format string,"); return; }
    if (S.nv->num_call >= NV_MAX_CALL) {
        nv_cap("call sites", (unsigned)NV_MAX_CALL);
        return;
    }

    switch (pfbuf(I, n, &base, &bad)) {
    case 1: {
        char tb[64];
        (void)bir_type_str(S.bir, bad, tb, (int)sizeof tb);
        (void)be_fail(BC_E760, tb);
        S.had_error = 1;
        return;
    }
    case 2:
        nv_refuse("a printf argument buffer this large");
        return;
    default:
        break;
    }

    pfpack(I, n, base);

    if (n > 1) {
        buf = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
        em1u(NV_LEA_LOCAL, buf, mop_imm((int32_t)base));
        gen = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
        em1u(NV_CVTA_LOC, gen, buf);
    } else {
        gen = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
        em1u(NV_MOV_U64, gen, mop_imm(0));
    }

    fmt = mat_const(fv, NV_RF_U64);
    if (!BIR_VAL_IS_CONST(fv)) {
        uint32_t si = BIR_VAL_INDEX(fv);
        if (si < S.bir->num_insts
         && S.bir->insts[si].op == BIR_GLOBAL_REF) {
            nv_opnd_t g = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
            em1u(NV_CVTA_GLB, g, fmt);
            fmt = g;
        }
    }

    pfcall(idx, I, fmt, gen);
}

/* ---- Per-Block Instruction Selection ---- */


#define MMA_N 16

static nv_opnd_t mma_u32(void)
{
    return mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
}

static nv_opnd_t mma_off(nv_opnd_t a, int32_t k)
{
    if (k == 0) return a;
    nv_opnd_t d = mma_u32();
    em1(NV_ADD_U32, d, a, mop_imm(k));
    return d;
}

static nv_opnd_t mma_adr(nv_opnd_t base, nv_opnd_t ld,
                         nv_opnd_t row, nv_opnd_t col, int32_t esz)
{
    nv_opnd_t t = mma_u32();
    em1(NV_MUL_LO_U32, t, row, ld);
    nv_opnd_t e = mma_u32();
    em1(NV_ADD_U32, e, t, col);
    nv_opnd_t w = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    em1u(NV_CVT_U64_U32, w, e);
    nv_opnd_t a = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    nv_opnd_t ops[4] = { a, w, mop_imm(esz), base };
    emit(NV_MAD_LO_U64, 1, 3, ops, 0);
    return a;
}

static uint16_t mma_frag(uint8_t rf, int n)
{
    uint16_t b = new_vreg(rf);
    for (int i = 1; i < n; i++) (void)new_vreg(rf);
    return b;
}

static void is_mma(const bir_inst_t *I)
{
    const nv_mmash_t *sh = &nv_mmash[I->subop % NV_MMA_NSHAPE];
    nv_opnd_t ap = mat_const(I->operands[0], NV_RF_U64);
    nv_opnd_t la = mat_const(I->operands[1], NV_RF_U32);
    nv_opnd_t bp = mat_const(I->operands[2], NV_RF_U64);
    nv_opnd_t lb = mat_const(I->operands[3], NV_RF_U32);
    nv_opnd_t dp = mat_const(I->operands[4], NV_RF_U64);
    nv_opnd_t ld = mat_const(I->operands[5], NV_RF_U32);

    em0(NV_BARWARP);

    nv_opnd_t lane = mma_u32();
    em1u(NV_MOV_U32, lane, mop_spec(NV_SPEC_LANEID));
    nv_opnd_t gid = mma_u32();
    em1(NV_SHR_U32, gid, lane, mop_imm(2));
    nv_opnd_t tig = mma_u32();
    em1(NV_AND_B32, tig, lane, mop_imm(3));
    nv_opnd_t t2 = mma_u32();
    em1(NV_SHL_B32, t2, tig, mop_imm(1));

    uint16_t abase = mma_frag(NV_RF_B32, sh->na / 2);
    for (uint8_t i = 0; i < sh->na; i += 2) {
        nv_opnd_t h[2];
        for (uint8_t j = 0; j < 2; j++) {
            uint8_t e = (uint8_t)(i + j);
            nv_opnd_t row = mma_off(gid, ((e & 2) != 0) ? 8 : 0);
            nv_opnd_t col = mma_off(t2, (e & 1) + ((e >= 4) ? 8 : 0));
            nv_opnd_t a = mma_adr(ap, la, row, col, 2);
            h[j] = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
            em1u(NV_LD_GLB_U16, h[j], a);
        }
        nv_opnd_t pk[3] = { mop_reg(NV_RF_B32, (uint16_t)(abase + i / 2)),
                            h[0], h[1] };
        emit(NV_MOV_PK16, 1, 2, pk, 0);
    }

    for (int nb = 0; nb < MMA_N; nb += 8) {
        uint16_t bbase = mma_frag(NV_RF_B32, sh->nb / 2);
        for (uint8_t i = 0; i < sh->nb; i += 2) {
            nv_opnd_t h[2];
            for (uint8_t j = 0; j < 2; j++) {
                uint8_t e = (uint8_t)(i + j);
                nv_opnd_t row = mma_off(t2, (e & 1) + ((e >= 2) ? 8 : 0));
                nv_opnd_t col = mma_off(gid, nb);
                nv_opnd_t a = mma_adr(bp, lb, row, col, 2);
                h[j] = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
                em1u(NV_LD_GLB_U16, h[j], a);
            }
            nv_opnd_t pk[3] = { mop_reg(NV_RF_B32, (uint16_t)(bbase + i / 2)),
                                h[0], h[1] };
            emit(NV_MOV_PK16, 1, 2, pk, 0);
        }

        nv_opnd_t da[4];
        uint16_t cbase = mma_frag(NV_RF_F32, 4);
        for (int i = 0; i < 4; i++) {
            nv_opnd_t row = mma_off(gid, (i >= 2) ? 8 : 0);
            nv_opnd_t col = mma_off(t2, (i & 1) + nb);
            da[i] = mma_adr(dp, ld, row, col, 4);
            em1u(NV_LD_GLB_F32, mop_reg(NV_RF_F32, (uint16_t)(cbase + i)),
                 da[i]);
        }
        uint16_t dbase = mma_frag(NV_RF_F32, 4);
        nv_opnd_t mo[4] = { mop_reg(NV_RF_F32, dbase),
                            mop_reg(NV_RF_B32, abase),
                            mop_reg(NV_RF_B32, bbase),
                            mop_reg(NV_RF_F32, cbase) };
        emit(NV_MMA, 1, 3, mo, (uint16_t)(I->subop % NV_MMA_NSHAPE));
        for (int i = 0; i < 4; i++) {
            nv_opnd_t st[2] = { da[i],
                                mop_reg(NV_RF_F32, (uint16_t)(dbase + i)) };
            emit(NV_ST_GLB_F32, 0, 2, st, 0);
        }
    }
}

static nv_opnd_t wm_ea(nv_opnd_t base, int32_t off)
{
    if (off == 0) return base;
    nv_opnd_t d = mop_reg(NV_RF_U64, new_vreg(NV_RF_U64));
    em1(NV_ADD_U64, d, base, mop_imm(off));
    return d;
}

static void wm_out(uint32_t fp, uint16_t tb, uint8_t rf, uint8_t n)
{
    int as = pas(fp);
    nv_opnd_t base = mat_const(fp, NV_RF_U64);
    int32_t w = (rf == NV_RF_F64) ? 8 : 4;

    for (uint8_t k = 0; k < n; k++) {
        nv_opnd_t ops[2] = { wm_ea(base, (int32_t)k * w),
                             mop_reg(rf, (uint16_t)(tb + k)) };
        emit(stop(as, rf, 4), 0, 2, ops, 0);
    }
}

static void wm_in(uint32_t fp, uint16_t tb, uint8_t rf, uint8_t n)
{
    int as = pas(fp);
    nv_opnd_t base = mat_const(fp, NV_RF_U64);
    int32_t w = (rf == NV_RF_F64) ? 8 : 4;

    for (uint8_t k = 0; k < n; k++)
        em1u(ldop(as, rf, 4), mop_reg(rf, (uint16_t)(tb + k)),
             wm_ea(base, (int32_t)k * w));
}

typedef struct {
    uint32_t lm, ln, lk;
    uint32_t ab, cb;
    uint32_t kind;
} wsh_t;

static int wslog(uint32_t v)
{
    for (int i = 0; i < 16; i++)
        if (v == (1u << i)) return i;
    return -1;
}

static uint32_t wsnum(const char *s, char tag, uint32_t *at)
{
    uint32_t v = 0u, i = *at;

    if (s[i] != tag) return 0u;
    i++;
    for (uint32_t g = 0; g < 4u && s[i] >= '0' && s[i] <= '9'; g++)
        v = v * 10u + (uint32_t)(s[i++] - '0');
    *at = i;
    return v;
}

static int wsdim(uint32_t row, wsh_t *h)
{
    const bir_wmma_t *W = &bir_wmma[row % BIR_WM_NROW];
    uint32_t at = 0u;
    int lm = wslog(wsnum(W->shp, 'm', &at));
    int ln = wslog(wsnum(W->shp, 'n', &at));
    int lk = wslog(wsnum(W->shp, 'k', &at));

    if (lm < 0 || ln < 0 || lk < 0 || W->shp[at] != 0) {
        nv_refuse("a wmma shape the SASS lowering cannot read");
        return 0;
    }
    h->lm = (uint32_t)lm;
    h->ln = (uint32_t)ln;
    h->lk = (uint32_t)lk;
    if (strcmp(W->abt, "f16") == 0)       { h->kind = 0u; h->ab = 16u; }
    else if (strcmp(W->abt, "bf16") == 0) { h->kind = 1u; h->ab = 16u; }
    else if (strcmp(W->abt, "s8") == 0)   { h->kind = 2u; h->ab = 8u; }
    else if (strcmp(W->abt, "u8") == 0)   { h->kind = 3u; h->ab = 8u; }
    else {
        nv_refuse("a wmma fragment of tf32, f64, four-bit or single-bit elements on the SASS backend");
        return 0;
    }
    h->cb = (strcmp(W->act, "f16") == 0) ? 16u : 32u;
    if ((h->kind < 2u) == (strcmp(W->act, "s32") == 0)) {
        nv_refuse("a wmma accumulator whose type does not match its multiplicands");
        return 0;
    }
    return 1;
}

static int wsfit(const wsh_t *h, unsigned role, uint32_t n, uint32_t epr,
                 uint32_t *lc)
{
    int l = wslog(n * epr);
    int ok = l >= 0;

    if (ok) *lc = (uint32_t)l;
    if (ok && role == NV_WM_ROLE_A)
        ok = h->lm + h->lk <= 5u + *lc && *lc <= h->lk;
    else if (ok && role == NV_WM_ROLE_B)
        ok = h->lk + h->ln <= 5u + *lc && *lc <= h->lk;
    else if (ok)
        ok = h->lm + h->ln == 5u + *lc && *lc <= h->ln;
    if (!ok)
        nv_refuse("a wmma shape the SASS lowering cannot distribute across one warp");
    return ok;
}

static nv_opnd_t wsu(void)
{
    return mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
}

static nv_opnd_t wsop(uint16_t op, nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t d = wsu();
    em1(op, d, a, b);
    return d;
}

static nv_opnd_t wsfop(uint16_t op, nv_opnd_t a, nv_opnd_t b)
{
    nv_opnd_t d = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
    em1(op, d, a, b);
    return d;
}

static nv_opnd_t wsbit(nv_opnd_t r, uint8_t rf)
{
    nv_opnd_t d = mop_reg(rf, new_vreg(rf));
    em1u(NV_MOV_B32, d, r);
    return d;
}

static nv_opnd_t wslane(uint32_t sh)
{
    nv_opnd_t l = wsu();
    em1u(NV_MOV_U32, l, mop_spec(NV_SPEC_LANEID));
    return (sh != 0u) ? wsop(NV_SHL_B32, l, mop_imm((int32_t)sh)) : l;
}

static nv_opnd_t wsshf(nv_opnd_t v, nv_opnd_t lane)
{
    nv_opnd_t ops[3] = { wsu(), v, lane };
    emit(NV_SHFL_IDX, 1, 2, ops, 0);
    return ops[0];
}

static void wsmap(const wsh_t *h, unsigned role, nv_opnd_t tl, uint32_t e,
                  nv_opnd_t *i, nv_opnd_t *j)
{
    nv_opnd_t t = wsop(NV_ADD_U32, tl, mop_imm((int32_t)e));
    int32_t km = (int32_t)((1u << h->lk) - 1u);

    if (role == NV_WM_ROLE_A) {
        t = wsop(NV_AND_B32, t, mop_imm((int32_t)((1u << (h->lm + h->lk)) - 1u)));
        *i = wsop(NV_SHR_U32, t, mop_imm((int32_t)h->lk));
        *j = wsop(NV_AND_B32, t, mop_imm(km));
    } else if (role == NV_WM_ROLE_B) {
        t = wsop(NV_AND_B32, t, mop_imm((int32_t)((1u << (h->lk + h->ln)) - 1u)));
        *i = wsop(NV_AND_B32, t, mop_imm(km));
        *j = wsop(NV_SHR_U32, t, mop_imm((int32_t)h->lk));
    } else {
        *i = wsop(NV_SHR_U32, t, mop_imm((int32_t)h->ln));
        *j = wsop(NV_AND_B32, t, mop_imm((int32_t)((1u << h->ln) - 1u)));
    }
}

static nv_opnd_t wsadr(nv_opnd_t base, nv_opnd_t std, unsigned lay,
                       nv_opnd_t i, nv_opnd_t j, uint32_t bits)
{
    int32_t esz = (int32_t)(bits / 8u);
    return (lay != 0u) ? mma_adr(base, std, j, i, esz)
                       : mma_adr(base, std, i, j, esz);
}

static nv_opnd_t wsget(int as, nv_opnd_t a, uint32_t bits)
{
    nv_opnd_t d = wsu();

    if (bits == 16u) {
        nv_opnd_t h = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
        em1u(ldop(as, NV_RF_U16, 2u), h, a);
        em1u(NV_CVT_U32_U16, d, h);
        return d;
    }
    em1u(ldop(as, NV_RF_U32, bits / 8u), d, a);
    return d;
}

static void wsput(int as, nv_opnd_t a, nv_opnd_t v, uint32_t bits)
{
    nv_opnd_t ops[2] = { a, v };

    if (bits == 16u) {
        ops[1] = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
        em1u(NV_CVT_U16_U32, ops[1], v);
    }
    emit(stop(as, bits == 16u ? NV_RF_U16 : NV_RF_U32, bits / 8u), 0, 2, ops, 0);
}

static nv_opnd_t wswide(uint32_t kind, nv_opnd_t r, uint32_t off)
{
    nv_opnd_t h, f, g;

    if (kind == 2u) {
        nv_opnd_t t = wsop(NV_SHL_B32, r, mop_imm((int32_t)(24u - off)));
        return wsop(NV_SHR_S32, t, mop_imm(24));
    }
    if (off != 0u) r = wsop(NV_SHR_U32, r, mop_imm((int32_t)off));
    if (kind == 3u) return wsop(NV_AND_B32, r, mop_imm(0xFF));
    h = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
    em1u(NV_CVT_U16_U32, h, r);
    f = mop_reg(NV_RF_F32, new_vreg(NV_RF_F32));
    if (kind == 1u) {
        em1u(NV_CVT_F32_BF16, f, h);
        return f;
    }
    g = mop_reg(NV_RF_F16, new_vreg(NV_RF_F16));
    em1u(NV_MOV_B16, g, h);
    em1u(NV_CVT_F32_F16, f, g);
    return f;
}

static nv_opnd_t wsacc(const wsh_t *h, nv_opnd_t r, uint32_t off)
{
    if (h->cb == 16u) return wswide(0u, r, off);
    if (h->kind < 2u) return wsbit(r, NV_RF_F32);
    return r;
}

static nv_opnd_t wsnar(const wsh_t *h, nv_opnd_t v)
{
    nv_opnd_t g, u, z;

    if (h->kind >= 2u) return v;
    if (h->cb == 32u) return wsbit(v, NV_RF_U32);
    g = mop_reg(NV_RF_F16, new_vreg(NV_RF_F16));
    em1u(NV_CVT_F16_F32, g, v);
    u = mop_reg(NV_RF_U16, new_vreg(NV_RF_U16));
    em1u(NV_MOV_B16, u, g);
    z = wsu();
    em1u(NV_CVT_U32_U16, z, u);
    return z;
}

static void wsld(const bir_inst_t *I)
{
    uint32_t row = (uint32_t)(I->subop & 31u);
    unsigned role = ((unsigned)I->subop >> 6) & 3u;
    unsigned lay = ((unsigned)I->subop >> 5) & 1u;
    uint8_t rf = nv_wmrf(row);
    uint8_t n = bir_wmn(row, role);
    uint32_t bits, epr, lc = 0u;
    wsh_t h;

    if (!wsdim(row, &h)) return;
    bits = (role == NV_WM_ROLE_C) ? h.cb : h.ab;
    epr = 32u / bits;
    if (!wsfit(&h, role, n, epr, &lc)) return;

    int as = pas(I->operands[1]);
    nv_opnd_t base = mat_const(I->operands[1], NV_RF_U64);
    nv_opnd_t std = mat_const(I->operands[2], NV_RF_U32);
    uint16_t tb = mma_frag(rf, n);

    em0(NV_BAR_SYNC);
    nv_opnd_t tl = wslane(lc);
    for (uint32_t r = 0; r < n; r++) {
        nv_opnd_t acc = mop_none();

        for (uint32_t x = 0; x < epr; x++) {
            nv_opnd_t i, j, v;

            wsmap(&h, role, tl, r * epr + x, &i, &j);
            v = wsget(as, wsadr(base, std, lay, i, j, bits), bits);
            if (x == 0u) { acc = v; continue; }
            v = wsop(NV_SHL_B32, v, mop_imm((int32_t)(x * bits)));
            acc = wsop(NV_OR_B32, acc, v);
        }
        em1u(NV_MOV_B32, mop_reg(rf, (uint16_t)(tb + r)), acc);
    }
    wm_out(I->operands[0], tb, rf, n);
}

static void wsst(const bir_inst_t *I)
{
    uint32_t row = (uint32_t)(I->subop & 31u);
    unsigned lay = ((unsigned)I->subop >> 5) & 1u;
    uint8_t rf = nv_wmrf(row);
    uint8_t n = bir_wmn(row, NV_WM_ROLE_D);
    uint32_t epr, lc = 0u;
    wsh_t h;

    if (!wsdim(row, &h)) return;
    epr = 32u / h.cb;
    if (!wsfit(&h, NV_WM_ROLE_D, n, epr, &lc)) return;

    uint16_t tb = mma_frag(rf, n);
    wm_in(I->operands[1], tb, rf, n);
    int as = pas(I->operands[0]);
    nv_opnd_t base = mat_const(I->operands[0], NV_RF_U64);
    nv_opnd_t std = mat_const(I->operands[2], NV_RF_U32);

    em0(NV_BAR_SYNC);
    nv_opnd_t tl = wslane(lc);
    for (uint32_t r = 0; r < n; r++) {
        nv_opnd_t g = wsbit(mop_reg(rf, (uint16_t)(tb + r)), NV_RF_U32);

        for (uint32_t x = 0; x < epr; x++) {
            nv_opnd_t i, j;
            nv_opnd_t v = (x != 0u)
                ? wsop(NV_SHR_U32, g, mop_imm((int32_t)(x * h.cb))) : g;

            wsmap(&h, NV_WM_ROLE_D, tl, r * epr + x, &i, &j);
            wsput(as, wsadr(base, std, lay, i, j, h.cb), v, h.cb);
        }
    }
}

static nv_opnd_t wsar[16][8], wsbr[16][8], wscr[8], wsdv[16];

static void wsbrow(uint8_t rf, uint16_t bbas, uint8_t nb, uint32_t sh,
                   uint32_t qb, nv_opnd_t c)
{
    nv_opnd_t cl = wsop(NV_SHL_B32, c, mop_imm((int32_t)sh));

    for (uint32_t q = 0; q < qb; q++) {
        nv_opnd_t ln = wsop(NV_ADD_U32, cl, mop_imm((int32_t)q));

        for (uint32_t j = 0; j < nb; j++)
            wsbr[q][j] = wsshf(mop_reg(rf, (uint16_t)(bbas + j)), ln);
    }
}

static nv_opnd_t wsdot(const wsh_t *h, uint32_t la, uint32_t lb, uint32_t e)
{
    uint32_t ea = 32u / h->ab, ed = 32u / h->cb;
    nv_opnd_t acc = wsacc(h, wscr[e / ed], (e % ed) * h->cb);

    for (uint32_t k = 0; k < (1u << h->lk); k++) {
        uint32_t ka = k & ((1u << la) - 1u);
        uint32_t kb = k & ((1u << lb) - 1u);
        nv_opnd_t a = wswide(h->kind, wsar[k >> la][ka / ea], (ka % ea) * h->ab);
        nv_opnd_t b = wswide(h->kind, wsbr[k >> lb][kb / ea], (kb % ea) * h->ab);

        if (h->kind < 2u)
            acc = wsfop(NV_ADD_F32, acc, wsfop(NV_MUL_F32, a, b));
        else
            acc = wsop(NV_ADD_S32, acc, wsop(NV_MUL_LO_S32, a, b));
    }
    return wsnar(h, acc);
}

static void wspack(const wsh_t *h, uint8_t rf, uint16_t dbas, uint8_t nc)
{
    uint32_t ed = 32u / h->cb;

    for (uint32_t j = 0; j < nc; j++) {
        nv_opnd_t v = wsdv[j * ed];

        for (uint32_t x = 1; x < ed; x++)
            v = wsop(NV_OR_B32, v, wsop(NV_SHL_B32, wsdv[j * ed + x],
                                        mop_imm((int32_t)(x * h->cb))));
        em1u(NV_MOV_B32, mop_reg(rf, (uint16_t)(dbas + j)), v);
    }
}

static void wsmm(const bir_inst_t *I)
{
    uint32_t row = (uint32_t)(I->subop & 31u);
    uint8_t rf = nv_wmrf(row);
    uint8_t na = bir_wmn(row, NV_WM_ROLE_A);
    uint8_t nb = bir_wmn(row, NV_WM_ROLE_B);
    uint8_t nc = bir_wmn(row, NV_WM_ROLE_C);
    uint32_t la = 0u, lb = 0u, ld = 0u;
    wsh_t h;

    if (!wsdim(row, &h)) return;
    if (!wsfit(&h, NV_WM_ROLE_A, na, 32u / h.ab, &la)
        || !wsfit(&h, NV_WM_ROLE_B, nb, 32u / h.ab, &lb)
        || !wsfit(&h, NV_WM_ROLE_C, nc, 32u / h.cb, &ld)) return;
    if (h.lk - la > 4u || h.lk - lb > 4u || na > 8u || nb > 8u || nc > 8u
        || ld > 4u) {
        nv_refuse("a wmma shape the SASS lowering cannot distribute across one warp");
        return;
    }

    uint16_t abas = mma_frag(rf, na);
    wm_in(I->operands[1], abas, rf, na);
    uint16_t bbas = mma_frag(rf, nb);
    wm_in(I->operands[2], bbas, rf, nb);
    uint16_t cbas = mma_frag(rf, nc);
    wm_in(I->operands[3], cbas, rf, nc);
    uint16_t dbas = mma_frag(rf, nc);

    em0(NV_BAR_SYNC);
    nv_opnd_t tl = wslane(ld);
    nv_opnd_t cs = wsop(NV_AND_B32, tl, mop_imm((int32_t)((1u << h.ln) - 1u)));
    nv_opnd_t rr = wsop(NV_SHR_U32, tl, mop_imm((int32_t)h.ln));
    nv_opnd_t rl = wsop(NV_SHL_B32, rr, mop_imm((int32_t)(h.lk - la)));

    for (uint32_t q = 0; q < (1u << (h.lk - la)); q++) {
        nv_opnd_t ln = wsop(NV_ADD_U32, rl, mop_imm((int32_t)q));

        for (uint32_t j = 0; j < na; j++)
            wsar[q][j] = wsshf(mop_reg(rf, (uint16_t)(abas + j)), ln);
    }
    for (uint32_t j = 0; j < nc; j++)
        wscr[j] = wsbit(mop_reg(rf, (uint16_t)(cbas + j)), NV_RF_U32);
    for (uint32_t e = 0; e < (1u << ld); e++) {
        wsbrow(rf, bbas, nb, h.lk - lb, 1u << (h.lk - lb),
               wsop(NV_ADD_U32, cs, mop_imm((int32_t)e)));
        wsdv[e] = wsdot(&h, la, lb, e);
    }
    wspack(&h, rf, dbas, nc);
    wm_out(I->operands[0], dbas, rf, nc);
}

static void is_wld(const bir_inst_t *I)
{
    if (S.nv->sass != 0u) { wsld(I); return; }
    uint32_t row = (uint32_t)(I->subop & 31u);
    unsigned role = ((unsigned)I->subop >> 6) & 3u;
    unsigned lay = ((unsigned)I->subop >> 5) & 1u;
    uint8_t rf = nv_wmrf(row);
    uint8_t n = bir_wmn(row, role);
    uint16_t tb = mma_frag(rf, n);
    nv_opnd_t ops[3] = { mop_reg(rf, tb),
                         mat_const(I->operands[1], NV_RF_U64),
                         mat_const(I->operands[2], NV_RF_U32) };

    emit(NV_WLD, 1, 2, ops, NV_WM_MKF(row, lay, 0u, role, 0u));
    wm_out(I->operands[0], tb, rf, n);
}

static void is_wst(const bir_inst_t *I)
{
    if (S.nv->sass != 0u) { wsst(I); return; }
    uint32_t row = (uint32_t)(I->subop & 31u);
    unsigned lay = ((unsigned)I->subop >> 5) & 1u;
    uint8_t rf = nv_wmrf(row);
    uint8_t n = bir_wmn(row, NV_WM_ROLE_D);
    uint16_t tb = mma_frag(rf, n);

    wm_in(I->operands[1], tb, rf, n);
    nv_opnd_t ops[3] = { mat_const(I->operands[0], NV_RF_U64),
                         mop_reg(rf, tb),
                         mat_const(I->operands[2], NV_RF_U32) };
    emit(NV_WST, 0, 3, ops, NV_WM_MKF(row, lay, 0u, NV_WM_ROLE_D, 0u));
}

static void is_wmma(const bir_inst_t *I)
{
    if (S.nv->sass != 0u) { wsmm(I); return; }
    uint32_t row = (uint32_t)(I->subop & 31u);
    unsigned al = ((unsigned)I->subop >> 5) & 1u;
    unsigned bl = ((unsigned)I->subop >> 6) & 1u;
    unsigned bop = ((unsigned)I->subop >> 7) & 1u;
    uint8_t rf = nv_wmrf(row);
    uint8_t na = bir_wmn(row, NV_WM_ROLE_A);
    uint8_t nb = bir_wmn(row, NV_WM_ROLE_B);
    uint8_t nc = bir_wmn(row, NV_WM_ROLE_C);
    uint16_t ab = mma_frag(rf, na);
    uint16_t bb, cb, db;

    wm_in(I->operands[1], ab, rf, na);
    bb = mma_frag(rf, nb);
    wm_in(I->operands[2], bb, rf, nb);
    cb = mma_frag(rf, nc);
    wm_in(I->operands[3], cb, rf, nc);
    db = mma_frag(rf, nc);

    nv_opnd_t ops[4] = { mop_reg(rf, db), mop_reg(rf, ab),
                         mop_reg(rf, bb), mop_reg(rf, cb) };
    emit(NV_WMMA, 1, 3, ops, NV_WM_MKF(row, al, bl, 0u, bop));
    wm_out(I->operands[0], db, rf, nc);
}

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
        case BIR_UMULHI:
            is_umulhi(idx, I); break;
        case BIR_POPCOUNT: case BIR_CTZ:
        case BIR_CLZ: case BIR_BREV:
            is_bitcount(idx, I); break;
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
            is_swch(I, bir_bi);
            break;
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
            is_barrier(I); break;
        case BIR_FENCE:
            is_fence(I); break;
        case BIR_NANOSLP:
            is_nslp(I); break;
        case BIR_BARRED:
            is_barred(idx, I); break;
        case BIR_GRIDBAR:
            is_gbar(bir_bi); break;

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

        case BIR_INLINE_ASM:
            is_asm(I); break;

        case BIR_CALL:
            is_call(idx, I); break;

        case BIR_GLOBAL_REF:
            is_gref(idx, I); break;

        case BIR_TRAP:
            em0(NV_TRAP); break;

        case BIR_FNREF:
            nv_refuse("taking the address of a device function, which "
                      "this backend inlines away,");
            break;

        case BIR_PRINTF:
            is_prntf(idx, I); break;

        case BIR_MFMA:
        case BIR_MFRG:
            nv_refuse("MFMA (AMD matrix intrinsic on a PTX target)");
            break;

        case BIR_MMA:
            is_mma(I);
            break;

        case BIR_WLD:
            is_wld(I); break;
        case BIR_WST:
            is_wst(I); break;
        case BIR_WMMA:
            is_wmma(I); break;

        default:
            (void)be_fail(BC_E542, "nvptx", bir_op_name(I->op));
            S.had_error = 1;
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
    if (S.nv->num_minst + count > NV_MAX_MINST) {
        nv_cap("machine instructions", (unsigned)NV_MAX_MINST);
        return;
    }
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

static nv_minst_t mk_selp(nv_opnd_t dst, nv_opnd_t src)
{
    nv_minst_t I;
    memset(&I, 0, sizeof(I));
    I.op = NV_SELP_U32;
    I.num_defs = 1;
    I.num_uses = 3;
    I.ops[0] = dst;
    I.ops[1] = mop_imm(1);
    I.ops[2] = mop_imm(0);
    I.ops[3] = src;
    return I;
}

/* ---- Emit one PHI copy into a bridge block ----
 * Appends the copy instruction(s) to the bridge block.
 * Called BEFORE the bridge's terminating bra is emitted. */
static void brg_copy(nv_pcopy_t *pc)
{
    if (pc->src.kind == NV_MOP_REG && pc->src.rfile == NV_RF_PRED
        && pc->mop != NV_MOV_PRED) {
        nv_opnd_t tmp = pc->dst;
        if (tmp.rfile != NV_RF_U32)
            tmp = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
        nv_opnd_t s_ops[4] = { tmp, mop_imm(1), mop_imm(0), pc->src };
        emit(NV_SELP_U32, 1, 3, s_ops, 0);
        if (tmp.reg_num != pc->dst.reg_num || tmp.rfile != pc->dst.rfile) {
            nv_opnd_t c_ops[2] = { pc->dst, tmp };
            emit(pc->mop, 1, 1, c_ops, 0);
        }
        return;
    }
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
    } else if (pc->src.kind == NV_MOP_REG
            && pc->src.rfile != pc->dst.rfile) {
        cocpy(pc->dst, pc->src, 1);
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
                    if (S_npc >= NV_MAX_PCOPY)
                        nv_cap("phi copies", (unsigned)NV_MAX_PCOPY);
                    if (S_npc < NV_MAX_PCOPY) {
                        uint16_t sop = movof(ci->src.rfile);
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
        if (S.nv->num_mblk >= NV_MAX_MBLK) {
            nv_cap("machine blocks", (unsigned)NV_MAX_MBLK);
            continue;
        }

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

        if (pc->src.kind == NV_MOP_REG && pc->src.rfile == NV_RF_PRED
            && pc->mop != NV_MOV_PRED) {
            nv_opnd_t tmp = pc->dst;
            if (tmp.rfile != NV_RF_U32)
                tmp = mop_reg(NV_RF_U32, new_vreg(NV_RF_U32));
            insts[0] = mk_selp(tmp, pc->src);
            count = 1;
            if (tmp.reg_num != pc->dst.reg_num
                || tmp.rfile != pc->dst.rfile) {
                insts[1] = mk_mov(pc->mop, pc->dst, tmp);
                count = 2;
            }
        } else if (pc->mop == NV_MOV_PRED && pc->src.kind == NV_MOP_IMM) {
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
        } else if (pc->src.kind == NV_MOP_REG
                && pc->src.rfile != pc->dst.rfile) {
            uint16_t cop = cvop(pc->src.rfile, pc->dst.rfile, 1);
            if (cop == NV_OP_COUNT) {
                (void)be_fail(BC_E844, curfn(), rfnam(pc->src.rfile),
                              rfnam(pc->dst.rfile));
                S.had_error = 1;
                cop = pc->mop;
            }
            insts[0] = mk_mov(cop, pc->dst, pc->src);
            count = 1;
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

static void fmark(void)
{
    uint32_t pass;
    for (pass = 0; pass < S.bir->num_funcs + 1u; pass++) {
        int grew = 0;
        for (uint32_t fi = 0; fi < S.bir->num_funcs; fi++) {
            const bir_func_t *F = &S.bir->funcs[fi];
            if (!(F->cuda_flags & CUDA_GLOBAL) && !S.fneed[fi]) continue;
            for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
                uint32_t bir_bi = F->first_block + bi;
                const bir_block_t *B;
                if (bir_bi >= S.bir->num_blocks) break;
                B = &S.bir->blocks[bir_bi];
                for (uint32_t ii = 0; ii < B->num_insts; ii++) {
                    const bir_inst_t *I = &S.bir->insts[B->first_inst + ii];
                    uint32_t cf;
                    if (I->op != BIR_CALL) continue;
                    cf = I->operands[0];
                    if (cf >= S.bir->num_funcs || S.fneed[cf]) continue;
                    if (S.bir->funcs[cf].num_blocks == 0) continue;
                    if (S.bir->funcs[cf].cuda_flags & CUDA_GLOBAL) continue;
                    S.fneed[cf] = 1;
                    grew = 1;
                }
            }
        }
        if (!grew) return;
    }
}

static void fsig(nv_mfunc_t *MF, const bir_func_t *F)
{
    uint32_t ft = F->type;
    const bir_type_t *FT;
    uint32_t np, i;

    if (ft >= S.bir->num_types) return;
    FT = &S.bir->types[ft];
    if (FT->kind != BIR_TYPE_FUNC) return;

    np = FT->num_fields;
    if (np > NV_MAX_PARAMS) np = NV_MAX_PARAMS;
    for (i = 0; i < np; i++) {
        uint32_t pt = (FT->count + i < S.bir->num_type_fields)
                    ? S.bir->type_fields[FT->count + i] : 0;
        if (!psinf(&MF->params[i], pt))
            nv_refuse("a parameter this large");
    }
    MF->num_params = np;

    if (FT->inner < S.bir->num_types
     && S.bir->types[FT->inner].kind != BIR_TYPE_VOID) {
        MF->hasret = 1;
        MF->retrf = bir_rfile(FT->inner);
    }
}

static int isel_func(uint32_t fi)
{
    const bir_func_t *F = &S.bir->funcs[fi];

    if (!(F->cuda_flags & CUDA_GLOBAL) && !S.fneed[fi]) return BC_OK;
    if (F->num_blocks == 0) return BC_OK;
    if (S.nv->num_mfunc >= NV_MAX_MFUNC) {
        nv_cap("machine functions", (unsigned)NV_MAX_MFUNC);
        return BC_ERR_NVIDIA;
    }

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
    MF->retrf = 0;
    MF->hasret = 0;
    memset(MF->rc, 0, sizeof(MF->rc));
    memset(MF->params, 0, sizeof(MF->params));

    S.cur_func = mfi;
    S.blk_lo = F->first_block;
    S.blk_hi = F->first_block + F->num_blocks;
    S.lcl_off = 0;
    S.shr_off = 0;
    S.lcl_alg = 8;
    S.shr_alg = 4;

    /* Reset per-function vreg counters */
    memset(S.nv->rc, 0, sizeof(S.nv->rc));
    /* Start at 1 so vreg 0 is sentinel */
    for (int r = 0; r < NV_RF_COUNT; r++)
        S.nv->rc[r] = 1;

    if (!MF->is_kern) fsig(MF, F);

    /* Build parameter descriptors from BIR */
    uint32_t first_block = F->first_block;
    if (MF->is_kern && first_block < S.bir->num_blocks) {
        const bir_block_t *entry = &S.bir->blocks[first_block];
        uint32_t pi = 0;
        int pg = 4 * NV_MAX_PARAMS;
        for (uint32_t ii = 0; ii < entry->num_insts && pg > 0; ii++, pg--) {
            const bir_inst_t *I = &S.bir->insts[entry->first_inst + ii];
            if (I->op != BIR_PARAM) continue;
            if (pi >= NV_MAX_PARAMS) {
                nv_cap("kernel parameters", (unsigned)NV_MAX_PARAMS);
                break;
            }
            if (!psinf(&MF->params[pi], I->type))
                nv_refuse("a parameter this large");
            pi++;
        }
        MF->num_params = pi;
    }

    /* Pre-create block map */
    {
        uint32_t ex = 0;
        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            uint32_t bir_bi = F->first_block + bi;

            S.block_map[bir_bi] = S.nv->num_mblk + bi + ex;
            if (S.nv->sass) ex += GB_NBLK * gbcnt(bir_bi);
            S.blk_lst[bir_bi] = S.nv->num_mblk + bi + ex;
        }
    }

    /* Select instructions per block */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        nv_mblk_t *MB;
        if (S.nv->num_mblk >= NV_MAX_MBLK) {
            nv_cap("machine blocks", (unsigned)NV_MAX_MBLK);
            break;
        }

        S.cmb = S.nv->num_mblk;
        MB = &S.nv->mblks[S.cmb];
        MB->first_inst = S.nv->num_minst;
        MB->bir_block = bir_bi;

        isel_blk(bir_bi);

        MB = &S.nv->mblks[S.cmb];
        MB->num_insts = S.nv->num_minst - MB->first_inst;
        S.nv->num_mblk = S.cmb + 1u;
    }

    /* Insert deferred PHI copies into predecessor blocks.
     * Must happen before register counts are recorded since
     * pred-to-pred copies allocate temporary U32 registers. */
    phi_fix();

    if (S.nv->num_mblk - MF->first_blk > 0xFFFFu) {
        nv_cap("blocks in one function", 0xFFFFu);
        return BC_ERR_NVIDIA;
    }
    MF->num_blks = (uint16_t)(S.nv->num_mblk - MF->first_blk);
    MF->lds_bytes = S.shr_off;
    MF->lcl_bytes = S.lcl_off;
    MF->lds_alg   = S.shr_alg;
    MF->lcl_alg   = S.lcl_alg;

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
    nv->num_asm = 0;
    nv->num_call = 0;
    nv->vprt = 0;
    nv->out_len = 0;

    memset(nv->rc, 0, sizeof(nv->rc));
    memset(nv->val_vreg, 0, (size_t)bir->num_insts * sizeof(nv->val_vreg[0]));
    memset(nv->val_rfile, 0, (size_t)bir->num_insts * sizeof(nv->val_rfile[0]));

    fmark();

    int guard = 8192;
    for (uint32_t fi = 0; fi < bir->num_funcs && guard > 0; fi++, guard--) {
        int rc = isel_func(fi);
        if (rc != BC_OK) return rc;
    }

    return (S.unsz || S.had_error) ? BC_ERR_NVIDIA : BC_OK;
}
