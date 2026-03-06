#include "amdgpu.h"
#include <string.h>

/*
 * AMDGPU instruction selection: BIR SSA -> AMDGCN machine IR.
 * Two passes: divergence analysis, then instruction selection.
 * Maps all BIR opcodes to RDNA3 machine instructions with the
 * grim determination of someone who's read the ISA manual twice.
 */

/* ---- Static State ---- */

/* Large state kept static to avoid stack overflow (~2 MB) */
static struct {
    amd_module_t    *amd;
    const bir_module_t *bir;

    /* Current function context */
    uint32_t        func_idx;
    uint32_t        func_first_inst;   /* BIR inst base for current func */
    uint32_t        func_first_block;  /* BIR block base for current func */
    uint16_t        num_params;
    uint16_t        is_kernel;

    /* Scratch offset tracking */
    uint32_t        scratch_offset;

    /* LDS (shared memory) offset tracking */
    uint32_t        lds_offset;

    /* Pointer param SGPR pair allocation */
    uint16_t        next_param_sgpr;

    /* Current BIR block being processed */
    uint32_t        current_bir_block;

    /* Divergent region tracking — EXEC mask save/restore */
    #define MAX_DIV_REGIONS 64
    struct {
        uint32_t saved_vreg;    /* virtual SGPR holding saved EXEC */
        uint32_t false_bir;     /* BIR block for else path */
        uint32_t merge_bir;     /* BIR block for merge (post-dominator) */
        int      has_else;      /* 1 = diamond (then + else), 0 = triangle */
    } div_stack[MAX_DIV_REGIONS];
    uint32_t        div_depth;

    /* Branch suppression for divergent diamonds: then-block's BR to merge
       gets suppressed so it falls through to the else-block. */
    uint32_t        suppress_src;   /* BIR block whose BR to suppress */
    uint32_t        suppress_dst;   /* target BIR block of the suppressed BR */

    /* Saved thread IDs: v0/v1/v2 must be copied before param loads clobber them */
    uint32_t        saved_tid[3];   /* virtual VGPR holding saved threadIdx.x/y/z */

    /* Dynamic SGPR layout — set per-kernel based on needs_dispatch */
    uint16_t        sgpr_kernarg;   /* SGPR pair base for kernarg ptr */
    uint16_t        sgpr_dispatch;  /* SGPR pair base for dispatch ptr (if enabled) */
    uint16_t        sgpr_wg_base;   /* first SGPR for workgroup IDs */
    uint16_t        kern_reserved;  /* first allocatable SGPR after system regs */
    uint8_t         needs_dispatch; /* this kernel uses dispatch_ptr */
    uint8_t         max_dim;        /* highest dim needed (0=x, 1=xy, 2=xyz) */

    /* Hidden kernarg offset for __device__/__constant__ global pointers */
    uint32_t        hkrarg;

    /* Block mapping: BIR block index -> machine block index */
    uint32_t        block_map[BIR_MAX_BLOCKS];
} S;

/* ---- Target Helpers ---- */

static int is_cdna(void) { return S.amd->target <= AMD_TARGET_GFX942; }

/* ---- Divergence Analysis ---- */

static int is_divergent(uint32_t bir_inst)
{
    uint32_t word = bir_inst / 32;
    uint32_t bit  = bir_inst % 32;
    return (S.amd->divergent[word] >> bit) & 1;
}

static void mark_divergent(uint32_t bir_inst)
{
    uint32_t word = bir_inst / 32;
    uint32_t bit  = bir_inst % 32;
    S.amd->divergent[word] |= (1u << bit);
}

static int val_is_divergent(uint32_t val)
{
    if (val == BIR_VAL_NONE) return 0;
    if (BIR_VAL_IS_CONST(val)) return 0;
    return is_divergent(BIR_VAL_INDEX(val));
}

/* Get operands for an instruction, handling overflow */
static uint32_t get_num_ops(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW)
        return I->operands[1];
    return I->num_operands;
}

static uint32_t get_op(const bir_inst_t *I, uint32_t idx)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t base = I->operands[0];
        if (base + idx < BIR_MAX_EXTRA_OPS)
            return S.bir->extra_operands[base + idx];
        return BIR_VAL_NONE;
    }
    if (idx < BIR_OPERANDS_INLINE)
        return I->operands[idx];
    return BIR_VAL_NONE;
}

/* Find the unconditional branch target of a BIR block, or 0xFFFFFFFF if none */
static uint32_t bir_block_successor(uint32_t bir_bi)
{
    if (bir_bi >= S.bir->num_blocks) return 0xFFFFFFFF;
    const bir_block_t *B = &S.bir->blocks[bir_bi];
    if (B->num_insts == 0) return 0xFFFFFFFF;
    const bir_inst_t *last = &S.bir->insts[B->first_inst + B->num_insts - 1];
    if (last->op == BIR_BR)
        return last->operands[0];
    return 0xFFFFFFFF;
}

/* Find the merge block for a divergent if-then or if-then-else.
   Triangle: then→false (false IS the merge).
   Diamond: then→merge AND false→merge (both converge). */
static uint32_t find_merge_block(uint32_t true_bir, uint32_t false_bir)
{
    uint32_t true_succ = bir_block_successor(true_bir);
    if (true_succ == false_bir)
        return false_bir; /* triangle: then branches to false/merge */
    uint32_t false_succ = bir_block_successor(false_bir);
    if (true_succ != 0xFFFFFFFF && true_succ == false_succ)
        return true_succ; /* diamond: both converge */
    return false_bir; /* fallback: treat as triangle */
}

/*
 * Forward dataflow. Seeds: THREAD_ID = divergent, BLOCK_ID/DIM/GRID_DIM = uniform,
 * constants = uniform, PARAMs = uniform. Propagate: any divergent input -> divergent output.
 * PHI: divergent if any incoming value is divergent.
 * Iterate until fixpoint (bounded: each bit set at most once).
 */
static void divergence_analysis(const bir_func_t *F)
{
    const bir_module_t *M = S.bir;
    int changed = 1;
    uint32_t guard = 0;

    /* Seed divergent values: thread IDs, shuffles, device func params */
    int is_device = !(S.is_kernel);
    for (uint32_t bi = 0; bi < F->num_blocks && guard < 1000000; bi++, guard++) {
        const bir_block_t *B = &M->blocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < B->num_insts && guard < 1000000; ii++, guard++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            switch (I->op) {
            case BIR_THREAD_ID:
            case BIR_SHFL: case BIR_SHFL_UP:
            case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            case BIR_ALLOCA: /* per-thread scratch — inherently divergent */
            case BIR_MFMA:  /* matrix result is a collective warp operation */
                mark_divergent(idx);
                break;
            case BIR_PARAM:
                /* Device function params arrive in VGPRs — divergent by nature */
                if (is_device)
                    mark_divergent(idx);
                break;
            default:
                break;
            }
        }
    }

    /* Iterate until fixpoint */
    while (changed && guard < 2000000) {
        changed = 0;
        for (uint32_t bi = 0; bi < F->num_blocks && guard < 2000000; bi++, guard++) {
            const bir_block_t *B = &M->blocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < B->num_insts && guard < 2000000; ii++, guard++) {
                uint32_t idx = B->first_inst + ii;
                const bir_inst_t *I = &M->insts[idx];

                if (is_divergent(idx)) continue;

                /* Skip instructions that are inherently uniform */
                if (I->op == BIR_BLOCK_ID || I->op == BIR_BLOCK_DIM ||
                    I->op == BIR_GRID_DIM || I->op == BIR_PARAM)
                    continue;

                /* Check if any operand is divergent */
                uint32_t nops = get_num_ops(I);
                int any_div = 0;

                if (I->op == BIR_PHI) {
                    /* PHI: check value operands (every other one) */
                    for (uint32_t k = 1; k < nops; k += 2) {
                        if (val_is_divergent(get_op(I, k))) {
                            any_div = 1;
                            break;
                        }
                    }
                } else if (I->op == BIR_LOAD) {
                    /* Load from divergent address -> divergent */
                    if (nops > 0 && val_is_divergent(get_op(I, 0)))
                        any_div = 1;
                } else if (I->op == BIR_BR || I->op == BIR_BR_COND ||
                           I->op == BIR_RET || I->op == BIR_UNREACHABLE ||
                           I->op == BIR_STORE) {
                    /* Terminators and stores don't produce values that need tracking */
                    continue;
                } else if (I->op == BIR_CALL) {
                    /* Conservative: calls are divergent if any arg is */
                    for (uint32_t k = 1; k < nops; k++) {
                        if (val_is_divergent(get_op(I, k))) {
                            any_div = 1;
                            break;
                        }
                    }
                } else {
                    /* General: any divergent operand -> divergent result */
                    for (uint32_t k = 0; k < nops; k++) {
                        uint32_t op = get_op(I, k);
                        /* Skip block references in branch targets */
                        if (I->op == BIR_SWITCH && k == 1) continue;
                        if (val_is_divergent(op)) {
                            any_div = 1;
                            break;
                        }
                    }
                }

                if (any_div) {
                    mark_divergent(idx);
                    changed = 1;
                }
            }
        }
    }
}

/* ---- Virtual Register Allocation ---- */

static uint32_t new_vreg(int is_vector)
{
    uint32_t v = S.amd->vreg_count;
    if (v >= AMD_MAX_VREGS - 1)
        return AMD_MAX_VREGS - 1; /* saturate — better than wandering into the void */
    S.amd->vreg_count = v + 1;
    S.amd->reg_file[v] = (uint8_t)is_vector;
    return v;
}

/* Map a BIR instruction result to a virtual register */
static uint32_t map_bir_val(uint32_t bir_inst, int is_vector)
{
    if (bir_inst < BIR_MAX_INSTS && S.amd->val_vreg[bir_inst] != 0xFFFFFFFF)
        return S.amd->val_vreg[bir_inst];
    uint32_t v = new_vreg(is_vector);
    if (bir_inst < BIR_MAX_INSTS) {
        S.amd->val_vreg[bir_inst] = v;
        S.amd->val_file[bir_inst] = (uint8_t)is_vector;
    }
    return v;
}

/* ---- Machine Instruction Emission ---- */

static moperand_t mop_none(void)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_NONE;
    return o;
}

static moperand_t mop_sgpr(uint16_t reg)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_SGPR;
    o.reg_num = reg;
    return o;
}

static moperand_t mop_vgpr(uint16_t reg)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_VGPR;
    o.reg_num = reg;
    return o;
}

static moperand_t mop_vreg_s(uint16_t vreg)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_VREG_S;
    o.reg_num = vreg;
    return o;
}

static moperand_t mop_vreg_v(uint16_t vreg)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_VREG_V;
    o.reg_num = vreg;
    return o;
}

static moperand_t mop_imm(int32_t val)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_IMM;
    o.imm = val;
    return o;
}

static moperand_t mop_label(uint32_t mblock)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_LABEL;
    o.imm = (int32_t)mblock;
    return o;
}

static moperand_t mop_special(int id)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_SPECIAL;
    o.imm = id;
    return o;
}

static moperand_t mop_vreg(uint16_t vreg, int is_vector)
{
    return is_vector ? mop_vreg_v(vreg) : mop_vreg_s(vreg);
}

/* Emit a machine instruction, returns its index */
static uint32_t emit_minst(uint16_t op, uint8_t ndefs, uint8_t nuses,
                           moperand_t *ops, uint16_t flags)
{
    amd_module_t *A = S.amd;
    if (A->num_minsts >= AMD_MAX_MINSTS) return A->num_minsts - 1;
    uint32_t idx = A->num_minsts++;
    minst_t *mi = &A->minsts[idx];
    mi->op = op;
    mi->num_defs = ndefs;
    mi->num_uses = nuses;
    mi->flags = flags;
    uint8_t total = ndefs + nuses;
    for (uint8_t i = 0; i < total && i < MINST_MAX_OPS; i++)
        mi->operands[i] = ops[i];
    for (uint8_t i = total; i < MINST_MAX_OPS; i++)
        mi->operands[i] = mop_none();
    return idx;
}

/* Convenience: emit 1-def, N-use instruction */
static uint32_t emit1(uint16_t op, moperand_t dst, moperand_t s0)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = dst; ops[1] = s0;
    return emit_minst(op, 1, 1, ops, 0);
}

static uint32_t emit2(uint16_t op, moperand_t dst, moperand_t s0, moperand_t s1)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = dst; ops[1] = s0; ops[2] = s1;
    return emit_minst(op, 1, 2, ops, 0);
}

static uint32_t emit3(uint16_t op, moperand_t dst,
                      moperand_t s0, moperand_t s1, moperand_t s2)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = dst; ops[1] = s0; ops[2] = s1; ops[3] = s2;
    return emit_minst(op, 1, 3, ops, 0);
}

/* Emit 0-def instruction (stores, branches, barriers) */
static uint32_t emit0_1(uint16_t op, moperand_t s0)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = s0;
    return emit_minst(op, 0, 1, ops, 0);
}

static uint32_t emit0_2(uint16_t op, moperand_t s0, moperand_t s1)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = s0; ops[1] = s1;
    return emit_minst(op, 0, 2, ops, 0);
}

static uint32_t emit0_0(uint16_t op, uint16_t flags)
{
    moperand_t ops[MINST_MAX_OPS];
    return emit_minst(op, 0, 0, ops, flags);
}

/* Emit with explicit flags */
static uint32_t emit2f(uint16_t op, moperand_t dst, moperand_t s0,
                       moperand_t s1, uint16_t flags)
{
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = dst; ops[1] = s0; ops[2] = s1;
    return emit_minst(op, 1, 2, ops, flags);
}

/* ---- Wait Helpers (GFX11 vs GFX12) ---- */

/* GFX12 splits s_waitcnt into per-counter instructions.
   These helpers pick the right one so isel doesn't have to care. */

static void emit_wait_vm(void)
{
    if (S.amd->target >= AMD_TARGET_GFX1200) {
        emit0_0(AMD_S_WAIT_LOADCNT, 0);
        emit0_0(AMD_S_WAIT_STORECNT, 0);
    } else {
        emit0_0(AMD_S_WAITCNT, AMD_WAIT_VMCNT0);
    }
}

static void emit_wait_smem(void)
{
    if (S.amd->target >= AMD_TARGET_GFX1200) {
        emit0_0(AMD_S_WAIT_KMCNT, 0);
    } else {
        emit0_0(AMD_S_WAITCNT, AMD_WAIT_LGKMCNT0);
    }
}

static void emit_wait_ds(void)
{
    if (S.amd->target >= AMD_TARGET_GFX1200) {
        emit0_0(AMD_S_WAIT_DSCNT, 0);
    } else {
        emit0_0(AMD_S_WAITCNT, AMD_WAIT_LGKMCNT0);
    }
}

static void emit_wait_all(void)
{
    if (S.amd->target >= AMD_TARGET_GFX1200) {
        emit0_0(AMD_S_WAIT_LOADCNT, 0);
        emit0_0(AMD_S_WAIT_STORECNT, 0);
        emit0_0(AMD_S_WAIT_KMCNT, 0);
        emit0_0(AMD_S_WAIT_DSCNT, 0);
    } else {
        emit0_0(AMD_S_WAITCNT, AMD_WAIT_ALL);
    }
}

/* ---- Resolve BIR Value to Machine Operand ---- */

static moperand_t resolve_val(uint32_t val, int want_vector)
{
    if (val == BIR_VAL_NONE) return mop_imm(0);

    if (BIR_VAL_IS_CONST(val)) {
        uint32_t ci = BIR_VAL_INDEX(val);
        if (ci < S.bir->num_consts) {
            const bir_const_t *C = &S.bir->consts[ci];
            switch (C->kind) {
            case BIR_CONST_INT:
                return mop_imm((int32_t)C->d.ival);
            case BIR_CONST_FLOAT: {
                /* Reinterpret float bits as int32 for immediate */
                float f = (float)C->d.fval;
                int32_t bits;
                memcpy(&bits, &f, 4);
                return mop_imm(bits);
            }
            case BIR_CONST_NULL:
            case BIR_CONST_ZERO:
                return mop_imm(0);
            case BIR_CONST_UNDEF:
                return mop_imm(0);
            default:
                return mop_imm(0);
            }
        }
        return mop_imm(0);
    }

    uint32_t idx = BIR_VAL_INDEX(val);
    if (idx < BIR_MAX_INSTS && S.amd->val_vreg[idx] != 0xFFFFFFFF) {
        uint32_t vreg = S.amd->val_vreg[idx];
        int is_vec = S.amd->val_file[idx];
        if (want_vector && !is_vec) {
            /* Need to move scalar to vector */
            uint32_t vv = new_vreg(1);
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vv), mop_vreg_s((uint16_t)vreg));
            return mop_vreg_v((uint16_t)vv);
        }
        return mop_vreg((uint16_t)vreg, is_vec);
    }

    return mop_imm(0);
}

/* Ensure an operand is in a VGPR (move from SGPR/imm if needed) */
static moperand_t ensure_vgpr(moperand_t op)
{
    if (op.kind == MOP_VREG_V || op.kind == MOP_VGPR) return op;
    uint32_t v = new_vreg(1);
    emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)v), op);
    return mop_vreg_v((uint16_t)v);
}

/* Get the BIR type width in bits. Default 32 for pointers, etc. */
static int bir_type_width(uint32_t tidx)
{
    if (tidx >= S.bir->num_types) return 32;
    const bir_type_t *t = &S.bir->types[tidx];
    if (t->kind == BIR_TYPE_INT || t->kind == BIR_TYPE_FLOAT
        || t->kind == BIR_TYPE_BFLOAT)
        return t->width;
    if (t->kind == BIR_TYPE_PTR) return 64;
    if (t->kind == BIR_TYPE_ARRAY)
        return (int)t->count * bir_type_width(t->inner);
    return 32;
}

static uint32_t arrsz(uint32_t tidx)
{
    if (tidx >= S.bir->num_types) return 4;
    const bir_type_t *t = &S.bir->types[tidx];
    if (t->kind == BIR_TYPE_ARRAY)
        return t->count * arrsz(t->inner);
    if (t->kind == BIR_TYPE_INT || t->kind == BIR_TYPE_FLOAT
        || t->kind == BIR_TYPE_BFLOAT)
        return t->width / 8;
    if (t->kind == BIR_TYPE_PTR) return 8;
    if (t->kind == BIR_TYPE_STRUCT) return t->num_fields * 4;
    return 4;
}

/* Get type kind */
static int bir_type_kind(uint32_t tidx)
{
    if (tidx >= S.bir->num_types) return BIR_TYPE_INT;
    return S.bir->types[tidx].kind;
}

/* Get the address space of a pointer type */
static int get_addrspace(uint32_t tidx)
{
    if (tidx >= S.bir->num_types) return BIR_AS_GLOBAL;
    const bir_type_t *t = &S.bir->types[tidx];
    if (t->kind == BIR_TYPE_PTR) return t->addrspace;
    return BIR_AS_GLOBAL;
}

/* Get pointee type's size in bytes */
static uint32_t pointee_size(uint32_t ptr_type)
{
    if (ptr_type >= S.bir->num_types) return 4;
    const bir_type_t *pt = &S.bir->types[ptr_type];
    if (pt->kind != BIR_TYPE_PTR || pt->inner >= S.bir->num_types) return 4;
    return arrsz(pt->inner);
}

/* ---- Instruction Selection: Individual BIR Opcodes ---- */

static void isel_arith(uint32_t idx, const bir_inst_t *I, int div)
{
    moperand_t src0 = resolve_val(I->operands[0], div);
    moperand_t src1 = resolve_val(I->operands[1], div);
    uint32_t vr = map_bir_val(idx, div);
    moperand_t dst = mop_vreg((uint16_t)vr, div);

    if (div) {
        src0 = ensure_vgpr(src0);
        /* For VOP2, src1 must be VGPR */
        switch (I->op) {
        case BIR_ADD:  emit2(AMD_V_ADD_U32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_SUB:  emit2(AMD_V_SUB_U32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_MUL:  emit2(AMD_V_MUL_LO_U32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FADD: emit2(AMD_V_ADD_F32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FSUB: emit2(AMD_V_SUB_F32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FMUL: emit2(AMD_V_MUL_F32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FMAX: emit2(AMD_V_MAX_F32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FMIN: emit2(AMD_V_MIN_F32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_FDIV: {
            /* v_rcp_f32 + v_mul_f32 */
            uint32_t tmp = new_vreg(1);
            emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)tmp), ensure_vgpr(src1));
            emit2(AMD_V_MUL_F32, dst, src0, mop_vreg_v((uint16_t)tmp));
            break;
        }
        case BIR_FREM: {
            /* floor(a/b)*b, then a - result. Use rcp approximation. */
            uint32_t rcp = new_vreg(1);
            uint32_t q = new_vreg(1);
            uint32_t t = new_vreg(1);
            moperand_t vs1 = ensure_vgpr(src1);
            emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), vs1);
            emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)q), src0, mop_vreg_v((uint16_t)rcp));
            emit1(AMD_V_CVT_I32_F32, mop_vreg_v((uint16_t)t), mop_vreg_v((uint16_t)q));
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)q), mop_vreg_v((uint16_t)t));
            emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)t), mop_vreg_v((uint16_t)q), vs1);
            emit2(AMD_V_SUB_F32, dst, src0, mop_vreg_v((uint16_t)t));
            break;
        }
        case BIR_AND:  emit2(AMD_V_AND_B32, dst, src0, ensure_vgpr(src1)); break;
        case BIR_OR:   emit2(AMD_V_OR_B32,  dst, src0, ensure_vgpr(src1)); break;
        case BIR_XOR:  emit2(AMD_V_XOR_B32, dst, src0, ensure_vgpr(src1)); break;
        /* Reversed operand order for vector shift: VOP2 shift uses REV encoding */
        case BIR_SHL:  emit2(AMD_V_LSHLREV_B32, dst, ensure_vgpr(src1), src0); break;
        case BIR_LSHR: emit2(AMD_V_LSHRREV_B32, dst, ensure_vgpr(src1), src0); break;
        case BIR_ASHR: emit2(AMD_V_ASHRREV_I32, dst, ensure_vgpr(src1), src0); break;
        /* Integer div/rem: no hardware instruction, use float approx for now */
        case BIR_SDIV: case BIR_UDIV: {
            uint32_t fa = new_vreg(1), fb = new_vreg(1), rcp = new_vreg(1);
            uint32_t fq = new_vreg(1);
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fa), src0);
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fb), ensure_vgpr(src1));
            emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), mop_vreg_v((uint16_t)fb));
            emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)fq),
                  mop_vreg_v((uint16_t)fa), mop_vreg_v((uint16_t)rcp));
            emit1(AMD_V_CVT_I32_F32, dst, mop_vreg_v((uint16_t)fq));
            break;
        }
        case BIR_SREM: case BIR_UREM: {
            /* a - (a/b)*b */
            uint32_t fa = new_vreg(1), fb = new_vreg(1), rcp = new_vreg(1);
            uint32_t fq = new_vreg(1), qi = new_vreg(1), prod = new_vreg(1);
            moperand_t vs1 = ensure_vgpr(src1);
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fa), src0);
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fb), vs1);
            emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), mop_vreg_v((uint16_t)fb));
            emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)fq),
                  mop_vreg_v((uint16_t)fa), mop_vreg_v((uint16_t)rcp));
            emit1(AMD_V_CVT_I32_F32, mop_vreg_v((uint16_t)qi), mop_vreg_v((uint16_t)fq));
            emit2(AMD_V_MUL_LO_U32, mop_vreg_v((uint16_t)prod),
                  mop_vreg_v((uint16_t)qi), vs1);
            emit2(AMD_V_SUB_U32, dst, src0, mop_vreg_v((uint16_t)prod));
            break;
        }
        default: break;
        }
    } else {
        /* Scalar (uniform) path */
        switch (I->op) {
        case BIR_ADD:
            if (is_cdna()) {
                /* GFX9/CDNA: s_add yields 0 when both operands come
                 * from SMEM loads on MI300X.  hipcc dodges this via
                 * scheduling; we promote to VALU.  Not optional. */
                S.amd->val_file[idx] = 1;
                S.amd->reg_file[vr] = 1;
                emit2(AMD_V_ADD_U32, mop_vreg_v((uint16_t)vr),
                      ensure_vgpr(src0), ensure_vgpr(src1));
            } else {
                emit2(AMD_S_ADD_I32, dst, src0, src1);
            }
            break;
        case BIR_SUB:
            if (is_cdna()) {
                S.amd->val_file[idx] = 1;
                S.amd->reg_file[vr] = 1;
                emit2(AMD_V_SUB_U32, mop_vreg_v((uint16_t)vr),
                      ensure_vgpr(src0), ensure_vgpr(src1));
            } else {
                emit2(AMD_S_SUB_U32, dst, src0, src1);
            }
            break;
        case BIR_MUL:  emit2(AMD_S_MUL_I32, dst, src0, src1); break;
        case BIR_AND:  emit2(AMD_S_AND_B32, dst, src0, src1); break;
        case BIR_OR:   emit2(AMD_S_OR_B32,  dst, src0, src1); break;
        case BIR_XOR:  emit2(AMD_S_XOR_B32, dst, src0, src1); break;
        case BIR_SHL:  emit2(AMD_S_LSHL_B32, dst, src0, src1); break;
        case BIR_LSHR: emit2(AMD_S_LSHR_B32, dst, src0, src1); break;
        case BIR_ASHR: emit2(AMD_S_ASHR_I32, dst, src0, src1); break;
        /* Float ops: no scalar float ALU on AMDGPU, always vector */
        case BIR_FADD: case BIR_FSUB: case BIR_FMUL:
        case BIR_FMAX: case BIR_FMIN:
        case BIR_FDIV: case BIR_FREM: {
            S.amd->val_file[idx] = 1;
            S.amd->reg_file[vr] = 1;
            moperand_t vdst = mop_vreg_v((uint16_t)vr);
            moperand_t vs0 = ensure_vgpr(src0);
            moperand_t vs1 = ensure_vgpr(src1);
            if (I->op == BIR_FADD) emit2(AMD_V_ADD_F32, vdst, vs0, vs1);
            else if (I->op == BIR_FSUB) emit2(AMD_V_SUB_F32, vdst, vs0, vs1);
            else if (I->op == BIR_FMUL) emit2(AMD_V_MUL_F32, vdst, vs0, vs1);
            else if (I->op == BIR_FMAX) emit2(AMD_V_MAX_F32, vdst, vs0, vs1);
            else if (I->op == BIR_FMIN) emit2(AMD_V_MIN_F32, vdst, vs0, vs1);
            else if (I->op == BIR_FDIV) {
                uint32_t rcp = new_vreg(1);
                emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), vs1);
                emit2(AMD_V_MUL_F32, vdst, vs0, mop_vreg_v((uint16_t)rcp));
            } else { /* FREM */
                uint32_t rcp = new_vreg(1), q = new_vreg(1), t = new_vreg(1);
                emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), vs1);
                emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)q), vs0, mop_vreg_v((uint16_t)rcp));
                emit1(AMD_V_CVT_I32_F32, mop_vreg_v((uint16_t)t), mop_vreg_v((uint16_t)q));
                emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)q), mop_vreg_v((uint16_t)t));
                emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)t), mop_vreg_v((uint16_t)q), vs1);
                emit2(AMD_V_SUB_F32, vdst, vs0, mop_vreg_v((uint16_t)t));
            }
            break;
        }
        /* Scalar int div/rem: use mul_i32 with float rcp approximation */
        case BIR_SDIV: case BIR_UDIV: case BIR_SREM: case BIR_UREM: {
            /* Promote to vector for the float intermediate work */
            S.amd->val_file[idx] = 1;
            S.amd->reg_file[vr] = 1;
            moperand_t vdst = mop_vreg_v((uint16_t)vr);
            uint32_t fa = new_vreg(1), fb = new_vreg(1), rcp = new_vreg(1);
            uint32_t fq = new_vreg(1);
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fa), ensure_vgpr(src0));
            emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)fb), ensure_vgpr(src1));
            emit1(AMD_V_RCP_F32, mop_vreg_v((uint16_t)rcp), mop_vreg_v((uint16_t)fb));
            emit2(AMD_V_MUL_F32, mop_vreg_v((uint16_t)fq),
                  mop_vreg_v((uint16_t)fa), mop_vreg_v((uint16_t)rcp));
            if (I->op == BIR_SDIV || I->op == BIR_UDIV) {
                emit1(AMD_V_CVT_I32_F32, vdst, mop_vreg_v((uint16_t)fq));
            } else {
                uint32_t qi = new_vreg(1), prod = new_vreg(1);
                emit1(AMD_V_CVT_I32_F32, mop_vreg_v((uint16_t)qi), mop_vreg_v((uint16_t)fq));
                emit2(AMD_V_MUL_LO_U32, mop_vreg_v((uint16_t)prod),
                      mop_vreg_v((uint16_t)qi), ensure_vgpr(src1));
                emit2(AMD_V_SUB_U32, vdst, ensure_vgpr(src0), mop_vreg_v((uint16_t)prod));
            }
            break;
        }
        default: break;
        }
    }
}

/* Map BIR integer comparison predicate to SOPC/VOPC opcode */
static uint16_t icmp_to_scmp(uint8_t pred)
{
    switch (pred) {
    case BIR_ICMP_EQ:  return AMD_S_CMP_EQ_U32;
    case BIR_ICMP_NE:  return AMD_S_CMP_NE_U32;
    case BIR_ICMP_SLT: return AMD_S_CMP_LT_I32;
    case BIR_ICMP_SLE: return AMD_S_CMP_LE_I32;
    case BIR_ICMP_SGT: return AMD_S_CMP_GT_I32;
    case BIR_ICMP_SGE: return AMD_S_CMP_GE_I32;
    case BIR_ICMP_ULT: return AMD_S_CMP_LT_U32;
    case BIR_ICMP_ULE: return AMD_S_CMP_LE_U32;
    case BIR_ICMP_UGT: return AMD_S_CMP_GT_U32;
    case BIR_ICMP_UGE: return AMD_S_CMP_GE_U32;
    default:           return AMD_S_CMP_EQ_U32;
    }
}

static uint16_t icmp_to_vcmp(uint8_t pred)
{
    switch (pred) {
    case BIR_ICMP_EQ:  return AMD_V_CMP_EQ_U32;
    case BIR_ICMP_NE:  return AMD_V_CMP_NE_U32;
    case BIR_ICMP_SLT: return AMD_V_CMP_LT_I32;
    case BIR_ICMP_SLE: return AMD_V_CMP_LE_I32;
    case BIR_ICMP_SGT: return AMD_V_CMP_GT_I32;
    case BIR_ICMP_SGE: return AMD_V_CMP_GE_I32;
    case BIR_ICMP_ULT: return AMD_V_CMP_LT_U32;
    case BIR_ICMP_ULE: return AMD_V_CMP_LE_U32;
    case BIR_ICMP_UGT: return AMD_V_CMP_GT_U32;
    case BIR_ICMP_UGE: return AMD_V_CMP_GE_U32;
    default:           return AMD_V_CMP_EQ_U32;
    }
}

static uint16_t fcmp_to_vcmp(uint8_t pred)
{
    switch (pred) {
    case BIR_FCMP_OEQ: return AMD_V_CMP_EQ_F32;
    case BIR_FCMP_ONE: return AMD_V_CMP_NE_F32;
    case BIR_FCMP_OLT: return AMD_V_CMP_LT_F32;
    case BIR_FCMP_OLE: return AMD_V_CMP_LE_F32;
    case BIR_FCMP_OGT: return AMD_V_CMP_GT_F32;
    case BIR_FCMP_OGE: return AMD_V_CMP_GE_F32;
    case BIR_FCMP_UEQ: return AMD_V_CMP_NLT_F32;  /* approximation */
    case BIR_FCMP_UNE: return AMD_V_CMP_NEQ_F32;
    case BIR_FCMP_ULT: return AMD_V_CMP_NGE_F32;
    case BIR_FCMP_ULE: return AMD_V_CMP_NGT_F32;
    case BIR_FCMP_UGT: return AMD_V_CMP_NLE_F32;
    case BIR_FCMP_UGE: return AMD_V_CMP_NLT_F32;
    case BIR_FCMP_ORD: return AMD_V_CMP_O_F32;
    case BIR_FCMP_UNO: return AMD_V_CMP_U_F32;
    default:           return AMD_V_CMP_EQ_F32;
    }
}

static void isel_icmp(uint32_t idx, const bir_inst_t *I, int div)
{
    moperand_t src0 = resolve_val(I->operands[0], div);
    moperand_t src1 = resolve_val(I->operands[1], div);
    uint32_t vr = map_bir_val(idx, div);

    if (div) {
        /* VOPC: v_cmp_* sets VCC. Materialize to VGPR via v_cndmask_b32 */
        moperand_t vs0 = ensure_vgpr(src0);
        moperand_t vs1 = ensure_vgpr(src1);
        uint16_t vcmp = icmp_to_vcmp(I->subop);
        /* VOPC writes VCC implicitly */
        emit0_2(vcmp, vs0, vs1);
        /* Materialize: vDst = vcc ? 1 : 0.
           VOP2 VSRC1 must be a VGPR — literals there silently become v0.
           Ask me how I know. */
        uint32_t one_vr = new_vreg(1);
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)one_vr), mop_imm(1));
        emit3(AMD_V_CNDMASK_B32, mop_vreg_v((uint16_t)vr),
              mop_imm(0), mop_vreg_v((uint16_t)one_vr),
              mop_special(AMD_SPEC_VCC));
    } else {
        /* SOPC: s_cmp_* sets SCC. Materialize via s_cselect_b32 */
        uint16_t scmp = icmp_to_scmp(I->subop);
        emit0_2(scmp, src0, src1);
        emit2(AMD_S_CSELECT_B32, mop_vreg_s((uint16_t)vr), mop_imm(1), mop_imm(0));
    }
}

static void isel_fcmp(uint32_t idx, const bir_inst_t *I)
{
    /* FCMP is always vector */
    moperand_t src0 = ensure_vgpr(resolve_val(I->operands[0], 1));
    moperand_t src1 = ensure_vgpr(resolve_val(I->operands[1], 1));
    uint32_t vr = map_bir_val(idx, 1);
    uint16_t vcmp = fcmp_to_vcmp(I->subop);
    emit0_2(vcmp, src0, src1);
    uint32_t one_vr = new_vreg(1);
    emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)one_vr), mop_imm(1));
    emit3(AMD_V_CNDMASK_B32, mop_vreg_v((uint16_t)vr),
          mop_imm(0), mop_vreg_v((uint16_t)one_vr),
          mop_special(AMD_SPEC_VCC));
}

static void isel_conversion(uint32_t idx, const bir_inst_t *I, int div)
{
    moperand_t src = resolve_val(I->operands[0], div);
    uint32_t vr = map_bir_val(idx, div);

    switch (I->op) {
    case BIR_TRUNC: {
        /* Truncate to narrower int: mask off high bits */
        int w = bir_type_width(I->type);
        if (w < 32) {
            uint32_t mask = (1u << w) - 1;
            if (div) {
                emit2(AMD_V_AND_B32, mop_vreg_v((uint16_t)vr),
                      ensure_vgpr(src), mop_imm((int32_t)mask));
            } else {
                emit2(AMD_S_AND_B32, mop_vreg_s((uint16_t)vr), src, mop_imm((int32_t)mask));
            }
        } else {
            /* 32-bit trunc from 64-bit: just take low word (copy) */
            if (div)
                emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
            else
                emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr), src);
        }
        break;
    }
    case BIR_ZEXT: {
        /* Zero-extend: for i1->i32, mask with 1. Otherwise just copy. */
        int src_w = 32;
        if (I->operands[0] != BIR_VAL_NONE && !BIR_VAL_IS_CONST(I->operands[0])) {
            uint32_t si = BIR_VAL_INDEX(I->operands[0]);
            if (si < S.bir->num_insts)
                src_w = bir_type_width(S.bir->insts[si].type);
        }
        if (src_w < 32) {
            uint32_t mask = (1u << src_w) - 1;
            if (div)
                emit2(AMD_V_AND_B32, mop_vreg_v((uint16_t)vr), mop_imm((int32_t)mask), ensure_vgpr(src));
            else
                emit2(AMD_S_AND_B32, mop_vreg_s((uint16_t)vr), src, mop_imm((int32_t)mask));
        } else {
            if (div)
                emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
            else
                emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr), src);
        }
        break;
    }
    case BIR_SEXT: {
        /* Sign-extend: use BFE (bit field extract signed) */
        int src_w = 32;
        if (I->operands[0] != BIR_VAL_NONE && !BIR_VAL_IS_CONST(I->operands[0])) {
            uint32_t si = BIR_VAL_INDEX(I->operands[0]);
            if (si < S.bir->num_insts)
                src_w = bir_type_width(S.bir->insts[si].type);
        }
        if (src_w < 32 && div) {
            /* v_bfe_i32 vDst, src, 0, width */
            S.amd->val_file[idx] = 1;
            S.amd->reg_file[vr] = 1;
            emit3(AMD_V_BFE_I32, mop_vreg_v((uint16_t)vr),
                  ensure_vgpr(src), mop_imm(0), mop_imm(src_w));
        } else if (src_w < 32) {
            emit3(AMD_S_BFE_I32, mop_vreg_s((uint16_t)vr),
                  src, mop_imm(src_w), mop_imm(0));
        } else {
            if (div)
                emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
            else
                emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr), src);
        }
        break;
    }
    case BIR_SITOFP: {
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit1(AMD_V_CVT_F32_I32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_UITOFP: {
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit1(AMD_V_CVT_F32_U32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_FPTOSI: {
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit1(AMD_V_CVT_I32_F32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_FPTOUI: {
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit1(AMD_V_CVT_U32_F32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_FPTRUNC: {
        /* f64->f32, f32->f16, or f32->bf16 */
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        uint8_t dkind = I->type < S.bir->num_types
                       ? S.bir->types[I->type].kind : 0;
        if (dkind == BIR_TYPE_BFLOAT)
            emit2(AMD_V_LSHRREV_B32, mop_vreg_v((uint16_t)vr),
                  mop_imm(16), ensure_vgpr(src));
        else if (bir_type_width(I->type) <= 16)
            emit1(AMD_V_CVT_F16_F32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        else
            emit1(AMD_V_CVT_F32_F64, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_FPEXT: {
        /* bf16->f32, f16->f32, or f32->f64 */
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        uint32_t opref = I->operands[0];
        uint8_t skind = 0;
        if (!BIR_VAL_IS_CONST(opref)) {
            uint32_t oidx = BIR_VAL_INDEX(opref);
            if (oidx < S.bir->num_insts) {
                uint32_t styp = S.bir->insts[oidx].type;
                if (styp < S.bir->num_types)
                    skind = S.bir->types[styp].kind;
            }
        }
        if (skind == BIR_TYPE_BFLOAT) {
            if (S.amd->target >= AMD_TARGET_GFX1100)
                emit1(AMD_V_CVT_F32_BF16, mop_vreg_v((uint16_t)vr),
                      ensure_vgpr(src));
            else
                emit2(AMD_V_LSHLREV_B32, mop_vreg_v((uint16_t)vr),
                      mop_imm(16), ensure_vgpr(src));
        } else if (bir_type_width(I->type) >= 64)
            emit1(AMD_V_CVT_F64_F32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        else
            emit1(AMD_V_CVT_F32_F16, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        break;
    }
    case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST: {
        if (div)
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
        else
            emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr), src);
        break;
    }
    case BIR_SQRT: case BIR_RSQ: case BIR_RCP:
    case BIR_EXP2: case BIR_LOG2:
    case BIR_SIN: case BIR_COS:
    case BIR_FLOOR: case BIR_CEIL: case BIR_FTRUNC: case BIR_RNDNE: {
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        static const struct { bir_op_t bo; amd_op_t ao; } m1[] = {
            {BIR_SQRT,AMD_V_SQRT_F32},{BIR_RSQ,AMD_V_RSQ_F32},
            {BIR_RCP,AMD_V_RCP_F32},{BIR_EXP2,AMD_V_EXP_F32},
            {BIR_LOG2,AMD_V_LOG_F32},{BIR_SIN,AMD_V_SIN_F32},
            {BIR_COS,AMD_V_COS_F32},{BIR_FLOOR,AMD_V_FLOOR_F32},
            {BIR_CEIL,AMD_V_CEIL_F32},{BIR_FTRUNC,AMD_V_TRUNC_F32},
            {BIR_RNDNE,AMD_V_RNDNE_F32},
        };
        for (int mi = 0; mi < 11; mi++) {
            if (m1[mi].bo == I->op) {
                emit1(m1[mi].ao, mop_vreg_v((uint16_t)vr), ensure_vgpr(src));
                break;
            }
        }
        break;
    }
    case BIR_FABS: {
        /* VOP2 VSRC1 is VGPR-only — literal in VSRC1 silently becomes v0.
           AND is commutative, so put the bitmask in SRC0. */
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit2(AMD_V_AND_B32, mop_vreg_v((uint16_t)vr),
              mop_imm(0x7FFFFFFF), ensure_vgpr(src));
        break;
    }
    default:
        break;
    }
}

static void isel_load(uint32_t idx, const bir_inst_t *I, int div)
{
    /* ops[0] = address */
    uint32_t ptr_type = 0;
    uint16_t sbase = 0xFFFF;
    if (I->operands[0] != BIR_VAL_NONE && !BIR_VAL_IS_CONST(I->operands[0])) {
        uint32_t si = BIR_VAL_INDEX(I->operands[0]);
        if (si < S.bir->num_insts) ptr_type = S.bir->insts[si].type;
        if (si < BIR_MAX_INSTS) sbase = S.amd->val_sbase[si];
    }
    int as = get_addrspace(ptr_type);

    int result_vec = (as != BIR_AS_CONSTANT);
    uint32_t vr = map_bir_val(idx, result_vec);

    switch (as) {
    case BIR_AS_GLOBAL: case BIR_AS_GENERIC: {
        if (sbase != 0xFFFF) {
            /* saddr form: global_load_dword vDst, vOffset, s[base:base+1] */
            moperand_t voff = ensure_vgpr(resolve_val(I->operands[0], 1));
            moperand_t ops[MINST_MAX_OPS];
            ops[0] = mop_vreg_v((uint16_t)vr);
            ops[1] = voff;
            ops[2] = mop_sgpr(sbase);
            emit_minst(AMD_GLOBAL_LOAD_DWORD, 1, 2, ops, 0);
        } else {
            moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[0], div));
            emit2(AMD_GLOBAL_LOAD_DWORD, mop_vreg_v((uint16_t)vr), vaddr, mop_imm(0));
        }
        emit_wait_vm();
        break;
    }
    case BIR_AS_SHARED: {
        moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[0], div));
        emit2(AMD_DS_READ_B32, mop_vreg_v((uint16_t)vr), vaddr, mop_imm(0));
        emit_wait_ds();
        break;
    }
    case BIR_AS_CONSTANT: {
        moperand_t addr = resolve_val(I->operands[0], 0);
        emit2(AMD_S_LOAD_DWORD, mop_vreg_s((uint16_t)vr), addr, mop_imm(0));
        emit_wait_smem();
        break;
    }
    case BIR_AS_PRIVATE: {
        moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[0], div));
        emit2(AMD_SCRATCH_LOAD_DWORD, mop_vreg_v((uint16_t)vr), vaddr, mop_imm(0));
        emit_wait_vm();
        break;
    }
    default:
        break;
    }
}

static void isel_store(const bir_inst_t *I, int div)
{
    /* BIR: store value, address — ops[0] = value, ops[1] = address */
    moperand_t val = resolve_val(I->operands[0], div);
    uint32_t ptr_type = 0;
    uint16_t sbase = 0xFFFF;
    if (I->operands[1] != BIR_VAL_NONE && !BIR_VAL_IS_CONST(I->operands[1])) {
        uint32_t si = BIR_VAL_INDEX(I->operands[1]);
        if (si < S.bir->num_insts) ptr_type = S.bir->insts[si].type;
        if (si < BIR_MAX_INSTS) sbase = S.amd->val_sbase[si];
    }
    int as = get_addrspace(ptr_type);

    switch (as) {
    case BIR_AS_GLOBAL: case BIR_AS_GENERIC: {
        moperand_t vval = ensure_vgpr(val);
        moperand_t ops[MINST_MAX_OPS];
        if (sbase != 0xFFFF) {
            /* saddr form: global_store_dword vOffset, vSrc, s[base:base+1] */
            moperand_t voff = ensure_vgpr(resolve_val(I->operands[1], 1));
            ops[0] = voff; ops[1] = vval; ops[2] = mop_sgpr(sbase);
        } else {
            moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[1], div));
            ops[0] = vaddr; ops[1] = vval; ops[2] = mop_imm(0);
        }
        emit_minst(AMD_GLOBAL_STORE_DWORD, 0, 3, ops, 0);
        break;
    }
    case BIR_AS_SHARED: {
        moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[1], div));
        emit0_2(AMD_DS_WRITE_B32, vaddr, ensure_vgpr(val));
        break;
    }
    case BIR_AS_PRIVATE: {
        moperand_t vaddr = ensure_vgpr(resolve_val(I->operands[1], div));
        moperand_t ops[MINST_MAX_OPS];
        ops[0] = vaddr; ops[1] = ensure_vgpr(val); ops[2] = mop_imm(0);
        emit_minst(AMD_SCRATCH_STORE_DWORD, 0, 3, ops, 0);
        break;
    }
    default:
        break;
    }
}

static void isel_gep(uint32_t idx, const bir_inst_t *I, int div)
{
    /* GEP: base + index * element_size */
    uint32_t nops = get_num_ops(I);
    if (nops < 2) return;

    uint32_t ptr_type = I->type;
    uint32_t elem_sz = pointee_size(ptr_type);
    uint32_t base_val = get_op(I, 0);

    /* Check if base pointer carries an SGPR pair (saddr mode) */
    uint16_t sbase = 0xFFFF;
    if (!BIR_VAL_IS_CONST(base_val) && base_val != BIR_VAL_NONE) {
        uint32_t bi = BIR_VAL_INDEX(base_val);
        if (bi < BIR_MAX_INSTS)
            sbase = S.amd->val_sbase[bi];
    }

    if (sbase != 0xFFFF) {
        /* saddr path: propagate SGPR pair, compute 32-bit VGPR offset */
        S.amd->val_sbase[idx] = sbase;

        moperand_t base_off = ensure_vgpr(resolve_val(base_val, 1));
        moperand_t acc = base_off;

        for (uint32_t k = 1; k < nops; k++) {
            moperand_t index = ensure_vgpr(resolve_val(get_op(I, k), 1));
            if (elem_sz != 1) {
                uint32_t scaled = new_vreg(1);
                emit2(AMD_V_MUL_LO_U32, mop_vreg_v((uint16_t)scaled),
                      index, mop_imm((int32_t)elem_sz));
                index = mop_vreg_v((uint16_t)scaled);
            }
            uint32_t tmp = new_vreg(1);
            emit2(AMD_V_ADD_U32, mop_vreg_v((uint16_t)tmp), acc, index);
            acc = mop_vreg_v((uint16_t)tmp);
        }

        uint32_t vr = map_bir_val(idx, 1);
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), acc);
        return;
    }

    /* Non-pointer or no sbase: full address arithmetic */
    moperand_t base = resolve_val(base_val, div);
    uint32_t vr = map_bir_val(idx, div);
    moperand_t acc;

    if (div) {
        acc = ensure_vgpr(base);
        for (uint32_t k = 1; k < nops; k++) {
            moperand_t index = ensure_vgpr(resolve_val(get_op(I, k), div));
            if (elem_sz != 1) {
                uint32_t scaled = new_vreg(1);
                emit2(AMD_V_MUL_LO_U32, mop_vreg_v((uint16_t)scaled),
                      index, mop_imm((int32_t)elem_sz));
                index = mop_vreg_v((uint16_t)scaled);
            }
            uint32_t tmp = new_vreg(1);
            emit2(AMD_V_ADD_U32, mop_vreg_v((uint16_t)tmp), acc, index);
            acc = mop_vreg_v((uint16_t)tmp);
        }
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), acc);
    } else if (is_cdna()) {
        /* CDNA GEP: the MI300X s_add_i32 zero-result errata strikes
         * again.  Two SMEM-sourced operands → s_add returns 0, your
         * pointer goes to la-la land, and the GPU segfaults with the
         * serenity of a mainframe ABEND.  Promote to VALU. */
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        acc = ensure_vgpr(base);
        for (uint32_t k = 1; k < nops; k++) {
            moperand_t index = ensure_vgpr(resolve_val(get_op(I, k), 0));
            if (elem_sz != 1) {
                uint32_t scaled = new_vreg(1);
                emit2(AMD_V_MUL_LO_U32, mop_vreg_v((uint16_t)scaled),
                      index, mop_imm((int32_t)elem_sz));
                index = mop_vreg_v((uint16_t)scaled);
            }
            uint32_t tmp = new_vreg(1);
            emit2(AMD_V_ADD_U32, mop_vreg_v((uint16_t)tmp), acc, index);
            acc = mop_vreg_v((uint16_t)tmp);
        }
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), acc);
    } else {
        acc = base;
        for (uint32_t k = 1; k < nops; k++) {
            moperand_t index = resolve_val(get_op(I, k), 0);
            if (elem_sz != 1) {
                uint32_t scaled = new_vreg(0);
                emit2(AMD_S_MUL_I32, mop_vreg_s((uint16_t)scaled),
                      index, mop_imm((int32_t)elem_sz));
                index = mop_vreg_s((uint16_t)scaled);
            }
            uint32_t tmp = new_vreg(0);
            emit2(AMD_S_ADD_U32, mop_vreg_s((uint16_t)tmp), acc, index);
            acc = mop_vreg_s((uint16_t)tmp);
        }
        emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr), acc);
    }
}

static void isel_alloca(uint32_t idx, const bir_inst_t *I)
{
    /* Compute scratch frame offset */
    uint32_t align = 1u << I->subop;
    S.scratch_offset = (S.scratch_offset + align - 1) & ~(align - 1);

    /* Allocate a vreg holding the scratch offset */
    uint32_t vr = map_bir_val(idx, 1);
    /* v_mov_b32 vr, scratch_offset */
    emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_imm((int32_t)S.scratch_offset));

    uint32_t sz = pointee_size(I->type);
    if (sz < 4) sz = 4;
    S.scratch_offset += sz;
}

static void isel_shared_alloc(uint32_t idx, const bir_inst_t *I)
{
    uint32_t sz = pointee_size(I->type);
    if (sz < 1) sz = 4;
    /* Align to 4 bytes */
    S.lds_offset = (S.lds_offset + 3u) & ~3u;
    uint32_t vr = map_bir_val(idx, 0);
    emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr),
          mop_imm((int32_t)S.lds_offset));
    S.lds_offset += sz;
}

static void isel_global_ref(uint32_t idx, const bir_inst_t *I)
{
    (void)I;
    /* Hidden kernarg: 64-bit pointer appended after explicit params.
       Load into SGPR pair for saddr, VGPR gets zero offset. */
    uint32_t offst = S.hkrarg;
    S.hkrarg += 8;

    uint16_t sbase = S.next_param_sgpr;
    if (sbase & 1) sbase++;
    if (sbase + 1 >= AMD_MAX_SGPRS) return;
    S.next_param_sgpr = sbase + 2;

    emit2(AMD_S_LOAD_DWORDX2, mop_sgpr(sbase),
          mop_sgpr(S.sgpr_kernarg), mop_imm((int32_t)offst));
    emit_wait_smem();
    S.amd->val_sbase[idx] = sbase;

    uint32_t vr = map_bir_val(idx, 1);
    S.amd->val_file[idx] = 1;
    S.amd->reg_file[vr] = 1;
    emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_imm(0));
}

static void isel_branch(const bir_inst_t *I)
{
    uint32_t target_bir = I->operands[0];
    if (target_bir >= BIR_MAX_BLOCKS) return;

    /* Suppress then→merge branches in divergent diamonds.
       The then-block falls through to the else-block instead. */
    if (S.suppress_src == S.current_bir_block &&
        S.suppress_dst == target_bir) {
        S.suppress_src = 0xFFFFFFFF;
        S.suppress_dst = 0xFFFFFFFF;
        return;
    }
    emit0_1(AMD_S_BRANCH, mop_label(S.block_map[target_bir]));
}

static void isel_br_cond(const bir_inst_t *I, int cond_div)
{
    uint32_t true_bir  = I->operands[1];
    uint32_t false_bir = I->operands[2];
    if (true_bir >= BIR_MAX_BLOCKS || false_bir >= BIR_MAX_BLOCKS) return;
    uint32_t true_mb   = S.block_map[true_bir];
    uint32_t false_mb  = S.block_map[false_bir];

    if (cond_div) {
        /* Divergent branch: EXEC mask save/restore pattern.
           Both paths execute sequentially — true lanes first, then false.
           Nature's way of saying "why not both?" */
        moperand_t cond = resolve_val(I->operands[0], 1);
        moperand_t vcond = ensure_vgpr(cond);
        /* VOPC: immediate must be SRC0 (VSRC1 is VGPR-only). NE is commutative. */
        emit0_2(AMD_V_CMP_NE_U32, mop_imm(0), vcond);

        uint32_t saved;
        if (is_cdna()) {
            /* Wave64: saveexec writes 64-bit pair. Regalloc doesn't do
               pairs, so pre-allocate even-aligned physical SGPRs.
               Like booking a double room — you can't just take one bed. */
            if (S.next_param_sgpr & 1) S.next_param_sgpr++;
            saved = S.next_param_sgpr;
            S.next_param_sgpr += 2;
            emit1(AMD_S_AND_SAVEEXEC_B64,
                  mop_sgpr((uint16_t)saved), mop_special(AMD_SPEC_VCC));
        } else {
            saved = new_vreg(0);
            emit1(AMD_S_AND_SAVEEXEC_B32,
                  mop_vreg_s((uint16_t)saved), mop_special(AMD_SPEC_VCC));
        }
        emit0_1(AMD_S_CBRANCH_EXECZ, mop_label(false_mb));
        /* Fall through to true block (next in layout) */

        /* Find the merge point and record the divergent region */
        uint32_t merge_bir = find_merge_block(true_bir, false_bir);
        int has_else = (merge_bir != false_bir);

        if (S.div_depth < MAX_DIV_REGIONS) {
            S.div_stack[S.div_depth].saved_vreg = saved;
            S.div_stack[S.div_depth].false_bir = false_bir;
            S.div_stack[S.div_depth].merge_bir = merge_bir;
            S.div_stack[S.div_depth].has_else = has_else;
            S.div_depth++;
        }

        /* For diamonds, suppress then-block's branch to merge
           so it falls through to the else-block */
        if (has_else) {
            S.suppress_src = true_bir;
            S.suppress_dst = merge_bir;
        }
    } else {
        /* Uniform branch: compare and branch on SCC */
        moperand_t cond = resolve_val(I->operands[0], 0);
        emit0_2(AMD_S_CMP_NE_U32, cond, mop_imm(0));
        emit0_1(AMD_S_CBRANCH_SCC1, mop_label(true_mb));
        emit0_1(AMD_S_BRANCH, mop_label(false_mb));
    }
}

static void isel_switch(const bir_inst_t *I)
{
    uint32_t nops = get_num_ops(I);
    int sel_div = val_is_divergent(get_op(I, 0));
    moperand_t sel;
    if (sel_div) {
        /* Divergent selector: yank it into SGPR land.
           Not ideal — only compares one lane's value — but switch on a divergent
           selector is unusual enough that correctness > cleverness here. */
        moperand_t vsel = resolve_val(get_op(I, 0), 1);
        uint32_t svreg = new_vreg(0);
        emit1(AMD_V_READFIRSTLANE_B32, mop_vreg_s((uint16_t)svreg), ensure_vgpr(vsel));
        sel = mop_vreg_s((uint16_t)svreg);
    } else {
        sel = resolve_val(get_op(I, 0), 0);
    }
    uint32_t default_bir = get_op(I, 1);
    if (default_bir >= BIR_MAX_BLOCKS) return;
    uint32_t default_mb = S.block_map[default_bir];

    for (uint32_t k = 2; k + 1 < nops; k += 2) {
        uint32_t case_val = get_op(I, k);
        uint32_t case_blk = get_op(I, k + 1);
        if (case_blk >= BIR_MAX_BLOCKS) continue;
        uint32_t case_mb = S.block_map[case_blk];
        moperand_t cv;
        if (BIR_VAL_IS_CONST(case_val)) {
            uint32_t ci = BIR_VAL_INDEX(case_val);
            if (ci < S.bir->num_consts)
                cv = mop_imm((int32_t)S.bir->consts[ci].d.ival);
            else
                cv = mop_imm(0);
        } else {
            cv = mop_imm((int32_t)case_val);
        }
        emit0_2(AMD_S_CMP_EQ_U32, sel, cv);
        emit0_1(AMD_S_CBRANCH_SCC1, mop_label(case_mb));
    }
    emit0_1(AMD_S_BRANCH, mop_label(default_mb));
}

static void isel_ret(const bir_inst_t *I, int nops)
{
    if (S.is_kernel) {
        emit0_0(AMD_S_ENDPGM, 0);
    } else {
        /* Device function: return value in v0, then s_setpc_b64 */
        if (nops > 0) {
            moperand_t val = resolve_val(I->operands[0], 1);
            emit1(AMD_V_MOV_B32, mop_vgpr(0), ensure_vgpr(val));
        }
        /* s_setpc_b64 s[30:31] (return address) */
        emit0_1(AMD_S_SETPC_B64, mop_sgpr(30));
    }
}

static void isel_phi(uint32_t idx, const bir_inst_t *I, int div)
{
    /* Emit pseudo-PHI, will be eliminated later */
    uint32_t vr = map_bir_val(idx, div);
    uint32_t nops = get_num_ops(I);
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = mop_vreg((uint16_t)vr, div);

    /* PHI operands are (block, value) pairs. Encode up to 2 pairs inline. */
    uint8_t npairs = 0;
    for (uint32_t k = 0; k + 1 < nops && npairs < 2; k += 2, npairs++) {
        uint32_t pred_blk = get_op(I, k);
        uint32_t pred_val = get_op(I, k + 1);
        if (pred_blk >= BIR_MAX_BLOCKS) continue;
        ops[1 + npairs * 2] = mop_label(S.block_map[pred_blk]);
        ops[2 + npairs * 2] = resolve_val(pred_val, div);
    }
    emit_minst(AMD_PSEUDO_PHI, 1, (uint8_t)(npairs * 2), ops, 0);
}

static void isel_param(uint32_t idx, const bir_inst_t *I)
{
    uint32_t param_idx = I->subop;

    if (S.is_kernel) {
        uint32_t offset = param_idx * 8; /* 8-byte aligned for pointers */
        int width = bir_type_width(I->type);
        int is_ptr = (bir_type_kind(I->type) == BIR_TYPE_PTR);

        if (width > 32 || is_ptr) {
            /* 64-bit pointer: load into a physical SGPR pair for saddr.
               These live outside the vreg world — real registers, real problems. */
            uint16_t base_sgpr = S.next_param_sgpr;
            if (base_sgpr + 1 >= AMD_MAX_SGPRS) return;
            /* Align to even SGPR for pair */
            if (base_sgpr & 1) base_sgpr++;
            S.next_param_sgpr = base_sgpr + 2;

            emit2(AMD_S_LOAD_DWORDX2, mop_sgpr(base_sgpr),
                  mop_sgpr(S.sgpr_kernarg), mop_imm((int32_t)offset));
            emit_wait_smem();
            S.amd->val_sbase[idx] = base_sgpr;

            /* VGPR offset starts at 0 (base + 0 = base address) */
            uint32_t vr = map_bir_val(idx, 1);
            S.amd->val_file[idx] = 1;
            S.amd->reg_file[vr] = 1;
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_imm(0));
        } else {
            uint32_t vr = map_bir_val(idx, 0);
            emit2(AMD_S_LOAD_DWORD, mop_vreg_s((uint16_t)vr),
                  mop_sgpr(S.sgpr_kernarg), mop_imm((int32_t)offset));
            emit_wait_smem();
        }
    } else {
        /* Device function: params in v0, v1, ... */
        uint32_t vr = map_bir_val(idx, 1);
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        if (param_idx < 16)
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_vgpr((uint16_t)param_idx));
    }
}

static void isel_thread_model(uint32_t idx, const bir_inst_t *I)
{
    uint32_t dim = I->subop; /* 0=x, 1=y, 2=z */

    switch (I->op) {
    case BIR_THREAD_ID: {
        /* Use saved copies — originals may have been clobbered by param loads */
        uint32_t vr = map_bir_val(idx, 1);
        if (S.is_kernel && dim < 3)
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr),
                  mop_vreg_v((uint16_t)S.saved_tid[dim]));
        else
            emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_vgpr((uint16_t)dim));
        break;
    }
    case BIR_BLOCK_ID: {
        /* Pre-loaded workgroup IDs at sgpr_wg_base + dim */
        uint32_t vr = map_bir_val(idx, 0);
        emit1(AMD_S_MOV_B32, mop_vreg_s((uint16_t)vr),
              mop_sgpr((uint16_t)(S.sgpr_wg_base + dim)));
        break;
    }
    case BIR_BLOCK_DIM: {
        /* Hidden kernarg: group_size_x/y/z are uint16 at hk_base+12/14/16.
         * Load 32 bits containing the pair, mask to 16. */
        uint32_t vr = map_bir_val(idx, 0);
        uint32_t hk_base = S.num_params * 8;  /* hidden args start here */
        uint32_t off = hk_base + 12 + (dim & ~1u) * 2;
        emit2(AMD_S_LOAD_DWORD, mop_vreg_s((uint16_t)vr),
              mop_sgpr(S.sgpr_kernarg), mop_imm((int32_t)off));
        emit_wait_smem();
        if (dim & 1) /* odd dim: shift right 16 */
            emit2(AMD_S_LSHR_B32, mop_vreg_s((uint16_t)vr),
                  mop_vreg_s((uint16_t)vr), mop_imm(16));
        else
            emit2(AMD_S_AND_B32, mop_vreg_s((uint16_t)vr),
                  mop_vreg_s((uint16_t)vr), mop_imm(0xFFFF));
        break;
    }
    case BIR_GRID_DIM: {
        /* Hidden kernarg: block_count_x/y/z are uint32 at hk_base+0/4/8.
         * This IS gridDim (num workgroups), not grid_size. */
        uint32_t vr = map_bir_val(idx, 0);
        uint32_t hk_base = S.num_params * 8;
        emit2(AMD_S_LOAD_DWORD, mop_vreg_s((uint16_t)vr),
              mop_sgpr(S.sgpr_kernarg), mop_imm((int32_t)(hk_base + dim * 4)));
        emit_wait_smem();
        break;
    }
    default:
        break;
    }
}

static void isel_barrier(void)
{
    emit_wait_all();
    emit0_0(AMD_S_BARRIER, 0);
}

static void isel_atomic(uint32_t idx, const bir_inst_t *I, int div)
{
    /* ops[0] = address, ops[1] = value (ops[2] = compare for CAS) */
    moperand_t addr = ensure_vgpr(resolve_val(I->operands[0], div));
    uint32_t nops = get_num_ops(I);
    uint32_t ptr_type = 0;
    if (I->operands[0] != BIR_VAL_NONE && !BIR_VAL_IS_CONST(I->operands[0])) {
        uint32_t si = BIR_VAL_INDEX(I->operands[0]);
        if (si < S.bir->num_insts) ptr_type = S.bir->insts[si].type;
    }
    int as = get_addrspace(ptr_type);

    uint32_t vr = map_bir_val(idx, 1);
    moperand_t dst = mop_vreg_v((uint16_t)vr);

    if (as == BIR_AS_SHARED) {
        /* DS atomics */
        moperand_t val = (nops > 1) ? ensure_vgpr(resolve_val(I->operands[1], 1)) : mop_imm(0);
        uint16_t ds_op;
        switch (I->op) {
        case BIR_ATOMIC_ADD:  ds_op = AMD_DS_ADD_RTN_U32; break;
        case BIR_ATOMIC_SUB:  ds_op = AMD_DS_SUB_RTN_U32; break;
        case BIR_ATOMIC_AND:  ds_op = AMD_DS_AND_RTN_B32; break;
        case BIR_ATOMIC_OR:   ds_op = AMD_DS_OR_RTN_B32;  break;
        case BIR_ATOMIC_XOR:  ds_op = AMD_DS_XOR_RTN_B32; break;
        case BIR_ATOMIC_MIN:  ds_op = AMD_DS_MIN_RTN_I32; break;
        case BIR_ATOMIC_MAX:  ds_op = AMD_DS_MAX_RTN_I32; break;
        case BIR_ATOMIC_XCHG: ds_op = AMD_DS_ADD_RTN_U32; break; /* placeholder */
        default: ds_op = AMD_DS_ADD_RTN_U32; break;
        }
        emit2(ds_op, dst, addr, val);
        emit_wait_ds();
    } else {
        /* Global atomics */
        moperand_t val = (nops > 1) ? ensure_vgpr(resolve_val(I->operands[1], 1)) : mop_imm(0);
        uint16_t glb_op;
        switch (I->op) {
        case BIR_ATOMIC_ADD:  glb_op = AMD_GLOBAL_ATOMIC_ADD;  break;
        case BIR_ATOMIC_SUB:  glb_op = AMD_GLOBAL_ATOMIC_SUB;  break;
        case BIR_ATOMIC_AND:  glb_op = AMD_GLOBAL_ATOMIC_AND;  break;
        case BIR_ATOMIC_OR:   glb_op = AMD_GLOBAL_ATOMIC_OR;   break;
        case BIR_ATOMIC_XOR:  glb_op = AMD_GLOBAL_ATOMIC_XOR;  break;
        case BIR_ATOMIC_MIN:  glb_op = AMD_GLOBAL_ATOMIC_SMIN; break;
        case BIR_ATOMIC_MAX:  glb_op = AMD_GLOBAL_ATOMIC_SMAX; break;
        case BIR_ATOMIC_XCHG: glb_op = AMD_GLOBAL_ATOMIC_SWAP; break;
        case BIR_ATOMIC_CAS:  glb_op = AMD_GLOBAL_ATOMIC_CMPSWAP; break;
        default: glb_op = AMD_GLOBAL_ATOMIC_ADD; break;
        }
        if (I->op == BIR_ATOMIC_CAS && nops > 2) {
            moperand_t cmp = ensure_vgpr(resolve_val(I->operands[2], 1));
            moperand_t ops[MINST_MAX_OPS];
            ops[0] = dst; ops[1] = addr; ops[2] = val; ops[3] = cmp;
            emit_minst(glb_op, 1, 3, ops, AMD_FLAG_GLC);
        } else {
            emit2f(glb_op, dst, addr, val, AMD_FLAG_GLC);
        }
        emit_wait_vm();
    }
}

static void isel_atomic_load(uint32_t idx, const bir_inst_t *I, int div)
{
    /* Atomic load: same as regular load but with glc for ordering */
    moperand_t addr = ensure_vgpr(resolve_val(I->operands[0], div));
    uint32_t vr = map_bir_val(idx, 1);
    emit2f(AMD_GLOBAL_LOAD_DWORD, mop_vreg_v((uint16_t)vr), addr, mop_imm(0), AMD_FLAG_GLC);
    emit_wait_vm();
}

static void isel_atomic_store(const bir_inst_t *I, int div)
{
    /* BIR atomic_store: ops[0] = value, ops[1] = address */
    moperand_t val  = ensure_vgpr(resolve_val(I->operands[0], div));
    moperand_t addr = ensure_vgpr(resolve_val(I->operands[1], div));
    moperand_t ops[MINST_MAX_OPS];
    ops[0] = addr; ops[1] = val; ops[2] = mop_imm(0);
    emit_minst(AMD_GLOBAL_STORE_DWORD, 0, 3, ops, AMD_FLAG_GLC);
}

static void isel_warp(uint32_t idx, const bir_inst_t *I)
{
    uint32_t vr = map_bir_val(idx, 1);

    switch (I->op) {
    case BIR_SHFL: case BIR_SHFL_UP: case BIR_SHFL_DOWN: case BIR_SHFL_XOR: {
        /* ds_bpermute_b32: the cross-lane shuffle.  AMD doesn't have
         * NVIDIA's nice warp-shuffle instructions, so we fake it via
         * LDS permute — like passing notes in class, but through the
         * shared memory subsystem.
         *
         * BIR ops: [0]=mask(ignored on AMD — all lanes or nothing),
         *          [1]=val, [2]=delta, [3]=width(ignored).
         * Lane computation per variant:
         *   SHFL      → lane = delta  (direct index)
         *   SHFL_DOWN → lane = tid + delta  (look ahead)
         *   SHFL_UP   → lane = tid - delta  (look behind)
         *   SHFL_XOR  → lane = tid ^ delta  (butterfly) */
        moperand_t val   = ensure_vgpr(resolve_val(I->operands[1], 1));
        moperand_t delta = ensure_vgpr(resolve_val(I->operands[2], 1));
        moperand_t tid   = mop_vreg_v((uint16_t)S.saved_tid[0]);
        uint32_t lane_v  = new_vreg(1);
        moperand_t lane  = mop_vreg_v((uint16_t)lane_v);
        switch (I->op) {
        case BIR_SHFL:      emit1(AMD_V_MOV_B32, lane, delta); break;
        case BIR_SHFL_DOWN: emit2(AMD_V_ADD_U32, lane, tid, delta); break;
        case BIR_SHFL_UP:   emit2(AMD_V_SUB_U32, lane, tid, delta); break;
        case BIR_SHFL_XOR:  emit2(AMD_V_XOR_B32, lane, tid, delta); break;
        default: break;
        }
        /* Byte address = lane * 4 */
        uint32_t addr_v = new_vreg(1);
        emit2(AMD_V_LSHLREV_B32, mop_vreg_v((uint16_t)addr_v), mop_imm(2), lane);
        emit2(AMD_DS_BPERMUTE_B32, mop_vreg_v((uint16_t)vr),
              mop_vreg_v((uint16_t)addr_v), val);
        emit_wait_ds();
        break;
    }
    case BIR_BALLOT: {
        /* v_cmp_ne_u32 vcc, 0, pred; v_mov_b32 vDst, vcc */
        moperand_t pred = ensure_vgpr(resolve_val(I->operands[0], 1));
        emit0_2(AMD_V_CMP_NE_U32, mop_imm(0), pred);
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vr), mop_special(AMD_SPEC_VCC));
        break;
    }
    case BIR_VOTE_ANY: {
        /* v_cmp_ne_u32 vcc, 0, pred; s_cmp_ne vcc, 0; materialize SCC */
        moperand_t pred = ensure_vgpr(resolve_val(I->operands[0], 1));
        emit0_2(AMD_V_CMP_NE_U32, mop_imm(0), pred);
        /* Check if VCC != 0 (any lane true) */
        uint32_t sv = new_vreg(0);
        emit1(AMD_V_READFIRSTLANE_B32, mop_vreg_s((uint16_t)sv), mop_special(AMD_SPEC_VCC));
        emit0_2(AMD_S_CMP_NE_U32, mop_vreg_s((uint16_t)sv), mop_imm(0));
        emit2(AMD_S_CSELECT_B32, mop_vreg_s((uint16_t)vr), mop_imm(1), mop_imm(0));
        /* Move to VGPR */
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        uint32_t vv = new_vreg(1);
        S.amd->val_vreg[idx] = vv;
        S.amd->val_file[idx] = 1;
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vv), mop_vreg_s((uint16_t)vr));
        break;
    }
    case BIR_VOTE_ALL: {
        /* v_cmp_ne vcc, 0, pred; check vcc == exec */
        moperand_t pred = ensure_vgpr(resolve_val(I->operands[0], 1));
        emit0_2(AMD_V_CMP_NE_U32, mop_imm(0), pred);
        uint32_t sv = new_vreg(0);
        emit1(AMD_V_READFIRSTLANE_B32, mop_vreg_s((uint16_t)sv), mop_special(AMD_SPEC_VCC));
        emit0_2(AMD_S_CMP_EQ_U32, mop_vreg_s((uint16_t)sv), mop_special(AMD_SPEC_EXEC));
        S.amd->val_file[idx] = 1;
        S.amd->reg_file[vr] = 1;
        emit2(AMD_S_CSELECT_B32, mop_vreg_s((uint16_t)vr), mop_imm(1), mop_imm(0));
        uint32_t vv = new_vreg(1);
        S.amd->val_vreg[idx] = vv;
        S.amd->val_file[idx] = 1;
        emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)vv), mop_vreg_s((uint16_t)vr));
        break;
    }
    default:
        break;
    }
}

/* ---- MFMA variant IDs (packed into BIR_MFMA subop) ---- */
#define MFMA_F16_4x4x4       0
#define MFMA_F16_16x16x16    1
#define MFMA_F16_32x32x8     2
#define MFMA_BF16_4x4x4      3
#define MFMA_BF16_16x16x16   4
#define MFMA_BF16_32x32x8    5
#define MFMA_F32_4x4x1       6
#define MFMA_F32_16x16x4     7
#define MFMA_F32_32x32x2     8
#define MFMA_I8_4x4x4        9
#define MFMA_I8_16x16x16    10
#define MFMA_I8_32x32x8     11
/* FP8/BF8 mixed-precision (gfx942) */
#define MFMA_FP8_FP8_16x16  12
#define MFMA_FP8_BF8_16x16  13
#define MFMA_BF8_FP8_16x16  14
#define MFMA_BF8_BF8_16x16  15
#define MFMA_FP8_FP8_32x32  16
#define MFMA_FP8_BF8_32x32  17
#define MFMA_BF8_FP8_32x32  18
#define MFMA_BF8_BF8_32x32  19
/* F64 matrix (gfx942) */
#define MFMA_F64_4x4x4      20
#define MFMA_F64_16x16x4    21

static void isel_mfma(uint32_t idx, const bir_inst_t *I)
{
    /* Map subop → AMD machine opcode. The hardware does the hard
       part; we just shuffle operands like a very expensive postman. */
    static const uint16_t mfma_ops[] = {
        [MFMA_F16_4x4x4]     = AMD_V_MFMA_F32_4X4X4_F16,
        [MFMA_F16_16x16x16]  = AMD_V_MFMA_F32_16X16X16_F16,
        [MFMA_F16_32x32x8]   = AMD_V_MFMA_F32_32X32X8_F16,
        [MFMA_BF16_4x4x4]    = AMD_V_MFMA_F32_4X4X4_BF16_1K,
        [MFMA_BF16_16x16x16] = AMD_V_MFMA_F32_16X16X16_BF16_1K,
        [MFMA_BF16_32x32x8]  = AMD_V_MFMA_F32_32X32X8_BF16_1K,
        [MFMA_F32_4x4x1]     = AMD_V_MFMA_F32_4X4X1_F32,
        [MFMA_F32_16x16x4]   = AMD_V_MFMA_F32_16X16X4_F32,
        [MFMA_F32_32x32x2]   = AMD_V_MFMA_F32_32X32X2_F32,
        [MFMA_I8_4x4x4]      = AMD_V_MFMA_I32_4X4X4_I8,
        [MFMA_I8_16x16x16]   = AMD_V_MFMA_I32_16X16X16_I8,
        [MFMA_I8_32x32x8]    = AMD_V_MFMA_I32_32X32X8_I8,
        [MFMA_FP8_FP8_16x16] = AMD_V_MFMA_F32_16X16X32_FP8_FP8,
        [MFMA_FP8_BF8_16x16] = AMD_V_MFMA_F32_16X16X32_FP8_BF8,
        [MFMA_BF8_FP8_16x16] = AMD_V_MFMA_F32_16X16X32_BF8_FP8,
        [MFMA_BF8_BF8_16x16] = AMD_V_MFMA_F32_16X16X32_BF8_BF8,
        [MFMA_FP8_FP8_32x32] = AMD_V_MFMA_F32_32X32X16_FP8_FP8,
        [MFMA_FP8_BF8_32x32] = AMD_V_MFMA_F32_32X32X16_FP8_BF8,
        [MFMA_BF8_FP8_32x32] = AMD_V_MFMA_F32_32X32X16_BF8_FP8,
        [MFMA_BF8_BF8_32x32] = AMD_V_MFMA_F32_32X32X16_BF8_BF8,
        [MFMA_F64_4x4x4]     = AMD_V_MFMA_F64_4X4X4_F64,
        [MFMA_F64_16x16x4]   = AMD_V_MFMA_F64_16X16X4_F64,
    };

    uint8_t var = I->subop;
    if (var > MFMA_F64_16x16x4) return;
    uint16_t mop = mfma_ops[var];

    /* MFMA: ops[0]=A, ops[1]=B, ops[2]=C(accum). All VGPR on gfx942. */
    moperand_t a = ensure_vgpr(resolve_val(I->operands[0], 1));
    moperand_t b = ensure_vgpr(resolve_val(I->operands[1], 1));
    moperand_t c = ensure_vgpr(resolve_val(I->operands[2], 1));
    uint32_t vr = map_bir_val(idx, 1);

    /* Wait for any pending VMEM before the matrix op — the hardware
       won't schedule around these for you */
    emit_wait_vm();

    emit3(mop, mop_vreg_v((uint16_t)vr), a, b, c);
}

static void isel_select(uint32_t idx, const bir_inst_t *I, int div)
{
    /* ops[0] = cond, ops[1] = true, ops[2] = false */
    moperand_t cond  = resolve_val(I->operands[0], div);
    moperand_t vtrue = resolve_val(I->operands[1], div);
    moperand_t vfalse = resolve_val(I->operands[2], div);
    uint32_t vr = map_bir_val(idx, div);

    if (div) {
        /* v_cmp_ne_u32 vcc, 0, cond; v_cndmask_b32 dst, false, true, vcc */
        moperand_t vcond = ensure_vgpr(cond);
        emit0_2(AMD_V_CMP_NE_U32, mop_imm(0), vcond);
        emit3(AMD_V_CNDMASK_B32, mop_vreg_v((uint16_t)vr),
              ensure_vgpr(vfalse), ensure_vgpr(vtrue), mop_special(AMD_SPEC_VCC));
    } else {
        /* s_cmp_ne cond, 0; s_cselect_b32 dst, true, false */
        emit0_2(AMD_S_CMP_NE_U32, cond, mop_imm(0));
        emit2(AMD_S_CSELECT_B32, mop_vreg_s((uint16_t)vr), vtrue, vfalse);
    }
}

static void isel_call(uint32_t idx, const bir_inst_t *I, int div)
{
    /* s_swappc_b64 needs a PC-relative offset, but we only have
     * a raw BIR function index — like being handed a phone number
     * with no country code.  Device function linking is a future
     * adventure; for now, die with dignity. */
    fprintf(stderr, "barracuda: device function calls not yet supported "
            "(BIR_CALL func=%u)\n", get_op(I, 0));
    (void)idx; (void)div;
}

/* ---- Pre-scan: determine kernel's SGPR layout needs ---- */

static void scan_kernel_needs(const bir_func_t *F)
{
    const bir_module_t *M = S.bir;
    S.needs_dispatch = 0;
    S.max_dim = 0;

    for (uint32_t bi = 0; bi < F->num_blocks && bi < BIR_MAX_BLOCKS; bi++) {
        const bir_block_t *B = &M->blocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            const bir_inst_t *I = &M->insts[B->first_inst + ii];
            switch (I->op) {
            case BIR_BLOCK_DIM:
            case BIR_GRID_DIM:
                S.needs_dispatch = 1;
                if (I->subop > S.max_dim) S.max_dim = (uint8_t)I->subop;
                break;
            case BIR_THREAD_ID:
            case BIR_BLOCK_ID:
                if (I->subop > S.max_dim) S.max_dim = (uint8_t)I->subop;
                break;
            default:
                break;
            }
        }
    }
    if (S.max_dim > 2) S.max_dim = 2; /* clamp */

    /* SGPR layout depends on dispatch usage.
     * With dispatch_ptr: s[0:1]=dispatch, s[2:3]=kernarg, s4+=TGID.
     * Without dispatch_ptr: s[0:1]=kernarg, s2+=TGID. */
    if (S.needs_dispatch) {
        S.sgpr_dispatch = 0;
        S.sgpr_kernarg = 2;
        S.sgpr_wg_base = 4;
        S.kern_reserved = 4 + 1 + S.max_dim;
    } else {
        S.sgpr_dispatch = 0xFFFF;
        S.sgpr_kernarg = 0;
        S.sgpr_wg_base = 2;
        S.kern_reserved = 2 + 1 + S.max_dim;
    }
    /* Align reserved to even for SGPR pair loads */
    if (S.kern_reserved & 1) S.kern_reserved++;
}

/* ---- Main Instruction Selection Loop ---- */

static void isel_function(uint32_t fi)
{
    const bir_module_t *M = S.bir;
    amd_module_t *A = S.amd;
    const bir_func_t *F = &M->funcs[fi];

    S.func_idx = fi;
    S.func_first_inst = (F->num_blocks > 0) ? M->blocks[F->first_block].first_inst : 0;
    S.func_first_block = F->first_block;
    S.num_params = F->num_params;
    S.is_kernel = (F->cuda_flags & CUDA_GLOBAL) ? 1 : 0;
    S.scratch_offset = 0;
    S.lds_offset = 0;
    S.hkrarg = F->num_params * 8;
    S.div_depth = 0;
    S.suppress_src = 0xFFFFFFFF;
    S.suppress_dst = 0xFFFFFFFF;
    S.current_bir_block = 0;

    /* Scan BIR to determine hidden kernarg needs */
    if (S.is_kernel)
        scan_kernel_needs(F);
    else {
        S.needs_dispatch = 0;
        S.max_dim = 0;
        S.sgpr_kernarg = 0;
        S.sgpr_dispatch = 0xFFFF;
        S.sgpr_wg_base = 2;
        S.kern_reserved = 2;
    }

    /* Reserve hidden kernarg space for block_count/group_size if needed.
     * Layout: [block_count_x/y/z (3×4B)] [group_size_x/y/z (3×2B)]
     *       = 18 bytes, rounded to 24 for alignment. */
    if (S.is_kernel && S.needs_dispatch)
        S.hkrarg = (uint32_t)(F->num_params * 8) + 24u;

    /* Pointer params get physical SGPR pairs starting after reserved regs */
    S.next_param_sgpr = S.kern_reserved;
    if (S.next_param_sgpr & 1) S.next_param_sgpr++; /* align to even */

    /* Skip host-only functions */
    if (!(F->cuda_flags & (CUDA_GLOBAL | CUDA_DEVICE))) return;

    /* Divergence analysis for this function */
    divergence_analysis(F);

    /* Create machine function */
    if (A->num_mfuncs >= AMD_MAX_MFUNCS) return;
    uint32_t mf_idx = A->num_mfuncs++;
    mfunc_t *MF = &A->mfuncs[mf_idx];
    MF->name = F->name;
    MF->first_block = A->num_mblocks;
    MF->is_kernel = S.is_kernel;
    MF->wavefront_size = is_cdna() ? AMD_WAVE64 : AMD_WAVE_SIZE;
    MF->lds_bytes = 0;
    MF->scratch_bytes = 0;
    MF->kernarg_bytes = F->num_params * 8;
    MF->launch_bounds_max = F->launch_bounds_max;
    MF->launch_bounds_min = F->launch_bounds_min;
    MF->needs_dispatch = S.needs_dispatch;
    MF->max_dim = S.max_dim;

    /* Pre-create machine blocks and build block map */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        uint32_t mb_idx = A->num_mblocks + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        S.block_map[bir_bi] = mb_idx;
        A->mblocks[mb_idx].bir_block = bir_bi;
    }

    /* Select instructions block by block */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bir_bi = F->first_block + bi;
        const bir_block_t *B = &M->blocks[bir_bi];
        uint32_t mb_idx = A->num_mblocks;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;

        mblock_t *MB = &A->mblocks[mb_idx];
        MB->first_inst = A->num_minsts;
        MB->bir_block = bir_bi;
        S.current_bir_block = bir_bi;

        /* EXEC mask restore for divergent regions.
           At else-block start: flip to false lanes.
           At merge-block start: restore all lanes.
           Like changing shifts — the work never stops, the crew just rotates. */
        for (uint32_t di = 0; di < S.div_depth; di++) {
            if (S.div_stack[di].has_else && S.div_stack[di].false_bir == bir_bi) {
                /* Else block: xor EXEC to get false lanes */
                uint16_t sv = (uint16_t)S.div_stack[di].saved_vreg;
                moperand_t sv_op = is_cdna() ? mop_sgpr(sv) : mop_vreg_s(sv);
                emit2(is_cdna() ? AMD_S_XOR_B64 : AMD_S_XOR_B32,
                      mop_special(AMD_SPEC_EXEC),
                      mop_special(AMD_SPEC_EXEC), sv_op);
                /* If all false lanes are off, skip else body */
                uint32_t merge_mb = S.block_map[S.div_stack[di].merge_bir];
                emit0_1(AMD_S_CBRANCH_EXECZ, mop_label(merge_mb));
            }
        }
        for (uint32_t di = S.div_depth; di > 0; di--) {
            if (S.div_stack[di - 1].merge_bir == bir_bi) {
                /* Merge block: restore all lanes */
                uint16_t sv = (uint16_t)S.div_stack[di - 1].saved_vreg;
                moperand_t sv_op = is_cdna() ? mop_sgpr(sv) : mop_vreg_s(sv);
                emit2(is_cdna() ? AMD_S_OR_B64 : AMD_S_OR_B32,
                      mop_special(AMD_SPEC_EXEC),
                      mop_special(AMD_SPEC_EXEC), sv_op);
                /* Pop the region */
                S.div_depth--;
            }
        }

        /* Save hardware thread IDs before params can clobber v0/v1/v2.
           First block of kernel only — like saving the black box before takeoff.
           Only save dims the kernel actually uses (max_dim). */
        if (bi == 0 && S.is_kernel) {
            for (uint32_t d = 0; d <= (uint32_t)S.max_dim && d < 3; d++) {
                S.saved_tid[d] = new_vreg(1);
                emit1(AMD_V_MOV_B32, mop_vreg_v((uint16_t)S.saved_tid[d]),
                      mop_vgpr((uint16_t)d));
            }
        }

        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            int div = is_divergent(idx);

            switch (I->op) {
            /* Arithmetic + Bitwise */
            case BIR_ADD: case BIR_SUB: case BIR_MUL:
            case BIR_SDIV: case BIR_UDIV: case BIR_SREM: case BIR_UREM:
            case BIR_FADD: case BIR_FSUB: case BIR_FMUL:
            case BIR_FDIV: case BIR_FREM:
            case BIR_FMAX: case BIR_FMIN:
            case BIR_AND: case BIR_OR: case BIR_XOR:
            case BIR_SHL: case BIR_LSHR: case BIR_ASHR:
                isel_arith(idx, I, div);
                break;

            /* Comparison */
            case BIR_ICMP:
                isel_icmp(idx, I, div);
                break;
            case BIR_FCMP:
                isel_fcmp(idx, I);
                break;

            /* Conversion */
            case BIR_TRUNC: case BIR_ZEXT: case BIR_SEXT:
            case BIR_FPTRUNC: case BIR_FPEXT:
            case BIR_FPTOSI: case BIR_FPTOUI:
            case BIR_SITOFP: case BIR_UITOFP:
            case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST:
            case BIR_SQRT: case BIR_RSQ: case BIR_RCP:
            case BIR_EXP2: case BIR_LOG2:
            case BIR_SIN: case BIR_COS:
            case BIR_FABS: case BIR_FLOOR: case BIR_CEIL:
            case BIR_FTRUNC: case BIR_RNDNE:
                isel_conversion(idx, I, div);
                break;

            /* Memory */
            case BIR_LOAD:
                isel_load(idx, I, div);
                break;
            case BIR_STORE:
                isel_store(I, div);
                break;
            case BIR_GEP:
                isel_gep(idx, I, div);
                break;
            case BIR_ALLOCA:
                isel_alloca(idx, I);
                break;
            case BIR_SHARED_ALLOC:
                isel_shared_alloc(idx, I);
                break;
            case BIR_GLOBAL_REF:
                isel_global_ref(idx, I);
                break;

            /* Control flow */
            case BIR_BR:
                isel_branch(I);
                break;
            case BIR_BR_COND: {
                int cond_div = val_is_divergent(I->operands[0]);
                isel_br_cond(I, cond_div);
                break;
            }
            case BIR_SWITCH:
                isel_switch(I);
                break;
            case BIR_RET:
                isel_ret(I, I->num_operands);
                break;
            case BIR_UNREACHABLE:
                emit0_0(AMD_S_ENDPGM, 0);
                break;

            /* SSA */
            case BIR_PHI:
                isel_phi(idx, I, div);
                break;
            case BIR_PARAM:
                isel_param(idx, I);
                break;

            /* Thread model */
            case BIR_THREAD_ID:
            case BIR_BLOCK_ID:
            case BIR_BLOCK_DIM:
            case BIR_GRID_DIM:
                isel_thread_model(idx, I);
                break;

            /* Barriers */
            case BIR_BARRIER:
            case BIR_BARRIER_GROUP:
                isel_barrier();
                break;

            /* Atomics */
            case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
            case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
            case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
            case BIR_ATOMIC_XCHG: case BIR_ATOMIC_CAS:
                isel_atomic(idx, I, div);
                break;
            case BIR_ATOMIC_LOAD:
                isel_atomic_load(idx, I, div);
                break;
            case BIR_ATOMIC_STORE:
                isel_atomic_store(I, div);
                break;

            /* Warp-level */
            case BIR_SHFL: case BIR_SHFL_UP:
            case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
                isel_warp(idx, I);
                break;

            /* Matrix */
            case BIR_MFMA:
                isel_mfma(idx, I);
                break;

            /* Misc */
            case BIR_SELECT:
                isel_select(idx, I, div);
                break;
            case BIR_CALL:
                isel_call(idx, I, div);
                break;
            case BIR_INLINE_ASM:
                /* Skip inline asm for now */
                break;

            default:
                break;
            }
        }

        MB->num_insts = A->num_minsts - MB->first_inst;
        if (A->num_mblocks < AMD_MAX_MBLOCKS)
            A->num_mblocks++;
    }

    MF->num_blocks = (uint16_t)(A->num_mblocks - MF->first_block);
    MF->scratch_bytes = S.scratch_offset;
    MF->kernarg_bytes = S.hkrarg;
    MF->lds_bytes = (uint16_t)S.lds_offset;
    MF->first_alloc_sgpr = S.next_param_sgpr;
}

/* ---- Public API ---- */

int amdgpu_compile(const bir_module_t *bir, amd_module_t *amd)
{
    memset(&S, 0, sizeof(S));
    S.bir = bir;
    S.amd = amd;

    /* Initialize module */
    amd->bir = bir;
    amd->num_minsts = 0;
    amd->num_mblocks = 0;
    amd->num_mfuncs = 0;
    amd->vreg_count = 0;
    amd->code_len = 0;
    amd->asm_len = 0;
    memset(amd->divergent, 0, sizeof(amd->divergent));
    memset(amd->val_vreg, 0xFF, sizeof(amd->val_vreg));
    memset(amd->val_file, 0, sizeof(amd->val_file));
    memset(amd->reg_map, 0, sizeof(amd->reg_map));
    memset(amd->reg_file, 0, sizeof(amd->reg_file));
    memset(amd->val_sbase, 0xFF, sizeof(amd->val_sbase));

    /* Process each function */
    for (uint32_t fi = 0; fi < bir->num_funcs; fi++) {
        isel_function(fi);
    }

    return BC_OK;
}
