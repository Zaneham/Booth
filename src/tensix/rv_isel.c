#include "rv_isel.h"
#include "rv_enc.h"
#include "rt_args.h"
#include "tdf.h"
#include <stdio.h>
#include <string.h>

/*
 * Bring-up isel: stack-spill every BIR value, walk the instruction stream
 * once, emit a small pattern per opcode. The frame keeps saved ra at sp+0
 * (always saved, leaf or not, to keep the prologue uniform; the host
 * loader does not care), reserved padding at sp+4 for 16-byte alignment,
 * then one 4-byte slot per BIR value from sp+8 onward. Each instruction's
 * slot lives at sp + ISEL_LOCALS_BASE + bir_inst_index * 4; constants get
 * no slot and are materialised into t0/t1 on demand. The frame size must
 * be 16-byte aligned per the RISC-V soft-float RV32 psABI (Unprivileged
 * ISA appendix on the integer calling convention).
 */

#define ISEL_LOCALS_BASE  8u           /* bytes; ra + pad */
#define ISEL_MAX_INSTS    1024u        /* per-function ceiling */
#define ISEL_FRAME_ALIGN  16u
#define ISEL_MAX_BLOCKS   256u
#define ISEL_MAX_PATCHES  512u
#define ISEL_MAX_PHI_COPIES 512u
#define ISEL_ALLOCA_ALIGN 4u           /* word-align every alloca */

/*
 * Branch patching scratch. Block positions get filled in as we
 * emit each basic block; branches to as-yet-unemitted blocks are
 * recorded here and resolved at the end of the function. Forward
 * references are common because BIR follows the natural source
 * order which usually places the consequent before the alternate.
 */
typedef struct {
    uint32_t word_idx;       /* placeholder location in code buffer */
    uint16_t target_block;   /* absolute index into M->blocks       */
    uint8_t  kind;           /* 0=BEQ rs1,zero; 1=BNE rs1,zero;     */
                             /* 2=JAL x0 (unconditional)            */
    uint8_t  rs1;            /* condition register for BEQ/BNE      */
} patch_t;

static uint32_t blkwidx[ISEL_MAX_BLOCKS];
static uint8_t  blkknwn[ISEL_MAX_BLOCKS];
static patch_t  patches[ISEL_MAX_PATCHES];
static uint32_t npatch;

/*
 * PHI destruction scratch. For each PHI (block, value) pair in the
 * function, we record a "copy" that the predecessor block must
 * emit just before its terminator: load the value, store it into
 * the PHI's stack slot. The PHI instruction itself then emits no
 * code in its own block because its slot is already populated by
 * whichever predecessor branched in.
 *
 * Classical SSA destruction; the swap and lost-copy problems are
 * avoided because every BIR instruction already has its own stack
 * slot and we detect within-batch cyclic reads at emission time
 * and refuse rather than corrupt.
 */
typedef struct {
    uint16_t pred_block;   /* predecessor block this copy fires in */
    uint16_t phi_inst;     /* destination slot (PHI's inst index)   */
    uint32_t src_val;      /* value handle (CONST or VAL)           */
} phicp_t;

static phicp_t phicp[ISEL_MAX_PHI_COPIES];
static uint32_t   nphicp;

/*
 * Per-function position tracking for inter-function calls. When a
 * BIR_CALL is emitted we don't necessarily know where the callee
 * lives in the code buffer yet, so the JAL goes in as a placeholder
 * and gets back-patched once every function has been laid out.
 * Same pattern as the block-branch patcher above; calls just have
 * a wider offset range (J-type is +/- 1 MiB) and target functions
 * instead of basic blocks.
 *
 * The recursion check is a separate pass over the instruction
 * stream before emission so we refuse with a clear message rather
 * than discovering halfway through that the prologue's saved ra
 * is about to be clobbered by a self-call.
 */
#define ISEL_MAX_FUNCS    32u
#define ISEL_MAX_CALLS    256u

typedef struct {
    uint32_t jal_word_idx;   /* placeholder location in the buffer */
    uint16_t callee_func;    /* target function index in M->funcs[] */
    uint16_t _pad;
} calpat_t;

static uint32_t     fnwidx[ISEL_MAX_FUNCS];
static uint8_t      fnknwn[ISEL_MAX_FUNCS];
static calpat_t calpat[ISEL_MAX_CALLS];
static uint32_t     ncalpat;
static uint32_t     curfn;     /* set per-function during emit */

/* Module-global index of this function's first instruction. Slot indices are
 * biased by it so function N>0 addresses its own frame rather than the
 * caller's, and so offsets stay inside the 12-bit immediate. */
static uint32_t     slotbas;

static void isel_rstcf(void)
{
    memset(blkwidx, 0, sizeof(blkwidx));
    memset(blkknwn,    0, sizeof(blkknwn));
    npatch    = 0;
    nphicp = 0;
}

static void isel_rstmod(void)
{
    memset(fnwidx, 0, sizeof(fnwidx));
    memset(fnknwn,    0, sizeof(fnknwn));
    ncalpat = 0;
    curfn = 0;
}

static int recpat(uint32_t word_idx, uint16_t target,
                        uint8_t kind, uint8_t rs1)
{
    if (npatch >= ISEL_MAX_PATCHES) {
        fprintf(stderr,
                "rv_isel: branch patch table full (%u entries)\n",
                ISEL_MAX_PATCHES);
        return BC_ERR_TDF;
    }
    patches[npatch].word_idx     = word_idx;
    patches[npatch].target_block = target;
    patches[npatch].kind         = kind;
    patches[npatch].rs1          = rs1;
    npatch++;
    return BC_OK;
}

/* ---- Helpers ---- */

static uint32_t rndup(uint32_t x, uint32_t a)
{
    return (x + (a - 1u)) & ~(a - 1u);
}

static int emit(rv_buf_t *out, uint32_t word)
{
    if (rv_buf_emit(out, word) < 0) return BC_ERR_OVERFLOW;
    return BC_OK;
}

/* Each BIR inst gets one 4-byte slot off sp, after the prologue's saved ra.
 * Refuses past 2047 because LW/SW take a 12-bit signed immediate and would
 * otherwise wrap to a negative offset and address below the frame. */
static int slotoff(uint32_t inst_idx, uint32_t *off)
{
    uint32_t o = ISEL_LOCALS_BASE + (inst_idx - slotbas) * 4u;
    if (o > 2047u) {
        fprintf(stderr,
                "rv_isel: stack slot for inst %u at offset %u exceeds the "
                "12-bit immediate; function too large for the bring-up isel\n",
                inst_idx, o);
        return BC_ERR_TDF;
    }
    *off = o;
    return BC_OK;
}

/*
 * Materialise a 32-bit integer constant into a register using the
 * canonical LUI/ADDI pair. The trick is in how RV32IM sign-extends
 * the I-type immediate: ADDI adds a 12-bit signed value, so if the
 * low 12 bits of our target value would be interpreted as negative
 * the ADDI will subtract that magnitude from the LUI result and we
 * need to bump the LUI's upper half up by one to compensate.
 *
 * For values that fit in 12-bit signed (-2048..2047) we skip the
 * LUI entirely and use a single ADDI from x0, which is the canonical
 * small-constant encoding (also reaches across the LI pseudoinstruction
 * coverage in standard assemblers). The spec citation is the
 * Unprivileged ISA section 2.4.1 (U-format) plus the sign-extension
 * note in section 2.3.
 */
static int ldimm32(uint8_t reg, int32_t v, rv_buf_t *out)
{
    if (v >= -2048 && v <= 2047) {
        return emit(out, rv_addi(reg, RV_ZERO, (int16_t)v));
    }
    /* Split into upper 20 bits and lower 12-bit signed remainder.
     * If the low half's top bit is set the remainder is negative,
     * so add 1 to the high half so the final ADDI subtracts back
     * down to the intended value. */
    uint32_t uv = (uint32_t)v;
    uint32_t lo = uv & 0xFFFu;
    uint32_t hi = uv >> 12;
    if (lo & 0x800u) hi = (hi + 1u) & 0xFFFFFu;
    int16_t lo_signed = (int16_t)(int32_t)((lo & 0x800u) ? (lo | 0xFFFFF000u)
                                                         : lo);
    int rc = emit(out, rv_lui(reg, hi & 0xFFFFFu));
    if (rc != BC_OK) return rc;
    if (lo_signed == 0) return BC_OK;
    return emit(out, rv_addi(reg, reg, lo_signed));
}

/* Load a BIR operand into the given temp register, leaving the
 * register holding the value. Handles constants by materialising
 * via ldimm32, and instruction values by loading their stack slot. */
static int ldopnd(const bir_module_t *M, uint32_t operand,
                        uint8_t reg, rv_buf_t *out)
{
    if (BIR_VAL_IS_CONST(operand)) {
        uint32_t ci = BIR_VAL_INDEX(operand);
        if (ci >= M->num_consts) {
            fprintf(stderr, "rv_isel: bad constant index %u\n", ci);
            return BC_ERR_TDF;
        }
        const bir_const_t *c = &M->consts[ci];
        if (c->kind == BIR_CONST_ZERO) {
            return emit(out, rv_addi(reg, RV_ZERO, 0));
        }
        if (c->kind == BIR_CONST_INT) {
            int64_t v = c->d.ival;
            if (v < INT32_MIN || v > UINT32_MAX) {
                fprintf(stderr,
                        "rv_isel: constant %lld out of i32 range\n",
                        (long long)v);
                return BC_ERR_TDF;
            }
            return ldimm32(reg, (int32_t)v, out);
        }
        if (c->kind == BIR_CONST_NULL) {
            return emit(out, rv_addi(reg, RV_ZERO, 0));
        }
        if (c->kind == BIR_CONST_FLOAT) {
            /* Float constants get materialised as their 32-bit bit
             * pattern. This is honest: the bits in the register are
             * the same bits the kernel would see for a fp32 value.
             * Any subsequent BIR_FADD or other float arithmetic
             * will refuse, because we have no soft-float runtime
             * yet, but plain "load a float constant and store it
             * to memory" works correctly without infrastructure. */
            float fv = (float)c->d.fval;
            uint32_t bits;
            memcpy(&bits, &fv, sizeof(bits));
            return ldimm32(reg, (int32_t)bits, out);
        }
        fprintf(stderr,
                "rv_isel: constant kind %u not supported for RV32IM\n",
                c->kind);
        return BC_ERR_TDF;
    }
    /* Instruction value: load from stack slot. */
    uint32_t idx = BIR_VAL_INDEX(operand);
    uint32_t off;
    int rc = slotoff(idx, &off);
    if (rc != BC_OK) return rc;
    return emit(out, rv_lw(reg, RV_SP, (int16_t)off));
}

static int stres(uint8_t reg, uint32_t inst_idx, rv_buf_t *out)
{
    uint32_t off;
    int rc = slotoff(inst_idx, &off);
    if (rc != BC_OK) return rc;
    return emit(out, rv_sw(reg, RV_SP, (int16_t)off));
}

/* ---- Per-opcode handlers ---- */

static int sel_binop(const bir_module_t *M, uint32_t inst_idx,
                     const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    if ((rc = ldopnd(M, I->operands[1], RV_T1, out)) != BC_OK) return rc;
    uint32_t op = 0;
    switch (I->op) {
    /* RV32I integer ALU. */
    case BIR_ADD:  op = rv_add (RV_T0, RV_T0, RV_T1); break;
    case BIR_SUB:  op = rv_sub (RV_T0, RV_T0, RV_T1); break;
    case BIR_AND:  op = rv_and (RV_T0, RV_T0, RV_T1); break;
    case BIR_OR:   op = rv_or  (RV_T0, RV_T0, RV_T1); break;
    case BIR_XOR:  op = rv_xor (RV_T0, RV_T0, RV_T1); break;
    case BIR_SHL:  op = rv_sll (RV_T0, RV_T0, RV_T1); break;
    case BIR_LSHR: op = rv_srl (RV_T0, RV_T0, RV_T1); break;
    case BIR_ASHR: op = rv_sra (RV_T0, RV_T0, RV_T1); break;
    /* RV32M integer multiply / divide. Divide is 2-33 cycles on
     * baby cores per the BabyRISCV README, so this is real but
     * sluggish; a future optimiser pass can recognise divides by
     * constant powers of two and rewrite to ASHR. */
    case BIR_MUL:  op = rv_mul (RV_T0, RV_T0, RV_T1); break;
    /* High half of the unsigned product; the keystone for wide-integer
     * carry chains, and RV32M gives it to us directly. */
    case BIR_UMULHI: op = rv_mulhu(RV_T0, RV_T0, RV_T1); break;
    case BIR_SDIV: op = rv_div (RV_T0, RV_T0, RV_T1); break;
    case BIR_UDIV: op = rv_divu(RV_T0, RV_T0, RV_T1); break;
    case BIR_SREM: op = rv_rem (RV_T0, RV_T0, RV_T1); break;
    case BIR_UREM: op = rv_remu(RV_T0, RV_T0, RV_T1); break;
    default:
        fprintf(stderr, "rv_isel: unreachable binop op %u\n", I->op);
        return BC_ERR_TDF;
    }
    if ((rc = emit(out, op)) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/* ---- Integer comparison ----
 *
 * BIR_ICMP's subop encodes the comparison variant. RV32I provides
 * SLT and SLTU for the two strictly-less-than directions; the other
 * eight relational variants compose out of those, often as a swap
 * and a 1-bit invert. EQ and NE use the (a XOR b) trick: the XOR
 * is zero iff equal, then SLTIU rd, rd, 1 produces a 0/1 result
 * suitable for storing in the BIR slot.
 *
 * Operand order in the BIR is (lhs, rhs); the spec says SLT rd,a,b
 * sets rd to 1 if a<b. We match that for SLT/SLTU and swap for the
 * "greater than" variants. The invert pattern (XORI rd, rd, 1) is
 * the canonical single-instruction logical not for a value already
 * constrained to {0, 1}. */
static int sel_icmp(const bir_module_t *M, uint32_t inst_idx,
                    const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    if ((rc = ldopnd(M, I->operands[1], RV_T1, out)) != BC_OK) return rc;

    switch (I->subop) {
    case BIR_ICMP_EQ:
        if ((rc = emit(out, rv_xor (RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
        rc = emit(out, rv_sltiu(RV_T0, RV_T0, 1));
        break;
    case BIR_ICMP_NE:
        if ((rc = emit(out, rv_xor (RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
        rc = emit(out, rv_sltu(RV_T0, RV_ZERO, RV_T0));
        break;
    case BIR_ICMP_SLT:
        rc = emit(out, rv_slt (RV_T0, RV_T0, RV_T1));
        break;
    case BIR_ICMP_SLE:
        if ((rc = emit(out, rv_slt(RV_T0, RV_T1, RV_T0))) != BC_OK) return rc;
        rc = emit(out, rv_xori(RV_T0, RV_T0, 1));
        break;
    case BIR_ICMP_SGT:
        rc = emit(out, rv_slt (RV_T0, RV_T1, RV_T0));
        break;
    case BIR_ICMP_SGE:
        if ((rc = emit(out, rv_slt(RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
        rc = emit(out, rv_xori(RV_T0, RV_T0, 1));
        break;
    case BIR_ICMP_ULT:
        rc = emit(out, rv_sltu(RV_T0, RV_T0, RV_T1));
        break;
    case BIR_ICMP_ULE:
        if ((rc = emit(out, rv_sltu(RV_T0, RV_T1, RV_T0))) != BC_OK) return rc;
        rc = emit(out, rv_xori(RV_T0, RV_T0, 1));
        break;
    case BIR_ICMP_UGT:
        rc = emit(out, rv_sltu(RV_T0, RV_T1, RV_T0));
        break;
    case BIR_ICMP_UGE:
        if ((rc = emit(out, rv_sltu(RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
        rc = emit(out, rv_xori(RV_T0, RV_T0, 1));
        break;
    default:
        fprintf(stderr,
                "rv_isel: unrecognised ICMP subop %u\n", I->subop);
        return BC_ERR_TDF;
    }
    if (rc != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/* ---- Identity casts ----
 *
 * On RV32IM both pointers and integers fit in a 32-bit register,
 * so PTRTOINT, INTTOPTR, and BITCAST between i32-sized types are
 * pure register copies. We still emit a load + store cycle so the
 * BIR slot bookkeeping stays uniform with the rest of the isel.
 * The compiler could fold these out at a later peephole pass once
 * we have one. */
static int sel_cast_id(const bir_module_t *M, uint32_t inst_idx,
                       const bir_inst_t *I, rv_buf_t *out)
{
    int rc = ldopnd(M, I->operands[0], RV_T0, out);
    if (rc != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/* ---- Width conversions ----
 *
 * Source and destination integer widths come from the operand and
 * the result type respectively. The lowering is straight out of the
 * Unprivileged ISA spec page 26 (shift idioms): a zero-extend masks with
 * (1<<src_w)-1, done via SLLI/SRLI to clear the high bits rather than an
 * ANDI whose immediate cannot reach 16 or 24 bits; a sign-extend is SLLI
 * by (32-src_w) then SRAI back; and a truncate uses the same clear-the-
 * high-bits trick. The shift-twice idiom works for any source width up to 31 and
 * gives the right answer regardless of whatever was in the high
 * bits before, which is the assumption we have to make because the
 * slot store/load pipeline keeps 32-bit values without tracking
 * provenance. */
static uint32_t valwid(const bir_module_t *M, uint32_t val)
{
    if (BIR_VAL_IS_CONST(val)) return 32u;
    uint32_t idx = BIR_VAL_INDEX(val);
    if (idx >= M->num_insts) return 32u;
    uint32_t ti = M->insts[idx].type;
    if (ti >= M->num_types) return 32u;
    const bir_type_t *t = &M->types[ti];
    if (t->kind == BIR_TYPE_INT) return t->width;
    return 32u;
}

static uint32_t instwid(const bir_module_t *M, const bir_inst_t *I)
{
    if (I->type >= M->num_types) return 32u;
    const bir_type_t *t = &M->types[I->type];
    if (t->kind == BIR_TYPE_INT) return t->width;
    return 32u;
}

static int sel_zext(const bir_module_t *M, uint32_t inst_idx,
                    const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    uint32_t src_w = valwid(M, I->operands[0]);
    if (src_w >= 32u) return stres(RV_T0, inst_idx, out);
    /* SLLI (32-src_w) then SRLI clears high bits without needing
     * a mask wider than ANDI's 12-bit immediate can express. */
    uint32_t sh = 32u - src_w;
    if ((rc = emit(out, rv_slli(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    if ((rc = emit(out, rv_srli(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

static int sel_sext(const bir_module_t *M, uint32_t inst_idx,
                    const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    uint32_t src_w = valwid(M, I->operands[0]);
    if (src_w >= 32u) return stres(RV_T0, inst_idx, out);
    uint32_t sh = 32u - src_w;
    if ((rc = emit(out, rv_slli(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    if ((rc = emit(out, rv_srai(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

static int sel_trunc(const bir_module_t *M, uint32_t inst_idx,
                     const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    uint32_t dst_w = instwid(M, I);
    if (dst_w >= 32u) return stres(RV_T0, inst_idx, out);
    uint32_t sh = 32u - dst_w;
    if ((rc = emit(out, rv_slli(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    if ((rc = emit(out, rv_srli(RV_T0, RV_T0, (uint8_t)sh))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/* ---- Branches ----
 *
 * Two emission paths: a backward branch where we already know the
 * target's word index can be encoded straight away, and a forward
 * branch where we drop a placeholder and resolve it at end-of-function.
 * The B-type offset is signed 13-bit (+/-4 KiB) which covers any
 * realistic function we'd put on one baby core, and JAL gets a
 * signed 21-bit (+/-1 MiB) offset for the same reason squared.
 *
 * BEQ/BNE encode a 2-byte-aligned offset; JAL encodes the same.
 * The rv_enc layer takes a byte offset and packs the scrambled
 * immediate per the spec. */
static int ebrcond(uint16_t target_block, uint8_t rs1,
                        int branch_kind, rv_buf_t *out)
{
    /* branch_kind: 0 = BEQ (branch if rs1 == zero, i.e. false),
     *              1 = BNE (branch if rs1 != zero, i.e. true) */
    if (target_block < ISEL_MAX_BLOCKS && blkknwn[target_block]) {
        uint32_t here = rv_buf_n_words(out);
        int32_t off = rv_buf_offset(out, here, blkwidx[target_block]);
        if (off < -4096 || off > 4094) {
            fprintf(stderr,
                    "rv_isel: branch offset %d out of B-type range\n", off);
            return BC_ERR_TDF;
        }
        uint32_t w = (branch_kind == 0)
                   ? rv_beq(rs1, RV_ZERO, (int16_t)off)
                   : rv_bne(rs1, RV_ZERO, (int16_t)off);
        return emit(out, w);
    }
    /* Forward reference: emit a placeholder and record the patch. */
    int slot = rv_buf_emit(out, 0u);
    if (slot < 0) return BC_ERR_OVERFLOW;
    return recpat((uint32_t)slot, target_block,
                        (uint8_t)branch_kind, rs1);
}

static int ejalbk(uint16_t target_block, rv_buf_t *out)
{
    if (target_block < ISEL_MAX_BLOCKS && blkknwn[target_block]) {
        uint32_t here = rv_buf_n_words(out);
        int32_t off = rv_buf_offset(out, here, blkwidx[target_block]);
        if (off < -1048576 || off > 1048574) {
            fprintf(stderr,
                    "rv_isel: JAL offset %d out of J-type range\n", off);
            return BC_ERR_TDF;
        }
        return emit(out, rv_jal(RV_ZERO, off));
    }
    int slot = rv_buf_emit(out, 0u);
    if (slot < 0) return BC_ERR_OVERFLOW;
    return recpat((uint32_t)slot, target_block, 2u, 0u);
}

/*
 * Walk every PHI in the function and queue one phicp_t entry
 * per incoming (block, value) pair. The PHI operand layout is
 * (block, value, block, value, ...) either inline or overflowed
 * into M->extra_operands; we handle both.
 */
static int phicoll(const bir_module_t *M, const bir_func_t *F)
{
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bk = F->first_block + bi;
        if (bk >= M->num_blocks) break;
        const bir_block_t *B = &M->blocks[bk];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            if (I->op != BIR_PHI) continue;
            uint32_t pair_count;
            const uint32_t *pairs;
            uint32_t inline_pairs[BIR_OPERANDS_INLINE];
            if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
                uint32_t start = I->operands[0];
                pair_count = I->operands[1];
                if (start + pair_count > M->num_extra_ops) {
                    fprintf(stderr,
                            "rv_isel: PHI overflow operands past extra_ops table\n");
                    return BC_ERR_TDF;
                }
                pairs = &M->extra_operands[start];
            } else {
                pair_count = I->num_operands;
                for (uint32_t k = 0; k < pair_count; k++)
                    inline_pairs[k] = I->operands[k];
                pairs = inline_pairs;
            }
            if (pair_count & 1u) {
                fprintf(stderr, "rv_isel: PHI has odd operand count\n");
                return BC_ERR_TDF;
            }
            for (uint32_t p = 0; p + 1 < pair_count; p += 2) {
                if (nphicp >= ISEL_MAX_PHI_COPIES) {
                    fprintf(stderr,
                            "rv_isel: PHI copy table full (%u entries)\n",
                            ISEL_MAX_PHI_COPIES);
                    return BC_ERR_TDF;
                }
                phicp[nphicp].pred_block = (uint16_t)pairs[p];
                phicp[nphicp].phi_inst   = (uint16_t)idx;
                phicp[nphicp].src_val    = pairs[p + 1];
                nphicp++;
            }
        }
    }
    return BC_OK;
}

/*
 * Emit all PHI copies whose pred_block matches the given block,
 * just before the block's terminator branch. Detects within-batch
 * cyclic reads: if a copy reads from a slot that an earlier copy
 * in the same batch wrote to, the resulting value would be wrong,
 * so we refuse rather than corrupt. The fix for that case is a
 * proper parallel-copy resolver with a temp register, which lives
 * in a future sitting; flagging it cleanly is the right thing
 * until that work lands.
 */
static int ephicp(const bir_module_t *M,
                                    uint16_t pred_block, rv_buf_t *out)
{
    static uint8_t written[ISEL_MAX_INSTS];
    /* Clear only what we use; the array is large enough that a
     * full memset every block would be wasteful. We walk the
     * batch twice: once to mark dests, once to emit; conflict
     * detection happens during the emit pass. */
    for (uint32_t p = 0; p < nphicp; p++) {
        if (phicp[p].pred_block != pred_block) continue;
        if (phicp[p].phi_inst >= ISEL_MAX_INSTS) continue;
        written[phicp[p].phi_inst] = 0;
    }

    for (uint32_t p = 0; p < nphicp; p++) {
        if (phicp[p].pred_block != pred_block) continue;
        uint32_t src = phicp[p].src_val;
        if (!BIR_VAL_IS_CONST(src)) {
            uint32_t sidx = BIR_VAL_INDEX(src);
            if (sidx < ISEL_MAX_INSTS && written[sidx]) {
                fprintf(stderr,
                        "rv_isel: cyclic PHI copy in pred block %u "
                        "(reads slot %u after writing it); "
                        "needs parallel-copy resolver\n",
                        pred_block, sidx);
                return BC_ERR_TDF;
            }
        }
        int rc = ldopnd(M, src, RV_T0, out);
        if (rc != BC_OK) return rc;
        if ((rc = stres(RV_T0, phicp[p].phi_inst, out)) != BC_OK)
            return rc;
        if (phicp[p].phi_inst < ISEL_MAX_INSTS)
            written[phicp[p].phi_inst] = 1;
    }
    return BC_OK;
}

static int sel_br(const bir_module_t *M, uint16_t cur_block,
                  const bir_inst_t *I, rv_buf_t *out)
{
    /* BIR_BR: ops[0] = absolute block index. JAL x0, target.
     * PHI copies for our successor's PHIs fire just before the
     * branch, while we're still in the current block. */
    if (I->num_operands < 1u) {
        fprintf(stderr, "rv_isel: BR with no target\n");
        return BC_ERR_TDF;
    }
    int rc = ephicp(M, cur_block, out);
    if (rc != BC_OK) return rc;
    return ejalbk((uint16_t)I->operands[0], out);
}

/* Operand count, resolving the overflow indirection. */
static uint32_t nopnd(const bir_inst_t *I)
{
    return I->num_operands == BIR_OPERANDS_OVERFLOW
         ? I->operands[1] : I->num_operands;
}

/* Operand j, resolving the overflow indirection. Returns BIR_VAL_NONE if the
 * index is out of range, which every caller must treat as malformed. */
static uint32_t opnd(const bir_module_t *M, const bir_inst_t *I, uint32_t j)
{
    if (j >= nopnd(I)) return BIR_VAL_NONE;
    if (I->num_operands != BIR_OPERANDS_OVERFLOW) return I->operands[j];
    uint32_t k = I->operands[0] + j;
    if (k >= M->num_extra_ops) return BIR_VAL_NONE;
    return M->extra_operands[k];
}

/*
 * Switch: operand 0 is the selector, 1 the default block, then (value, block)
 * pairs. We emit a compare chain rather than a jump table, since the case
 * values are arbitrary and a table would need a relocated base address that
 * XIP would have to rewrite.
 *
 * Each case subtracts its value from the selector and branches on zero, which
 * reuses the BEQ-against-zero patch the block patcher already knows how to
 * resolve instead of needing a two-register branch form.
 */
static int sel_switch(const bir_module_t *M, uint16_t cur_block,
                      const bir_inst_t *I, rv_buf_t *out)
{
    uint32_t n = nopnd(I);
    if (n < 2u || (n & 1u) != 0u) {
        fprintf(stderr,
                "rv_isel: SWITCH needs a selector, a default and whole "
                "(value, block) pairs, got %u operands\n", n);
        return BC_ERR_TDF;
    }
    uint32_t sel = opnd(M, I, 0);
    uint32_t dflt = opnd(M, I, 1);
    if (sel == BIR_VAL_NONE || dflt == BIR_VAL_NONE) {
        fprintf(stderr, "rv_isel: SWITCH operands out of range\n");
        return BC_ERR_TDF;
    }

    int rc = ephicp(M, cur_block, out);
    if (rc != BC_OK) return rc;

    for (uint32_t j = 2u; j + 1u < n; j += 2u) {
        uint32_t cval = opnd(M, I, j);
        uint32_t cblk = opnd(M, I, j + 1u);
        if (cval == BIR_VAL_NONE || cblk == BIR_VAL_NONE) {
            fprintf(stderr, "rv_isel: SWITCH case %u out of range\n", j);
            return BC_ERR_TDF;
        }
        if ((rc = ldopnd(M, sel, RV_T0, out)) != BC_OK) return rc;
        if ((rc = ldopnd(M, cval, RV_T1, out)) != BC_OK) return rc;
        if ((rc = emit(out, rv_sub(RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
        if ((rc = ebrcond((uint16_t)cblk, RV_T0, 0, out)) != BC_OK) return rc;
    }
    return ejalbk((uint16_t)dflt, out);
}

static int sel_br_cond(const bir_module_t *M, uint16_t cur_block,
                       const bir_inst_t *I, rv_buf_t *out)
{
    /* BIR_BR_COND: ops[0] = cond value, [1] = true block,
     *              [2] = false block, [3] = merge (ignored here).
     * PHI copies fire before the conditional split. Both branch
     * targets see the same set of copies because BIR PHIs only
     * record the predecessor block, not which arm of a cond
     * branch was taken; that semantics matches LLVM SSA. */
    if (I->num_operands < 3u) {
        fprintf(stderr, "rv_isel: BR_COND needs cond + true + false\n");
        return BC_ERR_TDF;
    }
    int rc = ldopnd(M, I->operands[0], RV_T0, out);
    if (rc != BC_OK) return rc;
    if ((rc = ephicp(M, cur_block, out)) != BC_OK) return rc;
    /* BNE T0, zero, true_block; JAL zero, false_block. */
    if ((rc = ebrcond((uint16_t)I->operands[1], RV_T0, 1, out)) != BC_OK)
        return rc;
    return ejalbk((uint16_t)I->operands[2], out);
}

/* ---- Select ----
 *
 * Conditional move with no architectural cmov. We materialise the
 * false value first, branch over the true-arm store if the cond is
 * false, otherwise overwrite with the true value. The branch is
 * a fixed +12 bytes (skip three instructions); a real optimiser
 * would use the standard SLT-and-AND-with-mask trick instead but
 * we are not that today. */
static int sel_select(const bir_module_t *M, uint32_t inst_idx,
                      const bir_inst_t *I, rv_buf_t *out)
{
    if (I->num_operands < 3u) {
        fprintf(stderr, "rv_isel: SELECT needs cond + true + false\n");
        return BC_ERR_TDF;
    }
    int rc;
    /* Load condition into T2 so we can keep T0/T1 free for the
     * value loads. */
    if ((rc = ldopnd(M, I->operands[0], RV_T2, out)) != BC_OK) return rc;
    /* Materialise the false value into T0 first. */
    if ((rc = ldopnd(M, I->operands[2], RV_T0, out)) != BC_OK) return rc;
    /* BEQ T2, zero, +N to skip past the true-arm load if cond is 0.
     * The true arm is the next op's ldopnd which can be 1 or
     * more instructions (constant materialisation may use LUI+ADDI).
     * Easiest: emit BEQ to a placeholder, load the true value,
     * back-patch the BEQ offset once we know how big the true arm is. */
    int beq_slot = rv_buf_emit(out, 0u);
    if (beq_slot < 0) return BC_ERR_OVERFLOW;
    uint32_t before_true = rv_buf_n_words(out);
    if ((rc = ldopnd(M, I->operands[1], RV_T0, out)) != BC_OK) return rc;
    uint32_t after_true = rv_buf_n_words(out);
    int32_t skip_bytes = (int32_t)(after_true - before_true) * 4 + 4;
    /* +4 because the branch is at beq_slot and we want to skip to
     * the instruction after the true arm. */
    if (skip_bytes < -4096 || skip_bytes > 4094) {
        fprintf(stderr, "rv_isel: SELECT skip too big\n");
        return BC_ERR_TDF;
    }
    if (rv_buf_patch(out, (uint32_t)beq_slot,
                     rv_beq(RV_T2, RV_ZERO, (int16_t)skip_bytes)) != 0)
        return BC_ERR_TDF;
    return stres(RV_T0, inst_idx, out);
}

/* ---- Function calls ----
 *
 * BIR_CALL: ops[0] is the callee's index into M->funcs[]; ops[1..]
 * are the argument values. We load the args into a0..a7 per the
 * RISC-V psABI, emit a JAL with the destination still up in the
 * air, and record a patch that gets resolved once every function's
 * starting position is known. The return value (if any) lives in
 * a0 by convention and gets spilled to this instruction's stack
 * slot.
 *
 * Caller-saved register hygiene: every BIR value is already on
 * the stack in our model, so there's nothing live in t0/t1/t2 or
 * a0..a7 across the call point. The function we are calling will
 * overwrite a0..a7, and that's exactly what we want it to do.
 */
static int sel_call(const bir_module_t *M, uint32_t inst_idx,
                    const bir_inst_t *I, rv_buf_t *out)
{
    if (I->num_operands < 1u) {
        fprintf(stderr, "rv_isel: CALL with no callee operand\n");
        return BC_ERR_TDF;
    }
    uint16_t callee = (uint16_t)I->operands[0];
    if (callee >= ISEL_MAX_FUNCS) {
        fprintf(stderr,
                "rv_isel: CALL to func %u beyond ISEL_MAX_FUNCS=%u\n",
                callee, ISEL_MAX_FUNCS);
        return BC_ERR_TDF;
    }
    if (callee == curfn) {
        /* Direct recursion would clobber the saved ra in our
         * one-slot prologue. A real recursion story needs a
         * deeper save area; until then, refusing is correct. */
        fprintf(stderr,
                "rv_isel: direct recursion (func %u calling itself) "
                "not supported by the bring-up isel\n", callee);
        return BC_ERR_TDF;
    }
    uint32_t nargs = (uint32_t)I->num_operands - 1u;
    if (nargs > 8u) {
        fprintf(stderr,
                "rv_isel: CALL with %u args (>8 needs stack-passed args, "
                "future work)\n", nargs);
        return BC_ERR_TDF;
    }

    /* Load each arg into the corresponding a-register. RISC-V
     * psABI says first arg in a0, second in a1, and so on. */
    for (uint32_t i = 0; i < nargs; i++) {
        int rc = ldopnd(M, I->operands[i + 1u],
                              (uint8_t)(RV_A0 + i), out);
        if (rc != BC_OK) return rc;
    }
    /* Emit a JAL placeholder and record the call patch. The
     * J-immediate is signed 21-bit (+/- 1 MiB) which covers any
     * realistic function distance within a single ELF. */
    int jal_slot = rv_buf_emit(out, 0u);
    if (jal_slot < 0) return BC_ERR_OVERFLOW;
    if (ncalpat >= ISEL_MAX_CALLS) {
        fprintf(stderr,
                "rv_isel: call patch table full at %u entries\n",
                ISEL_MAX_CALLS);
        return BC_ERR_TDF;
    }
    calpat[ncalpat].jal_word_idx = (uint32_t)jal_slot;
    calpat[ncalpat].callee_func  = callee;
    ncalpat++;

    /* Spill return value (in a0) to this instruction's slot, if
     * the call produces a result. A void call has no result slot
     * to write to, which is fine; the slot for BIR_CALL's inst
     * is simply never read. */
    return stres(RV_A0, inst_idx, out);
}

/* ---- Unreachable ----
 *
 * EBREAK is the right semantic: it stops the baby core at the
 * instruction that BIR_UNREACHABLE marks. Per the BabyRISCV README
 * EBREAK triggers a debug pause rather than a real trap on these
 * cores, which is actually closer to the BIR meaning (the
 * compiler thinks this code is unreachable; if execution gets
 * here, something is wrong and we want to halt for inspection). */
static int sel_unrch(rv_buf_t *out)
{
    return emit(out, rv_ebreak());
}

static uint32_t tybytes(const bir_module_t *M, uint32_t ti);

/* Access width in bytes from the pointer's pointee type. Returns 0 when the
 * type is not a pointer we can size, which is what an untyped pointer into L1
 * looks like, and the caller then assumes a word. Widths we cannot express in
 * one RV32 access come back as-is so the caller can refuse. */
static uint32_t accwid(const bir_module_t *M, uint32_t ptr_val)
{
    if (BIR_VAL_IS_CONST(ptr_val)) return 0u;
    uint32_t idx = BIR_VAL_INDEX(ptr_val);
    if (idx >= M->num_insts) return 0u;
    uint32_t pt = M->insts[idx].type;
    if (pt >= M->num_types || M->types[pt].kind != BIR_TYPE_PTR) return 0u;
    return tybytes(M, M->types[pt].inner);
}

/* Pick the load or store for an access width, or refuse. An i64 or an
 * aggregate would otherwise compile to a 4-byte access and quietly read or
 * write the wrong number of bytes. */
static int winsn(uint32_t w, int is_load, uint32_t *insn)
{
    switch (w) {
    case 0u:  /* unsizable pointee; assume a word, as the isel always has */
    case 4u: *insn = is_load ? rv_lw (RV_T0, RV_T1, 0) : rv_sw(RV_T0, RV_T1, 0); return BC_OK;
    case 2u: *insn = is_load ? rv_lhu(RV_T0, RV_T1, 0) : rv_sh(RV_T0, RV_T1, 0); return BC_OK;
    case 1u: *insn = is_load ? rv_lbu(RV_T0, RV_T1, 0) : rv_sb(RV_T0, RV_T1, 0); return BC_OK;
    default:
        fprintf(stderr,
                "rv_isel: %u-byte %s has no single RV32 access; i64 and "
                "aggregate memory operations are not supported\n",
                w, is_load ? "load" : "store");
        return BC_ERR_TDF;
    }
}

static int sel_load(const bir_module_t *M, uint32_t inst_idx,
                    const bir_inst_t *I, rv_buf_t *out)
{
    /* BIR_LOAD has ops[0] = address. Sub-word loads zero-extend; BIR carries no
     * signedness on the type, so a sign-extending source arrives as an explicit
     * BIR_SEXT, which re-derives the sign from the low bits anyway.
     *
     * Alignment is still the front end's problem. A misaligned address makes
     * the baby core round down silently, per the BabyRISCV README, and we have
     * no type info here to catch it. */
    int rc = ldopnd(M, I->operands[0], RV_T1, out);
    if (rc != BC_OK) return rc;
    uint32_t insn;
    if ((rc = winsn(accwid(M, I->operands[0]), 1, &insn)) != BC_OK) return rc;
    if ((rc = emit(out, insn)) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

static int sel_store(const bir_module_t *M,
                     const bir_inst_t *I, rv_buf_t *out)
{
    /* BIR_STORE ops[0] = value, ops[1] = address (per the memory
     * note in the Booth codebase: store operand order is
     * value, address, NOT address, value). */
    int rc;
    if ((rc = ldopnd(M, I->operands[0], RV_T0, out)) != BC_OK) return rc;
    if ((rc = ldopnd(M, I->operands[1], RV_T1, out)) != BC_OK) return rc;
    /* Narrow stores must not write the adjacent bytes. */
    uint32_t insn;
    if ((rc = winsn(accwid(M, I->operands[1]), 0, &insn)) != BC_OK) return rc;
    return emit(out, insn);
}

/*
 * GEP: base pointer plus an integer index, scaled by the element
 * size of the pointee. Bring-up handles the constant-index case
 * by folding (index * elem_size) into a single ADDI; non-constant
 * indices need a runtime multiply we are not yet emitting.
 */
/*
 * Natural alignment in bytes for a primitive size, capped at the
 * RV32 word size. i8 -> 1, i16 -> 2, i32/f32/ptr -> 4. Used by the
 * struct layout walker so each field starts at its own boundary,
 * matching the standard C ABI convention. Anything wider than 4
 * bytes (we have no i64 yet on baby cores) clamps to word align.
 */
static uint32_t natalg(uint32_t sz)
{
    if (sz >= 4u) return 4u;
    if (sz >= 2u) return 2u;
    return 1u;
}

static uint32_t algnup(uint32_t x, uint32_t a)
{
    return (x + (a - 1u)) & ~(a - 1u);
}

/*
 * Byte size of a BIR type. Primitives are immediate. Aggregates
 * recurse and apply natural alignment between fields, which is
 * the layout C/CUDA front ends produce in the absence of
 * explicit packing or alignment attributes.
 *
 * Returns 0 if any subcomponent is unknown (e.g. a struct that
 * contains a FUNC or an as-yet-unhandled type kind). Callers must
 * check and refuse rather than silently computing a wrong offset.
 */
static uint32_t tybytes(const bir_module_t *M, uint32_t ti)
{
    if (ti >= M->num_types) return 0u;
    const bir_type_t *t = &M->types[ti];
    switch (t->kind) {
    case BIR_TYPE_INT:    return (uint32_t)(t->width / 8u);
    case BIR_TYPE_FLOAT:  return (uint32_t)(t->width / 8u);
    case BIR_TYPE_BFLOAT: return 2u;
    case BIR_TYPE_PTR:    return 4u;
    case BIR_TYPE_ARRAY: {
        uint32_t es = tybytes(M, t->inner);
        if (es == 0u) return 0u;
        return es * t->count;
    }
    case BIR_TYPE_VECTOR: {
        uint32_t es = tybytes(M, t->inner);
        if (es == 0u) return 0u;
        return es * (uint32_t)t->width;     /* width = lane count for VECTOR */
    }
    case BIR_TYPE_STRUCT: {
        uint32_t off = 0u;
        for (uint16_t i = 0; i < t->num_fields; i++) {
            if (t->count + i >= M->num_type_fields) return 0u;
            uint32_t ft = M->type_fields[t->count + i];
            uint32_t fs = tybytes(M, ft);
            if (fs == 0u) return 0u;
            off = algnup(off, natalg(fs)) + fs;
        }
        return off;                         /* no tail padding for now */
    }
    default:
        return 0u;
    }
}

/*
 * Byte offset of field `fi` inside a STRUCT type, applying natural
 * alignment to each preceding field. Mirrors the tybytes layout
 * walker exactly so the struct's overall layout and any per-field
 * GEP agree on where each field sits.
 */
static uint32_t sfldof(const bir_module_t *M,
                                    uint32_t struct_ti,
                                    uint32_t fi)
{
    if (struct_ti >= M->num_types) return 0u;
    const bir_type_t *t = &M->types[struct_ti];
    if (t->kind != BIR_TYPE_STRUCT) return 0u;
    if (fi >= t->num_fields) return 0u;
    uint32_t off = 0u;
    for (uint16_t i = 0; i < (uint16_t)fi; i++) {
        if (t->count + i >= M->num_type_fields) return 0u;
        uint32_t ft = M->type_fields[t->count + i];
        uint32_t fs = tybytes(M, ft);
        if (fs == 0u) return 0u;
        off = algnup(off, natalg(fs)) + fs;
    }
    /* Align the START of the requested field to its own boundary. */
    if (t->count + fi < M->num_type_fields) {
        uint32_t cur = tybytes(M, M->type_fields[t->count + fi]);
        if (cur != 0u) off = algnup(off, natalg(cur));
    }
    return off;
}

/*
 * Trace the value handle of a GEP base operand back to the BIR
 * type of the underlying instruction. Returns 0 if the trail goes
 * cold or the base is a constant (which shouldn't happen for a
 * GEP base, but we handle it defensively rather than dereferencing
 * out of range).
 */
static uint32_t basety(const bir_module_t *M, uint32_t val)
{
    if (BIR_VAL_IS_CONST(val)) return 0u;
    uint32_t idx = BIR_VAL_INDEX(val);
    if (idx >= M->num_insts) return 0u;
    uint32_t ti = M->insts[idx].type;
    if (ti >= M->num_types) return 0u;
    const bir_type_t *t = &M->types[ti];
    if (t->kind != BIR_TYPE_PTR) return 0u;
    return t->inner;
}

/*
 * Materialise a byte offset into T0 (already holding the base
 * pointer). Folds into a single ADDI when the offset fits the
 * 12-bit signed immediate; otherwise builds the offset in T1 via
 * LUI+ADDI and uses an ADD. Used by both the struct-field and the
 * array-stride paths so the encoding choices live in one place.
 */
static int gepoff(int64_t off, rv_buf_t *out)
{
    if (off == 0) return BC_OK;
    if (off >= -2048 && off <= 2047) {
        return emit(out, rv_addi(RV_T0, RV_T0, (int16_t)off));
    }
    if (off < INT32_MIN || off > INT32_MAX) {
        fprintf(stderr,
                "rv_isel: GEP offset %lld too big for i32\n",
                (long long)off);
        return BC_ERR_TDF;
    }
    int rc;
    if ((rc = ldimm32(RV_T1, (int32_t)off, out)) != BC_OK) return rc;
    return emit(out, rv_add(RV_T0, RV_T0, RV_T1));
}

/*
 * GEP lowering comes in two shapes. A struct-field GEP has a pointer-to-
 * struct base and a constant field number for the index, and its offset
 * comes from the struct's layout. An array-stride GEP has any other base
 * (pointer to a primitive, decayed array, vector) and its offset is the
 * index times the element size of the GEP's result pointee.
 *
 * The distinction is made by inspecting the BASE operand's type,
 * not the result type, because the BIR encoding changes the
 * result type to the field pointee in the struct case but the
 * "index" semantics are different (field number vs stride
 * multiplier).
 */
static int sel_gep(const bir_module_t *M, uint32_t inst_idx,
                   const bir_inst_t *I, rv_buf_t *out)
{
    if (I->num_operands < 2u) {
        fprintf(stderr, "rv_isel: GEP needs at least base + 1 index\n");
        return BC_ERR_TDF;
    }
    if (I->type >= M->num_types ||
        M->types[I->type].kind != BIR_TYPE_PTR) {
        fprintf(stderr, "rv_isel: GEP result must be a pointer\n");
        return BC_ERR_TDF;
    }

    int rc = ldopnd(M, I->operands[0], RV_T0, out);
    if (rc != BC_OK) return rc;

    /* Identify the base's pointee type. A struct-field GEP has a
     * STRUCT pointee on the base AND a different pointee on the
     * result (because the GEP extracts a member of differing
     * type). When base and result pointees match, even if both
     * are structs, this is a stride GEP through an array-of-
     * struct, and the index is a byte multiplier rather than a
     * field number. The two are distinguished by the front end's
     * encoding choice and we honour that here. */
    uint32_t base_pointee = basety(M, I->operands[0]);
    uint32_t result_pointee = M->types[I->type].inner;
    int is_struct_gep = (base_pointee != 0u &&
                         base_pointee < M->num_types &&
                         M->types[base_pointee].kind == BIR_TYPE_STRUCT &&
                         result_pointee != base_pointee);

    /* Constant index path. Same shape for struct and stride: we
     * fold the index into a byte offset at compile time and emit
     * a single ADDI or LUI+ADDI sequence. */
    if (BIR_VAL_IS_CONST(I->operands[1])) {
        uint32_t ci = BIR_VAL_INDEX(I->operands[1]);
        if (ci >= M->num_consts) {
            fprintf(stderr, "rv_isel: bad GEP const index %u\n", ci);
            return BC_ERR_TDF;
        }
        const bir_const_t *c = &M->consts[ci];
        int64_t idx_v = (c->kind == BIR_CONST_ZERO) ? 0
                       : (c->kind == BIR_CONST_INT) ? c->d.ival
                       : INT64_MIN;
        if (idx_v == INT64_MIN) {
            fprintf(stderr, "rv_isel: GEP index must be INT/ZERO const\n");
            return BC_ERR_TDF;
        }
        /* Index zero is free regardless of pointee size, so we
         * short-circuit before computing elem_size. This unblocks
         * the common "gep base, 0" idiom that bir_lower emits as
         * the first step of a struct member chain. */
        if (idx_v == 0) return stres(RV_T0, inst_idx, out);

        int64_t off;
        if (is_struct_gep) {
            if (idx_v < 0 || idx_v >= 0x10000) {
                fprintf(stderr,
                        "rv_isel: struct GEP field index %lld out of range\n",
                        (long long)idx_v);
                return BC_ERR_TDF;
            }
            off = (int64_t)sfldof(M, base_pointee,
                                               (uint32_t)idx_v);
        } else {
            uint32_t elem_sz = tybytes(M, M->types[I->type].inner);
            if (elem_sz == 0u) {
                fprintf(stderr, "rv_isel: unknown GEP pointee size\n");
                return BC_ERR_TDF;
            }
            off = idx_v * (int64_t)elem_sz;
        }
        if ((rc = gepoff(off, out)) != BC_OK) return rc;
        return stres(RV_T0, inst_idx, out);
    }

    /* Non-constant index path: stride lowering only. Field-index
     * GEPs in BIR are always constant because they came from
     * source-level "s.field" syntax that names a specific field;
     * a non-constant struct-index is not a thing C/CUDA can
     * produce, so refusing it loudly is correct. */
    if (is_struct_gep) {
        fprintf(stderr,
                "rv_isel: non-constant struct GEP index is illegal\n");
        return BC_ERR_TDF;
    }
    uint32_t elem_sz = tybytes(M, M->types[I->type].inner);
    if (elem_sz == 0u) {
        fprintf(stderr, "rv_isel: unknown GEP pointee size\n");
        return BC_ERR_TDF;
    }
    if ((rc = ldopnd(M, I->operands[1], RV_T1, out)) != BC_OK) return rc;
    if (elem_sz != 1u) {
        if ((rc = ldimm32(RV_T2, (int32_t)elem_sz, out)) != BC_OK) return rc;
        if ((rc = emit(out, rv_mul(RV_T1, RV_T1, RV_T2))) != BC_OK) return rc;
    }
    if ((rc = emit(out, rv_add(RV_T0, RV_T0, RV_T1))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/*
 * Materialise TD_L1_RTARG_BASE into the given register via a single
 * LUI. The base is chosen with a zero low-12-bit half (0x00008000)
 * specifically so this is one instruction; if it ever moves we
 * need either a LUI+ADDI pair or a different free register slot.
 */
static int ldrtarg(uint8_t reg, rv_buf_t *out)
{
    return emit(out, rv_lui(reg, TD_L1_RTARG_BASE >> 12));
}

/*
 * Kernel parameter: load from the L1 runtime-args block at
 * RT_ARG_OFF_KARG(pi). On Tensix there is no caller invoking the
 * kernel with values in a0..a7; the host launcher writes the args
 * struct into L1 before dispatch and the kernel reads it from
 * there. RT_ARG_N_KERNEL_SLOTS bounds how many params we can
 * carry per kernel, currently 16.
 */
static int sel_param(uint32_t inst_idx, const bir_inst_t *I, rv_buf_t *out)
{
    uint8_t pi = I->subop;
    if (pi >= RT_ARG_N_KERNEL_SLOTS) {
        fprintf(stderr,
                "rv_isel: parameter %u beyond %u runtime-arg slots\n",
                pi, RT_ARG_N_KERNEL_SLOTS);
        return BC_ERR_TDF;
    }
    int rc;
    if ((rc = ldrtarg(RV_T1, out)) != BC_OK) return rc;
    if ((rc = emit(out, rv_lw(RV_T0, RV_T1,
            (int16_t)RT_ARG_OFF_KARG(pi)))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}

/*
 * CUDA coordinate intrinsics: threadIdx, blockIdx, blockDim,
 * gridDim. Each instruction has subop = dimension (0=x, 1=y, 2=z)
 * and the byte offset into the runtime args struct is the base for
 * that intrinsic plus subop*4. The launcher writes the appropriate
 * value into each slot before dispatch; single-thread-per-core
 * model means threadIdx is always (0,0,0) and the launcher records
 * exactly that.
 */
static int sel_intrin(uint32_t inst_idx, const bir_inst_t *I, rv_buf_t *out)
{
    uint8_t dim = I->subop;
    if (dim > 2u) {
        fprintf(stderr,
                "rv_isel: intrinsic dim %u must be 0..2\n", dim);
        return BC_ERR_TDF;
    }
    uint32_t off;
    switch (I->op) {
    case BIR_THREAD_ID: off = RT_ARG_OFF_TID_X  + (uint32_t)dim * 4u; break;
    case BIR_BLOCK_ID:  off = RT_ARG_OFF_BID_X  + (uint32_t)dim * 4u; break;
    case BIR_BLOCK_DIM: off = RT_ARG_OFF_BDIM_X + (uint32_t)dim * 4u; break;
    case BIR_GRID_DIM:  off = RT_ARG_OFF_GDIM_X + (uint32_t)dim * 4u; break;
    default:
        fprintf(stderr, "rv_isel: unreachable intrinsic op %u\n", I->op);
        return BC_ERR_TDF;
    }
    int rc;
    if ((rc = ldrtarg(RV_T1, out)) != BC_OK) return rc;
    if ((rc = emit(out, rv_lw(RV_T0, RV_T1, (int16_t)off))) != BC_OK) return rc;
    return stres(RV_T0, inst_idx, out);
}


static int sel_ret(const bir_module_t *M, uint32_t frame_sz,
                   const bir_inst_t *I, rv_buf_t *out)
{
    int rc;
    if (I->num_operands > 0u) {
        if ((rc = ldopnd(M, I->operands[0], RV_A0, out)) != BC_OK) return rc;
    } else {
        /* tt-metal calls the kernel entry as uint32_t(*)() and feeds a0 to
         * record_stack_usage, where zero means "not computed". Leaving a0
         * dirty would report a bogus watermark. */
        if ((rc = emit(out, rv_addi(RV_A0, RV_ZERO, 0))) != BC_OK) return rc;
    }
    /* Epilogue: restore ra, deallocate the frame, jalr to ra. */
    if ((rc = emit(out, rv_lw(RV_RA, RV_SP, 0))) != BC_OK) return rc;
    if ((rc = emit(out,
            rv_addi(RV_SP, RV_SP, (int16_t)frame_sz))) != BC_OK) return rc;
    return emit(out, rv_jalr(RV_ZERO, RV_RA, 0));
}

/* ---- Public entry ---- */

int rv_isel_func(const bir_module_t *M, uint32_t func_idx,
                 rv_buf_t *out)
{
    if (func_idx >= M->num_funcs) {
        fprintf(stderr, "rv_isel: bad func idx %u\n", func_idx);
        return BC_ERR_TDF;
    }
    /* Record this function's starting position so any other
     * function in the same module that calls into us can patch
     * the JAL offset later. curfn tracks which function
     * we're emitting so sel_call can detect direct recursion. */
    if (func_idx < ISEL_MAX_FUNCS) {
        fnwidx[func_idx] = rv_buf_n_words(out);
        fnknwn[func_idx] = 1;
    }
    curfn = (uint16_t)func_idx;
    const bir_func_t *F = &M->funcs[func_idx];

    /* Instruction indices are module-global, so bias them by this function's
     * first one to get frame-local slot numbers. */
    slotbas = 0;
    if (F->num_blocks > 0u && F->first_block < M->num_blocks) {
        slotbas = M->blocks[F->first_block].first_inst;
    }

    if (F->total_insts > ISEL_MAX_INSTS) {
        fprintf(stderr,
                "rv_isel: function has %u instructions, ceiling %u\n",
                F->total_insts, ISEL_MAX_INSTS);
        return BC_ERR_TDF;
    }

    /* Pre-walk: every BIR_ALLOCA needs a stack region for the
     * pointee. We compute their cumulative offset before the main
     * emit pass so the frame size includes both the per-inst slots
     * and the alloca region, and the main pass can hand each
     * BIR_ALLOCA its absolute frame offset. */
    static uint32_t alcoff[ISEL_MAX_INSTS];
    uint32_t alctot = 0;
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bk = F->first_block + bi;
        if (bk >= M->num_blocks) break;
        const bir_block_t *B = &M->blocks[bk];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            if (I->op != BIR_ALLOCA) continue;
            uint32_t lidx = idx - slotbas;
            if (lidx >= ISEL_MAX_INSTS) {
                fprintf(stderr,
                        "rv_isel: alloca at idx %u past table\n", idx);
                return BC_ERR_TDF;
            }
            /* Pointee type tells us how many bytes to reserve. */
            uint32_t pointee_sz = 0;
            if (I->type < M->num_types &&
                M->types[I->type].kind == BIR_TYPE_PTR) {
                pointee_sz = tybytes(M, M->types[I->type].inner);
            }
            if (pointee_sz == 0u) pointee_sz = 4u;  /* default i32-shaped */
            alctot = rndup(alctot, ISEL_ALLOCA_ALIGN);
            alcoff[lidx] = alctot;
            alctot += pointee_sz;
        }
    }

    /* Same walk for __shared__, but the region is a fixed slab at the top of
     * L1 rather than stack, and it is laid out once per kernel because a baby
     * core runs a single block. */
    static uint32_t shoff[ISEL_MAX_INSTS];
    uint32_t shtot = 0;
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bk = F->first_block + bi;
        if (bk >= M->num_blocks) break;
        const bir_block_t *B = &M->blocks[bk];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            if (I->op != BIR_SHARED_ALLOC) continue;
            uint32_t lidx = idx - slotbas;
            if (lidx >= ISEL_MAX_INSTS) {
                fprintf(stderr,
                        "rv_isel: shared_alloc at idx %u past table\n", idx);
                return BC_ERR_TDF;
            }
            uint32_t sz = 0;
            if (I->type < M->num_types &&
                M->types[I->type].kind == BIR_TYPE_PTR) {
                sz = tybytes(M, M->types[I->type].inner);
            }
            if (sz == 0u) {
                fprintf(stderr,
                        "rv_isel: shared_alloc at idx %u has unsizable "
                        "pointee type\n", idx);
                return BC_ERR_TDF;
            }
            shtot = rndup(shtot, TD_L1_ALIGN);
            shoff[lidx] = shtot;
            shtot += sz;
            if (shtot > TD_L1_SHARED_SIZE) {
                fprintf(stderr,
                        "rv_isel: __shared__ needs %u bytes, slab is %u\n",
                        shtot, TD_L1_SHARED_SIZE);
                return BC_ERR_TDF;
            }
        }
    }

    /* Frame size: header (ra + pad) + 4 bytes per BIR instruction
     * slot + the alloca region, rounded up to the 16-byte alignment
     * the RISC-V soft-float psABI requires for the stack pointer
     * at function entry. */
    uint32_t frame_sz = rndup(
        ISEL_LOCALS_BASE + F->total_insts * 4u + alctot,
        ISEL_FRAME_ALIGN);
    /* Alloca region starts immediately after the per-inst slots. */
    uint32_t alcbase = ISEL_LOCALS_BASE + F->total_insts * 4u;

    /* The prologue and epilogue adjust sp with a single ADDI, so the frame has
     * to fit the 12-bit signed immediate. */
    if (frame_sz > 2047u) {
        fprintf(stderr,
                "rv_isel: frame of %u bytes exceeds the ADDI immediate range\n",
                frame_sz);
        return BC_ERR_TDF;
    }

    /* Prologue: drop sp, save ra. */
    int rc;
    if ((rc = emit(out,
            rv_addi(RV_SP, RV_SP, (int16_t)-(int32_t)frame_sz))) != BC_OK)
        return rc;
    if ((rc = emit(out, rv_sw(RV_RA, RV_SP, 0))) != BC_OK) return rc;

    /* Walk every instruction in every block of this function. We
     * use the raw instruction index inside the module (insts[idx])
     * as the slot index since the BIR module is single-function in
     * the bring-up path; once multi-function lands this will need
     * a per-function renumbering. */
    isel_rstcf();
    rc = phicoll(M, F);
    if (rc != BC_OK) return rc;
    int saw_ret = 0;
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t bk = F->first_block + bi;
        if (bk >= M->num_blocks) break;
        /* Record where this block's first instruction lands in the
         * code buffer so subsequent branches to it can resolve. */
        if (bk < ISEL_MAX_BLOCKS) {
            blkwidx[bk] = rv_buf_n_words(out);
            blkknwn[bk] = 1;
        }
        const bir_block_t *B = &M->blocks[bk];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            uint32_t idx = B->first_inst + ii;
            const bir_inst_t *I = &M->insts[idx];
            switch (I->op) {
            case BIR_PARAM:
                rc = sel_param(idx, I, out);
                break;
            case BIR_THREAD_ID:
            case BIR_BLOCK_ID:
            case BIR_BLOCK_DIM:
            case BIR_GRID_DIM:
                rc = sel_intrin(idx, I, out);
                break;
            case BIR_LOAD:
                rc = sel_load(M, idx, I, out);
                break;
            case BIR_STORE:
                rc = sel_store(M, I, out);
                break;
            case BIR_ADD:
            case BIR_SUB:
            case BIR_MUL:
            case BIR_UMULHI:
            case BIR_SDIV:
            case BIR_UDIV:
            case BIR_SREM:
            case BIR_UREM:
            case BIR_AND:
            case BIR_OR:
            case BIR_XOR:
            case BIR_SHL:
            case BIR_LSHR:
            case BIR_ASHR:
                rc = sel_binop(M, idx, I, out);
                break;
            case BIR_ICMP:
                rc = sel_icmp(M, idx, I, out);
                break;
            case BIR_PTRTOINT:
            case BIR_INTTOPTR:
            case BIR_BITCAST:
                rc = sel_cast_id(M, idx, I, out);
                break;
            case BIR_ZEXT:
                rc = sel_zext(M, idx, I, out);
                break;
            case BIR_SEXT:
                rc = sel_sext(M, idx, I, out);
                break;
            case BIR_TRUNC:
                rc = sel_trunc(M, idx, I, out);
                break;
            case BIR_UNREACHABLE:
                rc = sel_unrch(out);
                break;
            case BIR_ALLOCA: {
                /* Compute sp + alcbase + alcoff[idx], store
                 * the resulting pointer into this inst's slot. */
                uint32_t total = alcbase + alcoff[idx - slotbas];
                if (total > 2047u) {
                    fprintf(stderr,
                            "rv_isel: alloca offset %u out of ADDI range\n",
                            total);
                    rc = BC_ERR_TDF;
                    break;
                }
                rc = emit(out, rv_addi(RV_T0, RV_SP, (int16_t)total));
                if (rc == BC_OK) rc = stres(RV_T0, idx, out);
                break;
            }
            case BIR_SHARED_ALLOC:
                /* Absolute L1 address, materialised inline. No relocation is
                 * involved: this is a data address, and XIP only rewrites
                 * text-relative references. */
                rc = ldimm32(RV_T0,
                             (int32_t)(TD_L1_SHARED_BASE + shoff[idx - slotbas]),
                             out);
                if (rc == BC_OK) rc = stres(RV_T0, idx, out);
                break;
            case BIR_BR:
                rc = sel_br(M, (uint16_t)bk, I, out);
                break;
            case BIR_BR_COND:
                rc = sel_br_cond(M, (uint16_t)bk, I, out);
                break;
            case BIR_SWITCH:
                rc = sel_switch(M, (uint16_t)bk, I, out);
                break;
            case BIR_GLOBAL_REF:
                /* Computing the address is easy; the problem is that nothing
                 * puts a value there. The ELF has one PT_LOAD and no .data,
                 * and the kernel is a bare function that never runs a crt1 to
                 * stage an image, so the pointer would reference uninitialised
                 * L1. Even a zero-init global would be wrong. */
                fprintf(stderr,
                        "rv_isel: __device__ and __constant__ globals need a "
                        "data segment and an initialisation path, neither of "
                        "which this backend emits yet\n");
                rc = BC_ERR_TDF;
                break;
            case BIR_BARRIER:
            case BIR_BARRIER_GROUP:
                /* A baby core is one hardware thread with no SIMT, so a
                 * block-level barrier has nothing to wait on. FENCE is a
                 * documented no-op here and would not order anything, so
                 * emitting one would only mislead. Cross-core coordination
                 * goes through CB semaphores, which is a different opcode. */
                rc = BC_OK;
                break;
            case BIR_PHI:
                /* PHI emits no code in its own block; its slot is
                 * populated by predecessor blocks before they
                 * branch here. See phicoll / emit_phi_copies. */
                rc = BC_OK;
                break;
            case BIR_CALL:
                rc = sel_call(M, idx, I, out);
                break;
            case BIR_SELECT:
                rc = sel_select(M, idx, I, out);
                break;
            case BIR_GEP:
                rc = sel_gep(M, idx, I, out);
                break;
            case BIR_RET:
                rc = sel_ret(M, frame_sz, I, out);
                saw_ret = 1;
                break;
            default:
                fprintf(stderr,
                        "rv_isel: BIR op %u not yet supported "
                        "(bring-up isel)\n", I->op);
                return BC_ERR_TDF;
            }
            if (rc != BC_OK) return rc;
        }
    }

    /* If the function falls off the end without an explicit return,
     * emit a void epilogue so the code still finishes cleanly. */
    if (!saw_ret) {
        bir_inst_t fake;
        memset(&fake, 0, sizeof(fake));
        fake.op = BIR_RET;
        fake.num_operands = 0;
        rc = sel_ret(M, frame_sz, &fake, out);
        if (rc != BC_OK) return rc;
    }

    /* Resolve any branches we deferred when we hit a forward
     * reference. By now every block we might branch to has been
     * recorded in blkwidx, so each patch turns into a single
     * offset computation and a buffer rewrite. */
    for (uint32_t p = 0; p < npatch; p++) {
        uint16_t tb = patches[p].target_block;
        if (tb >= ISEL_MAX_BLOCKS || !blkknwn[tb]) {
            fprintf(stderr,
                    "rv_isel: unresolved branch to block %u\n", tb);
            return BC_ERR_TDF;
        }
        int32_t off = rv_buf_offset(out, patches[p].word_idx,
                                    blkwidx[tb]);
        uint32_t word;
        if (patches[p].kind == 2u) {
            if (off < -1048576 || off > 1048574) {
                fprintf(stderr,
                        "rv_isel: JAL patch offset %d out of range\n", off);
                return BC_ERR_TDF;
            }
            word = rv_jal(RV_ZERO, off);
        } else {
            if (off < -4096 || off > 4094) {
                fprintf(stderr,
                        "rv_isel: branch patch offset %d out of range\n", off);
                return BC_ERR_TDF;
            }
            word = (patches[p].kind == 0u)
                 ? rv_beq(patches[p].rs1, RV_ZERO, (int16_t)off)
                 : rv_bne(patches[p].rs1, RV_ZERO, (int16_t)off);
        }
        if (rv_buf_patch(out, patches[p].word_idx, word) != 0)
            return BC_ERR_TDF;
    }
    return BC_OK;
}

/*
 * Module-level entry: emit every function in M back-to-back into
 * the buffer, then resolve inter-function JAL patches by looking
 * up each callee's recorded starting position. Function 0 is the
 * entry point and is emitted first so the host loader's PC value
 * lands on it; the rest are reachable via JAL.
 *
 * The call-patch resolution mirrors the branch-patch path in
 * rv_isel_func but operates across function boundaries. JAL's
 * 21-bit signed J-immediate gives us +/- 1 MiB of reach, which
 * is comfortably more than the 16 KiB code budget any single
 * baby-core kernel actually has.
 */
int rv_isel_module(const bir_module_t *M, rv_buf_t *out)
{
    if (!M || !out) {
        fprintf(stderr, "rv_isel: NULL module or output buffer\n");
        return BC_ERR_TDF;
    }
    if (M->num_funcs > ISEL_MAX_FUNCS) {
        fprintf(stderr,
                "rv_isel: module has %u functions, ceiling %u\n",
                M->num_funcs, ISEL_MAX_FUNCS);
        return BC_ERR_TDF;
    }

    /* tt-metal enters a kernel by calling the first byte of .text, so the
     * __global__ function has to be emitted first. Source order puts any
     * __device__ helpers ahead of it, and emitting those first would run a
     * helper as the kernel. */
    uint32_t kfn = M->num_funcs;
    uint32_t nglob = 0;
    for (uint32_t fi = 0; fi < M->num_funcs; fi++) {
        if (!(M->funcs[fi].cuda_flags & CUDA_GLOBAL)) continue;
        if (nglob == 0u) kfn = fi;
        nglob++;
    }
    if (nglob == 0u) {
        fprintf(stderr, "rv_isel: module has no __global__ kernel to enter\n");
        return BC_ERR_TDF;
    }
    if (nglob > 1u) {
        fprintf(stderr,
                "rv_isel: module has %u __global__ kernels; entering the "
                "first one. Split them into separate ELFs to pick another\n",
                nglob);
    }

    isel_rstmod();

    int rc = rv_isel_func(M, kfn, out);
    if (rc != BC_OK) return rc;
    for (uint32_t fi = 0; fi < M->num_funcs; fi++) {
        if (fi == kfn) continue;
        rc = rv_isel_func(M, fi, out);
        if (rc != BC_OK) return rc;
    }

    /* Resolve every call placeholder. By now every callee's
     * starting position is in fnwidx, so each patch is a
     * single offset computation and a buffer rewrite. */
    for (uint32_t p = 0; p < ncalpat; p++) {
        uint16_t cf = calpat[p].callee_func;
        if (cf >= ISEL_MAX_FUNCS || !fnknwn[cf]) {
            fprintf(stderr,
                    "rv_isel: unresolved call to function %u\n", cf);
            return BC_ERR_TDF;
        }
        int32_t off = rv_buf_offset(out, calpat[p].jal_word_idx,
                                    fnwidx[cf]);
        if (off < -1048576 || off > 1048574) {
            fprintf(stderr,
                    "rv_isel: call offset %d busts JAL range\n", off);
            return BC_ERR_TDF;
        }
        /* RA = next instruction's address, so JAL must use ra
         * (x1) as the link register, not zero like the
         * branch-style jumps. */
        uint32_t word = rv_jal(RV_RA, off);
        if (rv_buf_patch(out, calpat[p].jal_word_idx, word) != 0)
            return BC_ERR_TDF;
    }
    return BC_OK;
}
