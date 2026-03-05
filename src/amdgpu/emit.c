#include "amdgpu.h"
#include "encode.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/*
 * AMDGPU emitter: phi elimination, register allocation, assembly printer,
 * and ELF code object writer.
 * Targets CDNA 2/3 (gfx90a/gfx942, Wave64), RDNA 2/3/4 (gfx1030/1100/1200, Wave32).
 * Dependencies: libc, optimism, tea.
 */

/* ---- Phi Elimination ---- */

/*
 * Proper phi elimination: insert copies at predecessor block ends.
 * For each PSEUDO_PHI with operands [(pred, src), ...], insert
 * PSEUDO_COPY dst, srcN at the end of each predecessor (before its
 * terminator). Then NOP the PHI. Cycles are mercifully rare from our
 * frontend, so we don't break them — yet.
 */

#define PHI_MAX_COPIES 4096

typedef struct {
    uint32_t   pred_mb;
    moperand_t dst;
    moperand_t src;
} phi_copy_t;

static phi_copy_t phi_copies[PHI_MAX_COPIES];

/* Is this instruction a block terminator? */
static int is_terminator(uint16_t op)
{
    return op == AMD_S_BRANCH || op == AMD_S_CBRANCH_SCC0 ||
           op == AMD_S_CBRANCH_SCC1 || op == AMD_S_CBRANCH_EXECZ ||
           op == AMD_S_CBRANCH_EXECNZ || op == AMD_S_ENDPGM ||
           op == AMD_S_SETPC_B64;
}

void amdgpu_phi_elim(amd_module_t *A)
{
    uint32_t nc = 0;

    /* Phase 1: collect copies from PHIs, NOP the PHIs */
    for (uint32_t i = 0; i < A->num_minsts; i++) {
        minst_t *mi = &A->minsts[i];
        if (mi->op != AMD_PSEUDO_PHI) continue;

        moperand_t dst = mi->operands[0];
        for (uint8_t p = 0; p + 1 < mi->num_uses && nc < PHI_MAX_COPIES; p += 2) {
            uint32_t off = mi->num_defs + p;
            if (off + 1 >= MINST_MAX_OPS) break;
            if (mi->operands[off].kind != MOP_LABEL) continue;

            phi_copies[nc].pred_mb = (uint32_t)mi->operands[off].imm;
            phi_copies[nc].dst = dst;
            phi_copies[nc].src = mi->operands[off + 1];
            nc++;
        }
        mi->op = AMD_S_NOP;
        mi->num_defs = 0;
        mi->num_uses = 0;
    }

    if (nc == 0) return;

    /* Phase 2: count copies per predecessor block */
    static uint32_t cpb[AMD_MAX_MBLOCKS]; /* copies per block */
    memset(cpb, 0, A->num_mblocks * sizeof(uint32_t));
    for (uint32_t i = 0; i < nc; i++) {
        if (phi_copies[i].pred_mb < A->num_mblocks)
            cpb[phi_copies[i].pred_mb]++;
    }

    /* Phase 3: insert copies before terminators, processing blocks in
       reverse order so shifts don't affect already-processed blocks. */
    for (uint32_t mb = A->num_mblocks; mb > 0; mb--) {
        uint32_t b = mb - 1;
        uint32_t copies_here = cpb[b];
        if (copies_here == 0) continue;
        if (A->num_minsts + copies_here > AMD_MAX_MINSTS) continue;

        mblock_t *B = &A->mblocks[b];

        /* Find insertion point: before trailing terminators */
        uint32_t insert_rel = B->num_insts;
        for (uint32_t ii = B->num_insts; ii > 0; ii--) {
            if (is_terminator(A->minsts[B->first_inst + ii - 1].op))
                insert_rel = ii - 1;
            else
                break;
        }
        uint32_t insert_abs = B->first_inst + insert_rel;

        /* Shift tail of instruction array to make room */
        uint32_t tail_len = A->num_minsts - insert_abs;
        memmove(&A->minsts[insert_abs + copies_here],
                &A->minsts[insert_abs],
                tail_len * sizeof(minst_t));

        /* Insert copies */
        uint32_t ci = 0;
        for (uint32_t i = 0; i < nc && ci < copies_here; i++) {
            if (phi_copies[i].pred_mb != b) continue;
            minst_t *copy = &A->minsts[insert_abs + ci];
            memset(copy, 0, sizeof(minst_t));
            copy->op = AMD_PSEUDO_COPY;
            copy->num_defs = 1;
            copy->num_uses = 1;
            copy->operands[0] = phi_copies[i].dst;
            copy->operands[1] = phi_copies[i].src;
            ci++;
        }

        A->num_minsts += copies_here;
        B->num_insts += copies_here;

        /* Update first_inst for all subsequent blocks */
        for (uint32_t later = b + 1; later < A->num_mblocks; later++)
            A->mblocks[later].first_inst += copies_here;
    }
}

/* ---- Register Allocation (Linear Scan) ---- */

/* Live interval for a virtual register */
typedef struct {
    uint32_t vreg;
    uint32_t start;    /* first def */
    uint32_t end;      /* last use */
    uint16_t phys;     /* allocated physical reg */
    uint8_t  file;     /* 0=SGPR, 1=VGPR */
    uint8_t  spilled;
} live_interval_t;

/* Static storage for regalloc (~4 MB) */
static struct {
    live_interval_t intervals[AMD_MAX_VREGS];
    uint32_t        num_intervals;

    /* Sort index */
    uint32_t        sorted[AMD_MAX_VREGS];

    /* Free register pools */
    uint8_t         sgpr_free[AMD_MAX_SGPRS];
    uint8_t         vgpr_free[AMD_MAX_VGPRS];
    uint32_t        num_sgpr_free;
    uint32_t        num_vgpr_free;

    /* Active intervals sorted by end point */
    uint32_t        active[AMD_MAX_VREGS];
    uint32_t        num_active;

    /* Track max used */
    uint16_t        max_sgpr;
    uint16_t        max_vgpr;
} RA;

static int interval_cmp_start(const void *a, const void *b)
{
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (RA.intervals[ia].start != RA.intervals[ib].start)
        return (RA.intervals[ia].start < RA.intervals[ib].start) ? -1 : 1;
    return 0;
}

/* Get the vreg referenced by an operand, or 0xFFFF if not a vreg */
static uint16_t operand_vreg(const moperand_t *op)
{
    if (op->kind == MOP_VREG_S || op->kind == MOP_VREG_V)
        return op->reg_num;
    return 0xFFFF;
}

static void compute_live_intervals(const amd_module_t *A, const mfunc_t *F)
{
    RA.num_intervals = 0;

    /* Initialize: one interval per vreg, with start=MAX, end=0 */
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        RA.intervals[v].vreg = v;
        RA.intervals[v].start = 0xFFFFFFFF;
        RA.intervals[v].end = 0;
        RA.intervals[v].phys = 0xFFFF;
        RA.intervals[v].file = A->reg_file[v];
        RA.intervals[v].spilled = 0;
    }

    /* Walk all instructions in the function */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            const minst_t *mi = &A->minsts[mi_idx];

            /* Defs */
            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t vr = operand_vreg(&mi->operands[d]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }

            /* Uses */
            for (uint8_t u = mi->num_defs; u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t vr = operand_vreg(&mi->operands[u]);
                if (vr != 0xFFFF) {
                    if (mi_idx < RA.intervals[vr].start)
                        RA.intervals[vr].start = mi_idx;
                    if (mi_idx > RA.intervals[vr].end)
                        RA.intervals[vr].end = mi_idx;
                }
            }
        }
    }

    /* Collect valid intervals */
    RA.num_intervals = 0;
    for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
        if (RA.intervals[v].start != 0xFFFFFFFF) {
            RA.sorted[RA.num_intervals++] = v;
        }
    }

    /* Sort by start point */
    qsort(RA.sorted, RA.num_intervals, sizeof(uint32_t), interval_cmp_start);
}

static void expire_old(uint32_t point)
{
    /* Remove intervals that have ended before this point */
    uint32_t j = 0;
    for (uint32_t i = 0; i < RA.num_active; i++) {
        uint32_t v = RA.active[i];
        if (RA.intervals[v].end >= point) {
            RA.active[j++] = v;
        } else {
            /* Free the register */
            uint16_t phys = RA.intervals[v].phys;
            if (RA.intervals[v].file == 0 && phys < AMD_MAX_SGPRS) {
                RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)phys;
            } else if (phys < AMD_MAX_VGPRS) {
                RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)phys;
            }
        }
    }
    RA.num_active = j;
}

static void regalloc_function(amd_module_t *A, uint32_t mf_idx)  /* called from amdgpu_regalloc */
{
    mfunc_t *F = &A->mfuncs[mf_idx];

    /* Initialize free pools */
    RA.num_sgpr_free = 0;
    RA.num_vgpr_free = 0;
    RA.max_sgpr = 0;
    RA.max_vgpr = 0;
    RA.num_active = 0;

    /* Push high regs first so low regs are popped first (stack order) */
    uint16_t sgpr_start = F->is_kernel ? F->first_alloc_sgpr : 0;
    if (sgpr_start < AMD_KERN_MIN_RESERVED && F->is_kernel)
        sgpr_start = AMD_KERN_MIN_RESERVED;
    for (uint16_t r = AMD_MAX_SGPRS; r-- > sgpr_start; )
        RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)r;
    for (uint16_t r = AMD_MAX_VGPRS; r-- > 0; )
        RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)r;

    compute_live_intervals(A, F);

    /* Linear scan */
    for (uint32_t i = 0; i < RA.num_intervals; i++) {
        uint32_t v = RA.sorted[i];
        live_interval_t *iv = &RA.intervals[v];

        expire_old(iv->start);

        uint16_t phys = 0xFFFF;
        if (iv->file == 0) {
            /* SGPR */
            if (RA.num_sgpr_free > 0) {
                phys = RA.sgpr_free[--RA.num_sgpr_free];
                if (phys >= RA.max_sgpr) RA.max_sgpr = phys + 1;
            }
        } else {
            /* VGPR */
            if (RA.num_vgpr_free > 0) {
                phys = RA.vgpr_free[--RA.num_vgpr_free];
                if (phys >= RA.max_vgpr) RA.max_vgpr = phys + 1;
            }
        }

        if (phys == 0xFFFF) {
            /* Spill: find the active interval with the farthest end */
            uint32_t farthest = 0, farthest_idx = 0;
            for (uint32_t a = 0; a < RA.num_active; a++) {
                uint32_t av = RA.active[a];
                if (RA.intervals[av].file == iv->file &&
                    RA.intervals[av].end > farthest) {
                    farthest = RA.intervals[av].end;
                    farthest_idx = a;
                }
            }
            if (farthest > iv->end && RA.num_active > 0) {
                /* Spill the farthest, give its reg to us */
                uint32_t sv = RA.active[farthest_idx];
                phys = RA.intervals[sv].phys;
                RA.intervals[sv].spilled = 1;
                RA.intervals[sv].phys = 0xFFFF;
                /* Remove from active */
                RA.active[farthest_idx] = RA.active[--RA.num_active];
            } else {
                /* Spill ourselves */
                iv->spilled = 1;
                phys = 0; /* fallback */
            }
        }

        iv->phys = phys;
        A->reg_map[v] = phys;

        /* Add to active */
        if (RA.num_active < AMD_MAX_VREGS)
            RA.active[RA.num_active++] = v;
    }

    /* Record usage for kernel descriptor.
     * Regalloc only tracks its own assigned SGPRs, but kernels also
     * use system SGPRs (kernarg, TGID) and param pair SGPRs.
     * first_alloc_sgpr is the floor — everything below it is spoken for. */
    F->num_sgprs = RA.max_sgpr;
    if (F->is_kernel && F->num_sgprs < F->first_alloc_sgpr)
        F->num_sgprs = F->first_alloc_sgpr;
    F->num_vgprs = RA.max_vgpr;

    /* Minimum 1 SGPR/VGPR for the descriptor */
    if (F->num_sgprs == 0) F->num_sgprs = 1;
    if (F->num_vgprs == 0) F->num_vgprs = 1;

    /* __launch_bounds__ VGPR cap. More threads = fewer registers.
       The maths of sharing: 256 VGPRs divided among the waves you
       promised the hardware you'd run. Break the promise at your peril. */
    if (F->launch_bounds_max > 0 && F->launch_bounds_max < 1024) {
        int w64 = (A->target <= AMD_TARGET_GFX942);
        uint32_t wsz = w64 ? 64u : 32u;
        uint32_t desired_waves = (F->launch_bounds_max + wsz - 1) / wsz;
        if (desired_waves > 0) {
            uint32_t gran = w64 ? ~3u : ~7u;
            uint32_t vgpr_cap = (256 / desired_waves) & gran;
            if (vgpr_cap < (w64 ? 4u : 8u)) vgpr_cap = w64 ? 4u : 8u;
            if (F->num_vgprs > vgpr_cap)
                F->num_vgprs = (uint16_t)vgpr_cap;
        }
    }

    /* Rewrite virtual reg operands to physical */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = 0; k < total; k++) {
                moperand_t *op = &mi->operands[k];
                if (op->kind == MOP_VREG_S) {
                    op->kind = MOP_SGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                } else if (op->kind == MOP_VREG_V) {
                    op->kind = MOP_VGPR;
                    op->reg_num = A->reg_map[op->reg_num];
                }
            }

            /* Convert PSEUDO_COPY to actual MOV */
            if (mi->op == AMD_PSEUDO_COPY) {
                if (mi->operands[0].kind == MOP_VGPR)
                    mi->op = AMD_V_MOV_B32;
                else
                    mi->op = AMD_S_MOV_B32;
            }
        }
    }

    /* Dead copy elimination: kill MOVs where src == dst.
       These appear when regalloc assigns the same phys reg to both sides
       of a copy. Harmless but noisy — like a postman delivering a letter
       back to the sender. */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            if ((mi->op == AMD_V_MOV_B32 || mi->op == AMD_S_MOV_B32) &&
                mi->num_defs == 1 && mi->num_uses == 1 &&
                mi->operands[0].kind == mi->operands[1].kind &&
                mi->operands[0].reg_num == mi->operands[1].reg_num) {
                /* Convert to NOP — the emitter already handles these */
                mi->op = AMD_PSEUDO_DEF;
                mi->num_defs = 0;
                mi->num_uses = 0;
            }
        }
    }
}

/* ---- Assembly Text Printer ---- */

static void asm_append(amd_module_t *A, const char *fmt, ...)
{
    if (A->asm_len >= AMD_ASM_SIZE - 256) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(A->asm_buf + A->asm_len, AMD_ASM_SIZE - A->asm_len, fmt, ap);
    va_end(ap);
    if (n > 0) A->asm_len += (uint32_t)n;
}

static void print_operand(amd_module_t *A, const moperand_t *op)
{
    switch (op->kind) {
    case MOP_SGPR:
        asm_append(A, "s%u", op->reg_num);
        break;
    case MOP_VGPR:
        asm_append(A, "v%u", op->reg_num);
        break;
    case MOP_VREG_S:
        asm_append(A, "%%vs%u", op->reg_num);
        break;
    case MOP_VREG_V:
        asm_append(A, "%%vv%u", op->reg_num);
        break;
    case MOP_IMM:
        /* Print hex for large values, decimal for small */
        if (op->imm >= -16 && op->imm <= 64)
            asm_append(A, "%d", op->imm);
        else
            asm_append(A, "0x%x", (uint32_t)op->imm);
        break;
    case MOP_LABEL:
        asm_append(A, ".LBB%u", (uint32_t)op->imm);
        break;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:
            asm_append(A, A->target <= AMD_TARGET_GFX942 ? "vcc" : "vcc_lo");
            break;
        case AMD_SPEC_EXEC:
            asm_append(A, A->target <= AMD_TARGET_GFX942 ? "exec" : "exec_lo");
            break;
        case AMD_SPEC_SCC:  asm_append(A, "scc"); break;
        case AMD_SPEC_M0:   asm_append(A, "m0"); break;
        default:            asm_append(A, "???"); break;
        }
        break;
    default:
        break;
    }
}

static void print_sgpr_pair(amd_module_t *A, uint16_t base)
{
    asm_append(A, "s[%u:%u]", base, base + 1);
}

static void print_minst(amd_module_t *A, const minst_t *mi)
{
    if (mi->op >= AMD_OP_COUNT) return;
    const amd_enc_entry_t *tbl = get_enc_table(A);
    const amd_enc_entry_t *enc = &tbl[mi->op];
    if (enc->mnemonic == NULL) return;

    /* Skip pseudo-instructions that survived */
    if (enc->fmt == AMD_FMT_PSEUDO) return;

    asm_append(A, "    %s", enc->mnemonic);

    /* Format-specific operand printing */
    uint8_t total = mi->num_defs + mi->num_uses;
    if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

    switch (enc->fmt) {
    case AMD_FMT_SMEM: {
        /* s_load_dword[x2/x4] sDst, sBase, offset */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            if (mi->op == AMD_S_LOAD_DWORDX2 || mi->op == AMD_S_LOAD_DWORDX4) {
                uint16_t base = mi->operands[0].reg_num;
                uint16_t cnt = (mi->op == AMD_S_LOAD_DWORDX2) ? 2 : 4;
                asm_append(A, "s[%u:%u]", base, base + cnt - 1);
            } else {
                print_operand(A, &mi->operands[0]);
            }
            asm_append(A, ", ");
            /* Base is a pair */
            if (mi->operands[1].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[1].reg_num);
            else
                print_operand(A, &mi->operands[1]);
            if (mi->num_uses > 1) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[2]);
            }
        }
        break;
    }
    case AMD_FMT_SOPP: {
        /* s_branch target / s_waitcnt encoding / s_endpgm / s_barrier */
        if (mi->op == AMD_S_WAITCNT) {
            uint16_t w = mi->flags;
            int vm = (w & AMD_WAIT_VMCNT0) != 0;
            int lgkm = (w & AMD_WAIT_LGKMCNT0) != 0;
            if (vm && lgkm)
                asm_append(A, " vmcnt(0) lgkmcnt(0)");
            else if (vm)
                asm_append(A, " vmcnt(0)");
            else if (lgkm)
                asm_append(A, " lgkmcnt(0)");
            else
                asm_append(A, " 0x%04x", w);
        } else if (mi->op == AMD_S_WAIT_LOADCNT ||
                   mi->op == AMD_S_WAIT_STORECNT ||
                   mi->op == AMD_S_WAIT_DSCNT ||
                   mi->op == AMD_S_WAIT_KMCNT) {
            asm_append(A, " 0x%x", mi->flags);
        } else if (mi->num_uses > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
        }
        break;
    }
    case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR: {
        /* Load:  global_load_dword  vDst, vOffset, sBase|off */
        /* Store: global_store_dword vOffset, vSrc, sBase|off */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            asm_append(A, ", ");
            print_operand(A, &mi->operands[1]);
            asm_append(A, ", ");
            if (mi->num_uses > 1 && mi->operands[2].kind == MOP_SGPR)
                print_sgpr_pair(A, mi->operands[2].reg_num);
            else
                asm_append(A, "off");
        } else {
            if (mi->num_uses >= 2) {
                asm_append(A, " ");
                print_operand(A, &mi->operands[0]);
                asm_append(A, ", ");
                print_operand(A, &mi->operands[1]);
                asm_append(A, ", ");
                if (mi->num_uses > 2 && mi->operands[2].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[2].reg_num);
                else
                    asm_append(A, "off");
            }
        }
        if (mi->flags & AMD_FLAG_GLC) asm_append(A, " glc");
        break;
    }
    case AMD_FMT_VOP3P_MAI: {
        /* v_mfma_*  vDst, vSrc0, vSrc1, vAccum */
        for (uint8_t k = 0; k < total; k++) {
            if (k > 0) asm_append(A, ",");
            asm_append(A, " ");
            print_operand(A, &mi->operands[k]);
        }
        break;
    }
    case AMD_FMT_DS: {
        /* ds_read_b32 vDst, vAddr [, offset] */
        /* ds_write_b32 vAddr, vSrc [, offset] */
        if (mi->num_defs > 0) {
            asm_append(A, " ");
            print_operand(A, &mi->operands[0]);
            for (uint8_t k = mi->num_defs; k < total; k++) {
                asm_append(A, ", ");
                print_operand(A, &mi->operands[k]);
            }
        } else {
            for (uint8_t k = 0; k < mi->num_uses && k < MINST_MAX_OPS; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    case AMD_FMT_SOP1: {
        /* s_mov_b32 sDst, sSrc */
        /* s_setpc_b64 sBase */
        if (mi->op == AMD_S_SETPC_B64 || mi->op == AMD_S_SWAPPC_B64) {
            if (mi->num_uses > 0) {
                asm_append(A, " ");
                if (mi->operands[mi->num_defs].kind == MOP_SGPR)
                    print_sgpr_pair(A, mi->operands[mi->num_defs].reg_num);
                else
                    print_operand(A, &mi->operands[mi->num_defs]);
            }
        } else {
            for (uint8_t k = 0; k < total; k++) {
                if (k > 0) asm_append(A, ",");
                asm_append(A, " ");
                print_operand(A, &mi->operands[k]);
            }
        }
        break;
    }
    default: {
        /* Generic: dst, src0, src1, ... */
        for (uint8_t k = 0; k < total; k++) {
            if (k > 0) asm_append(A, ",");
            asm_append(A, " ");
            print_operand(A, &mi->operands[k]);
        }
        break;
    }
    }

    asm_append(A, "\n");
}

static void emit_asm_function(amd_module_t *A, uint32_t mf_idx)
{
    const mfunc_t *F = &A->mfuncs[mf_idx];
    const char *name = A->bir->strings + F->name;

    if (F->is_kernel) {
        asm_append(A, "    .globl %s\n", name);
        asm_append(A, "    .p2align 8\n");
        asm_append(A, "    .type %s,@function\n", name);
    } else {
        asm_append(A, "    .type %s,@function\n", name);
    }
    asm_append(A, "%s:\n", name);
    asm_append(A, "    ; %u SGPRs, %u VGPRs, %u LDS bytes, %u scratch bytes\n",
               F->num_sgprs, F->num_vgprs, F->lds_bytes, F->scratch_bytes);

    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        const mblock_t *MB = &A->mblocks[mb_idx];

        asm_append(A, ".LBB%u:\n", mb_idx);

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            print_minst(A, &A->minsts[mi_idx]);
        }
    }
    asm_append(A, "\n");
}

void amdgpu_regalloc(amd_module_t *A)
{
    amdgpu_phi_elim(A);
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++)
        regalloc_function(A, fi);
}

void amdgpu_emit_asm(const amd_module_t *amd, FILE *out)
{
    /* We need to cast away const for the asm buffer operations */
    amd_module_t *A = (amd_module_t *)amd;

    A->asm_len = 0;
    asm_append(A, "    .amdgcn_target \"amdgcn-amd-amdhsa--%s\"\n",
               A->chip_name);
    asm_append(A, "    .text\n\n");

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        emit_asm_function(A, fi);
    }

    /* Write to output */
    fwrite(A->asm_buf, 1, A->asm_len, out);
}

/* ---- Msgpack Encoder (minimal, bounded) ---- */

#define MP_BUF_MAX 8192

static void mp_write(uint8_t *buf, uint32_t *pos, const void *data, uint32_t len)
{
    if (*pos + len > MP_BUF_MAX) return;
    memcpy(buf + *pos, data, len);
    *pos += len;
}

static void mp_fixmap(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x80 | count);
}

static void mp_fixarray(uint8_t *buf, uint32_t *pos, uint8_t count)
{
    if (*pos >= MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0x90 | count);
}

static void mp_fixstr(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint8_t len = (uint8_t)strlen(s);
    if (len > 31) len = 31;
    if (*pos + 1 + len > MP_BUF_MAX) return;
    buf[(*pos)++] = (uint8_t)(0xA0 | len);
    mp_write(buf, pos, s, len);
}

static void mp_str(uint8_t *buf, uint32_t *pos, const char *s)
{
    uint32_t len = (uint32_t)strlen(s);
    if (len <= 31) {
        mp_fixstr(buf, pos, s);
    } else if (len <= 255) {
        if (*pos + 2 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xD9;
        buf[(*pos)++] = (uint8_t)len;
        mp_write(buf, pos, s, len);
    } else {
        if (*pos + 3 + len > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xDA;
        buf[(*pos)++] = (uint8_t)(len >> 8);
        buf[(*pos)++] = (uint8_t)(len);
        mp_write(buf, pos, s, len);
    }
}

static void mp_uint(uint8_t *buf, uint32_t *pos, uint32_t val)
{
    if (*pos >= MP_BUF_MAX - 5) return;  /* worst case: 5 bytes */
    if (val <= 127) {
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFF) {
        buf[(*pos)++] = 0xCC;
        buf[(*pos)++] = (uint8_t)val;
    } else if (val <= 0xFFFF) {
        buf[(*pos)++] = 0xCD;
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    } else {
        buf[(*pos)++] = 0xCE;
        buf[(*pos)++] = (uint8_t)(val >> 24);
        buf[(*pos)++] = (uint8_t)(val >> 16);
        buf[(*pos)++] = (uint8_t)(val >> 8);
        buf[(*pos)++] = (uint8_t)val;
    }
}

/* ---- ELF Code Object Writer ---- */



/* ELF64 types */
typedef struct {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} elf64_ehdr_t;   /* 64 bytes */

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} elf64_shdr_t;   /* 64 bytes */

typedef struct {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
} elf64_phdr_t;   /* 56 bytes */

typedef struct {
    uint32_t st_name;
    uint8_t  st_info;
    uint8_t  st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} elf64_sym_t;    /* 24 bytes */

typedef struct {
    uint32_t n_namesz;
    uint32_t n_descsz;
    uint32_t n_type;
} elf64_nhdr_t;   /* 12 bytes */

typedef struct {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
} elf64_phdr_t;   /* 56 bytes */

typedef struct {
    int64_t  d_tag;
    uint64_t d_val;
} elf64_dyn_t;    /* 16 bytes */

#define SHT_NULL     0
#define SHT_PROGBITS 1
#define SHT_SYMTAB   2
#define SHT_STRTAB   3
#define SHT_HASH     5
#define SHT_DYNAMIC  6
#define SHT_NOTE     7
#define SHT_DYNSYM   11
#define SHF_WRITE    1
#define SHF_ALLOC    2
#define SHF_EXECINSTR 4
#define STB_GLOBAL   1
#define STT_FUNC     2
#define STT_OBJECT   1
#define NT_AMDGPU_METADATA 32

#define PT_LOAD      1
#define PT_DYNAMIC   2
#define PT_NOTE      4
#define PT_PHDR      6
#define PF_X         1
#define PF_W         2
#define PF_R         4

#define DT_NULL      0
#define DT_HASH      4
#define DT_STRTAB    5
#define DT_SYMTAB    6
#define DT_STRSZ     10
#define DT_SYMENT    11

/* Pad file to alignment boundary (bounded to avoid infinite loops) */
static void fwrite_pad(FILE *fp, uint32_t align)
{
    long pos = ftell(fp);
    if (pos < 0) return;
    uint32_t pad = ((uint32_t)pos + align - 1) & ~(align - 1);
    uint32_t n = pad - (uint32_t)pos;
    if (n > 256) n = 256; /* sanity cap */
    for (uint32_t i = 0; i < n; i++)
        fputc(0, fp);
}

int amdgpu_emit_elf(amd_module_t *A, const char *path)
{
    /* First, encode all functions to binary */
    A->code_len = 0;

    /* Build .rodata (kernel descriptors) and .text (code) separately.
     * The HSA runtime wants KDs in .rodata — data and deeds, separated
     * like a well-organised criminal enterprise. */
    static uint8_t rodata[16384];   /* up to ~256 KDs with alignment */
    uint32_t rodata_len = 0;
    static uint32_t rodata_kd_off[64]; /* KD offset within .rodata */
    static uint32_t code_offsets[64];  /* code offset within .text */
    uint32_t num_kernels = 0;

    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        if (num_kernels >= 64) break;

        mfunc_t *F = &A->mfuncs[fi];
        int cdna = (A->target <= AMD_TARGET_GFX942);

        /* ---- KD → .rodata (64-byte aligned, CP microcode demands it) ---- */
        while (rodata_len % 64 != 0 && rodata_len < sizeof(rodata))
            rodata[rodata_len++] = 0;
        rodata_kd_off[num_kernels] = rodata_len;

        amd_kernel_descriptor_t kd;
        memset(&kd, 0, sizeof(kd));
        kd.group_segment_fixed_size = F->lds_bytes;
        kd.private_segment_fixed_size = F->scratch_bytes;
        kd.kernarg_size = F->kernarg_bytes;
        kd.kernel_code_entry_byte_offset = 0; /* patched after layout */

        /* compute_pgm_rsrc1 — VGPR gran=8 (GFX90A+), SGPR gran=16 (GFX9).
         * GFX10+ ignores the SGPR field entirely.
         * GFX9/CDNA: VCC, FLAT_SCRATCH, XNACK_MASK are carved from the
         * RSRC1 SGPR allocation.  MI300X needs SGPR_BLOCKS >= 2 (i.e.
         * >= 48 physical) or kernels with 12+ user SGPRs get Error 700.
         * The +6 from the ISA manual is necessary but not sufficient —
         * 48 is the empirically proven floor.  Don't fly with less. */
        uint32_t vgpr_blocks = (F->num_vgprs > 0)
            ? (uint32_t)((F->num_vgprs + 7) / 8 - 1) : 0;
        uint32_t sgpr_gran = cdna ? 16u : 8u;
        uint32_t total_sgprs = F->num_sgprs;
        if (cdna && total_sgprs < 33) total_sgprs = 33;
        uint32_t sgpr_blocks = (total_sgprs > 0)
            ? (uint32_t)((total_sgprs + sgpr_gran - 1) / sgpr_gran - 1) : 0;
        kd.compute_pgm_rsrc1 = (vgpr_blocks & 0x3F) |
                               ((sgpr_blocks & 0xF) << 6) |
                               (3u << 16) |   /* FLOAT_DENORM_MODE_32 = preserve all */
                               (3u << 18) |   /* FLOAT_DENORM_MODE_16_64 = preserve all */
                               (1u << 21) |   /* ENABLE_DX10_CLAMP */
                               (1u << 23);    /* ENABLE_IEEE_MODE */
        if (!cdna) {
            kd.compute_pgm_rsrc1 |= (1u << 26) |  /* WGP_MODE (RDNA only) */
                                    (1u << 27);    /* MEM_ORDERED (RDNA only) */
        }

        /* compute_pgm_rsrc2.
         * CDNA/GFX9 uses TGID bits at 11/12/13. RDNA keeps 7/8/9 and
         * VGPR_WORKITEM_ID at 12:11. */
        {
            uint32_t tgid_x = cdna ? 11u : 7u;
            uint32_t tgid_y = cdna ? 12u : 8u;
            uint32_t tgid_z = cdna ? 13u : 9u;
            uint32_t user_sgpr = 4u; /* dispatch_ptr + kernarg_ptr */
            uint32_t rsrc2 = ((F->scratch_bytes > 0) ? 1u : 0u) |
                             (user_sgpr << 1) |
                             (1u << tgid_x);
            if (F->max_dim >= 1) rsrc2 |= (1u << tgid_y);
            if (F->max_dim >= 2) rsrc2 |= (1u << tgid_z);
            if (!cdna)
                rsrc2 |= ((uint32_t)F->max_dim << 11);
            kd.compute_pgm_rsrc2 = rsrc2;
        }

        /* compute_pgm_rsrc3 — ACCUM_OFFSET for CDNA (GFX90A/GFX942).
         * Tells the HW where ArchVGPRs end and AccVGPRs begin.
         * GFX942 unified VGPRs: all are ArchVGPR, so offset = vgpr_blocks. */
        if (cdna) {
            uint32_t ao_gran = (A->target == AMD_TARGET_GFX942) ? 8u : 4u;
            uint32_t accum_off = (F->num_vgprs > 0)
                ? (uint32_t)((F->num_vgprs + ao_gran - 1) / ao_gran - 1) : 0;
            kd.compute_pgm_rsrc3 = accum_off & 0x3F;
        }

        /* kernel_code_properties (amd_hsa_kernel_code.h) */
        kd.kernel_code_properties = (1u << 1) |  /* ENABLE_SGPR_DISPATCH_PTR */
                                    (1u << 3) |  /* ENABLE_SGPR_KERNARG_PTR */
                                    (1u << 19);  /* IS_PTR64 */
        if (!cdna)
            kd.kernel_code_properties |= (1u << 10); /* WAVEFRONT_SIZE32 */

        if (rodata_len + 64 <= sizeof(rodata)) {
            memcpy(rodata + rodata_len, &kd, 64);
            rodata_len += 64;
        }

        /* ---- Code → A->code (.text, 256-byte aligned for HW prefetcher) ---- */
        for (uint32_t pad = 0; A->code_len % 256 != 0 && A->code_len < AMD_CODE_SIZE && pad < 256; pad++)
            A->code[A->code_len++] = 0;
        code_offsets[num_kernels] = A->code_len;
        num_kernels++;

        encode_function(A, fi);
    }

    /* Also encode device functions */
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++) {
        if (A->mfuncs[fi].is_kernel) continue;
        encode_function(A, fi);
    }

    /* Build note section (msgpack metadata) */
    static uint8_t note_buf[16384];
    uint32_t note_len = 0;

    /* Note header: name = "AMDGPU", type = NT_AMDGPU_METADATA */
    const char *note_name = "AMDGPU\0\0"; /* 8 bytes aligned */
    uint32_t note_name_len = 7; /* including null */

    /* Build msgpack payload */
    static uint8_t mp_buf[8192];
    uint32_t mp_pos = 0;

    mp_fixmap(mp_buf, &mp_pos, 3);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.version");
    mp_fixarray(mp_buf, &mp_pos, 2);
    mp_uint(mp_buf, &mp_pos, 1);
    mp_uint(mp_buf, &mp_pos, 2);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.target");
    char mp_tgt[40];
    snprintf(mp_tgt, sizeof(mp_tgt), "amdgcn-amd-amdhsa--%s", A->chip_name);
    mp_str(mp_buf, &mp_pos, mp_tgt);

    mp_fixstr(mp_buf, &mp_pos, "amdhsa.kernels");
    uint8_t nk = (num_kernels > 15) ? 15 : (uint8_t)num_kernels;
    mp_fixarray(mp_buf, &mp_pos, nk);

    uint32_t ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < nk; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        mfunc_t *F = &A->mfuncs[fi];
        const char *name = A->bir->strings + F->name;

        /* Build symbol name: "name.kd" */
        char kd_name[256];
        snprintf(kd_name, sizeof(kd_name), "%s.kd", name);

        mp_fixmap(mp_buf, &mp_pos, 15);

        mp_fixstr(mp_buf, &mp_pos, ".name");
        mp_str(mp_buf, &mp_pos, name);

        mp_fixstr(mp_buf, &mp_pos, ".symbol");
        mp_str(mp_buf, &mp_pos, kd_name);

        mp_str(mp_buf, &mp_pos, ".kernarg_segment_size");
        mp_uint(mp_buf, &mp_pos, F->kernarg_bytes);

        mp_str(mp_buf, &mp_pos, ".kernarg_segment_align");
        mp_uint(mp_buf, &mp_pos, 8);

        mp_str(mp_buf, &mp_pos, ".group_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->lds_bytes);

        mp_str(mp_buf, &mp_pos, ".private_segment_fixed_size");
        mp_uint(mp_buf, &mp_pos, F->scratch_bytes);

        mp_str(mp_buf, &mp_pos, ".wavefront_size");
        mp_uint(mp_buf, &mp_pos, F->wavefront_size);

        mp_fixstr(mp_buf, &mp_pos, ".sgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_sgprs);

        mp_fixstr(mp_buf, &mp_pos, ".vgpr_count");
        mp_uint(mp_buf, &mp_pos, F->num_vgprs);

        mp_str(mp_buf, &mp_pos, ".agpr_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".sgpr_spill_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".vgpr_spill_count");
        mp_uint(mp_buf, &mp_pos, 0);

        mp_str(mp_buf, &mp_pos, ".max_flat_workgroup_size");
        mp_uint(mp_buf, &mp_pos, F->launch_bounds_max > 0 ? F->launch_bounds_max : 1024);

        mp_str(mp_buf, &mp_pos, ".uses_dynamic_stack");
        mp_buf[mp_pos++] = 0xC2; /* msgpack false */

        /* .args — the runtime needs this to map kernarg buffer properly.
         * Without it, hipModuleLaunchKernel refuses to dispatch. */
        mp_fixstr(mp_buf, &mp_pos, ".args");
        {
            /* Find matching BIR function for param type info */
            const bir_func_t *BF = NULL;
            for (uint32_t bfi = 0; bfi < A->bir->num_funcs; bfi++)
                if (A->bir->funcs[bfi].name == F->name) {
                    BF = &A->bir->funcs[bfi]; break;
                }
            uint32_t np = BF ? BF->num_params : 0;
            if (np > 15) np = 15;
            /* 6 hidden args for block_count + group_size if needed */
            uint32_t n_hidden = F->needs_dispatch ? 6 : 0;
            mp_fixarray(mp_buf, &mp_pos, (uint8_t)(np + n_hidden));
            for (uint32_t pi = 0; pi < np; pi++) {
                int is_ptr = 0;
                uint32_t arg_sz = 8; /* default 8-byte aligned */
                if (BF) {
                    const bir_type_t *ft = &A->bir->types[BF->type];
                    uint32_t pt_idx = A->bir->type_fields[ft->count + pi];
                    const bir_type_t *pt = &A->bir->types[pt_idx];
                    is_ptr = (pt->kind == BIR_TYPE_PTR);
                    if (!is_ptr)
                        arg_sz = (pt->width > 0) ? (uint32_t)(pt->width / 8) : 4;
                }
                if (is_ptr) {
                    mp_fixmap(mp_buf, &mp_pos, 4);
                    mp_str(mp_buf, &mp_pos, ".address_space");
                    mp_fixstr(mp_buf, &mp_pos, "global");
                } else {
                    mp_fixmap(mp_buf, &mp_pos, 3);
                }
                mp_fixstr(mp_buf, &mp_pos, ".offset");
                mp_uint(mp_buf, &mp_pos, pi * 8);
                mp_fixstr(mp_buf, &mp_pos, ".size");
                mp_uint(mp_buf, &mp_pos, arg_sz);
                mp_str(mp_buf, &mp_pos, ".value_kind");
                mp_str(mp_buf, &mp_pos, is_ptr ? "global_buffer" : "by_value");
            }
            /* Hidden dispatch args — runtime populates these automatically */
            if (F->needs_dispatch) {
                uint32_t hk = np * 8;
                static const struct { const char *kind; uint32_t off; uint32_t sz; } hargs[] = {
                    { "hidden_block_count_x", 0,  4 },
                    { "hidden_block_count_y", 4,  4 },
                    { "hidden_block_count_z", 8,  4 },
                    { "hidden_group_size_x",  12, 2 },
                    { "hidden_group_size_y",  14, 2 },
                    { "hidden_group_size_z",  16, 2 },
                };
                for (uint32_t hi = 0; hi < 6; hi++) {
                    mp_fixmap(mp_buf, &mp_pos, 3);
                    mp_fixstr(mp_buf, &mp_pos, ".offset");
                    mp_uint(mp_buf, &mp_pos, hk + hargs[hi].off);
                    mp_fixstr(mp_buf, &mp_pos, ".size");
                    mp_uint(mp_buf, &mp_pos, hargs[hi].sz);
                    mp_str(mp_buf, &mp_pos, ".value_kind");
                    mp_str(mp_buf, &mp_pos, hargs[hi].kind);
                }
            }
        }

        ki++;
    }

    /* Assemble note section */
    elf64_nhdr_t nhdr;
    nhdr.n_namesz = note_name_len;
    nhdr.n_descsz = mp_pos;
    nhdr.n_type = NT_AMDGPU_METADATA;
    memcpy(note_buf + note_len, &nhdr, 12);
    note_len += 12;
    memcpy(note_buf + note_len, note_name, 8); /* padded to 4-byte align */
    note_len += 8;
    memcpy(note_buf + note_len, mp_buf, mp_pos);
    note_len += mp_pos;
    /* Pad to 4 bytes */
    while (note_len % 4 != 0 && note_len < sizeof(note_buf))
        note_buf[note_len++] = 0;

    /* ---- Build the DSO envelope ----
     *
     * The HSA runtime loads code objects like a drunk bouncer inspects IDs:
     * it WILL check program headers, dynamic symbols, and ABI version,
     * and it WILL reject you if any are missing. Our previous bare ELF
     * worked fine for the emulator but real hardware has standards.
     *
     * Sections: 0=NULL 1=.note 2=.dynsym 3=.hash 4=.dynstr
     *           5=.text 6=.dynamic 7=.symtab 8=.strtab 9=.shstrtab
     *
     * Program headers: PT_PHDR, PT_LOAD(R), PT_LOAD(RX), PT_LOAD(RW),
     *                  PT_NOTE, PT_DYNAMIC
     */

    /* ---- .shstrtab: section names ---- */
    #define SHSTRTAB_MAX 256
    static char shstrtab[SHSTRTAB_MAX];
    uint32_t shstrtab_len = 0;
    #define SHSTR(var, s) do { \
        var = shstrtab_len; \
        uint32_t l = (uint32_t)sizeof(s); \
        if (shstrtab_len + l <= SHSTRTAB_MAX) { \
            memcpy(shstrtab + shstrtab_len, s, l); shstrtab_len += l; } \
    } while(0)
    shstrtab[shstrtab_len++] = '\0';
    uint32_t sn_note, sn_dynsym, sn_hash, sn_dynstr, sn_rodata;
    uint32_t sn_text, sn_dynamic, sn_symtab, sn_strtab, sn_shstrtab;
    SHSTR(sn_note,     ".note");
    SHSTR(sn_dynsym,   ".dynsym");
    SHSTR(sn_hash,     ".hash");
    SHSTR(sn_dynstr,   ".dynstr");
    SHSTR(sn_rodata,   ".rodata");
    SHSTR(sn_text,     ".text");
    SHSTR(sn_dynamic,  ".dynamic");
    SHSTR(sn_symtab,   ".symtab");
    SHSTR(sn_strtab,   ".strtab");
    SHSTR(sn_shstrtab, ".shstrtab");
    #undef SHSTR

    /* ---- .dynstr + .strtab: kernel name strings ---- */
    #define STRTAB_MAX 4096
    static char dynstr[STRTAB_MAX];
    static char strtab[STRTAB_MAX];
    uint32_t dynstr_len = 0, strtab_len = 0;
    dynstr[dynstr_len++] = '\0';
    strtab[strtab_len++] = '\0';

    /* Kernel name indices — need these before layout for symbol building */
    static uint32_t dk_name[64], df_name[64]; /* .dynstr offsets */
    static uint32_t sk_name[64], sf_name[64]; /* .strtab offsets */

    ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < num_kernels && ki < 64; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;
        const char *name = A->bir->strings + A->mfuncs[fi].name;
        char kd[256];
        snprintf(kd, sizeof(kd), "%s.kd", name);
        uint32_t kl = (uint32_t)strlen(kd) + 1;
        uint32_t nl = (uint32_t)strlen(name) + 1;

        /* .dynstr */
        dk_name[ki] = dynstr_len;
        if (dynstr_len + kl <= STRTAB_MAX) { memcpy(dynstr + dynstr_len, kd, kl); dynstr_len += kl; }
        df_name[ki] = dynstr_len;
        if (dynstr_len + nl <= STRTAB_MAX) { memcpy(dynstr + dynstr_len, name, nl); dynstr_len += nl; }

        /* .strtab (same names, separate table) */
        sk_name[ki] = strtab_len;
        if (strtab_len + kl <= STRTAB_MAX) { memcpy(strtab + strtab_len, kd, kl); strtab_len += kl; }
        sf_name[ki] = strtab_len;
        if (strtab_len + nl <= STRTAB_MAX) { memcpy(strtab + strtab_len, name, nl); strtab_len += nl; }
        ki++;
    }

    /* ---- Compute sizes ---- */
    uint32_t ndynsym = 1 + 2 * num_kernels; /* null + (kd + func) per kernel */
    uint32_t dynsym_size = ndynsym * 24;
    /* SysV hash: 1 bucket, all symbols chained. Simple as a bucket. */
    uint32_t hash_size = (2 + 1 + ndynsym) * 4; /* nbucket + nchain + bucket[1] + chain[ndynsym] */
    uint32_t dyn_nent = 6; /* HASH, SYMTAB, STRTAB, STRSZ, SYMENT, NULL */
    uint32_t dyn_size = dyn_nent * 16;

    /* ---- Compute file layout ----
     * R segment (VA = file offset): ehdr + phdrs + .note + .dynsym + .hash + .dynstr + .rodata
     * RX segment (VA = file_offset + 0x1000): .text
     * RW segment (VA = file_offset + 0x2000): .dynamic */

    #define N_PHDR 6
    uint64_t phdr_off  = 64;
    uint64_t phdr_size = N_PHDR * 56;
    uint64_t note_off  = (phdr_off + phdr_size + 3) & ~3ULL;
    uint64_t dsym_off  = (note_off + note_len + 7) & ~7ULL;
    uint64_t hash_off  = (dsym_off + dynsym_size + 3) & ~3ULL;
    uint64_t dstr_off  = hash_off + hash_size;

    uint64_t rod_off   = (dstr_off + dynstr_len + 63) & ~63ULL; /* 64-align for KD */
    uint64_t rod_va    = rod_off; /* R segment: VA = file offset */
    uint64_t seg_r_end = rod_off + rodata_len;

    uint64_t text_off  = (seg_r_end + 255) & ~255ULL; /* 256-align for code */
    uint64_t text_va   = text_off + 0x1000;
    uint64_t text_size = A->code_len;

    uint64_t dyn_off   = (text_off + text_size + 7) & ~7ULL;
    uint64_t dyn_va    = dyn_off + 0x2000;

    uint64_t sym_off   = (dyn_off + dyn_size + 7) & ~7ULL;
    uint64_t sym_size  = ndynsym * 24; /* .symtab mirrors .dynsym */
    uint64_t str_off   = sym_off + sym_size;
    uint64_t shs_off   = str_off + strtab_len;
    uint64_t shdr_off  = (shs_off + shstrtab_len + 7) & ~7ULL;

    /* Fix up kernel_code_entry_byte_offset now that VAs are known.
     * Offset 16 in the KD = signed distance from KD (.rodata) to code (.text). */
    for (uint32_t ri = 0; ri < num_kernels; ri++) {
        int64_t entry_off = (int64_t)((text_va + code_offsets[ri]) -
                                      (rod_va + rodata_kd_off[ri]));
        memcpy(rodata + rodata_kd_off[ri] + 16, &entry_off, 8);
    }

    /* ---- Build .dynsym + .symtab (now we know VAs) ---- */
    static elf64_sym_t dynsym[256];
    static elf64_sym_t symtab[256];
    memset(&dynsym[0], 0, 24);
    memset(&symtab[0], 0, 24);
    uint32_t si = 1;

    ki = 0;
    for (uint32_t fi = 0; fi < A->num_mfuncs && ki < num_kernels && ki < 64; fi++) {
        if (!A->mfuncs[fi].is_kernel) continue;

        /* .kd descriptor (STT_OBJECT) in .rodata (section 5) */
        uint64_t kd_va = rod_va + rodata_kd_off[ki];
        dynsym[si].st_name = dk_name[ki];
        dynsym[si].st_info = (STB_GLOBAL << 4) | STT_OBJECT;
        dynsym[si].st_shndx = 5; /* .rodata */
        dynsym[si].st_value = kd_va;
        dynsym[si].st_size = 64;

        symtab[si].st_name = sk_name[ki];
        symtab[si].st_info = (STB_GLOBAL << 4) | STT_OBJECT;
        symtab[si].st_shndx = 5;
        symtab[si].st_value = kd_va;
        symtab[si].st_size = 64;
        si++;

        /* Function entry (STT_FUNC) in .text (section 6) */
        uint64_t fn_va = text_va + code_offsets[ki];
        dynsym[si].st_name = df_name[ki];
        dynsym[si].st_info = (STB_GLOBAL << 4) | STT_FUNC;
        dynsym[si].st_shndx = 6; /* .text */
        dynsym[si].st_value = fn_va;
        dynsym[si].st_size = A->code_len - code_offsets[ki];

        symtab[si].st_name = sf_name[ki];
        symtab[si].st_info = (STB_GLOBAL << 4) | STT_FUNC;
        symtab[si].st_shndx = 6;
        symtab[si].st_value = fn_va;
        symtab[si].st_size = A->code_len - code_offsets[ki];
        si++;
        ki++;
    }
    uint32_t num_syms = si;

    /* ---- Build .hash (SysV, 1 bucket — all symbols in one chain) ---- */
    static uint32_t hash_buf[256];
    hash_buf[0] = 1;         /* nbucket */
    hash_buf[1] = num_syms;  /* nchain  */
    hash_buf[2] = (num_syms > 1) ? 1 : 0; /* bucket[0] = first real sym */
    hash_buf[3] = 0;         /* chain[0] = end (null sym) */
    for (uint32_t hi = 1; hi < num_syms && hi + 3 < 256; hi++)
        hash_buf[3 + hi] = (hi + 1 < num_syms) ? hi + 1 : 0;

    /* ---- Build .dynamic ---- */
    static elf64_dyn_t dynamic[8];
    dynamic[0].d_tag = DT_HASH;   dynamic[0].d_val = hash_off; /* R seg: VA = file off */
    dynamic[1].d_tag = DT_SYMTAB; dynamic[1].d_val = dsym_off;
    dynamic[2].d_tag = DT_STRTAB; dynamic[2].d_val = dstr_off;
    dynamic[3].d_tag = DT_STRSZ;  dynamic[3].d_val = dynstr_len;
    dynamic[4].d_tag = DT_SYMENT; dynamic[4].d_val = 24;
    dynamic[5].d_tag = DT_NULL;   dynamic[5].d_val = 0;

    /* ---- Write ELF ---- */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    /* ELF header */
    elf64_ehdr_t ehdr;
    memset(&ehdr, 0, sizeof(ehdr));
    ehdr.e_ident[0] = 0x7F;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    ehdr.e_ident[4] = 2;    /* ELFCLASS64 */
    ehdr.e_ident[5] = 1;    /* ELFDATA2LSB */
    ehdr.e_ident[6] = 1;    /* EV_CURRENT */
    ehdr.e_ident[7] = ELFOSABI_AMDGPU_HSA;
    ehdr.e_ident[8] = 1;    /* ELFABIVERSION_AMDGPU_HSA_V3 */
    ehdr.e_type = 3;         /* ET_DYN */
    ehdr.e_machine = EM_AMDGPU;
    ehdr.e_version = 1;
    ehdr.e_entry = 0;
    ehdr.e_phoff = phdr_off;
    ehdr.e_shoff = shdr_off;
    ehdr.e_flags = A->elf_mach;
    ehdr.e_ehsize = 64;
    ehdr.e_phentsize = 56;
    ehdr.e_phnum = N_PHDR;
    ehdr.e_shentsize = 64;
    ehdr.e_shnum = 11;      /* +.rodata */
    ehdr.e_shstrndx = 10;   /* .shstrtab */
    fwrite(&ehdr, 1, 64, fp);

    /* Program headers */
    elf64_phdr_t phdrs[N_PHDR];
    memset(phdrs, 0, sizeof(phdrs));

    /* 0: PT_PHDR — the program headers themselves */
    phdrs[0].p_type   = PT_PHDR;
    phdrs[0].p_flags  = PF_R;
    phdrs[0].p_offset = phdr_off;
    phdrs[0].p_vaddr  = phdr_off;
    phdrs[0].p_paddr  = phdr_off;
    phdrs[0].p_filesz = phdr_size;
    phdrs[0].p_memsz  = phdr_size;
    phdrs[0].p_align  = 8;

    /* 1: PT_LOAD (R) — ehdr + phdrs + note + dynsym + hash + dynstr */
    phdrs[1].p_type   = PT_LOAD;
    phdrs[1].p_flags  = PF_R;
    phdrs[1].p_offset = 0;
    phdrs[1].p_vaddr  = 0;
    phdrs[1].p_paddr  = 0;
    phdrs[1].p_filesz = seg_r_end;
    phdrs[1].p_memsz  = seg_r_end;
    phdrs[1].p_align  = 0x1000;

    /* 2: PT_LOAD (RX) — .text (KDs + code) */
    phdrs[2].p_type   = PT_LOAD;
    phdrs[2].p_flags  = PF_R | PF_X;
    phdrs[2].p_offset = text_off;
    phdrs[2].p_vaddr  = text_va;
    phdrs[2].p_paddr  = text_va;
    phdrs[2].p_filesz = text_size;
    phdrs[2].p_memsz  = text_size;
    phdrs[2].p_align  = 0x1000;

    /* 3: PT_LOAD (RW) — .dynamic */
    phdrs[3].p_type   = PT_LOAD;
    phdrs[3].p_flags  = PF_R | PF_W;
    phdrs[3].p_offset = dyn_off;
    phdrs[3].p_vaddr  = dyn_va;
    phdrs[3].p_paddr  = dyn_va;
    phdrs[3].p_filesz = dyn_size;
    phdrs[3].p_memsz  = dyn_size;
    phdrs[3].p_align  = 0x1000;

    /* 4: PT_NOTE — metadata */
    phdrs[4].p_type   = PT_NOTE;
    phdrs[4].p_flags  = PF_R;
    phdrs[4].p_offset = note_off;
    phdrs[4].p_vaddr  = note_off;
    phdrs[4].p_paddr  = note_off;
    phdrs[4].p_filesz = note_len;
    phdrs[4].p_memsz  = note_len;
    phdrs[4].p_align  = 4;

    /* 5: PT_DYNAMIC — so the loader finds .dynsym et al */
    phdrs[5].p_type   = PT_DYNAMIC;
    phdrs[5].p_flags  = PF_R | PF_W;
    phdrs[5].p_offset = dyn_off;
    phdrs[5].p_vaddr  = dyn_va;
    phdrs[5].p_paddr  = dyn_va;
    phdrs[5].p_filesz = dyn_size;
    phdrs[5].p_memsz  = dyn_size;
    phdrs[5].p_align  = 8;

    fwrite(phdrs, 56, N_PHDR, fp);

    /* .note */
    fwrite_pad(fp, 4);
    fwrite(note_buf, 1, note_len, fp);

    /* .dynsym */
    fwrite_pad(fp, 8);
    fwrite(dynsym, 24, num_syms, fp);

    /* .hash */
    fwrite_pad(fp, 4);
    fwrite(hash_buf, 4, 3 + num_syms, fp);

    /* .dynstr */
    fwrite(dynstr, 1, dynstr_len, fp);

    /* .rodata (kernel descriptors, 64-byte aligned) */
    fwrite_pad(fp, 64);
    fwrite(rodata, 1, rodata_len, fp);

    /* .text (code only, 256-byte aligned for HW prefetcher) */
    fwrite_pad(fp, 256);
    fwrite(A->code, 1, A->code_len, fp);

    /* .dynamic */
    fwrite_pad(fp, 8);
    fwrite(dynamic, 16, dyn_nent, fp);

    /* .symtab (non-loaded, no PT_LOAD needed) */
    fwrite_pad(fp, 8);
    fwrite(symtab, 24, num_syms, fp);

    /* .strtab */
    fwrite(strtab, 1, strtab_len, fp);

    /* .shstrtab */
    fwrite(shstrtab, 1, shstrtab_len, fp);
    fwrite_pad(fp, 8);

    /* ---- Section header table ----
     * 0=NULL 1=.note 2=.dynsym 3=.hash 4=.dynstr 5=.rodata
     * 6=.text 7=.dynamic 8=.symtab 9=.strtab 10=.shstrtab */
    elf64_shdr_t shdrs[11];
    memset(shdrs, 0, sizeof(shdrs));

    /* 0: NULL (already zeroed) */

    /* 1: .note */
    shdrs[1].sh_name  = sn_note;
    shdrs[1].sh_type  = SHT_NOTE;
    shdrs[1].sh_flags = SHF_ALLOC;
    shdrs[1].sh_addr  = note_off;
    shdrs[1].sh_offset = note_off;
    shdrs[1].sh_size  = note_len;
    shdrs[1].sh_addralign = 4;

    /* 2: .dynsym */
    shdrs[2].sh_name  = sn_dynsym;
    shdrs[2].sh_type  = SHT_DYNSYM;
    shdrs[2].sh_flags = SHF_ALLOC;
    shdrs[2].sh_addr  = dsym_off;
    shdrs[2].sh_offset = dsym_off;
    shdrs[2].sh_size  = dynsym_size;
    shdrs[2].sh_link  = 4;  /* .dynstr */
    shdrs[2].sh_info  = 1;  /* first global */
    shdrs[2].sh_addralign = 8;
    shdrs[2].sh_entsize = 24;

    /* 3: .hash */
    shdrs[3].sh_name  = sn_hash;
    shdrs[3].sh_type  = SHT_HASH;
    shdrs[3].sh_flags = SHF_ALLOC;
    shdrs[3].sh_addr  = hash_off;
    shdrs[3].sh_offset = hash_off;
    shdrs[3].sh_size  = hash_size;
    shdrs[3].sh_link  = 2;  /* .dynsym */
    shdrs[3].sh_addralign = 4;
    shdrs[3].sh_entsize = 4;

    /* 4: .dynstr */
    shdrs[4].sh_name  = sn_dynstr;
    shdrs[4].sh_type  = SHT_STRTAB;
    shdrs[4].sh_flags = SHF_ALLOC;
    shdrs[4].sh_addr  = dstr_off;
    shdrs[4].sh_offset = dstr_off;
    shdrs[4].sh_size  = dynstr_len;
    shdrs[4].sh_addralign = 1;

    /* 5: .rodata (kernel descriptors) */
    shdrs[5].sh_name  = sn_rodata;
    shdrs[5].sh_type  = SHT_PROGBITS;
    shdrs[5].sh_flags = SHF_ALLOC;
    shdrs[5].sh_addr  = rod_off;
    shdrs[5].sh_offset = rod_off;
    shdrs[5].sh_size  = rodata_len;
    shdrs[5].sh_addralign = 64;

    /* 6: .text */
    shdrs[6].sh_name  = sn_text;
    shdrs[6].sh_type  = SHT_PROGBITS;
    shdrs[6].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    shdrs[6].sh_addr  = text_va;
    shdrs[6].sh_offset = text_off;
    shdrs[6].sh_size  = text_size;
    shdrs[6].sh_addralign = 256;

    /* 7: .dynamic */
    shdrs[7].sh_name  = sn_dynamic;
    shdrs[7].sh_type  = SHT_DYNAMIC;
    shdrs[7].sh_flags = SHF_ALLOC | SHF_WRITE;
    shdrs[7].sh_addr  = dyn_va;
    shdrs[7].sh_offset = dyn_off;
    shdrs[7].sh_size  = dyn_size;
    shdrs[7].sh_link  = 4;  /* .dynstr */
    shdrs[7].sh_addralign = 8;
    shdrs[7].sh_entsize = 16;

    /* 8: .symtab (non-loaded) */
    shdrs[8].sh_name  = sn_symtab;
    shdrs[8].sh_type  = SHT_SYMTAB;
    shdrs[8].sh_offset = sym_off;
    shdrs[8].sh_size  = sym_size;
    shdrs[8].sh_link  = 9;  /* .strtab */
    shdrs[8].sh_info  = 1;
    shdrs[8].sh_addralign = 8;
    shdrs[8].sh_entsize = 24;

    /* 9: .strtab */
    shdrs[9].sh_name  = sn_strtab;
    shdrs[9].sh_type  = SHT_STRTAB;
    shdrs[9].sh_offset = str_off;
    shdrs[9].sh_size  = strtab_len;
    shdrs[9].sh_addralign = 1;

    /* 10: .shstrtab */
    shdrs[10].sh_name  = sn_shstrtab;
    shdrs[10].sh_type  = SHT_STRTAB;
    shdrs[10].sh_offset = shs_off;
    shdrs[10].sh_size  = shstrtab_len;
    shdrs[10].sh_addralign = 1;

    fwrite(shdrs, 64, 11, fp);

    fclose(fp);

    fprintf(stderr, "wrote %s (%u bytes code, %u kernels)\n",
            path, A->code_len, num_kernels);
    return BC_OK;
}
