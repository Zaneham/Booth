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

/* ---- Register Allocation ---- */

/* Live interval for a virtual register */
typedef struct {
    uint32_t vreg;
    uint32_t start;    /* first def */
    uint32_t end;      /* last use */
    uint16_t phys;     /* allocated physical reg */
    uint8_t  file;     /* 0=SGPR, 1=VGPR */
    uint8_t  spilled;
} live_interval_t;

/* Spill relay registers, reserved physical registers that shuttle
 * values between scratch memory and the instruction stream.
 * Like a postal depot: your letter (value) gets loaded from the
 * warehouse (scratch) into the van (relay), delivered to the
 * recipient (instruction), and the van goes back for more.
 *
 * Both RDNA and CDNA: v250-v252 relays, allocatable v0..v249.
 * GFX942 "512 unified VGPRs" is a lie — 256 ArchVGPR + 256 AccVGPR.
 * AccVGPRs only speak MFMA.  Regular VOP/FLAT encoding is 8-bit,
 * so v256+ silently wraps to v0+.  We learned this the hard way.
 *
 * SGPR s99/s98: scalar relays.  v_readfirstlane promotes VGPR to SGPR.
 * Two relays because some instructions have two spilled SGPR sources.
 * One relay? The second load clobbers the first.  We learned this
 * when k_eff came back 0.000 because every spilled comparison
 * compared a value with itself. Whack. */
#define RA_RELAY_V0   250   /* v250-v252 */
#define RA_VGPR_CEIL  250   /* v0..v249 allocatable */
#define RA_NUM_RELAY  3
#define RA_RELAY_S    99    /* first SGPR relay */
#define RA_RELAY_S2   98    /* second SGPR relay */

/* Spill slot map -- one scratch offset per evicted vreg.
 * 512 spills should suffice for any kernel that isn't trying
 * to simulate the entire observable universe in registers. */
#define RA_MAX_SPILL  512
static struct {
    uint16_t vreg;
    uint16_t off;     /* byte offset in scratch memory */
} ra_spills[RA_MAX_SPILL];
static uint32_t ra_nspill;

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
uint16_t op_vreg(const moperand_t *op)
{
    if (op->kind == MOP_VREG_S || op->kind == MOP_VREG_V)
        return op->reg_num;
    return 0xFFFF;
}
/* Legacy name used throughout this file */
static uint16_t operand_vreg(const moperand_t *op) { return op_vreg(op); }

static uint32_t coalesce(amd_module_t *A, const mfunc_t *F);

static void compute_live_intervals(amd_module_t *A, const mfunc_t *F)
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

    /* ---- Coalesce on raw intervals ----
     * Do this BEFORE the back-edge and exec-mask extensions.
     * The extensions inflate intervals conservatively, which makes
     * nearly every copy pair interfere.  On raw [first_def, last_use]
     * intervals, the copy point is where src dies and dst is born —
     * no overlap.  Coalesce first, extend the merged intervals after.
     * Order matters: Chaitin before conservatism. */
    coalesce(A, F);

    /* ---- Loop back-edge extension ----
     * The linear scan above computes intervals as [first_def, last_use],
     * blissfully ignorant of control flow.  A value used inside a loop
     * body has its register freed after the last use — but the loop
     * iterates, and the next pass finds the register holding someone
     * else's laundry.  MEMORY_APERTURE_VIOLATION ensues.
     *
     * Fix: scan for back edges (branch from block B to earlier block H).
     * Any interval alive inside the loop [H.first_inst, B.last_inst]
     * must extend to B.last_inst.  Iterate to fixpoint for nested loops.
     */
    {
        /* Collect back edges: (header_inst, tail_inst) pairs */
        struct { uint32_t hdr; uint32_t tail; } bedge[64];
        uint32_t n_be = 0;

        for (uint32_t bi = 0; bi < F->num_blocks && n_be < 64; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            if (MB->num_insts == 0) continue;
            uint32_t last = MB->first_inst + MB->num_insts - 1;
            const minst_t *mi = &A->minsts[last];
            /* Check all operands for label targets */
            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;
            for (uint8_t k = 0; k < total; k++) {
                if (mi->operands[k].kind == MOP_LABEL) {
                    uint32_t tgt = (uint32_t)mi->operands[k].imm;
                    if (tgt < F->first_block + bi && tgt >= F->first_block) {
                        const mblock_t *H = &A->mblocks[tgt];
                        bedge[n_be].hdr  = H->first_inst;
                        bedge[n_be].tail = last;
                        n_be++;
                    }
                }
            }
        }

        /* Extend intervals across back edges — iterate to fixpoint */
        int changed = 1;
        int guard = 32;
        while (changed && guard-- > 0) {
            changed = 0;
            for (uint32_t e = 0; e < n_be; e++) {
                uint32_t hdr  = bedge[e].hdr;
                uint32_t tail = bedge[e].tail;
                for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
                    if (RA.intervals[v].start == 0xFFFFFFFF) continue;
                    /* Only extend values defined AT or BEFORE the loop
                     * header — these are loop-invariant or PHI results
                     * that must survive the back edge.  Values defined
                     * INSIDE the loop body get re-materialised each
                     * iteration and don't need extension.  The previous
                     * condition (start <= tail) was catastrophically
                     * aggressive: every temporary in the loop got
                     * extended, register pressure went through the roof,
                     * the spill path panicked, and the compiler segfaulted
                     * into a pile of smoking silicon. */
                    if (RA.intervals[v].start <= hdr  &&
                        RA.intervals[v].end   >= hdr  &&
                        RA.intervals[v].end   <  tail) {
                        RA.intervals[v].end = tail;
                        changed = 1;
                    }
                }
            }
        }
    }

    /* ---- Exec mask region extension ----
     * Same principle as above, different villain.  Values alive
     * across a saveexec→restore pair must survive the entire
     * masked region.  Without this, the linear scan sees the
     * last use inside the mask, frees the register, and some
     * RNG temporary moves in.  Then exec restores, the formula
     * reads the RNG's leftovers, and k_eff goes to zero.
     * The hardware does not care about your feelings. */
    {
        /* Pair saveexec/restore using a bounded nesting stack.
         * Structured control flow means they nest properly. */
        struct { uint32_t save; uint32_t rest; } eregion[64];
        uint32_t n_er = 0;
        uint32_t estack[32];
        uint32_t esp = 0;

        for (uint32_t bi = 0; bi < F->num_blocks && n_er < 64; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                uint32_t mi_idx = MB->first_inst + ii;
                const minst_t *mi = &A->minsts[mi_idx];

                /* saveexec → push */
                if (mi->op == AMD_S_AND_SAVEEXEC_B64 ||
                    mi->op == AMD_S_AND_SAVEEXEC_B32) {
                    if (esp < 32)
                        estack[esp++] = mi_idx;
                    continue;
                }

                /* exec restore (OR/XOR to EXEC) → pop and record */
                if ((mi->op == AMD_S_OR_B64  || mi->op == AMD_S_OR_B32 ||
                     mi->op == AMD_S_XOR_B64 || mi->op == AMD_S_XOR_B32) &&
                    mi->num_defs > 0 &&
                    mi->operands[0].kind == MOP_SPECIAL &&
                    mi->operands[0].imm == AMD_SPEC_EXEC) {
                    if (esp > 0 && n_er < 64) {
                        eregion[n_er].save = estack[--esp];
                        eregion[n_er].rest = mi_idx;
                        n_er++;
                    }
                }
            }
        }

        /* Extend intervals that straddle a masked region.
         * If you were alive before the saveexec and your last
         * use is inside the mask, you need to survive until
         * the restore — even if nobody mentions you in between. */
        for (uint32_t e = 0; e < n_er; e++) {
            uint32_t sav = eregion[e].save;
            uint32_t rst = eregion[e].rest;
            for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
                if (RA.intervals[v].start == 0xFFFFFFFF) continue;
                if (RA.intervals[v].start <= sav &&
                    RA.intervals[v].end   >= sav &&
                    RA.intervals[v].end   <  rst) {
                    RA.intervals[v].end = rst;
                }
            }
        }
    }

    /* ---- Prologue SGPR extension ----
     * Function parameters and system SGPRs are defined in the entry
     * block and semantically live for the entire kernel — they're
     * inputs, not temporaries.  The back-edge and exec-mask extensions
     * usually get this right, but not always: a scalar param used
     * once deep in a nested loop can fall through the cracks when
     * diamond patterns eat the extension stack.
     *
     * Fix: any SGPR defined in the first block whose value escapes
     * to a later block gets pinned to the function's last instruction.
     * This costs a handful of SGPRs worth of pressure.  Cheap
     * insurance against the allocator recycling k_eff's register
     * to store someone else's loop counter. */
    {
        uint32_t last_inst = 0;
        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            if (MB->num_insts > 0) {
                uint32_t ei = MB->first_inst + MB->num_insts - 1;
                if (ei > last_inst) last_inst = ei;
            }
        }
        if (F->num_blocks > 0 && last_inst > 0) {
            const mblock_t *MB0 = &A->mblocks[F->first_block];
            uint32_t blk0_end = MB0->first_inst + MB0->num_insts;
            for (uint32_t v = 0; v < A->vreg_count && v < AMD_MAX_VREGS; v++) {
                if (RA.intervals[v].start == 0xFFFFFFFF) continue;
                if (RA.intervals[v].file != 0) continue; /* SGPRs only */
                if (RA.intervals[v].start < blk0_end &&
                    RA.intervals[v].end   >= blk0_end &&
                    RA.intervals[v].end   <  last_inst) {
                    RA.intervals[v].end = last_inst;
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

/* ---- Chaitin Coalescing (IBM Research, 1981) ----
 * Two vregs joined by a copy whose live ranges don't overlap can
 * share one register.  The copy evaporates.  Chaitin figured this
 * out while building the PL.8 register allocator at Yorktown
 * Heights; Dewar arrived at the same insight from the SPITBOL
 * direction.  We're 45 years late to the party but the drinks
 * are still good.
 *
 * This eliminates PHI copies that phi_elim inserted, which is
 * the main source of register pressure inflation.  A kernel
 * that needs 350 vregs often has 150 that are just copy aliases
 * of each other.  After coalescing, the linear scan sees the
 * truth: maybe 200 unique live values, which fit in 250 VGPRs
 * without spilling. */
static uint32_t coalesce(amd_module_t *A, const mfunc_t *F)
{
    int changed = 1;
    int guard = 16;
    uint32_t n_coal = 0;
    uint32_t n_cand = 0;

    while (changed && guard-- > 0) {
        changed = 0;
        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                uint32_t mi_idx = MB->first_inst + ii;
                minst_t *mi = &A->minsts[mi_idx];

                if (mi->op != AMD_PSEUDO_COPY) continue;
                if (mi->num_defs != 1 || mi->num_uses != 1) continue;

                uint16_t dst = operand_vreg(&mi->operands[0]);
                uint16_t src = operand_vreg(&mi->operands[1]);
                if (dst == 0xFFFF || src == 0xFFFF) continue;
                if (dst == src) continue;

                /* Same register file — can't merge SGPR with VGPR */
                if (RA.intervals[dst].file != RA.intervals[src].file)
                    continue;

                /* Both must be live (not already killed by prior merge) */
                if (RA.intervals[dst].start == 0xFFFFFFFF ||
                    RA.intervals[src].start == 0xFFFFFFFF) continue;

                n_cand++;

                /* Interference: strict overlap of closed intervals. */
                uint32_t s1 = RA.intervals[src].start;
                uint32_t e1 = RA.intervals[src].end;
                uint32_t s2 = RA.intervals[dst].start;
                uint32_t e2 = RA.intervals[dst].end;

                if (s1 < e2 && s2 < e1) { continue; }

                /* Coalesce: src absorbs dst */
                if (s2 < s1) RA.intervals[src].start = s2;
                if (e2 > e1) RA.intervals[src].end = e2;
                RA.intervals[dst].start = 0xFFFFFFFF;
                RA.intervals[dst].end = 0;

                /* Rename dst→src across the entire function */
                for (uint32_t rb = 0; rb < F->num_blocks; rb++) {
                    const mblock_t *RB = &A->mblocks[F->first_block + rb];
                    for (uint32_t rj = 0; rj < RB->num_insts; rj++) {
                        minst_t *mj = &A->minsts[RB->first_inst + rj];
                        uint8_t tot = mj->num_defs + mj->num_uses;
                        if (tot > MINST_MAX_OPS) tot = MINST_MAX_OPS;
                        for (uint8_t k = 0; k < tot; k++) {
                            if (operand_vreg(&mj->operands[k]) == dst)
                                mj->operands[k].reg_num = src;
                        }
                    }
                }

                /* Kill the copy — its work is done */
                mi->op = AMD_S_NOP;
                mi->num_defs = 0;
                mi->num_uses = 0;

                n_coal++;
                changed = 1;
            }
        }
    }
    if (n_coal > 0)
        fprintf(stderr, "  coalesce: %u/%u copies merged\n", n_coal, n_cand);
    return n_coal;
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

/* Rewrite virtual reg operands to physical */
void rw_ops(amd_module_t *A, const mfunc_t *F)
{
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            minst_t *mi = &A->minsts[mi_idx];

            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = 0; k < total; k++) {
                moperand_t *op = &mi->operands[k];
                if (op->kind == MOP_VREG_S || op->kind == MOP_VREG_V) {
                    uint16_t vr = op->reg_num;
                    /* Use reg_file[] as the authority — operand kind
                     * can disagree after SMEM→VALU hazard promotion
                     * (vreg promoted to VGPR file but still referenced
                     * as MOP_VREG_S at some use sites). */
                    if (A->reg_file[vr])
                        op->kind = MOP_VGPR;
                    else
                        op->kind = MOP_SGPR;
                    uint16_t phys = A->reg_map[vr];
                    if (phys == 0xFFFF) {
                        /* Spilled vreg: encode slot index with bit 15
                         * so the spill resolution pass can find it.
                         * Without this the 0xFFFF passes straight through
                         * and the verifier rightfully complains. */
                        for (uint32_t si = 0; si < ra_nspill; si++) {
                            if (ra_spills[si].vreg == vr) {
                                phys = (uint16_t)(0x8000u | si);
                                break;
                            }
                        }
                    }
                    op->reg_num = phys;
                }
            }

            /* Convert PSEUDO_COPY to actual MOV.
             * SGPR←VGPR needs v_readfirstlane (uniform value
             * that ended up in a VGPR — thanks, CDNA hazard). */
            if (mi->op == AMD_PSEUDO_COPY) {
                if (mi->operands[0].kind == MOP_VGPR)
                    mi->op = AMD_V_MOV_B32;
                else if (mi->operands[1].kind == MOP_VGPR)
                    mi->op = AMD_V_READFIRSTLANE_B32;
                else
                    mi->op = AMD_S_MOV_B32;
            }
        }
    }
}

/* Dead copy elimination: kill MOVs where src == dst.
   These appear when regalloc assigns the same phys reg to both sides
   of a copy. Harmless but noisy — like a postman delivering a letter
   back to the sender. */
void dce_copy(amd_module_t *A, const mfunc_t *F)
{
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
                continue;
            }
        }
    }
}

void fin_regs(const amd_module_t *A, mfunc_t *F)
{
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
}

/* ---- Linear Scan (fallback) ---- */

static void ra_lin(amd_module_t *A, uint32_t mf_idx)
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
    for (uint16_t r = AMD_MAX_SGPRS; r-- > sgpr_start; ) {
        if (r == RA_RELAY_S || r == RA_RELAY_S2) continue; /* reserved for spill relays */
        RA.sgpr_free[RA.num_sgpr_free++] = (uint8_t)r;
    }

    /* VGPR pool: v0..v249 on both targets.  v250-v252 reserved
     * for spill relays.  GFX942's AccVGPRs are MFMA-only — the
     * 8-bit encoding fields in VOP/FLAT literally can't see them.
     * We tried.  The hardware was unimpressed. */
    for (uint16_t r = RA_VGPR_CEIL; r-- > 0; )
        RA.vgpr_free[RA.num_vgpr_free++] = (uint8_t)r;

    ra_nspill = 0;

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
                /* Evict the farthest — commandeer its register.
                 * The evicted vreg gets a scratch slot and will be
                 * loaded/stored via relay VGPRs at every use/def.
                 * Expensive, but better than two values sharing
                 * one register like flatmates sharing one toothbrush. */
                uint32_t sv = RA.active[farthest_idx];
                phys = RA.intervals[sv].phys;
                RA.intervals[sv].spilled = 1;
                RA.intervals[sv].phys = 0xFFFF;
                A->reg_map[sv] = 0xFFFF;
                if (ra_nspill < RA_MAX_SPILL) {
                    ra_spills[ra_nspill].vreg = (uint16_t)sv;
                    ra_spills[ra_nspill].off = (uint16_t)(F->scratch_bytes +
                                               ra_nspill * 4u);
                    ra_nspill++;
                }
                RA.active[farthest_idx] = RA.active[--RA.num_active];
            } else {
                /* Spill ourselves — no register, straight to scratch.
                 * The compiler equivalent of being told the hotel is
                 * full and you'll be sleeping in the car park. */
                iv->spilled = 1;
                phys = 0xFFFF;
                if (ra_nspill < RA_MAX_SPILL) {
                    ra_spills[ra_nspill].vreg = (uint16_t)v;
                    ra_spills[ra_nspill].off = (uint16_t)(F->scratch_bytes +
                                               ra_nspill * 4u);
                    ra_nspill++;
                }
            }
        }

        iv->phys = phys;
        A->reg_map[v] = phys;

        /* Add to active — but NOT self-spilled intervals.  Their
         * phys=0 "fallback" is a lie, and expire_old would free
         * register 0 back to the pool when it shouldn't.  That
         * corrupts the free list and eventually overflows the
         * vgpr_free[256] buffer.  Ask me how I know. */
        if (!iv->spilled && RA.num_active < AMD_MAX_VREGS)
            RA.active[RA.num_active++] = v;
    }

    if (ra_nspill > 0) {
        uint32_t vs = 0, ss = 0;
        for (uint32_t si = 0; si < ra_nspill; si++) {
            if (RA.intervals[ra_spills[si].vreg].file == 0) ss++;
            else vs++;
        }
        fprintf(stderr, "  regalloc: %u spills (%uV %uS)\n",
                ra_nspill, vs, ss);
    }

    /* Record usage for kernel descriptor.
     * Regalloc only tracks its own assigned SGPRs, but kernels also
     * use system SGPRs (kernarg, TGID) and param pair SGPRs.
     * first_alloc_sgpr is the floor — everything below it is spoken for. */
    F->num_sgprs = RA.max_sgpr;
    if (F->is_kernel && F->num_sgprs < F->first_alloc_sgpr)
        F->num_sgprs = F->first_alloc_sgpr;
    F->num_vgprs = RA.max_vgpr;

    /* Match old regalloc_function order exactly:
     * 1. min SGPR/VGPR + launch_bounds
     * 2. rw_ops (virtual→physical)
     * 3. DCE (self-copy only)
     * 4. spill resolution */
    fin_regs(A, F);

    /* Rewrite virtual→physical BEFORE spill resolution.
     * Spill resolution scans for 0x8000-encoded reg_nums to find
     * spilled operands, and reads physical SGPRs from scratch ops
     * to find scr_sgpr.  Without this, everything is still virtual
     * and the spill code sees nothing to resolve. */
    rw_ops(A, F);

    /* Dead copy elimination BEFORE spill resolution — matches old
     * regalloc_function order.  Kill self-copies so spill resolution
     * doesn't waste time inserting load/store plumbing around no-ops. */
    dce_copy(A, F);

    /* ---- Spill Resolution ----
     * When the register file runs dry, evicted values get parked in
     * scratch memory.  Now we walk the instruction stream and insert
     * the actual load/store plumbing -- scratch_load before every use,
     * scratch_store after every def.  The relay VGPRs (v250-v252) are
     * the middlemen: values hop from scratch to relay to instruction
     * and back.
     *
     * This is the register allocator's overflow car park at the
     * airport: slow, far from the terminal, but at least your car
     * doesn't get towed.  Without this, two values share one register
     * and corrupt each other silently.  We know because k_eff read
     * 0.000 for three days while n_mat quietly aliased to zero. */
    if (ra_nspill > 0) {
        /* Find the scratch frame-pointer SGPR by scanning for an
         * existing scratch op.  If we're spilling, we have scratch. */
        uint16_t scr_sgpr = 0;
        for (uint32_t bi = 0; bi < F->num_blocks && scr_sgpr == 0; bi++) {
            mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                minst_t *mi = &A->minsts[MB->first_inst + ii];
                if (mi->op == AMD_SCRATCH_LOAD_DWORD ||
                    mi->op == AMD_SCRATCH_STORE_DWORD) {
                    scr_sgpr = mi->operands[1].reg_num;
                    break;
                }
            }
        }

        /* Account for spill area in scratch allocation */
        F->scratch_bytes += ra_nspill * 4u;

        /* Hoist scratch FP init to the very first instruction.
         * The scheduler may have moved kernarg loads ahead of the
         * original s_mov_b32 scrfp,0 — and spill ops around those
         * early instructions would fire before the FP is set,
         * sending scratch accesses to whatever address the CP
         * left in the register.  We learned this the hard way:
         * TEA pointed to a host address, which is the GPU's
         * polite way of saying "that's not yours." */
        if (scr_sgpr > 0) {
            mblock_t *B0 = &A->mblocks[F->first_block];
            uint32_t pos = B0->first_inst;
            if (A->num_minsts + 1 < AMD_MAX_MINSTS) {
                uint32_t tail = A->num_minsts - pos;
                memmove(&A->minsts[pos + 1], &A->minsts[pos],
                        tail * sizeof(minst_t));
                A->num_minsts++;
                B0->num_insts++;
                for (uint32_t lb = F->first_block + 1;
                     lb < F->first_block + F->num_blocks; lb++)
                    A->mblocks[lb].first_inst++;
                minst_t *fp = &A->minsts[pos];
                memset(fp, 0, sizeof(minst_t));
                fp->op = AMD_S_MOV_B32;
                fp->num_defs = 1;
                fp->num_uses = 1;
                fp->operands[0].kind = MOP_SGPR;
                fp->operands[0].reg_num = scr_sgpr;
                fp->operands[1].kind = MOP_IMM;
                fp->operands[1].imm = 0;
            }
        }

        /* Process blocks backwards -- insertions shift later addresses,
         * so going backwards means we never revisit shifted territory. */
        for (int bi = (int)F->num_blocks - 1; bi >= 0; bi--) {
            mblock_t *MB = &A->mblocks[F->first_block + (uint32_t)bi];

            for (int ii = (int)MB->num_insts - 1; ii >= 0; ii--) {
                uint32_t mi_idx = MB->first_inst + (uint32_t)ii;
                minst_t *mi = &A->minsts[mi_idx];
                if (mi->op == AMD_PSEUDO_DEF) continue;

                uint8_t total = mi->num_defs + mi->num_uses;
                if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

                /* Detect spilled operands (bit 15 set by rewrite).
                 * VGPR source = 2 insns (load + wait).
                 * SGPR source = 3 insns (load + wait + readfirstlane).
                 * VGPR dest   = 1 insn  (store).
                 * SGPR dest   = 2 insns (v_mov + store). */
                uint32_t n_insert = 0;
                int has_spill = 0;
                for (uint8_t k = 0; k < total; k++) {
                    if (!(mi->operands[k].reg_num & 0x8000u)) continue;
                    if (mi->operands[k].kind != MOP_VGPR &&
                        mi->operands[k].kind != MOP_SGPR) continue;
                    has_spill = 1;
                    int is_sgpr = (mi->operands[k].kind == MOP_SGPR);
                    if (k < mi->num_defs)
                        n_insert += is_sgpr ? 2u : 1u;
                    else
                        n_insert += is_sgpr ? 3u : 2u;
                }
                if (!has_spill) continue;

                /* Account for the trailing s_waitcnt after dest stores.
                 * n_post gets +1 later (line ~955), but n_insert feeds
                 * the memmove — miss this and the waitcnt writes one
                 * slot past the gap, overwriting the next instruction.
                 * The off-by-one that ate six hours of debugging. */
                {
                    int has_dest_spill = 0;
                    for (uint8_t k2 = 0; k2 < mi->num_defs; k2++) {
                        if ((mi->operands[k2].reg_num & 0x8000u) &&
                            (mi->operands[k2].kind == MOP_VGPR ||
                             mi->operands[k2].kind == MOP_SGPR)) {
                            has_dest_spill = 1;
                            break;
                        }
                    }
                    if (has_dest_spill) n_insert += 1;
                }

                uint32_t tail_start = mi_idx + 1;
                uint32_t tail_len = A->num_minsts - tail_start;
                if (A->num_minsts + n_insert >= AMD_MAX_MINSTS) continue;

                memmove(&A->minsts[tail_start + n_insert],
                        &A->minsts[tail_start],
                        tail_len * sizeof(minst_t));
                A->num_minsts += n_insert;
                MB->num_insts += n_insert;

                for (uint32_t later = F->first_block + (uint32_t)bi + 1;
                     later < F->first_block + F->num_blocks; later++)
                    A->mblocks[later].first_inst += n_insert;

                mi = &A->minsts[mi_idx];

                /* Assign relay VGPRs and remember who was scalar.
                 * Scratch is VMEM — all loads land in VGPRs first.
                 * Scalar operands then hop across the VGPR/SGPR border
                 * via v_readfirstlane (sources) or v_mov (dests).
                 * Two bus changes and a customs declaration, but at
                 * least nothing gets silently aliased to zero. */
                uint16_t relay[MINST_MAX_OPS];
                uint16_t soff[MINST_MAX_OPS];
                uint8_t  was_sgpr[MINST_MAX_OPS];
                uint16_t sgpr_relay[MINST_MAX_OPS]; /* which SGPR relay */
                uint8_t rn = 0;
                uint8_t sgpr_rn = 0;  /* tracks SGPR relay assignment */
                uint32_t n_pre = 0, n_post = 0;
                for (uint8_t k = 0; k < total; k++) {
                    relay[k] = 0xFFFF;
                    soff[k] = 0;
                    was_sgpr[k] = 0;
                    sgpr_relay[k] = RA_RELAY_S;
                    if ((mi->operands[k].kind == MOP_VGPR ||
                         mi->operands[k].kind == MOP_SGPR) &&
                        (mi->operands[k].reg_num & 0x8000u)) {
                        uint16_t si = mi->operands[k].reg_num & 0x7FFFu;
                        if (si < ra_nspill) soff[k] = ra_spills[si].off;
                        was_sgpr[k] = (mi->operands[k].kind == MOP_SGPR);
                        relay[k] = (uint16_t)(RA_RELAY_V0 +
                                   (rn % RA_NUM_RELAY));
                        rn++;
                        if (rn > RA_NUM_RELAY)
                            fprintf(stderr, "  WARN: %u spilled ops in one insn (op %u)\n", rn, mi->op);
                        if (k < mi->num_defs) {
                            /* Dest: instruction writes SGPR relay or
                             * relay VGPR, then we store to scratch. */
                            if (was_sgpr[k]) {
                                sgpr_relay[k] = (sgpr_rn % 2) ? RA_RELAY_S2 : RA_RELAY_S;
                                sgpr_rn++;
                                mi->operands[k].kind = MOP_SGPR;
                                mi->operands[k].reg_num = sgpr_relay[k];
                                n_post += 2; /* v_mov + store */
                            } else {
                                mi->operands[k].kind = MOP_VGPR;
                                mi->operands[k].reg_num = relay[k];
                                n_post += 1; /* store */
                            }
                        } else {
                            /* Source: load from scratch into relay VGPR,
                             * then readfirstlane if the instruction
                             * expects an SGPR. */
                            if (was_sgpr[k]) {
                                sgpr_relay[k] = (sgpr_rn % 2) ? RA_RELAY_S2 : RA_RELAY_S;
                                sgpr_rn++;
                                mi->operands[k].kind = MOP_SGPR;
                                mi->operands[k].reg_num = sgpr_relay[k];
                                n_pre += 3; /* load + wait + rfl */
                            } else {
                                mi->operands[k].kind = MOP_VGPR;
                                mi->operands[k].reg_num = relay[k];
                                n_pre += 2; /* load + wait */
                            }
                        }
                    }
                }
                n_post++; /* waitcnt after stores */

                /* Slide instruction right to make room for source loads.
                 * Layout: [loads+waits+rfl] [instruction] [mov+stores] */
                uint32_t inst_pos = mi_idx + n_pre;
                if (n_pre > 0) {
                    A->minsts[inst_pos] = A->minsts[mi_idx];
                    mi = &A->minsts[inst_pos];
                }

                /* ---- Source loads ----
                 * Each spilled source: scratch_load → s_waitcnt → (rfl).
                 * The scratch→VGPR→SGPR journey: data climbs out of the
                 * scratch cellar through the VGPR ground floor, then
                 * takes the lift to the SGPR penthouse.  v_readfirstlane
                 * is the lift — it copies lane 0 from the VGPR to an
                 * SGPR, which is exactly what you want for a uniform
                 * scalar value that got evicted during a register
                 * shortage.  All lanes have the same value anyway. */
                uint32_t lp = mi_idx;
                for (uint8_t k = mi->num_defs; k < total; k++) {
                    if (relay[k] == 0xFFFF) continue;

                    minst_t *ld = &A->minsts[lp++];
                    memset(ld, 0, sizeof(minst_t));
                    ld->op = AMD_SCRATCH_LOAD_DWORD;
                    ld->num_defs = 1;
                    ld->num_uses = 2;
                    ld->operands[0].kind = MOP_VGPR;
                    ld->operands[0].reg_num = relay[k];
                    ld->operands[1].kind = MOP_SGPR;
                    ld->operands[1].reg_num = scr_sgpr;
                    ld->operands[2].kind = MOP_IMM;
                    ld->operands[2].imm = (int32_t)soff[k];

                    minst_t *wt = &A->minsts[lp++];
                    memset(wt, 0, sizeof(minst_t));
                    wt->op = AMD_S_WAITCNT;
                    wt->flags = AMD_WAIT_VMCNT0;

                    /* SGPR source: one more hop.  readfirstlane ferries
                     * lane 0 from the VGPR relay to the SGPR relay.
                     * Two SGPR relays (s99/s98) prevent the second load
                     * from clobbering the first. */
                    if (was_sgpr[k]) {
                        minst_t *rf = &A->minsts[lp++];
                        memset(rf, 0, sizeof(minst_t));
                        rf->op = AMD_V_READFIRSTLANE_B32;
                        rf->num_defs = 1;
                        rf->num_uses = 1;
                        rf->operands[0].kind = MOP_SGPR;
                        rf->operands[0].reg_num = sgpr_relay[k];
                        rf->operands[1].kind = MOP_VGPR;
                        rf->operands[1].reg_num = relay[k];
                    }
                }

                /* ---- Dest stores ----
                 * Each spilled dest: (v_mov for SGPR) → scratch_store.
                 * SGPR dests write s99, then v_mov copies s99 into a
                 * VGPR relay for the scratch_store.  This is the
                 * reverse of the source journey — value descends from
                 * the SGPR penthouse back down to the scratch cellar. */
                uint32_t sp = inst_pos + 1;
                for (uint8_t k = 0; k < mi->num_defs; k++) {
                    if (relay[k] == 0xFFFF) continue;

                    uint16_t store_vgpr = relay[k];

                    if (was_sgpr[k]) {
                        /* SGPR relay → VGPR relay so scratch_store can reach it */
                        minst_t *mv = &A->minsts[sp++];
                        memset(mv, 0, sizeof(minst_t));
                        mv->op = AMD_V_MOV_B32;
                        mv->num_defs = 1;
                        mv->num_uses = 1;
                        mv->operands[0].kind = MOP_VGPR;
                        mv->operands[0].reg_num = relay[k];
                        mv->operands[1].kind = MOP_SGPR;
                        mv->operands[1].reg_num = sgpr_relay[k];
                    }

                    minst_t *st = &A->minsts[sp++];
                    memset(st, 0, sizeof(minst_t));
                    st->op = AMD_SCRATCH_STORE_DWORD;
                    st->num_defs = 0;
                    st->num_uses = 3;
                    st->operands[0].kind = MOP_VGPR;
                    st->operands[0].reg_num = store_vgpr;
                    st->operands[1].kind = MOP_SGPR;
                    st->operands[1].reg_num = scr_sgpr;
                    st->operands[2].kind = MOP_IMM;
                    st->operands[2].imm = (int32_t)soff[k];
                }

                /* Fence: wait for all dest stores to land before any
                 * later instruction tries to reload from the same slot.
                 * Without this, scratch_load races scratch_store and
                 * the relay delivers last Tuesday's value. */
                if (n_post > 1) {
                    minst_t *wt2 = &A->minsts[sp++];
                    memset(wt2, 0, sizeof(minst_t));
                    wt2->op = AMD_S_WAITCNT;
                    wt2->flags = AMD_WAIT_VMCNT0;
                }
            }
        }
    }

    /* fin_regs already called before rw_ops (matching old order) */
}

/* ---- Graph Coloring Register Allocation ---- */

#define RA_MAX_NODES 8192

typedef struct {
    uint32_t vreg;
    uint32_t degree;
    uint32_t cost;
    uint16_t color;
    uint16_t alias;
    uint8_t  file;
    uint8_t  spilled;
    uint8_t  in_graph;
} ra_node_t;

/* Static pools for graph coloring -- sized to RA_MAX_NODES upper bound.
   Non-reentrant; fine for single-threaded compilation. */
#define RA_IFG_WORDS  ((RA_MAX_NODES * RA_MAX_NODES + 31) / 32)
static uint32_t  ra_ifg[RA_IFG_WORDS];
static ra_node_t ra_nodes[RA_MAX_NODES];
static uint32_t  ra_stack[RA_MAX_NODES];
static uint32_t  ra_num_nodes;
static uint32_t  ra_stride;
static uint32_t  ra_stack_top;

static uint16_t ra_vreg_to_node[AMD_MAX_VREGS];

static void ra_ifg_set(uint32_t i, uint32_t j)
{
    uint64_t bit;
    bit = (uint64_t)i * ra_stride + j;
    ra_ifg[bit / 32] |= 1u << (bit % 32);
    bit = (uint64_t)j * ra_stride + i;
    ra_ifg[bit / 32] |= 1u << (bit % 32);
}

static int ra_ifg_test(uint32_t i, uint32_t j)
{
    uint64_t bit = (uint64_t)i * ra_stride + j;
    return (ra_ifg[bit / 32] >> (bit % 32)) & 1;
}

static uint16_t ra_find(uint16_t n)
{
    while (ra_nodes[n].alias != n) {
        ra_nodes[n].alias = ra_nodes[ra_nodes[n].alias].alias;
        n = ra_nodes[n].alias;
    }
    return n;
}

static void ra_coalesce(uint16_t a, uint16_t b)
{
    ra_nodes[b].alias = a;
    for (uint32_t k = 0; k < ra_num_nodes; k++) {
        if (k == a || k == b) continue;
        if (ra_find((uint16_t)k) != k) continue;
        if (ra_ifg_test(b, k)) {
            if (!ra_ifg_test(a, k)) {
                ra_ifg_set(a, k);
                ra_nodes[a].degree++;
            }
        }
    }
    ra_nodes[b].in_graph = 0;
}

/* Bitvector operations for liveness sets.
   One bit per vreg, stored as uint32_t words. */

static void bv_set(uint32_t *bv, uint32_t bit)
{
    bv[bit / 32] |= 1u << (bit % 32);
}

static void bv_clear(uint32_t *bv, uint32_t bit)
{
    bv[bit / 32] &= ~(1u << (bit % 32));
}

static int bv_test(const uint32_t *bv, uint32_t bit)
{
    return (int)((bv[bit / 32] >> (bit % 32)) & 1u);
}

static int bv_or(uint32_t *dst, const uint32_t *src, uint32_t nwords)
{
    int changed = 0;
    for (uint32_t w = 0; w < nwords; w++) {
        uint32_t old = dst[w];
        dst[w] |= src[w];
        if (dst[w] != old) changed = 1;
    }
    return changed;
}

/* Per-block liveness arrays -- statically sized to RA_MAX_BLOCKS upper
   bound.  Accessed as ra_live_in[bi * ra_bv_words + w]. */
#define RA_MAX_BLOCKS   4096
#define RA_BV_MAX_WORDS ((RA_MAX_NODES + 31) / 32)

static uint32_t ra_live_in [RA_MAX_BLOCKS * RA_BV_MAX_WORDS];
static uint32_t ra_live_out[RA_MAX_BLOCKS * RA_BV_MAX_WORDS];
static uint32_t ra_blk_def [RA_MAX_BLOCKS * RA_BV_MAX_WORDS];
static uint32_t ra_blk_use [RA_MAX_BLOCKS * RA_BV_MAX_WORDS];
static uint16_t ra_succs   [RA_MAX_BLOCKS * 2];
static uint8_t  ra_nsuccs  [RA_MAX_BLOCKS];
static uint32_t ra_bv_words;

#define RA_BV(arr, bi)  ((arr) + (size_t)(bi) * ra_bv_words)
#define RA_SUCCS(bi)    (ra_succs + (size_t)(bi) * 2)

static void ra_build_cfg(const amd_module_t *A, const mfunc_t *F)
{
    uint32_t nb = F->num_blocks;
    memset(ra_nsuccs, 0, nb * sizeof(uint8_t));

    for (uint32_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        int has_unconditional = 0;

        for (uint32_t ii = MB->num_insts; ii > 0; ii--) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii - 1];
            if (!is_terminator(mi->op)) break;

            if (mi->op == AMD_S_ENDPGM || mi->op == AMD_S_SETPC_B64) {
                has_unconditional = 1;
            } else if (mi->op == AMD_S_BRANCH) {
                has_unconditional = 1;
                if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
                    uint32_t target = (uint32_t)mi->operands[0].imm;
                    if (target >= F->first_block &&
                        target < F->first_block + nb &&
                        ra_nsuccs[bi] < 2) {
                        RA_SUCCS(bi)[ra_nsuccs[bi]++] =
                            (uint16_t)(target - F->first_block);
                    }
                }
            } else if (mi->op == AMD_S_CBRANCH_SCC0 ||
                       mi->op == AMD_S_CBRANCH_SCC1 ||
                       mi->op == AMD_S_CBRANCH_EXECZ ||
                       mi->op == AMD_S_CBRANCH_EXECNZ) {
                if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
                    uint32_t target = (uint32_t)mi->operands[0].imm;
                    if (target >= F->first_block &&
                        target < F->first_block + nb &&
                        ra_nsuccs[bi] < 2) {
                        RA_SUCCS(bi)[ra_nsuccs[bi]++] =
                            (uint16_t)(target - F->first_block);
                    }
                }
            }
        }

        /* Fallthrough: if block doesn't end with an unconditional branch
           or endpgm, the next block is a successor */
        if (!has_unconditional && bi + 1 < nb && ra_nsuccs[bi] < 2) {
            RA_SUCCS(bi)[ra_nsuccs[bi]++] = (uint16_t)(bi + 1);
        }
    }
}

/* Compute per-block def/use sets, then solve live_in/live_out via
   backward dataflow iteration:
     live_in[B]  = use[B] ∪ (live_out[B] − def[B])
     live_out[B] = ∪ { live_in[S] : S ∈ successors(B) }
   Iterates until fixpoint. */
static void ra_cpliv(const amd_module_t *A, const mfunc_t *F,
                                uint32_t nv)
{
    uint32_t nb = F->num_blocks;
    uint32_t bv_words = ra_bv_words;
    for (uint32_t bi = 0; bi < nb; bi++) {
        memset(RA_BV(ra_blk_def, bi), 0, bv_words * sizeof(uint32_t));
        memset(RA_BV(ra_blk_use, bi), 0, bv_words * sizeof(uint32_t));

        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];

            for (uint8_t u = mi->num_defs;
                 u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t vr = operand_vreg(&mi->operands[u]);
                if (vr != 0xFFFF && vr < nv && !bv_test(RA_BV(ra_blk_def, bi), vr))
                    bv_set(RA_BV(ra_blk_use, bi), vr);
            }
            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t vr = operand_vreg(&mi->operands[d]);
                if (vr != 0xFFFF && vr < nv)
                    bv_set(RA_BV(ra_blk_def, bi), vr);
            }
        }
    }

    for (uint32_t bi = 0; bi < nb; bi++) {
        memset(RA_BV(ra_live_in, bi),  0, bv_words * sizeof(uint32_t));
        memset(RA_BV(ra_live_out, bi), 0, bv_words * sizeof(uint32_t));
    }

    for (int pass = 0; pass < 200; pass++) {
        int changed = 0;

        for (uint32_t bi2 = nb; bi2 > 0; bi2--) {
            uint32_t bi = bi2 - 1;

            for (uint8_t s = 0; s < ra_nsuccs[bi]; s++) {
                uint16_t succ = RA_SUCCS(bi)[s];
                if (succ < nb)
                    changed |= bv_or(RA_BV(ra_live_out, bi), RA_BV(ra_live_in, succ), bv_words);
            }

            for (uint32_t w = 0; w < bv_words; w++) {
                uint32_t new_in = RA_BV(ra_blk_use, bi)[w] |
                                  (RA_BV(ra_live_out, bi)[w] & ~RA_BV(ra_blk_def, bi)[w]);
                if (new_in != RA_BV(ra_live_in, bi)[w]) {
                    RA_BV(ra_live_in, bi)[w] = new_in;
                    changed = 1;
                }
            }
        }

        if (!changed) break;
    }
}

static void ra_blifg(const amd_module_t *A,
                                       const mfunc_t *F,
                                       uint32_t nv)
{
    uint32_t bv_words = ra_bv_words;
    static uint32_t live[RA_BV_MAX_WORDS];
    memset(live, 0, bv_words * sizeof(uint32_t));

    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];

        memcpy(live, RA_BV(ra_live_out, bi), bv_words * sizeof(uint32_t));

        for (uint32_t ii = MB->num_insts; ii > 0; ii--) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii - 1];

            /* For PSEUDO_COPY D = S: don't add D↔S interference even
               though S is live at D's def.  This is what makes coalescing
               possible --without it, copy-related pairs always interfere
               and can never be merged. */
            int is_copy = (mi->op == AMD_PSEUDO_COPY);
            uint16_t copy_src = 0xFFFF;
            if (is_copy && mi->num_uses >= 1)
                copy_src = operand_vreg(&mi->operands[mi->num_defs]);

            for (uint8_t d = 0; d < mi->num_defs && d < MINST_MAX_OPS; d++) {
                uint16_t dv = operand_vreg(&mi->operands[d]);
                if (dv == 0xFFFF || dv >= nv) continue;
                uint16_t dn = ra_vreg_to_node[dv];
                if (dn == 0xFFFF) continue;

                for (uint32_t w = 0; w < bv_words; w++) {
                    uint32_t bits = live[w];
                    while (bits) {
                        uint32_t lsb = bits & (uint32_t)(-(int32_t)bits);
                        uint32_t bit_idx = w * 32;
                        /* CTZ -- portable across MSVC/MinGW */
                        { uint32_t tmp = lsb; while (!(tmp & 1)) { tmp >>= 1; bit_idx++; } }
                        bits &= bits - 1;

                        if (bit_idx >= nv) continue;
                        uint16_t lv = (uint16_t)bit_idx;
                        if (lv == dv) continue;
                        if (is_copy && lv == copy_src) continue;
                        uint16_t ln = ra_vreg_to_node[lv];
                        if (ln == 0xFFFF) continue;

                        /* Only interfere within same register file */
                        if (ra_nodes[dn].file != ra_nodes[ln].file) continue;

                        if (!ra_ifg_test(dn, ln)) {
                            ra_ifg_set(dn, ln);
                            ra_nodes[dn].degree++;
                            ra_nodes[ln].degree++;
                        }
                    }
                }

                bv_clear(live, dv);
            }

            for (uint8_t u = mi->num_defs;
                 u < mi->num_defs + mi->num_uses && u < MINST_MAX_OPS; u++) {
                uint16_t uv = operand_vreg(&mi->operands[u]);
                if (uv != 0xFFFF && uv < nv)
                    bv_set(live, uv);
            }
        }
    }

}

static void ra_gc(amd_module_t *A, uint32_t mf_idx)
{
    mfunc_t *F = &A->mfuncs[mf_idx];
    uint32_t max_iters = 4;
    int gc_success = 0;

    if (F->num_blocks > RA_MAX_BLOCKS) {
        /* Too many blocks --fall back to linear scan */
        ra_lin(A, mf_idx);
        return;
    }

    for (uint32_t iter = 0; iter < max_iters; iter++) {
        uint32_t nv = A->vreg_count;
        if (nv > RA_MAX_NODES) nv = RA_MAX_NODES;

        /* Zero the static IFG bitmatrix and node table for this iteration */
        {
            size_t ifg_words = ((uint64_t)nv * nv + 31) / 32;
            memset(ra_ifg, 0, ifg_words * sizeof(uint32_t));
            memset(ra_nodes, 0, nv * sizeof(ra_node_t));
            ra_stride = nv;
        }

        /* --- Build node table from all vregs used in this function --- */
        ra_num_nodes = 0;
        memset(ra_vreg_to_node, 0xFF, nv * sizeof(uint16_t));

        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                const minst_t *mi = &A->minsts[MB->first_inst + ii];
                uint8_t total = mi->num_defs + mi->num_uses;
                if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;
                for (uint8_t k = 0; k < total; k++) {
                    uint16_t vr = operand_vreg(&mi->operands[k]);
                    if (vr != 0xFFFF && vr < nv) {
                        if (ra_vreg_to_node[vr] == 0xFFFF &&
                            ra_num_nodes < RA_MAX_NODES) {
                            uint16_t nidx = (uint16_t)ra_num_nodes++;
                            ra_nodes[nidx].vreg = vr;
                            ra_nodes[nidx].degree = 0;
                            ra_nodes[nidx].cost = 1;
                            ra_nodes[nidx].color = 0xFFFF;
                            ra_nodes[nidx].alias = nidx;
                            ra_nodes[nidx].file = A->reg_file[vr];
                            ra_nodes[nidx].spilled = 0;
                            ra_nodes[nidx].in_graph = 1;
                            ra_vreg_to_node[vr] = nidx;
                        } else if (ra_vreg_to_node[vr] != 0xFFFF) {
                            ra_nodes[ra_vreg_to_node[vr]].cost++;
                        }
                    }
                }
            }
        }

        if (ra_num_nodes == 0) { gc_success = 1; break; }

        /* --- Zero per-block liveness arrays --- */
        {
            uint32_t nb = F->num_blocks;
            ra_bv_words = (nv + 31) / 32;
            size_t bv_sz = (size_t)nb * ra_bv_words * sizeof(uint32_t);
            memset(ra_live_in,  0, bv_sz);
            memset(ra_live_out, 0, bv_sz);
            memset(ra_blk_def,  0, bv_sz);
            memset(ra_blk_use,  0, bv_sz);
            memset(ra_succs,    0, nb * 2 * sizeof(uint16_t));
            memset(ra_nsuccs,   0, nb * sizeof(uint8_t));
        }

        /* --- Build CFG and compute per-block liveness --- */
        ra_build_cfg(A, F);
        ra_cpliv(A, F, nv);

        /* --- Build interference graph from liveness --- */
        ra_blifg(A, F, nv);

        /* --- Compute K (available physical regs per file) --- */
        uint16_t sgpr_start = F->is_kernel ? F->first_alloc_sgpr : 0;
        if (sgpr_start < AMD_KERN_MIN_RESERVED && F->is_kernel)
            sgpr_start = AMD_KERN_MIN_RESERVED;
        uint32_t K_sgpr = (AMD_MAX_SGPRS > sgpr_start) ? AMD_MAX_SGPRS - sgpr_start : 0;
        uint32_t K_vgpr = (amd_max_vgpr > 0 && amd_max_vgpr < AMD_MAX_VGPRS)
                           ? (uint32_t)amd_max_vgpr : AMD_MAX_VGPRS;

        /* --- Copy coalescing (conservative, Briggs criterion) ---
           Only merge two non-interfering copy-related nodes if the
           resulting node would have fewer than K neighbors with
           degree >= K.  This guarantees the merge can't turn a
           colorable graph into one that needs spills --important on
           GPU where a spill means scratch memory (hundreds of cycles
           vs 1 cycle for a register MOV).
           */
        for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                uint32_t mi_idx = MB->first_inst + ii;
                const minst_t *mi = &A->minsts[mi_idx];

                if (mi->op != AMD_PSEUDO_COPY) continue;
                if (mi->num_defs < 1 || mi->num_uses < 1) continue;

                uint16_t dst_vr = operand_vreg(&mi->operands[0]);
                uint16_t src_vr = operand_vreg(&mi->operands[1]);
                if (dst_vr == 0xFFFF || src_vr == 0xFFFF) continue;

                uint16_t dn = ra_vreg_to_node[dst_vr];
                uint16_t sn = ra_vreg_to_node[src_vr];
                if (dn == 0xFFFF || sn == 0xFFFF) continue;

                dn = ra_find(dn);
                sn = ra_find(sn);
                if (dn == sn) continue;

                if (ra_nodes[dn].file != ra_nodes[sn].file) continue;
                if (ra_ifg_test(dn, sn)) continue;

                /* Briggs test: count neighbors of the merged node that
                   would have degree >= K.  A neighbor of the merged node
                   is anything adjacent to a OR b (de-duped). */
                uint32_t K = (ra_nodes[dn].file == 0) ? K_sgpr : K_vgpr;
                uint32_t high_deg = 0;
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (ra_find((uint16_t)m) != m) continue;
                    if (m == (uint32_t)dn || m == (uint32_t)sn) continue;
                    if (!ra_nodes[m].in_graph) continue;
                    if (ra_nodes[m].file != ra_nodes[dn].file) continue;
                    if (!ra_ifg_test(dn, m) && !ra_ifg_test(sn, m)) continue;
                    uint32_t mdeg = ra_nodes[m].degree;
                    if (ra_ifg_test(dn, m) && ra_ifg_test(sn, m) && mdeg > 0)
                        mdeg--;
                    if (mdeg >= K) high_deg++;
                }

                if (high_deg < K)
                    ra_coalesce(dn, sn);
            }
        }

        /* --- Recompute degrees after coalescing (only for representatives) --- */
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            if (ra_find((uint16_t)n) != n) { ra_nodes[n].in_graph = 0; continue; }
            if (!ra_nodes[n].in_graph) continue;
            uint32_t deg = 0;
            for (uint32_t m = 0; m < ra_num_nodes; m++) {
                if (m == n) continue;
                if (ra_find((uint16_t)m) != m) continue;
                if (!ra_nodes[m].in_graph) continue;
                if (ra_ifg_test(n, m) && ra_nodes[m].file == ra_nodes[n].file)
                    deg++;
            }
            ra_nodes[n].degree = deg;
        }

        /* --- Simplify: repeatedly remove nodes with degree < K --- */
        ra_stack_top = 0;
        uint32_t nodes_left = 0;
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            if (ra_nodes[n].in_graph) nodes_left++;
        }

        for (uint32_t guard = 0; guard < RA_MAX_NODES && nodes_left > 0; guard++) {
            int found = 0;
            for (uint32_t n = 0; n < ra_num_nodes; n++) {
                if (!ra_nodes[n].in_graph) continue;
                uint32_t K = (ra_nodes[n].file == 0) ? K_sgpr : K_vgpr;
                if (ra_nodes[n].degree < K) {
                    ra_nodes[n].in_graph = 0;
                    ra_stack[ra_stack_top++] = n;
                    nodes_left--;
                    for (uint32_t m = 0; m < ra_num_nodes; m++) {
                        if (!ra_nodes[m].in_graph) continue;
                        if (ra_ifg_test(n, m) &&
                            ra_nodes[m].file == ra_nodes[n].file &&
                            ra_nodes[m].degree > 0)
                            ra_nodes[m].degree--;
                    }
                    found = 1;
                    break;
                }
            }

            if (!found) {
                /* No simplify candidate --pick lowest cost/degree
                   as potential spill (optimistic: it might still color).
                   Nodes with many uses are expensive to spill; nodes with
                   high degree free the most neighbors when removed.
                   Compare via cross-multiply to stay integer:
                     cost_n/(deg_n+1) < cost_best/(deg_best+1)
                   ⟺ cost_n*(deg_best+1) < cost_best*(deg_n+1)  */
                uint32_t best = 0;
                uint32_t best_cost = 0, best_deg = 0;
                int have_best = 0;
                for (uint32_t n = 0; n < ra_num_nodes; n++) {
                    if (!ra_nodes[n].in_graph) continue;
                    uint32_t cn = ra_nodes[n].cost;
                    uint32_t dn = ra_nodes[n].degree + 1;
                    if (!have_best || cn * (best_deg + 1) < best_cost * dn) {
                        best = n;
                        best_cost = cn;
                        best_deg = ra_nodes[n].degree;
                        have_best = 1;
                    }
                }
                if (!have_best) break;
                ra_nodes[best].in_graph = 0;
                ra_stack[ra_stack_top++] = best;
                nodes_left--;
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_nodes[m].in_graph) continue;
                    if (ra_ifg_test(best, m) &&
                        ra_nodes[m].file == ra_nodes[best].file &&
                        ra_nodes[m].degree > 0)
                        ra_nodes[m].degree--;
                }
            }
        }

        /* --- Select: pop stack, assign colors --- */
        uint16_t max_sgpr = 0, max_vgpr = 0;
        int any_spilled = 0;

        /* Bitvectors for used colors among neighbors */
        static uint32_t used_sgpr[(AMD_MAX_SGPRS + 31) / 32];
        static uint32_t used_vgpr[(AMD_MAX_VGPRS + 31) / 32];

        while (ra_stack_top > 0) {
            uint32_t n = ra_stack[--ra_stack_top];
            uint16_t rep = ra_find((uint16_t)n);
            if (rep != n) {
                /* Coalesced away --will get color from representative */
                continue;
            }

            /* Collect colors used by already-colored neighbors.
               Check interference against original node indices (what the
               matrix was built with), then resolve to the representative
               to get the actual assigned color.  An original edge n-m
               means n interferes with whatever m got coalesced into. */
            if (ra_nodes[n].file == 0) {
                memset(used_sgpr, 0, sizeof(used_sgpr));
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_ifg_test(n, m)) continue;
                    uint16_t mr = ra_find((uint16_t)m);
                    if (mr == (uint16_t)n) continue; /* coalesced into us */
                    if (ra_nodes[mr].color != 0xFFFF && ra_nodes[mr].file == 0) {
                        uint16_t c = ra_nodes[mr].color;
                        used_sgpr[c / 32] |= 1u << (c % 32);
                    }
                }
                uint16_t picked = 0xFFFF;
                for (uint16_t r = sgpr_start; r < AMD_MAX_SGPRS; r++) {
                    if (!(used_sgpr[r / 32] & (1u << (r % 32)))) {
                        picked = r;
                        break;
                    }
                }
                if (picked != 0xFFFF) {
                    ra_nodes[n].color = picked;
                    if (picked + 1 > max_sgpr) max_sgpr = picked + 1;
                } else {
                    ra_nodes[n].spilled = 1;
                    any_spilled = 1;
                }
            } else {
                memset(used_vgpr, 0, sizeof(used_vgpr));
                for (uint32_t m = 0; m < ra_num_nodes; m++) {
                    if (!ra_ifg_test(n, m)) continue;
                    uint16_t mr = ra_find((uint16_t)m);
                    if (mr == (uint16_t)n) continue;
                    if (ra_nodes[mr].color != 0xFFFF && ra_nodes[mr].file == 1) {
                        uint16_t c = ra_nodes[mr].color;
                        used_vgpr[c / 32] |= 1u << (c % 32);
                    }
                }
                uint16_t picked = 0xFFFF;
                for (uint16_t r = 0; r < AMD_MAX_VGPRS; r++) {
                    if (!(used_vgpr[r / 32] & (1u << (r % 32)))) {
                        picked = r;
                        break;
                    }
                }
                if (picked != 0xFFFF) {
                    ra_nodes[n].color = picked;
                    if (picked + 1 > max_vgpr) max_vgpr = picked + 1;
                } else {
                    ra_nodes[n].spilled = 1;
                    any_spilled = 1;
                }
            }
        }

        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            uint16_t rep = ra_find((uint16_t)n);
            if (rep != n) {
                ra_nodes[n].color = ra_nodes[rep].color;
                ra_nodes[n].spilled = ra_nodes[rep].spilled;
            }
        }

        /* --- Spill code insertion ---
           For each block, expand instructions into a full-function output
           buffer, inserting scratch_load before uses and scratch_store after
           defs of spilled vregs.  Then shift the tail once and copy all
           expanded blocks back in a single pass. */
        if (any_spilled) {
            static uint32_t spill_vreg[RA_MAX_NODES];
            static uint8_t  spill_file[RA_MAX_NODES];
            static uint32_t spill_off[RA_MAX_NODES];
            uint32_t num_spills = 0;

            for (uint32_t n = 0; n < ra_num_nodes; n++) {
                if (!ra_nodes[n].spilled) continue;
                if (num_spills >= RA_MAX_NODES) break;
                spill_vreg[num_spills] = ra_nodes[n].vreg;
                spill_file[num_spills] = ra_nodes[n].file;
                spill_off[num_spills] = F->scratch_bytes;
                F->scratch_bytes += 4;
                num_spills++;
            }

            /* Static output buffer for spill expansion --sized to
               AMD_MAX_MINSTS (the hard ceiling on total instructions). */
            static minst_t  ra_output[AMD_MAX_MINSTS];
            static uint32_t blk_out_start[RA_MAX_BLOCKS];
            static uint32_t blk_out_count[RA_MAX_BLOCKS];
            uint32_t out_cap = AMD_MAX_MINSTS;
            uint32_t out_count = 0;

            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++) {
                mblock_t *MB = &A->mblocks[F->first_block + bi2];
                blk_out_start[bi2] = out_count;

                for (uint32_t ii2 = 0; ii2 < MB->num_insts; ii2++) {
                    minst_t orig = A->minsts[MB->first_inst + ii2];

                    for (uint32_t si = 0; si < num_spills; si++) {
                        uint32_t sv = spill_vreg[si];
                        int has_use = 0;
                        for (uint8_t u = orig.num_defs;
                             u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                            if (operand_vreg(&orig.operands[u]) == sv) has_use = 1;
                        }
                        if (has_use) {
                            uint32_t new_vr = A->vreg_count;
                            if (new_vr >= AMD_MAX_VREGS) continue;
                            A->vreg_count++;

                            if (spill_file[si] == 0 &&
                                A->vreg_count < AMD_MAX_VREGS) {
                                /* SGPR spill reload: scratch is vector memory,
                                   so load into a temp VGPR then readfirstlane
                                   into the SGPR.
                                   scratch_load_dword vTmp, off, spill_off
                                   v_readfirstlane_b32 sNew, vTmp */
                                uint32_t vtmp = A->vreg_count;
                                A->reg_file[vtmp] = 1; /* VGPR */
                                A->vreg_count++;

                                A->reg_file[new_vr] = 0; /* SGPR */

                                for (uint8_t u = orig.num_defs;
                                     u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                                    if (operand_vreg(&orig.operands[u]) == sv)
                                        orig.operands[u].reg_num = (uint16_t)new_vr;
                                }

                                /* scratch_load_dword vTmp */
                                if (out_count < out_cap) {
                                    minst_t *ld = &ra_output[out_count++];
                                    memset(ld, 0, sizeof(minst_t));
                                    ld->op = AMD_SCRATCH_LOAD_DWORD;
                                    ld->num_defs = 1;
                                    ld->num_uses = 2;
                                    ld->operands[0].kind = MOP_VREG_V;
                                    ld->operands[0].reg_num = (uint16_t)vtmp;
                                    ld->operands[1].kind = MOP_IMM;
                                    ld->operands[1].imm = 0;
                                    ld->operands[2].kind = MOP_IMM;
                                    ld->operands[2].imm = (int32_t)spill_off[si];
                                }
                                /* v_readfirstlane_b32 sNew, vTmp */
                                if (out_count < out_cap) {
                                    minst_t *rfl = &ra_output[out_count++];
                                    memset(rfl, 0, sizeof(minst_t));
                                    rfl->op = AMD_V_READFIRSTLANE_B32;
                                    rfl->num_defs = 1;
                                    rfl->num_uses = 1;
                                    rfl->operands[0].kind = MOP_VREG_S;
                                    rfl->operands[0].reg_num = (uint16_t)new_vr;
                                    rfl->operands[1].kind = MOP_VREG_V;
                                    rfl->operands[1].reg_num = (uint16_t)vtmp;
                                }
                            } else {
                                A->reg_file[new_vr] = 1; /* VGPR */

                                for (uint8_t u = orig.num_defs;
                                     u < orig.num_defs + orig.num_uses && u < MINST_MAX_OPS; u++) {
                                    if (operand_vreg(&orig.operands[u]) == sv)
                                        orig.operands[u].reg_num = (uint16_t)new_vr;
                                }

                                if (out_count < out_cap) {
                                    minst_t *ld = &ra_output[out_count++];
                                    memset(ld, 0, sizeof(minst_t));
                                    ld->op = AMD_SCRATCH_LOAD_DWORD;
                                    ld->num_defs = 1;
                                    ld->num_uses = 2;
                                    ld->operands[0].kind = MOP_VREG_V;
                                    ld->operands[0].reg_num = (uint16_t)new_vr;
                                    ld->operands[1].kind = MOP_IMM;
                                    ld->operands[1].imm = 0;
                                    ld->operands[2].kind = MOP_IMM;
                                    ld->operands[2].imm = (int32_t)spill_off[si];
                                }
                            }
                        }
                    }

                    if (out_count < out_cap)
                        ra_output[out_count++] = orig;

                    for (uint32_t si = 0; si < num_spills; si++) {
                        uint32_t sv = spill_vreg[si];
                        int has_def = 0;
                        for (uint8_t d = 0; d < orig.num_defs && d < MINST_MAX_OPS; d++) {
                            if (operand_vreg(&orig.operands[d]) == sv) has_def = 1;
                        }
                        if (has_def) {
                            if (spill_file[si] == 0) {
                                /* SGPR spill store: move SGPR into a temp
                                   VGPR, then scratch_store from it.
                                   v_mov_b32 vTmp, sDef
                                   scratch_store_dword vOff, vTmp, spill_off */
                                uint32_t vtmp = A->vreg_count;
                                if (vtmp >= AMD_MAX_VREGS) continue;
                                A->reg_file[vtmp] = 1; /* VGPR */
                                A->vreg_count++;

                                /* v_mov_b32 vTmp, sDef */
                                if (out_count < out_cap) {
                                    minst_t *mv = &ra_output[out_count++];
                                    memset(mv, 0, sizeof(minst_t));
                                    mv->op = AMD_V_MOV_B32;
                                    mv->num_defs = 1;
                                    mv->num_uses = 1;
                                    mv->operands[0].kind = MOP_VREG_V;
                                    mv->operands[0].reg_num = (uint16_t)vtmp;
                                    mv->operands[1].kind = MOP_VREG_S;
                                    mv->operands[1].reg_num = (uint16_t)sv;
                                }
                                /* scratch_store_dword vOff, vTmp, off */
                                if (out_count < out_cap) {
                                    minst_t *st = &ra_output[out_count++];
                                    memset(st, 0, sizeof(minst_t));
                                    st->op = AMD_SCRATCH_STORE_DWORD;
                                    st->num_defs = 0;
                                    st->num_uses = 3;
                                    st->operands[0].kind = MOP_IMM;
                                    st->operands[0].imm = 0;
                                    st->operands[1].kind = MOP_VREG_V;
                                    st->operands[1].reg_num = (uint16_t)vtmp;
                                    st->operands[2].kind = MOP_IMM;
                                    st->operands[2].imm = (int32_t)spill_off[si];
                                }
                            } else {
                                if (out_count < out_cap) {
                                    minst_t *st = &ra_output[out_count++];
                                    memset(st, 0, sizeof(minst_t));
                                    st->op = AMD_SCRATCH_STORE_DWORD;
                                    st->num_defs = 0;
                                    st->num_uses = 3;
                                    st->operands[0].kind = MOP_IMM;
                                    st->operands[0].imm = 0;
                                    st->operands[1].kind = MOP_VREG_V;
                                    st->operands[1].reg_num = (uint16_t)sv;
                                    st->operands[2].kind = MOP_IMM;
                                    st->operands[2].imm = (int32_t)spill_off[si];
                                }
                            }
                        }
                    }
                }

                blk_out_count[bi2] = out_count - blk_out_start[bi2];
            }

            /* Single bulk-copy: compute total delta, shift tail once,
               then copy all expanded blocks into the instruction array. */
            uint32_t old_total = 0;
            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++)
                old_total += A->mblocks[F->first_block + bi2].num_insts;

            int32_t total_delta = (int32_t)out_count - (int32_t)old_total;
            uint32_t func_end = A->mblocks[F->first_block].first_inst + old_total;
            uint32_t tail = A->num_minsts - func_end;

            if (total_delta != 0 &&
                A->num_minsts + (uint32_t)total_delta <= AMD_MAX_MINSTS) {
                if (tail > 0) {
                    memmove(&A->minsts[(uint32_t)((int32_t)func_end + total_delta)],
                            &A->minsts[func_end],
                            tail * sizeof(minst_t));
                }
                A->num_minsts = (uint32_t)((int32_t)A->num_minsts + total_delta);
            }

            uint32_t write_pos = A->mblocks[F->first_block].first_inst;
            for (uint32_t bi2 = 0; bi2 < F->num_blocks; bi2++) {
                mblock_t *MB2 = &A->mblocks[F->first_block + bi2];
                MB2->first_inst = write_pos;
                MB2->num_insts = blk_out_count[bi2];
                memcpy(&A->minsts[write_pos],
                       &ra_output[blk_out_start[bi2]],
                       blk_out_count[bi2] * sizeof(minst_t));
                write_pos += blk_out_count[bi2];
            }

            for (uint32_t later = F->first_block + F->num_blocks;
                 later < A->num_mblocks; later++)
                A->mblocks[later].first_inst =
                    (uint32_t)((int32_t)A->mblocks[later].first_inst + total_delta);

            continue;
        }

        /* --- Write reg_map from coloring --- */
        for (uint32_t n = 0; n < ra_num_nodes; n++) {
            uint32_t vreg = ra_nodes[n].vreg;
            uint16_t color = ra_nodes[n].color;
            if (color != 0xFFFF && vreg < AMD_MAX_VREGS)
                A->reg_map[vreg] = color;
            else if (vreg < AMD_MAX_VREGS)
                A->reg_map[vreg] = 0;
        }

        F->num_sgprs = max_sgpr;
        F->num_vgprs = max_vgpr;
        gc_success = 1;
        break; /* success, no spills */
    }

    if (!gc_success) {
        /* Spill iterations exhausted --fall back to linear scan */
        ra_lin(A, mf_idx);
        return;
    }

    fin_regs(A, F);
    rw_ops(A, F);
    dce_copy(A, F);
}

/* Global flag: set by --no-graphcolor to force linear scan */
int amd_ra_lin = 0;
/* SSA-based divergence-aware allocator */
int amd_ra_ssa = 0;
/* If non-zero, cap available VGPRs for regalloc (forces spills for testing) */
int amd_max_vgpr = 0;

static void ra_func(amd_module_t *A, uint32_t mf_idx)
{
    if (amd_ra_ssa) {
        ra_ssa(A, mf_idx);
    } else if (amd_ra_lin || A->vreg_count > RA_MAX_NODES) {
        ra_lin(A, mf_idx);
    } else {
        ra_gc(A, mf_idx);
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
        case AMD_SPEC_SCC:       asm_append(A, "scc"); break;
        case AMD_SPEC_M0:        asm_append(A, "m0"); break;
        case AMD_SPEC_PRIV_BASE: asm_append(A, "src_private_base"); break;
        default:                 asm_append(A, "???"); break;
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
    case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR: case AMD_FMT_FLAT: {
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
    /* SSA path does its own phi elimination post-allocation */
    if (!amd_ra_ssa)
        amdgpu_phi_elim(A);
    for (uint32_t fi = 0; fi < A->num_mfuncs; fi++)
        ra_func(A, fi);
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

/* array16 format for counts > 15 (fixarray only handles 0-15) */
static void mp_array(uint8_t *buf, uint32_t *pos, uint32_t count)
{
    if (count <= 15) {
        mp_fixarray(buf, pos, (uint8_t)count);
    } else {
        if (*pos + 3 > MP_BUF_MAX) return;
        buf[(*pos)++] = 0xDC;
        buf[(*pos)++] = (uint8_t)(count >> 8);
        buf[(*pos)++] = (uint8_t)(count);
    }
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

        /* compute_pgm_rsrc1 — VGPR gran=8 (GFX90A+).
         * SGPR encoding gran=8 (ALL GFX9), alloc gran=16 from pool.
         * GFX10+ ignores the SGPR field entirely.
         * GFX9/CDNA: VCC, FLAT_SCRATCH, XNACK_MASK are carved from the
         * RSRC1 SGPR allocation, from the top down.  The block count
         * must cover user SGPRs + 6 (VCC=2, FLAT_SCRATCH=2, XNACK=2)
         * or the CP silently aliases them with your live registers.
         * That's how you get aperture faults from perfectly valid code. */
        uint32_t vgpr_blocks = (F->num_vgprs > 0)
            ? (uint32_t)((F->num_vgprs + 7) / 8 - 1) : 0;
        /* GFX9 SGPR encoding granularity is 8, NOT 16.  The ALLOCATION
         * granule from the physical pool is 16, but the RSRC1 field
         * encodes in units of 8.  LLVM getSGPREncodingGranule() = 8.
         * Using 16 here causes the hardware to see fewer SGPRs than
         * we actually write, aliasing user regs with VCC/FLAT_SCRATCH. */
        uint32_t sgpr_gran = 8u;
        uint32_t total_sgprs = F->num_sgprs + F->imp_sgp;
        uint32_t sgpr_blocks = (total_sgprs > 0)
            ? (uint32_t)((total_sgprs + sgpr_gran - 1) / sgpr_gran - 1) : 0;
        if (F->sgp_min && sgpr_blocks < F->sgp_min)
            sgpr_blocks = F->sgp_min;
        kd.compute_pgm_rsrc1 = (vgpr_blocks & 0x3F) |
                               ((sgpr_blocks & 0xF) << 6) |
                               F->r1_mode;

        /* compute_pgm_rsrc2 — [0] SCRATCH_EN, [5:1] USER_SGPR_COUNT,
           [7] TGID_X, [8] TGID_Y, [9] TGID_Z, [12:11] VGPR_WORKITEM_ID.
           Layout matches what isel's scan_kernel_needs() decided. */
        {
            uint32_t user_sgpr = 2u; /* s[0:1] = kernarg only */
            uint32_t rsrc2 = ((F->scratch_bytes > 0) ? 1u : 0u) |
                             (user_sgpr << 1) |
                             (1u << 7);       /* TGID_X always enabled */
            if (F->max_dim >= 1) rsrc2 |= (1u << 8);  /* TGID_Y */
            if (F->max_dim >= 2) rsrc2 |= (1u << 9);  /* TGID_Z */
            rsrc2 |= ((uint32_t)F->max_dim << 11);    /* VGPR_WORKITEM_ID */
            kd.compute_pgm_rsrc2 = rsrc2;
        }

        /* compute_pgm_rsrc3 — ACCUM_OFFSET for CDNA (GFX90A/GFX942).
         * Tells the HW where ArchVGPRs end and AccVGPRs begin.
         * GFX942 unified VGPRs: all are ArchVGPR, so offset = vgpr_blocks. */
        if (F->exec_w) {
            uint32_t ao_gran = (A->target == AMD_TARGET_GFX942) ? 8u : 4u;
            uint32_t accum_off = (F->num_vgprs > 0)
                ? (uint32_t)((F->num_vgprs + ao_gran - 1) / ao_gran - 1) : 0;
            kd.compute_pgm_rsrc3 = accum_off & 0x3F;
        }

        /* kernel_code_properties */
        kd.kernel_code_properties = (1u << 3);   /* ENABLE_SGPR_KERNARG_PTR */
        /* Bit 0 = ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER — shifts SGPR
         * layout on GFX9. SCRATCH_EN (RSRC2) + private_segment_fixed_size
         * handle scratch allocation. Tested: bit 0 ON did not fix
         * the y=1.0 scratch bug and may cause SGPR shift. */

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

    /* ---- Build .debug_bc section ----
     * Maps code offsets to source lines. Currently we emit what we know,
     * which is per-instruction byte offsets. Line numbers are zero until
     * the frontend grows source location tracking. But the section exists,
     * the format is defined, and the ABEND dump machinery can read it.
     * Patience, grasshopper. */
    static uint8_t dbcbuf[65536];
    uint32_t dbc_len = 0;

    {
        /* Count total instructions across all functions */
        uint32_t n_ent = 0;
        for (uint32_t fi = 0; fi < A->num_mfuncs && n_ent < 8000; fi++) {
            const mfunc_t *F = &A->mfuncs[fi];
            for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
                uint32_t mb_idx = F->first_block + bi;
                if (mb_idx >= AMD_MAX_MBLOCKS) break;
                n_ent += A->mblocks[mb_idx].num_insts;
            }
        }
        if (n_ent > 8000) n_ent = 8000;

        /* Header: "BCDB" + count */
        if (8 + n_ent * 8 <= sizeof(dbcbuf)) {
            dbcbuf[0] = 'B'; dbcbuf[1] = 'C';
            dbcbuf[2] = 'D'; dbcbuf[3] = 'B';
            memcpy(dbcbuf + 4, &n_ent, 4);
            dbc_len = 8;

            /* Entries: offset + line (line=0 for now) */
            uint32_t ei = 0;
            for (uint32_t fi = 0; fi < A->num_mfuncs && ei < n_ent; fi++) {
                const mfunc_t *F2 = &A->mfuncs[fi];
                for (uint32_t bi = 0; bi < F2->num_blocks && ei < n_ent; bi++) {
                    uint32_t mb_idx = F2->first_block + bi;
                    if (mb_idx >= AMD_MAX_MBLOCKS) break;
                    const mblock_t *MB = &A->mblocks[mb_idx];
                    for (uint32_t ii = 0; ii < MB->num_insts && ei < n_ent; ii++) {
                        uint32_t mi_idx = MB->first_inst + ii;
                        if (mi_idx >= AMD_MAX_MINSTS) break;
                        uint32_t off = A->inst_off[mi_idx];
                        uint32_t ln  = 0; /* no source lines yet */
                        memcpy(dbcbuf + dbc_len,     &off, 4);
                        memcpy(dbcbuf + dbc_len + 4, &ln,  4);
                        dbc_len += 8;
                        ei++;
                    }
                }
            }
        }
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
            /* 6 hidden args for block_count + group_size if needed */
            uint32_t n_hidden = F->needs_dispatch ? 6 : 0;
            mp_array(mp_buf, &mp_pos, np + n_hidden);
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
     *           5=.rodata 6=.text 7=.dynamic 8=.symtab 9=.strtab
     *           10=.shstrtab 11=.debug_bc
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
    uint32_t sn_dbgbc;
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
    SHSTR(sn_dbgbc,    ".debug_bc");
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
    uint64_t dbc_off   = (str_off + strtab_len + 3) & ~3ULL;
    uint64_t shs_off   = dbc_off + dbc_len;
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
    ehdr.e_ident[8] = 4;    /* ABI version 4 — code object v6 */
    ehdr.e_type = 3;         /* ET_DYN */
    ehdr.e_machine = EM_AMDGPU;
    ehdr.e_version = 1;
    ehdr.e_phoff = phdr_off;
    ehdr.e_shoff = shdr_off;
    ehdr.e_flags = A->elf_mach;
    ehdr.e_ehsize = 64;
    ehdr.e_phentsize = 56;
    ehdr.e_phnum = N_PHDR;
    ehdr.e_shentsize = 64;
    ehdr.e_shnum = 12;      /* +.rodata +.debug_bc */
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

    /* .debug_bc */
    fwrite_pad(fp, 4);
    if (dbc_len > 0)
        fwrite(dbcbuf, 1, dbc_len, fp);

    /* .shstrtab */
    fwrite(shstrtab, 1, shstrtab_len, fp);
    fwrite_pad(fp, 8);

    /* ---- Section header table ----
     * 0=NULL 1=.note 2=.dynsym 3=.hash 4=.dynstr 5=.rodata
     * 6=.text 7=.dynamic 8=.symtab 9=.strtab 10=.shstrtab 11=.debug_bc */
    elf64_shdr_t shdrs[12];
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

    /* 11: .debug_bc (non-loaded debug section) */
    shdrs[11].sh_name  = sn_dbgbc;
    shdrs[11].sh_type  = SHT_PROGBITS;
    shdrs[11].sh_offset = dbc_off;
    shdrs[11].sh_size  = dbc_len;
    shdrs[11].sh_addralign = 4;

    fwrite(shdrs, 64, 12, fp);

    fclose(fp);

    fprintf(stderr, "wrote %s (%u bytes code, %u kernels)\n",
            path, A->code_len, num_kernels);
    return BC_OK;
}
