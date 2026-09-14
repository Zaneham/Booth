/* sched.c -- instruction scheduler for AMDGPU backend
 *
 * Per-block list scheduling that reorders instructions to hide memory
 * latency. Strips wait instructions, builds a dependency DAG, schedules
 * by critical-path priority, then re-inserts waits just before the
 * first consumer of each load.
 *
 * Runs on virtual registers (between isel and regalloc), so dependency
 * tracking is straightforward and regalloc is unaffected.
 */

#include "sched.h"
#include <string.h>

/* ---- Constants ---- */

#define SCHED_MAX_DEPS      32
#define SCHED_MAX_BLOCK     4096   /* max instructions per block we schedule */
#define SCHED_MAX_BARRIERS  64
#define SCHED_LATENCY_VMEM  80
#define SCHED_LATENCY_SMEM  20
#define SCHED_LATENCY_DS    4
#define SCHED_LATENCY_ALU   1
#define SCHED_MAX_SMEM_FLY  10     /* GFX942: >11 outstanding SMEM → fault */

/* Wait kind tags */
#define WAIT_VMEM  0
#define WAIT_SMEM  1
#define WAIT_DS    2

/* Special register tracking: stored at top of last_def/last_use arrays.
 * We reserve 4 slots for VCC, EXEC, SCC, M0. */
#define SPEC_KEY(sid) ((uint32_t)(AMD_MAX_VREGS - 4 + (uint32_t)(sid)))

/* Physical register tracking: offset into last_def/last_use arrays.
 * Physical SGPRs/VGPRs in the pre-regalloc IR (kernel params, saddr
 * pairs, etc.) need dependency tracking just like virtual registers. */
#define PHYS_SGPR_KEY(r) ((uint32_t)(AMD_MAX_VREGS - 4 - 512 + (uint32_t)(r)))
#define PHYS_VGPR_KEY(r) ((uint32_t)(AMD_MAX_VREGS - 4 - 256 + (uint32_t)(r)))

/* ---- DAG node ---- */

typedef struct {
    uint32_t num_preds;              /* unscheduled predecessor count */
    uint32_t latency;
    uint32_t priority;               /* critical path to block exit */
    uint16_t succs[SCHED_MAX_DEPS];
    uint16_t epoch;                  /* scheduling region (split at barriers) */
    uint8_t  num_succs;
    uint8_t  is_pinned_start;        /* pseudo-ops: pin at block start */
    uint8_t  is_pinned_end;          /* terminators: pin at block end */
    uint8_t  is_load;                /* produces result from memory */
    uint8_t  wait_kind;              /* WAIT_VMEM / WAIT_SMEM / WAIT_DS */
    uint8_t  is_store;               /* memory store */
    uint8_t  is_barrier;             /* s_barrier: full fence */
    uint8_t  deps_overflow;          /* too many deps, pin in place */
} sched_node_t;

/* ---- Scheduler state (static to avoid stack issues) ---- */

static sched_node_t  s_nodes[SCHED_MAX_BLOCK];
static minst_t       s_scheduled[SCHED_MAX_BLOCK];
static uint16_t      s_order[SCHED_MAX_BLOCK];
static uint16_t      s_last_def[AMD_MAX_VREGS];
static uint16_t      s_last_use[AMD_MAX_VREGS];
static uint16_t      s_vreg_load[AMD_MAX_VREGS];
static minst_t       s_stripped[SCHED_MAX_BLOCK];
static minst_t       s_final[SCHED_MAX_BLOCK * 2];
static uint16_t      s_ready[SCHED_MAX_BLOCK];
static uint8_t       s_load_waited[SCHED_MAX_BLOCK];
static uint16_t      s_barrier_pos[SCHED_MAX_BARRIERS];
static uint16_t      s_num_barriers;
static minst_t       s_output[AMD_MAX_MINSTS];

/* Per-VGPR VMEM pending-write tracker.  Turns out "is this
 * a load" is the wrong question when 200 of them are in the
 * post simultaneously.  The right question is "has this
 * particular register arrived yet."  Wisdom is retrospective. */
static uint8_t       s_vmem_pending[AMD_MAX_VREGS];
static uint8_t       s_has_vmem_store;  /* VMEM store in flight */

/* ---- Instruction classification ---- */

static int is_wait_op(uint16_t op)
{
    return op == AMD_S_WAITCNT ||
           op == AMD_S_WAIT_LOADCNT ||
           op == AMD_S_WAIT_STORECNT ||
           op == AMD_S_WAIT_DSCNT ||
           op == AMD_S_WAIT_KMCNT;
}

static int is_terminator(uint16_t op)
{
    return op == AMD_S_BRANCH || op == AMD_S_CBRANCH_SCC0 ||
           op == AMD_S_CBRANCH_SCC1 || op == AMD_S_CBRANCH_EXECZ ||
           op == AMD_S_CBRANCH_EXECNZ || op == AMD_S_ENDPGM ||
           op == AMD_S_SETPC_B64;
}

static int is_pseudo_start(uint16_t op)
{
    return op == AMD_PSEUDO_PHI || op == AMD_PSEUDO_COPY ||
           op == AMD_PSEUDO_DEF;
}

/* Atomic opcodes must stay contiguous in amd_op_t.  If you
 * rearrange the enum the range check quietly stops working
 * and you get to explain to someone why their atomics return
 * feelings instead of values.  This catches it at compile time. */
typedef char sched_assert_atomics_contiguous_[
    (AMD_GLOBAL_ATOMIC_CMPSWAP - AMD_GLOBAL_ATOMIC_ADD == 8) ? 1 : -1
];

static int is_vmem_write(uint16_t op)
{
    /* Anything that writes a VGPR through the VMEM pipe.
     * Classified by pipeline behaviour, not by an opcode list
     * that someone has to remember to update.  Memory is short
     * and schedules are long. */
    if (op == AMD_GLOBAL_LOAD_DWORD || op == AMD_GLOBAL_LOAD_DWORDX2)
        return 1;
    if (op == AMD_SCRATCH_LOAD_DWORD || op == AMD_FLAT_LOAD_DWORD)
        return 1;
    /* Atomic-with-return: contiguous range (guarded above) */
    if (op >= AMD_GLOBAL_ATOMIC_ADD && op <= AMD_GLOBAL_ATOMIC_CMPSWAP)
        return 1;
    return 0;
}

static int is_vmem_store(uint16_t op)
{
    /* Stores don't write VGPRs but they do occupy the VMEM pipe.
     * A load that arrives before the store commits reads whatever
     * was there previously, which is never what you wanted. */
    return op == AMD_GLOBAL_STORE_DWORD || op == AMD_GLOBAL_STORE_DWORDX2 ||
           op == AMD_SCRATCH_STORE_DWORD || op == AMD_FLAT_STORE_DWORD;
}

static int is_scalar_load(uint16_t op)
{
    return op == AMD_S_LOAD_DWORD || op == AMD_S_LOAD_DWORDX2 ||
           op == AMD_S_LOAD_DWORDX4;
}

static int is_ds_load(uint16_t op)
{
    return op == AMD_DS_READ_B32 || op == AMD_DS_ADD_RTN_U32 ||
           op == AMD_DS_SUB_RTN_U32 || op == AMD_DS_AND_RTN_B32 ||
           op == AMD_DS_OR_RTN_B32 || op == AMD_DS_XOR_RTN_B32 ||
           op == AMD_DS_MIN_RTN_I32 || op == AMD_DS_MAX_RTN_I32 ||
           op == AMD_DS_SWIZZLE_B32 || op == AMD_DS_BPERMUTE_B32;
}

static int is_store(uint16_t op)
{
    return op == AMD_GLOBAL_STORE_DWORD || op == AMD_GLOBAL_STORE_DWORDX2 ||
           op == AMD_SCRATCH_STORE_DWORD || op == AMD_DS_WRITE_B32 ||
           op == AMD_FLAT_STORE_DWORD;
}

/* SOP2 instructions that implicitly set SCC */
static int sop2_writes_scc(uint16_t op)
{
    return op == AMD_S_ADD_U32  || op == AMD_S_SUB_U32  ||
           op == AMD_S_AND_B32  || op == AMD_S_OR_B32   ||
           op == AMD_S_XOR_B32  || op == AMD_S_LSHL_B32 ||
           op == AMD_S_LSHR_B32 || op == AMD_S_ASHR_I32 ||
           op == AMD_S_ANDN2_B32 || op == AMD_S_ORN2_B32 ||
           op == AMD_S_BFE_I32  ||
           op == AMD_S_AND_B64  || op == AMD_S_OR_B64   ||
           op == AMD_S_XOR_B64  || op == AMD_S_ANDN2_B64 ||
           op == AMD_S_ORN2_B64;
}

static uint32_t inst_latency(uint16_t op)
{
    if (is_vmem_write(op))   return SCHED_LATENCY_VMEM;
    if (is_scalar_load(op))  return SCHED_LATENCY_SMEM;
    if (is_ds_load(op))      return SCHED_LATENCY_DS;
    return SCHED_LATENCY_ALU;
}

static uint8_t load_wait_kind(uint16_t op)
{
    if (is_vmem_write(op))   return WAIT_VMEM;
    if (is_scalar_load(op))  return WAIT_SMEM;
    return WAIT_DS;
}

/* ---- Operand helpers ---- */

static int is_trackable(const moperand_t *op)
{
    return op->kind == MOP_VREG_S || op->kind == MOP_VREG_V ||
           op->kind == MOP_SGPR   || op->kind == MOP_VGPR   ||
           op->kind == MOP_SPECIAL;
}

static uint32_t op_key(const moperand_t *op)
{
    switch (op->kind) {
    case MOP_VREG_S: case MOP_VREG_V: return (uint32_t)op->reg_num;
    case MOP_SGPR:   return PHYS_SGPR_KEY(op->reg_num);
    case MOP_VGPR:   return PHYS_VGPR_KEY(op->reg_num);
    case MOP_SPECIAL: return SPEC_KEY(op->imm);
    default: return 0xFFFFFFFF;
    }
}

/* ---- Add dependency edge ---- */

static void add_edge(uint16_t from_idx, uint16_t to_idx)
{
    sched_node_t *from = &s_nodes[from_idx];
    for (uint8_t i = 0; i < from->num_succs; i++)
        if (from->succs[i] == to_idx) return;

    if (from->num_succs < SCHED_MAX_DEPS) {
        from->succs[from->num_succs++] = to_idx;
    } else {
        from->deps_overflow = 1;
        s_nodes[to_idx].deps_overflow = 1;
    }
}

/* Record a definition: adds WAW edge from previous def, WAR from last use */
static void track_def(uint32_t key, uint16_t i)
{
    if (s_last_def[key] != 0xFFFF)
        add_edge(s_last_def[key], i);
    if (s_last_use[key] != 0xFFFF && s_last_use[key] != s_last_def[key]
        && s_last_use[key] != i)
        add_edge(s_last_use[key], i);
    s_last_def[key] = i;
    s_last_use[key] = 0xFFFF;
}

/* Record a use: adds RAW edge from the defining instruction */
static void track_use(uint32_t key, uint16_t i)
{
    if (s_last_def[key] != 0xFFFF)
        add_edge(s_last_def[key], i);
    s_last_use[key] = i;
}

/* ---- Build dependency DAG ---- */

static int build_dag(uint16_t n)
{
    memset(s_last_def, 0xFF, sizeof(s_last_def));
    memset(s_last_use, 0xFF, sizeof(s_last_use));
    uint16_t last_store = 0xFFFF;
    uint16_t cur_epoch = 0;
    s_num_barriers = 0;

    for (uint16_t i = 0; i < n; i++) {
        sched_node_t *nd = &s_nodes[i];
        const minst_t *mi = &s_stripped[i];
        uint16_t op = mi->op;

        nd->latency = inst_latency(op);
        nd->is_pinned_start = (uint8_t)is_pseudo_start(op);
        nd->is_pinned_end = (uint8_t)is_terminator(op);
        nd->is_store = (uint8_t)is_store(op);
        /* Exec mask modifiers are scheduling barriers — every vector
         * instruction implicitly depends on which lanes are active.
         * Moving a flat_store past an exec restore turns it into a NOP
         * (exec=0), which is how we learned this the hard way. */
        nd->is_barrier = 0;
        if (op == AMD_S_BARRIER ||
            op == AMD_S_AND_SAVEEXEC_B32 ||
            op == AMD_S_AND_SAVEEXEC_B64)
            nd->is_barrier = 1;
        for (uint8_t d = 0; !nd->is_barrier && d < mi->num_defs; d++) {
            if (mi->operands[d].kind == MOP_SPECIAL &&
                mi->operands[d].imm == AMD_SPEC_EXEC)
                nd->is_barrier = 1;
        }
        nd->is_load = 0;
        nd->wait_kind = 0;
        nd->epoch = cur_epoch;
        nd->deps_overflow = 0;

        if (is_vmem_write(op) || is_scalar_load(op) || is_ds_load(op)) {
            nd->is_load = 1;
            nd->wait_kind = load_wait_kind(op);
        }

        /* Barrier: implicit fence between scheduling regions.
         * No explicit edges -- epoch boundaries enforce ordering. */
        if (nd->is_barrier) {
            if (s_num_barriers >= SCHED_MAX_BARRIERS)
                return -1;
            s_barrier_pos[s_num_barriers++] = i;
            cur_epoch++;
            last_store = 0xFFFF;
            memset(s_last_def, 0xFF, sizeof(s_last_def));
            memset(s_last_use, 0xFF, sizeof(s_last_use));
            continue;
        }

        /* Register dependencies: uses (RAW) */
        uint8_t total = mi->num_defs + mi->num_uses;
        for (uint8_t u = mi->num_defs; u < total; u++) {
            const moperand_t *mop = &mi->operands[u];
            if (is_trackable(mop)) {
                track_use(op_key(mop), i);
                /* SGPR operands are often 64-bit pairs (saddr, sbase) --
                 * conservatively track the second register too. */
                if (mop->kind == MOP_SGPR)
                    track_use(PHYS_SGPR_KEY(mop->reg_num + 1), i);
            }
        }

        /* Register dependencies: defs (WAW + WAR) */
        for (uint8_t d = 0; d < mi->num_defs; d++) {
            const moperand_t *mop = &mi->operands[d];
            if (is_trackable(mop))
                track_def(op_key(mop), i);
        }

        /* 64-bit scalar ops define consecutive physical SGPRs */
        if ((op == AMD_S_MOV_B64 || op == AMD_S_LOAD_DWORDX2 ||
             op == AMD_GADDR) &&
            mi->num_defs > 0 && mi->operands[0].kind == MOP_SGPR)
            track_def(PHYS_SGPR_KEY(mi->operands[0].reg_num + 1), i);

        if (op == AMD_GADDR)
            track_def(SPEC_KEY(AMD_SPEC_SCC), i);
        if (op == AMD_S_LOAD_DWORDX4 && mi->num_defs > 0 &&
            mi->operands[0].kind == MOP_SGPR) {
            track_def(PHYS_SGPR_KEY(mi->operands[0].reg_num + 1), i);
            track_def(PHYS_SGPR_KEY(mi->operands[0].reg_num + 2), i);
            track_def(PHYS_SGPR_KEY(mi->operands[0].reg_num + 3), i);
        }

        /* VOPC implicitly writes VCC */
        if (op >= AMD_V_CMP_EQ_U32 && op <= AMD_V_CMP_NEQ_F32)
            track_def(SPEC_KEY(AMD_SPEC_VCC), i);

        /* v_cndmask_b32 implicitly reads VCC */
        if (op == AMD_V_CNDMASK_B32)
            track_use(SPEC_KEY(AMD_SPEC_VCC), i);

        /* SOPC implicitly writes SCC */
        if (op >= AMD_S_CMP_EQ_U32 && op <= AMD_S_CMP_NE_I32)
            track_def(SPEC_KEY(AMD_SPEC_SCC), i);

        /* SOP2 instructions that set SCC */
        if (sop2_writes_scc(op))
            track_def(SPEC_KEY(AMD_SPEC_SCC), i);

        /* s_cselect_b32 implicitly reads SCC */
        if (op == AMD_S_CSELECT_B32)
            track_use(SPEC_KEY(AMD_SPEC_SCC), i);

        /* s_cbranch_scc0/scc1 read SCC (handled by pinned_end, but
         * add the edge for completeness within the DAG) */
        if (op == AMD_S_CBRANCH_SCC0 || op == AMD_S_CBRANCH_SCC1)
            track_use(SPEC_KEY(AMD_SPEC_SCC), i);

        /* s_cbranch_execz/execnz read EXEC */
        if (op == AMD_S_CBRANCH_EXECZ || op == AMD_S_CBRANCH_EXECNZ)
            track_use(SPEC_KEY(AMD_SPEC_EXEC), i);

        /* s_and_saveexec reads and writes EXEC */
        if (op == AMD_S_AND_SAVEEXEC_B32 ||
            op == AMD_S_AND_SAVEEXEC_B64) {
            track_use(SPEC_KEY(AMD_SPEC_EXEC), i);
            track_def(SPEC_KEY(AMD_SPEC_EXEC), i);
            track_def(SPEC_KEY(AMD_SPEC_SCC), i);
        }

        /* Store ordering: conservative -- no reorder past other memory */
        if (nd->is_store) {
            if (last_store != 0xFFFF)
                add_edge(last_store, i);
            for (uint16_t j = 0; j < i; j++) {
                if (s_nodes[j].is_load && !s_nodes[j].is_pinned_start)
                    add_edge(j, i);
            }
            last_store = i;
        }

        /* Loads after a store depend on the store */
        if (nd->is_load && last_store != 0xFFFF)
            add_edge(last_store, i);
    }

    return 0;
}

/* ---- Compute priorities (critical path from node to exit) ---- */

static void compute_priorities(uint16_t n)
{
    for (int i = (int)n - 1; i >= 0; i--) {
        sched_node_t *nd = &s_nodes[i];
        uint32_t max_succ = 0;
        for (uint8_t s = 0; s < nd->num_succs; s++) {
            uint32_t sp = s_nodes[nd->succs[s]].priority;
            if (sp > max_succ) max_succ = sp;
        }
        nd->priority = nd->latency + max_succ;
    }
}

/* ---- Count predecessors from successor lists ---- */

static void count_preds(uint16_t n)
{
    for (uint16_t i = 0; i < n; i++)
        s_nodes[i].num_preds = 0;

    for (uint16_t i = 0; i < n; i++) {
        sched_node_t *nd = &s_nodes[i];
        for (uint8_t s = 0; s < nd->num_succs; s++)
            s_nodes[nd->succs[s]].num_preds++;
    }
}

/* ---- Emit a wait instruction into s_final ---- */

static uint16_t emit_wait(uint16_t pos, int kind, amd_target_t target)
{
    if (pos >= SCHED_MAX_BLOCK * 2) return pos;
    minst_t *w = &s_final[pos];
    memset(w, 0, sizeof(minst_t));
    if (target >= AMD_TARGET_GFX1200) {
        if (kind == WAIT_VMEM)      w->op = AMD_S_WAIT_LOADCNT;
        else if (kind == WAIT_SMEM) w->op = AMD_S_WAIT_KMCNT;
        else                        w->op = AMD_S_WAIT_DSCNT;
    } else {
        w->op = AMD_S_WAITCNT;
        if (kind == WAIT_VMEM) w->flags = AMD_WAIT_VMCNT0;
        else                   w->flags = AMD_WAIT_LGKMCNT0;
    }
    return (uint16_t)(pos + 1);
}

/* ---- Schedule one basic block ---- */

/* Takes input instructions and count, writes scheduled output to s_final.
 * Returns number of output instructions. Returns 0 if the block was
 * not worth scheduling (caller should copy input as-is). */
static uint32_t schedule_block(const minst_t *insts, uint32_t n,
                               amd_target_t target)
{
    if (n <= 1 || n > SCHED_MAX_BLOCK) return 0;

    /* Step 1: Strip waits and compact */
    uint16_t sn = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!is_wait_op(insts[i].op))
            s_stripped[sn++] = insts[i];
    }

    if (sn <= 1) return 0;

    /* Step 2: Initialize nodes and build DAG */
    memset(s_nodes, 0, sizeof(sched_node_t) * (size_t)sn);
    if (build_dag(sn) < 0)
        return 0;
    count_preds(sn);
    compute_priorities(sn);

    /* Pin any node that overflowed its dep list */
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].deps_overflow) {
            s_nodes[i].is_pinned_start = 1;
        }
    }

    /* Step 3: List schedule with epoch-based barrier handling.
     * Barriers divide the block into epochs. Instructions in epoch N
     * are fully scheduled before the barrier, then epoch N+1 begins. */
    uint16_t out = 0;
    uint16_t cur_epoch = 0;
    uint16_t barrier_cursor = 0;

    /* Pinned-start first (always epoch 0) */
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].is_pinned_start) {
            s_scheduled[out] = s_stripped[i];
            s_order[out] = i;
            out++;
            for (uint8_t s = 0; s < s_nodes[i].num_succs; s++)
                s_nodes[s_nodes[i].succs[s]].num_preds--;
            s_nodes[i].num_preds = 0xFFFFFFFF;
        }
    }

    /* Build ready queue for epoch 0 */
    uint16_t nready = 0;
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].num_preds == 0 && !s_nodes[i].is_pinned_start &&
            !s_nodes[i].is_pinned_end && !s_nodes[i].is_barrier &&
            s_nodes[i].epoch == 0)
            s_ready[nready++] = i;
    }

    /* Schedule epoch by epoch */
    for (uint32_t guard = 0; guard < SCHED_MAX_BLOCK * 2; guard++) {
        /* Drain current epoch's ready queue */
        while (nready > 0) {
            uint16_t best = 0;
            for (uint16_t r = 1; r < nready; r++) {
                if (s_nodes[s_ready[r]].priority >
                    s_nodes[s_ready[best]].priority)
                    best = r;
            }
            uint16_t pick = s_ready[best];
            s_ready[best] = s_ready[--nready];

            if (out >= SCHED_MAX_BLOCK) return 0;

            s_scheduled[out] = s_stripped[pick];
            s_order[out] = pick;
            out++;
            s_nodes[pick].num_preds = 0xFFFFFFFF;

            for (uint8_t s = 0; s < s_nodes[pick].num_succs; s++) {
                uint16_t si = s_nodes[pick].succs[s];
                if (s_nodes[si].num_preds != 0xFFFFFFFF) {
                    s_nodes[si].num_preds--;
                    if (s_nodes[si].num_preds == 0 &&
                        !s_nodes[si].is_pinned_end &&
                        !s_nodes[si].is_barrier &&
                        s_nodes[si].epoch == cur_epoch) {
                        if (nready < SCHED_MAX_BLOCK)
                            s_ready[nready++] = si;
                    }
                }
            }
        }

        /* Epoch drained -- emit barrier and advance */
        if (barrier_cursor < s_num_barriers) {
            uint16_t bpos = s_barrier_pos[barrier_cursor++];
            if (out >= SCHED_MAX_BLOCK) return 0;
            s_scheduled[out] = s_stripped[bpos];
            s_order[out] = bpos;
            out++;
            s_nodes[bpos].num_preds = 0xFFFFFFFF;
            cur_epoch++;

            for (uint16_t i = 0; i < sn; i++) {
                if (s_nodes[i].epoch == cur_epoch &&
                    s_nodes[i].num_preds == 0 &&
                    !s_nodes[i].is_pinned_start &&
                    !s_nodes[i].is_pinned_end &&
                    !s_nodes[i].is_barrier) {
                    if (nready < SCHED_MAX_BLOCK)
                        s_ready[nready++] = i;
                }
            }
            continue;
        }

        break;
    }

    /* Pinned-end (terminators) in original order */
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].is_pinned_end) {
            if (out >= SCHED_MAX_BLOCK) return 0;
            s_scheduled[out] = s_stripped[i];
            s_order[out] = i;
            out++;
        }
    }

    /* Step 4: Re-insert waits */
    memset(s_vreg_load, 0xFF, sizeof(s_vreg_load));
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].is_load) {
            const minst_t *mi = &s_stripped[i];
            for (uint8_t d = 0; d < mi->num_defs; d++) {
                if (is_trackable(&mi->operands[d])) {
                    uint32_t k = op_key(&mi->operands[d]);
                    if (k < AMD_MAX_VREGS)
                        s_vreg_load[k] = i;
                }
            }
            /* Multi-dword loads define consecutive physical SGPRs */
            uint16_t lop = mi->op;
            if (lop == AMD_S_LOAD_DWORDX2 && mi->num_defs > 0 &&
                mi->operands[0].kind == MOP_SGPR)
                s_vreg_load[PHYS_SGPR_KEY(mi->operands[0].reg_num + 1)] = i;
            if (lop == AMD_S_LOAD_DWORDX4 && mi->num_defs > 0 &&
                mi->operands[0].kind == MOP_SGPR) {
                s_vreg_load[PHYS_SGPR_KEY(mi->operands[0].reg_num + 1)] = i;
                s_vreg_load[PHYS_SGPR_KEY(mi->operands[0].reg_num + 2)] = i;
                s_vreg_load[PHYS_SGPR_KEY(mi->operands[0].reg_num + 3)] = i;
            }
        }
    }

    memset(s_load_waited, 0, (size_t)sn);
    memset(s_vmem_pending, 0, sizeof(s_vmem_pending));
    s_has_vmem_store = 0;
    uint16_t fn = 0;
    uint16_t smem_fly = 0;

    for (uint16_t i = 0; i < out; i++) {
        const minst_t *mi = &s_scheduled[i];
        uint16_t orig = s_order[i];

        uint8_t needs_wait[3] = {0, 0, 0};
        uint8_t total_ops = mi->num_defs + mi->num_uses;

        /* ---- Store-to-load ordering ----
         * Scratch stores are async.  A scratch load issued after
         * a store can arrive first, like a letter posted Tuesday
         * arriving before Monday's.  Royal Mail would be proud.
         * Only wait when there's actually a store outstanding. */
        if ((mi->op == AMD_FLAT_LOAD_DWORD ||
             mi->op == AMD_SCRATCH_LOAD_DWORD) && s_has_vmem_store)
            needs_wait[WAIT_VMEM] = 1;

        /* ---- Per-VGPR RAW hazard check ----
         * If a source register is still waiting on VMEM, reading
         * it now gets you whatever was in there before.  Quite
         * rude of the hardware, but there it is. */
        for (uint8_t u = mi->num_defs; u < total_ops; u++) {
            const moperand_t *mop = &mi->operands[u];
            if (!is_trackable(mop)) continue;
            uint32_t k = op_key(mop);
            if (k < AMD_MAX_VREGS) {
                /* Has this register's post arrived yet? */
                if (s_vmem_pending[k])
                    needs_wait[WAIT_VMEM] = 1;
                /* Also check the per-load tracker for good measure */
                uint16_t li = s_vreg_load[k];
                if (li != 0xFFFF && !s_load_waited[li]) {
                    needs_wait[s_nodes[li].wait_kind] = 1;
                    s_load_waited[li] = 1;
                }
            }
            /* SGPR pair: also check the second register */
            if (mop->kind == MOP_SGPR) {
                uint32_t k2 = PHYS_SGPR_KEY(mop->reg_num + 1);
                if (k2 < AMD_MAX_VREGS) {
                    if (s_vmem_pending[k2])
                        needs_wait[WAIT_VMEM] = 1;
                    uint16_t li2 = s_vreg_load[k2];
                    if (li2 != 0xFFFF && !s_load_waited[li2]) {
                        needs_wait[s_nodes[li2].wait_kind] = 1;
                        s_load_waited[li2] = 1;
                    }
                }
            }
        }

        if (orig < sn && s_nodes[orig].is_barrier) {
            /* Only wait on loads already emitted — same principle as
             * the regular wait handler below.  Marking future loads
             * as waited caused missing vmcnt after exec restores. */
            for (uint16_t p = 0; p < i; p++) {
                uint16_t oi = s_order[p];
                if (oi < sn && s_nodes[oi].is_load && !s_load_waited[oi]) {
                    needs_wait[s_nodes[oi].wait_kind] = 1;
                    s_load_waited[oi] = 1;
                }
            }
        }

        for (int k = 0; k < 3; k++) {
            if (needs_wait[k]) {
                fn = emit_wait(fn, k, target);
                if (k == WAIT_SMEM) smem_fly = 0;
                /* vmcnt(0): everything in the VMEM pipe retires.
                 * Clean slate.  Almost therapeutic. */
                if (k == WAIT_VMEM) {
                    memset(s_vmem_pending, 0, sizeof(s_vmem_pending));
                    s_has_vmem_store = 0;
                }
                /* Mark loads already emitted as waited */
                for (uint16_t p = 0; p < i; p++) {
                    uint16_t oi = s_order[p];
                    if (oi < sn && s_nodes[oi].is_load &&
                        s_nodes[oi].wait_kind == (uint8_t)k)
                        s_load_waited[oi] = 1;
                }
            }
        }

        if (fn >= SCHED_MAX_BLOCK * 2) return 0;
        s_final[fn++] = *mi;

        /* Note which VGPRs this instruction owes us.
         * We'll collect when someone tries to read them. */
        if (is_vmem_write(mi->op)) {
            for (uint8_t d = 0; d < mi->num_defs; d++) {
                if (is_trackable(&mi->operands[d])) {
                    uint32_t k = op_key(&mi->operands[d]);
                    if (k < AMD_MAX_VREGS)
                        s_vmem_pending[k] = 1;
                }
            }
        }
        if (is_vmem_store(mi->op))
            s_has_vmem_store = 1;

        /* GFX942 errata: >11 outstanding SMEM loads causes
         * MEMORY_APERTURE_VIOLATION. Drain periodically. */
        if (orig < sn && s_nodes[orig].is_load &&
            s_nodes[orig].wait_kind == WAIT_SMEM) {
            smem_fly++;
            if (smem_fly >= SCHED_MAX_SMEM_FLY) {
                fn = emit_wait(fn, WAIT_SMEM, target);
                smem_fly = 0;
                for (uint16_t p = 0; p <= i; p++) {
                    uint16_t oi = s_order[p];
                    if (oi < sn && s_nodes[oi].is_load &&
                        s_nodes[oi].wait_kind == WAIT_SMEM)
                        s_load_waited[oi] = 1;
                }
            }
        }
    }

    /* Trailing waits: loads consumed in successor blocks */
    uint16_t term_pos = fn;
    for (uint16_t i = 0; i < fn; i++) {
        if (is_terminator(s_final[i].op)) {
            term_pos = i;
            break;
        }
    }

    uint8_t trailing[3] = {0, 0, 0};
    for (uint16_t i = 0; i < sn; i++) {
        if (s_nodes[i].is_load && !s_load_waited[i])
            trailing[s_nodes[i].wait_kind] = 1;
    }

    uint16_t nwaits = (uint16_t)(trailing[0] + trailing[1] + trailing[2]);
    if (nwaits > 0 && fn + nwaits < SCHED_MAX_BLOCK * 2) {
        for (int i = (int)fn - 1; i >= (int)term_pos; i--)
            s_final[i + nwaits] = s_final[i];

        uint16_t wp = term_pos;
        for (int k = 0; k < 3; k++) {
            if (trailing[k])
                wp = emit_wait(wp, k, target);
        }
        fn = (uint16_t)(fn + nwaits);
    }

    return fn;
}

/* ---- Public API ---- */

void amdgpu_sched(amd_module_t *A)
{
    uint32_t out_pos = 0;

    for (uint32_t bi = 0; bi < A->num_mblocks; bi++) {
        mblock_t *blk = &A->mblocks[bi];
        uint32_t orig_base = blk->first_inst;
        uint32_t orig_n = blk->num_insts;

        blk->first_inst = out_pos;

        if (out_pos + orig_n * 2 > AMD_MAX_MINSTS) {
            memcpy(&s_output[out_pos], &A->minsts[orig_base],
                   sizeof(minst_t) * orig_n);
            out_pos += orig_n;
            blk->num_insts = orig_n;
            continue;
        }

        uint32_t written = schedule_block(&A->minsts[orig_base], orig_n,
                                          A->target);
        if (written > 0 && out_pos + written <= AMD_MAX_MINSTS) {
            memcpy(&s_output[out_pos], s_final, sizeof(minst_t) * written);
            blk->num_insts = written;
            out_pos += written;
        } else {
            memcpy(&s_output[out_pos], &A->minsts[orig_base],
                   sizeof(minst_t) * orig_n);
            blk->num_insts = orig_n;
            out_pos += orig_n;
        }
    }

    if (out_pos <= AMD_MAX_MINSTS) {
        memcpy(A->minsts, s_output, sizeof(minst_t) * out_pos);
        A->num_minsts = out_pos;
    }
}
