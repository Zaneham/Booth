#include "bir_mem2reg.h"
#include <string.h>

/*
 * mem2reg: promote stack allocas to SSA registers with phi nodes.
 *
 * Classic algorithm: compute dominance frontiers, insert phis at the
 * iterated dominance frontier of each alloca's store set, then rename
 * variables via iterative domtree walk.  No recursion.  No malloc.
 *
 * Seven steps per function:
 *   1. Identify promotable allocas (scalar, only load/store uses)
 *   2. Build predecessor lists from terminators
 *   3. Compute dominator tree (Cooper-Harvey-Kennedy iterative)
 *   4. Compute dominance frontiers
 *   5. Insert phi nodes at iterated dominance frontier
 *   6. Rename variables (iterative domtree walk)
 *   7. Compact: delete dead alloca/store/load, remap references
 */

/* ---- Limits ---- */

#define M2R_MAX_PROMOTE   256
#define M2R_MAX_BLOCKS    4096
#define M2R_MAX_PREDS     16
#define M2R_MAX_DF        64
#define M2R_MAX_STACK     4096
#define M2R_MAX_PHIS      8192
#define M2R_MAX_CHANGES   65536

#define M2R_UNDEF         0xFFFFFFFFu
#define M2R_UNVISITED     0xFFFFFFFEu

/* ---- Phi Record ---- */

typedef struct {
    uint32_t inst;     /* absolute instruction index in M->insts[] */
    uint32_t block;    /* block-local index this phi belongs to */
    int      alloca;   /* promote[] index */
} phi_rec_t;

/* ---- Save/restore entry for rename ---- */

typedef struct {
    int      alloca_idx;
    uint32_t old_val;
} val_change_t;

/* ---- Working State ---- */

typedef struct {
    bir_module_t *M;

    /* Per-function scope */
    uint32_t func_idx;
    uint32_t base_block, num_blocks;
    uint32_t base_inst;
    uint32_t orig_num_insts;   /* original instruction count for this func */

    /* Step 1: promotable allocas */
    struct {
        uint32_t inst;       /* absolute instruction index of alloca */
        uint32_t type;       /* promoted type (pointee of the ptr type) */
    } promote[M2R_MAX_PROMOTE];
    int num_promote;

    /* Step 2: predecessors (block-local indices) */
    uint32_t preds[M2R_MAX_BLOCKS][M2R_MAX_PREDS];
    uint8_t  npreds[M2R_MAX_BLOCKS];

    /* Step 3: dominators + RPO (block-local indices) */
    uint32_t idom[M2R_MAX_BLOCKS];
    uint32_t rpo[M2R_MAX_BLOCKS];
    uint32_t rpo_num[M2R_MAX_BLOCKS];
    uint32_t rpo_count;

    /* Step 4: dominance frontiers (block-local indices) */
    uint32_t df[M2R_MAX_BLOCKS][M2R_MAX_DF];
    uint8_t  ndf[M2R_MAX_BLOCKS];

    /* Step 3 cont: domtree children lists */
    uint32_t dt_child[M2R_MAX_BLOCKS];
    uint32_t dt_sibling[M2R_MAX_BLOCKS];

    /* Step 5: inserted phis */
    phi_rec_t phis[M2R_MAX_PHIS];
    int       num_phis;

    /* Step 6: current reaching definition per alloca */
    uint32_t  val_top[M2R_MAX_PROMOTE];

    /* Step 6: val_remap[abs_inst] = replacement value for dead loads/phis */
    uint32_t  val_remap[BIR_MAX_INSTS];

    /* Deletion marks */
    uint8_t   dead[BIR_MAX_INSTS];

    /* Step 6: save/restore stack for rename */
    val_change_t changes[M2R_MAX_CHANGES];
    int          change_sp;
} m2r_t;

static m2r_t G;

/* ---- Helpers ---- */

static uint32_t ptr_inner(const bir_module_t *M, uint32_t tidx)
{
    if (tidx < M->num_types && M->types[tidx].kind == BIR_TYPE_PTR)
        return M->types[tidx].inner;
    return 0;
}

static int is_scalar_type(const bir_module_t *M, uint32_t tidx)
{
    if (tidx >= M->num_types) return 0;
    uint8_t k = M->types[tidx].kind;
    return k == BIR_TYPE_INT || k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT
        || k == BIR_TYPE_PTR;
}

static int find_promote_idx(const m2r_t *S, uint32_t alloca_inst)
{
    for (int i = 0; i < S->num_promote; i++)
        if (S->promote[i].inst == alloca_inst)
            return i;
    return -1;
}

static uint32_t get_alloca_ref(const bir_module_t *M, const bir_inst_t *I,
                               int is_store)
{
    uint32_t ref = is_store ? I->operands[1] : I->operands[0];
    if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE) return 0;
    uint32_t idx = BIR_VAL_INDEX(ref);
    if (idx >= M->num_insts) return 0;
    if (M->insts[idx].op == BIR_ALLOCA)
        return idx;
    return 0;
}

static uint32_t make_undef(bir_module_t *M, uint32_t type)
{
    for (uint32_t i = 0; i < M->num_consts; i++) {
        if (M->consts[i].kind == BIR_CONST_UNDEF && M->consts[i].type == type)
            return BIR_MAKE_CONST(i);
    }
    if (M->num_consts >= BIR_MAX_CONSTS)
        return BIR_VAL_NONE;
    uint32_t ci = M->num_consts++;
    M->consts[ci].kind = BIR_CONST_UNDEF;
    memset(M->consts[ci].pad, 0, sizeof(M->consts[ci].pad));
    M->consts[ci].type = type;
    M->consts[ci].d.ival = 0;
    return BIR_MAKE_CONST(ci);
}

/* Get successors of block-local index bi.  Returns absolute block indices. */
static int get_succs(const m2r_t *S, uint32_t bi, uint32_t *out)
{
    int n = 0;
    uint32_t bidx = S->base_block + bi;
    const bir_block_t *B = &S->M->blocks[bidx];
    if (B->num_insts == 0) return 0;
    uint32_t li = B->first_inst + B->num_insts - 1;
    const bir_inst_t *I = &S->M->insts[li];
    switch (I->op) {
    case BIR_BR:
        out[n++] = I->operands[0];
        break;
    case BIR_BR_COND:
        out[n++] = I->operands[1];
        out[n++] = I->operands[2];
        break;
    case BIR_SWITCH:
        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t st = I->operands[0];
            uint32_t ct = I->operands[1];
            if (ct >= 2) {
                out[n++] = S->M->extra_operands[st + 1];
                for (uint32_t j = 3; j < ct && n < M2R_MAX_PREDS; j += 2)
                    out[n++] = S->M->extra_operands[st + j];
            }
        } else {
            out[n++] = I->operands[1];
        }
        break;
    default:
        break;
    }
    return n;
}

/* ---- Step 1: Identify Promotable Allocas ---- */

static void step1_find_promotable(m2r_t *S)
{
    bir_module_t *M = S->M;
    uint32_t end = S->base_inst + S->orig_num_insts;
    S->num_promote = 0;

    for (uint32_t i = S->base_inst; i < end
         && S->num_promote < M2R_MAX_PROMOTE; i++) {
        if (M->insts[i].op != BIR_ALLOCA) continue;
        uint32_t pointee = ptr_inner(M, M->insts[i].type);
        if (!is_scalar_type(M, pointee)) continue;
        S->promote[S->num_promote].inst = i;
        S->promote[S->num_promote].type = pointee;
        S->num_promote++;
    }

    /* Disqualify allocas used by anything other than load/store */
    for (uint32_t i = S->base_inst; i < end; i++) {
        bir_inst_t *I = &M->insts[i];

        if (I->op == BIR_LOAD) continue;
        if (I->op == BIR_BR || I->op == BIR_BR_COND
            || I->op == BIR_SWITCH) continue;
        if (I->op == BIR_STORE) {
            /* Alloca as stored VALUE (not dest) means address escapes */
            uint32_t vr = I->operands[0];
            if (!BIR_VAL_IS_CONST(vr) && vr != BIR_VAL_NONE) {
                uint32_t vi = BIR_VAL_INDEX(vr);
                if (vi >= S->base_inst && vi < end
                    && M->insts[vi].op == BIR_ALLOCA) {
                    int pi = find_promote_idx(S, vi);
                    if (pi >= 0)
                        S->promote[pi] = S->promote[--S->num_promote];
                }
            }
            continue;
        }

        /* Any other opcode: check all operands */
        uint8_t nops = I->num_operands;
        if (nops == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            for (uint32_t j = 0; j < count
                 && (start + j) < M->num_extra_ops; j++) {
                uint32_t ref = M->extra_operands[start + j];
                if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE) continue;
                uint32_t ri = BIR_VAL_INDEX(ref);
                if (ri >= S->base_inst && ri < end
                    && M->insts[ri].op == BIR_ALLOCA) {
                    int pi = find_promote_idx(S, ri);
                    if (pi >= 0)
                        S->promote[pi] = S->promote[--S->num_promote];
                }
            }
        } else {
            for (uint8_t j = 0; j < nops && j < BIR_OPERANDS_INLINE; j++) {
                uint32_t ref = I->operands[j];
                if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE) continue;
                uint32_t ri = BIR_VAL_INDEX(ref);
                if (ri >= S->base_inst && ri < end
                    && M->insts[ri].op == BIR_ALLOCA) {
                    int pi = find_promote_idx(S, ri);
                    if (pi >= 0)
                        S->promote[pi] = S->promote[--S->num_promote];
                }
            }
        }
    }
}

/* ---- Step 2: Build Predecessor Lists ---- */

static void step2_build_preds(m2r_t *S)
{
    memset(S->npreds, 0, S->num_blocks * sizeof(S->npreds[0]));

    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        uint32_t succs[M2R_MAX_PREDS];
        int ns = get_succs(S, bi, succs);
        for (int si = 0; si < ns; si++) {
            uint32_t s = succs[si] - S->base_block;
            if (s < S->num_blocks && S->npreds[s] < M2R_MAX_PREDS) {
                /* Avoid duplicate predecessors */
                int dup = 0;
                for (uint8_t k = 0; k < S->npreds[s]; k++) {
                    if (S->preds[s][k] == bi) { dup = 1; break; }
                }
                if (!dup)
                    S->preds[s][S->npreds[s]++] = bi;
            }
        }
    }
}

/* ---- Step 3: Compute Dominator Tree (Cooper-Harvey-Kennedy) ---- */

static void step3_compute_dominators(m2r_t *S)
{
    uint32_t nb = S->num_blocks;

    /* RPO via iterative DFS */
    {
        uint8_t visited[M2R_MAX_BLOCKS];
        uint32_t stack[M2R_MAX_STACK];
        uint32_t post[M2R_MAX_BLOCKS];
        uint32_t post_count = 0;
        int sp = 0;

        memset(visited, 0, nb);
        stack[sp++] = 0;
        visited[0] = 1;

        while (sp > 0) {
            uint32_t cur = stack[sp - 1];
            uint32_t succs[M2R_MAX_PREDS];
            int ns = get_succs(S, cur, succs);
            int pushed = 0;

            for (int si = 0; si < ns && !pushed; si++) {
                uint32_t s = succs[si] - S->base_block;
                if (s < nb && !visited[s] && sp < M2R_MAX_STACK) {
                    visited[s] = 1;
                    stack[sp++] = s;
                    pushed = 1;
                }
            }
            if (!pushed) {
                sp--;
                if (post_count < nb)
                    post[post_count++] = cur;
            }
        }

        S->rpo_count = post_count;
        for (uint32_t i = 0; i < post_count; i++) {
            S->rpo[i] = post[post_count - 1 - i];
            S->rpo_num[post[post_count - 1 - i]] = i;
        }
        for (uint32_t i = 0; i < nb; i++) {
            if (!visited[i])
                S->rpo_num[i] = M2R_UNDEF;
        }
    }

    /* Iterative dominator computation */
    for (uint32_t i = 0; i < nb; i++)
        S->idom[i] = M2R_UNDEF;
    S->idom[0] = 0;

    int changed = 1;
    int max_iter = (int)nb * 4 + 16;
    while (changed && max_iter-- > 0) {
        changed = 0;
        for (uint32_t ri = 1; ri < S->rpo_count; ri++) {
            uint32_t b = S->rpo[ri];
            uint32_t new_idom = M2R_UNDEF;

            for (uint8_t pi = 0; pi < S->npreds[b]; pi++) {
                uint32_t p = S->preds[b][pi];
                if (S->idom[p] == M2R_UNDEF) continue;
                if (new_idom == M2R_UNDEF) {
                    new_idom = p;
                } else {
                    uint32_t a = new_idom, bb = p;
                    int guard = (int)nb * 2;
                    while (a != bb && guard-- > 0) {
                        while (S->rpo_num[a] > S->rpo_num[bb] && guard-- > 0)
                            a = S->idom[a];
                        while (S->rpo_num[bb] > S->rpo_num[a] && guard-- > 0)
                            bb = S->idom[bb];
                    }
                    new_idom = a;
                }
            }

            if (new_idom != M2R_UNDEF && S->idom[b] != new_idom) {
                S->idom[b] = new_idom;
                changed = 1;
            }
        }
    }

    /* Build domtree children lists */
    for (uint32_t i = 0; i < nb; i++) {
        S->dt_child[i] = M2R_UNDEF;
        S->dt_sibling[i] = M2R_UNDEF;
    }
    for (uint32_t i = 1; i < nb; i++) {
        if (S->idom[i] != M2R_UNDEF && S->idom[i] != i) {
            uint32_t parent = S->idom[i];
            S->dt_sibling[i] = S->dt_child[parent];
            S->dt_child[parent] = i;
        }
    }
}

/* ---- Step 4: Compute Dominance Frontiers ---- */

static void step4_compute_df(m2r_t *S)
{
    uint32_t nb = S->num_blocks;
    memset(S->ndf, 0, nb * sizeof(S->ndf[0]));

    for (uint32_t b = 0; b < nb; b++) {
        if (S->npreds[b] < 2) continue;
        for (uint8_t pi = 0; pi < S->npreds[b]; pi++) {
            uint32_t runner = S->preds[b][pi];
            int guard = (int)nb * 2;
            while (runner != S->idom[b] && runner != M2R_UNDEF
                   && guard-- > 0) {
                int found = 0;
                for (uint8_t k = 0; k < S->ndf[runner]; k++) {
                    if (S->df[runner][k] == b) { found = 1; break; }
                }
                if (!found && S->ndf[runner] < M2R_MAX_DF)
                    S->df[runner][S->ndf[runner]++] = b;
                runner = S->idom[runner];
            }
        }
    }
}

/* ---- Step 5: Insert Phi Nodes ---- */

static void step5_insert_phis(m2r_t *S)
{
    bir_module_t *M = S->M;
    uint32_t end = S->base_inst + S->orig_num_insts;
    S->num_phis = 0;

    for (int ai = 0; ai < S->num_promote; ai++) {
        uint32_t alloca_inst = S->promote[ai].inst;
        uint32_t ptype = S->promote[ai].type;

        /* Collect def_blocks: blocks containing a store to this alloca */
        uint32_t def_blocks[M2R_MAX_BLOCKS];
        uint32_t ndef = 0;

        for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
            uint32_t bidx = S->base_block + bi;
            const bir_block_t *B = &M->blocks[bidx];
            for (uint32_t j = 0; j < B->num_insts; j++) {
                uint32_t ii = B->first_inst + j;
                if (ii >= end) break;
                const bir_inst_t *I = &M->insts[ii];
                if (I->op == BIR_STORE
                    && get_alloca_ref(M, I, 1) == alloca_inst) {
                    if (ndef < M2R_MAX_BLOCKS)
                        def_blocks[ndef++] = bi;
                    break;
                }
            }
        }

        /* Iterated dominance frontier */
        uint8_t in_wl[M2R_MAX_BLOCKS];
        uint8_t has_phi[M2R_MAX_BLOCKS];
        memset(in_wl, 0, S->num_blocks);
        memset(has_phi, 0, S->num_blocks);

        uint32_t wl[M2R_MAX_BLOCKS];
        uint32_t wh = 0, wt = 0;
        for (uint32_t di = 0; di < ndef; di++) {
            wl[wt++] = def_blocks[di];
            in_wl[def_blocks[di]] = 1;
        }

        while (wh < wt) {
            uint32_t w = wl[wh++];
            for (uint8_t di = 0; di < S->ndf[w]; di++) {
                uint32_t d = S->df[w][di];
                if (has_phi[d]) continue;
                has_phi[d] = 1;

                if (M->num_insts >= BIR_MAX_INSTS) continue;
                if (S->num_phis >= M2R_MAX_PHIS) continue;

                uint32_t phi_idx = M->num_insts++;
                bir_inst_t *phi = &M->insts[phi_idx];
                memset(phi, 0, sizeof(*phi));
                phi->op = BIR_PHI;
                phi->type = ptype;

                uint8_t np = S->npreds[d];
                uint32_t need = (uint32_t)np * 2;

                if (need <= BIR_OPERANDS_INLINE) {
                    phi->num_operands = (uint8_t)need;
                    for (uint8_t k = 0; k < np; k++) {
                        phi->operands[k * 2] =
                            S->preds[d][k] + S->base_block;
                        phi->operands[k * 2 + 1] = BIR_VAL_NONE;
                    }
                } else if (M->num_extra_ops + need <= BIR_MAX_EXTRA_OPS) {
                    uint32_t start = M->num_extra_ops;
                    phi->num_operands = BIR_OPERANDS_OVERFLOW;
                    phi->operands[0] = start;
                    phi->operands[1] = need;
                    for (uint8_t k = 0; k < np; k++) {
                        M->extra_operands[M->num_extra_ops++] =
                            S->preds[d][k] + S->base_block;
                        M->extra_operands[M->num_extra_ops++] = BIR_VAL_NONE;
                    }
                } else {
                    M->num_insts--;
                    continue;
                }

                S->phis[S->num_phis].inst = phi_idx;
                S->phis[S->num_phis].block = d;
                S->phis[S->num_phis].alloca = ai;
                S->num_phis++;

                if (!in_wl[d] && wt < M2R_MAX_BLOCKS) {
                    wl[wt++] = d;
                    in_wl[d] = 1;
                }
            }
        }
    }
}

/* ---- Step 6: Rename Variables ---- */

static void set_phi_incoming(bir_module_t *M, uint32_t phi_inst,
                             uint32_t pred_abs_block, uint32_t val)
{
    bir_inst_t *I = &M->insts[phi_inst];
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t start = I->operands[0];
        uint32_t count = I->operands[1];
        for (uint32_t j = 0; j + 1 < count; j += 2) {
            if (M->extra_operands[start + j] == pred_abs_block) {
                M->extra_operands[start + j + 1] = val;
                return;
            }
        }
    } else {
        for (uint8_t j = 0; j + 1 < I->num_operands; j += 2) {
            if (I->operands[j] == pred_abs_block) {
                I->operands[j + 1] = val;
                return;
            }
        }
    }
}

/* Save val_top[ai] before modifying, push to change stack */
static void save_val(m2r_t *S, int ai)
{
    if (S->change_sp < M2R_MAX_CHANGES) {
        S->changes[S->change_sp].alloca_idx = ai;
        S->changes[S->change_sp].old_val = S->val_top[ai];
        S->change_sp++;
    }
}

static void rename_block(m2r_t *S, uint32_t b)
{
    bir_module_t *M = S->M;
    uint32_t abs_b = S->base_block + b;

    /* 1. Process phis belonging to this block — each defines a new value */
    for (int pi = 0; pi < S->num_phis; pi++) {
        if (S->phis[pi].block != b) continue;
        int ai = S->phis[pi].alloca;
        save_val(S, ai);
        S->val_top[ai] = BIR_MAKE_VAL(S->phis[pi].inst);
    }

    /* 2. Scan original instructions in the block */
    const bir_block_t *B = &M->blocks[abs_b];
    for (uint32_t j = 0; j < B->num_insts; j++) {
        uint32_t ii = B->first_inst + j;
        bir_inst_t *I = &M->insts[ii];

        if (I->op == BIR_STORE) {
            uint32_t dest = get_alloca_ref(M, I, 1);
            if (dest == 0) continue;
            int ai = find_promote_idx(S, dest);
            if (ai < 0) continue;
            save_val(S, ai);
            S->val_top[ai] = I->operands[0];
            S->dead[ii] = 1;
        } else if (I->op == BIR_LOAD) {
            uint32_t src = get_alloca_ref(M, I, 0);
            if (src == 0) continue;
            int ai = find_promote_idx(S, src);
            if (ai < 0) continue;
            S->val_remap[ii] = S->val_top[ai];
            S->dead[ii] = 1;
        }
    }

    /* 3. Fill in phi operands in CFG successors */
    uint32_t succs[M2R_MAX_PREDS];
    int ns = get_succs(S, b, succs);
    for (int si = 0; si < ns; si++) {
        uint32_t succ_abs = succs[si];
        uint32_t succ_local = succ_abs - S->base_block;
        if (succ_local >= S->num_blocks) continue;
        for (int pi = 0; pi < S->num_phis; pi++) {
            if (S->phis[pi].block != succ_local) continue;
            int ai = S->phis[pi].alloca;
            set_phi_incoming(M, S->phis[pi].inst, abs_b, S->val_top[ai]);
        }
    }
}

static void step6_rename(m2r_t *S)
{
    bir_module_t *M = S->M;
    /* Initialize val_remap to identity (BIR_VAL_NONE = no remap) */
    for (uint32_t i = 0; i < M->num_insts; i++)
        S->val_remap[i] = BIR_VAL_NONE;

    /* Initialize reaching definitions to undef */
    for (int ai = 0; ai < S->num_promote; ai++)
        S->val_top[ai] = make_undef(M, S->promote[ai].type);

    S->change_sp = 0;

    /* Mark promotable allocas as dead */
    for (int ai = 0; ai < S->num_promote; ai++)
        S->dead[S->promote[ai].inst] = 1;

    /* Iterative domtree DFS with save/restore */
    struct { uint32_t block; uint32_t child; int save_point; } stack[M2R_MAX_STACK];
    int sp = 0;

    /* Push entry block */
    stack[0].block = 0;
    stack[0].child = M2R_UNVISITED;
    stack[0].save_point = 0;
    sp = 1;

    while (sp > 0) {
        uint32_t b = stack[sp - 1].block;

        if (stack[sp - 1].child == M2R_UNVISITED) {
            /* First visit: save state and process block */
            stack[sp - 1].save_point = S->change_sp;
            rename_block(S, b);
            stack[sp - 1].child = S->dt_child[b];
        }

        /* Find next child to visit */
        uint32_t c = stack[sp - 1].child;
        if (c != M2R_UNDEF) {
            stack[sp - 1].child = S->dt_sibling[c];
            if (sp < M2R_MAX_STACK) {
                stack[sp].block = c;
                stack[sp].child = M2R_UNVISITED;
                stack[sp].save_point = 0;
                sp++;
            }
        } else {
            /* All children processed — restore and pop */
            int save = stack[sp - 1].save_point;
            while (S->change_sp > save) {
                S->change_sp--;
                int ai = S->changes[S->change_sp].alloca_idx;
                S->val_top[ai] = S->changes[S->change_sp].old_val;
            }
            sp--;
        }
    }
}

/* ---- Step 7: Compact ---- */

/*
 * Resolve val_remap chains: if val_remap[a] = BIR_MAKE_VAL(b) and
 * val_remap[b] != NONE, follow the chain.
 */
static uint32_t resolve_val(const m2r_t *S, uint32_t ref,
                            uint32_t max_inst)
{
    int guard = 64;
    while (guard-- > 0) {
        if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref))
            return ref;
        uint32_t idx = BIR_VAL_INDEX(ref);
        if (idx >= max_inst) return ref;
        if (S->val_remap[idx] == BIR_VAL_NONE) return ref;
        ref = S->val_remap[idx];
    }
    return ref;
}

/*
 * Remap a single operand: apply val_remap, then renumber surviving insts.
 * inst_renum[old_abs] = new_abs for surviving instructions.
 * max_inst = old M->num_insts (before compaction shrunk it).
 * Returns the remapped operand value.
 */
static uint32_t remap_operand(const m2r_t *S, uint32_t ref,
                              const uint32_t *inst_renum,
                              uint32_t max_inst)
{
    if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref))
        return ref;

    /* First apply val_remap (dead loads → reaching defs) */
    ref = resolve_val(S, ref, max_inst);

    if (ref == BIR_VAL_NONE || BIR_VAL_IS_CONST(ref))
        return ref;

    /* Then renumber instruction index */
    uint32_t idx = BIR_VAL_INDEX(ref);
    if (idx < max_inst && inst_renum[idx] != M2R_UNDEF)
        return BIR_MAKE_VAL(inst_renum[idx]);

    return ref;
}

static void step7_compact(m2r_t *S)
{
    bir_module_t *M = S->M;
    uint32_t old_num_insts = M->num_insts;
    bir_func_t *F = &M->funcs[S->func_idx];

    /* Build inst_renum: for each old instruction, what's its new index?
     * New layout per block: [phis for block] [surviving original insts] */
    static uint32_t inst_renum[BIR_MAX_INSTS];
    for (uint32_t i = 0; i < old_num_insts; i++)
        inst_renum[i] = M2R_UNDEF;

    /* First pass: count new instructions and assign new indices */
    uint32_t new_idx = S->base_inst;
    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        /* Phis for this block */
        for (int pi = 0; pi < S->num_phis; pi++) {
            if (S->phis[pi].block != bi) continue;
            inst_renum[S->phis[pi].inst] = new_idx++;
        }
        /* Original block instructions */
        uint32_t abs_b = S->base_block + bi;
        const bir_block_t *B = &M->blocks[abs_b];
        for (uint32_t j = 0; j < B->num_insts; j++) {
            uint32_t ii = B->first_inst + j;
            if (S->dead[ii]) continue;
            inst_renum[ii] = new_idx++;
        }
    }

    uint32_t new_total = new_idx - S->base_inst;

    /* Second pass: copy instructions into new positions.
     * Use a static scratch buffer to avoid overlap issues. */
    static bir_inst_t scratch[BIR_MAX_INSTS];
    /* Lines travel with their instructions or they end up describing
     * whatever landed in the slot instead. */
    static uint32_t scratch_lines[BIR_MAX_INSTS];
    uint32_t si = 0;

    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        for (int pi = 0; pi < S->num_phis; pi++) {
            if (S->phis[pi].block != bi) continue;
            scratch_lines[si] = M->inst_lines[S->phis[pi].inst];
            scratch[si++] = M->insts[S->phis[pi].inst];
        }
        uint32_t abs_b = S->base_block + bi;
        const bir_block_t *B = &M->blocks[abs_b];
        for (uint32_t j = 0; j < B->num_insts; j++) {
            uint32_t ii = B->first_inst + j;
            if (S->dead[ii]) continue;
            scratch_lines[si] = M->inst_lines[ii];
            scratch[si++] = M->insts[ii];
        }
    }

    /* Copy scratch back */
    memcpy(&M->insts[S->base_inst], scratch, si * sizeof(bir_inst_t));
    memcpy(&M->inst_lines[S->base_inst], scratch_lines,
           si * sizeof(scratch_lines[0]));

    /* Update block boundaries */
    uint32_t cursor = S->base_inst;
    for (uint32_t bi = 0; bi < S->num_blocks; bi++) {
        uint32_t abs_b = S->base_block + bi;
        uint32_t count = 0;

        /* Count phis */
        for (int pi = 0; pi < S->num_phis; pi++) {
            if (S->phis[pi].block == bi) count++;
        }
        /* Count surviving original insts */
        const bir_block_t *B = &M->blocks[abs_b];
        for (uint32_t j = 0; j < B->num_insts; j++) {
            uint32_t ii = B->first_inst + j;
            if (!S->dead[ii]) count++;
        }

        M->blocks[abs_b].first_inst = cursor;
        M->blocks[abs_b].num_insts = count;
        cursor += count;
    }

    /* Update function total_insts */
    F->total_insts = new_total;

    /* Don't shift subsequent functions' instructions — leave gaps.
     * Each function's block boundaries remain valid.  The gaps are
     * cleaned up by a final compaction in bir_mem2reg. */

    /* Remap all operands in this function's instructions */
    for (uint32_t i = S->base_inst; i < S->base_inst + new_total; i++) {
        bir_inst_t *I = &M->insts[i];

        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            for (uint32_t j = 0; j < count
                 && (start + j) < M->num_extra_ops; j++) {
                uint32_t ref = M->extra_operands[start + j];
                /* PHI operands alternate: block_idx, value, block_idx, value
                 * Branch operands are block indices (not values).
                 * For PHI: even slots are block indices, odd are values.
                 * For SWITCH: slot 0 is value, slot 1 is block, then
                 *   (value, block) pairs.
                 * We remap values but not block indices. */
                if (I->op == BIR_PHI) {
                    if (j % 2 == 1) /* value slot */
                        M->extra_operands[start + j] =
                            remap_operand(S, ref, inst_renum, old_num_insts);
                } else if (I->op == BIR_SWITCH) {
                    if (j == 0) /* switched value */
                        M->extra_operands[start + j] =
                            remap_operand(S, ref, inst_renum, old_num_insts);
                    else if (j >= 2 && j % 2 == 0) /* case values */
                        M->extra_operands[start + j] =
                            remap_operand(S, ref, inst_renum, old_num_insts);
                } else {
                    /* CALL etc: remap all */
                    M->extra_operands[start + j] =
                        remap_operand(S, ref, inst_renum, old_num_insts);
                }
            }
        } else {
            /* Determine which operands are values vs block refs */
            for (uint8_t j = 0; j < I->num_operands
                 && j < BIR_OPERANDS_INLINE; j++) {
                int is_block_ref = 0;
                switch (I->op) {
                case BIR_BR:
                    is_block_ref = (j == 0); break;
                case BIR_BR_COND:
                    is_block_ref = (j >= 1 && j <= 3); break;
                case BIR_SWITCH:
                    is_block_ref = (j == 1); break;
                case BIR_PHI:
                    is_block_ref = (j % 2 == 0); break;
                default:
                    break;
                }
                if (!is_block_ref) {
                    I->operands[j] =
                        remap_operand(S, I->operands[j], inst_renum,
                                        old_num_insts);
                }
            }
        }
    }

}

/* ---- Per-Function Driver ---- */

static int m2r_run_func(m2r_t *S, uint32_t fi)
{
    bir_module_t *M = S->M;
    const bir_func_t *F = &M->funcs[fi];
    if (F->num_blocks == 0) return 0;

    S->func_idx      = fi;
    S->base_block    = F->first_block;
    S->num_blocks    = F->num_blocks;
    S->base_inst     = M->blocks[F->first_block].first_inst;
    S->orig_num_insts = F->total_insts;

    if (S->num_blocks > M2R_MAX_BLOCKS) return 0;

    memset(S->dead + S->base_inst, 0, S->orig_num_insts);

    step1_find_promotable(S);
    if (S->num_promote == 0) return 0;

    step2_build_preds(S);
    step3_compute_dominators(S);
    step4_compute_df(S);
    step5_insert_phis(S);
    step6_rename(S);
    step7_compact(S);

    return S->num_promote;
}

/* ---- Public API ---- */

int bir_mem2reg(bir_module_t *M)
{
    m2r_t *S = &G;
    memset(S, 0, sizeof(*S));
    S->M = M;
    int total = 0;
    for (uint32_t fi = 0; fi < M->num_funcs; fi++)
        total += m2r_run_func(S, fi);

    /* Close gaps: compact all function instructions into contiguous
     * storage.  Each function's blocks and operands already reference
     * correct absolute instruction indices within their own compacted
     * range.  We now slide each function's block down to close gaps
     * between functions, remapping intra-function references. */
    if (total > 0) {
        uint32_t dst = 0;
        for (uint32_t fi = 0; fi < M->num_funcs; fi++) {
            bir_func_t *F = &M->funcs[fi];
            if (F->num_blocks == 0) continue;

            uint32_t src = M->blocks[F->first_block].first_inst;
            uint32_t count = F->total_insts;

            if (src == dst) {
                dst += count;
                continue;
            }

            int32_t shift = (int32_t)dst - (int32_t)src;

            /* Move instructions */
            memmove(&M->insts[dst], &M->insts[src],
                    count * sizeof(bir_inst_t));
            memmove(&M->inst_lines[dst], &M->inst_lines[src],
                    count * sizeof(M->inst_lines[0]));

            /* Update block boundaries */
            for (uint16_t bi = 0; bi < F->num_blocks; bi++) {
                uint32_t abs_b = F->first_block + bi;
                M->blocks[abs_b].first_inst =
                    (uint32_t)((int32_t)M->blocks[abs_b].first_inst + shift);
            }

            /* Remap instruction-value references within this function */
            for (uint32_t i = dst; i < dst + count; i++) {
                bir_inst_t *I = &M->insts[i];
                if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
                    uint32_t start = I->operands[0];
                    uint32_t cnt = I->operands[1];
                    for (uint32_t j = 0; j < cnt
                         && (start + j) < M->num_extra_ops; j++) {
                        uint32_t ref = M->extra_operands[start + j];
                        if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE)
                            continue;
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= src && idx < src + count)
                            M->extra_operands[start + j] =
                                BIR_MAKE_VAL((uint32_t)((int32_t)idx + shift));
                    }
                } else {
                    for (uint8_t j = 0; j < I->num_operands
                         && j < BIR_OPERANDS_INLINE; j++) {
                        uint32_t ref = I->operands[j];
                        if (BIR_VAL_IS_CONST(ref) || ref == BIR_VAL_NONE)
                            continue;
                        uint32_t idx = BIR_VAL_INDEX(ref);
                        if (idx >= src && idx < src + count)
                            I->operands[j] =
                                BIR_MAKE_VAL((uint32_t)((int32_t)idx + shift));
                    }
                }
            }

            dst += count;
        }
        M->num_insts = dst;
    }

    return total;
}
