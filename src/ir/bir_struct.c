#include "bir_struct.h"

#include <string.h>

/*
 * The structuriser, in the flesh.
 *
 * The shape of a working day in here: take one function, learn the CFG by
 * heart (who jumps to whom), sort the blocks into reverse postorder, work
 * out the dominators by the same Cooper-Harvey-Kennedy method mem2reg
 * already trusts, and from those two facts read off the loops and the merge
 * points. Then a recursive descent rebuilds the nesting, paying off phi
 * nodes with plain copies on the edges as it goes. Nothing in BIR moves a
 * millimetre. The output is a structure tree and an opinion (ok or not ok).
 *
 * Block indices come in two flavours and mixing them up is the quickest way
 * to a bad afternoon. ABSOLUTE indices point into M->blocks[] across the
 * whole module. LOCAL indices are 0-based within the one function being
 * worked on, which is what the dominator and RPO tables are keyed on,
 * exactly as mem2reg keeps them. Helpers spell out which they trade in.
 */

/* ---- Working State ----
 * One big file-static lot of scratch, same arrangement as mem2reg's G. It
 * only has to survive the building of a single function's tree, so reusing
 * it call after call is fine and saves the caller lugging it about. */

typedef struct {
    const bir_module_t *M;
    bst_tree_t *T;

    uint32_t func;
    uint32_t base_block;        /* absolute index of this function's block 0 */
    uint32_t nb;                /* block count for the function */

    /* CFG, all in LOCAL block indices */
    uint32_t succ[BST_MAX_BLOCKS][2];   /* up to two for BR / BR_COND */
    uint8_t  nsucc[BST_MAX_BLOCKS];
    uint32_t preds[BST_MAX_BLOCKS][BST_MAX_PREDS];
    uint8_t  npreds[BST_MAX_BLOCKS];

    /* dominators + reverse postorder, LOCAL */
    uint32_t idom[BST_MAX_BLOCKS];
    uint32_t rpo[BST_MAX_BLOCKS];
    uint32_t rpo_num[BST_MAX_BLOCKS];
    uint32_t rpo_count;

    /* the lie of the land, LOCAL */
    uint8_t  is_header[BST_MAX_BLOCKS];     /* a back edge points here */
    uint32_t follow[BST_MAX_BLOCKS];        /* for a header, the block it spills to on exit */
    uint8_t  is_merge[BST_MAX_BLOCKS];      /* two or more forward edges arrive */
    uint32_t fwd_preds[BST_MAX_BLOCKS];     /* count of forward (non-back) predecessors */

    /* the loop stack the recursion carries, LOCAL headers + their follows */
    uint32_t ls_header[BST_MAX_LOOPS];
    uint32_t ls_follow[BST_MAX_LOOPS];

    uint32_t rdepth;            /* live recursion depth of build_region, capped */
    int bailed;                 /* tripped the moment something cannot be structured */
    const char *bail_reason;
} bst_ctx_t;

static bst_ctx_t G;

/* ---- Tiny conversions ---- */

static uint32_t to_local(const bst_ctx_t *S, uint32_t abs_block)
{
    return abs_block - S->base_block;
}

static uint32_t to_abs(const bst_ctx_t *S, uint32_t local_block)
{
    return S->base_block + local_block;
}

/* Down tools, leaving a note. The first complaint is the one that sticks,
 * since later confusion is usually just the first problem wearing a hat. */
static void bail(bst_ctx_t *S, const char *why)
{
    if (!S->bailed) {
        S->bailed = 1;
        S->bail_reason = why;
    }
}

/* ---- Node and pool minting ----
 * Hand out a fresh node, defaulted to nothing-in-particular so a half-set
 * node still reads sensibly. Overflow trips the bail, and from then on
 * every alloc keeps returning node 0 so the recursion can run itself out
 * without scribbling off the end of the pool. */

static uint32_t mk_node(bst_ctx_t *S, bst_kind_t kind)
{
    bst_tree_t *T = S->T;
    if (T->num_nodes >= BST_MAX_NODES) {
        bail(S, "structure-tree node pool overflow");
        return 0;
    }
    uint32_t id = T->num_nodes++;
    bst_node_t *n = &T->nodes[id];
    n->kind = (uint16_t)kind;
    n->nkids = 0;
    n->block = BST_NONE;
    n->cond = BIR_VAL_NONE;
    n->then_kid = BST_NONE;
    n->else_kid = BST_NONE;
    n->kid_start = 0;
    n->copy_start = 0;
    n->copy_count = 0;
    return id;
}

/* ---- Successors ----
 * Read a block's terminator and report where control can go next, in
 * ABSOLUTE block indices, the same shape mem2reg's get_succs uses. RET and
 * UNREACHABLE go nowhere and say so. BIR_SWITCH is on the roadmap, not the
 * menu, so spotting one trips the bail rather than guessing. */

static int term_succs(bst_ctx_t *S, uint32_t local_block, uint32_t *out)
{
    const bir_module_t *M = S->M;
    const bir_block_t *B = &M->blocks[to_abs(S, local_block)];
    if (B->num_insts == 0) return 0;
    const bir_inst_t *I = &M->insts[B->first_inst + B->num_insts - 1];
    switch (I->op) {
    case BIR_BR:
        out[0] = I->operands[0];
        return 1;
    case BIR_BR_COND:
        out[0] = I->operands[1];     /* the true target */
        out[1] = I->operands[2];     /* the false target */
        return 2;
    case BIR_RET:
    case BIR_UNREACHABLE:
        return 0;
    case BIR_SWITCH:
        bail(S, "BIR_SWITCH is not structured yet (roadmap)");
        return 0;
    default:
        /* A block that wanders off the end with no terminator is not a
         * river anyone can map, so refuse rather than invent an ending. */
        bail(S, "basic block has no terminator");
        return 0;
    }
}

/* ---- Step 1: build the CFG (succ + pred lists, all LOCAL) ---- */

static void build_cfg(bst_ctx_t *S)
{
    memset(S->nsucc, 0, S->nb * sizeof(S->nsucc[0]));
    memset(S->npreds, 0, S->nb * sizeof(S->npreds[0]));

    for (uint32_t b = 0; b < S->nb; b++) {
        uint32_t abs_succ[2];
        int ns = term_succs(S, b, abs_succ);
        if (S->bailed) return;
        S->nsucc[b] = (uint8_t)ns;
        for (int i = 0; i < ns; i++) {
            uint32_t s = to_local(S, abs_succ[i]);
            if (s >= S->nb) { bail(S, "branch target outside the function"); return; }
            S->succ[b][i] = s;
            /* stitch the predecessor side, dodging duplicates so a block
             * that branches to the same place twice does not show up as
             * its own twin */
            int dup = 0;
            for (uint8_t k = 0; k < S->npreds[s]; k++)
                if (S->preds[s][k] == b) { dup = 1; break; }
            if (!dup) {
                if (S->npreds[s] >= BST_MAX_PREDS) {
                    bail(S, "a block gathered more predecessors than the pool holds");
                    return;
                }
                S->preds[s][S->npreds[s]++] = b;
            }
        }
    }
}

/* ---- Step 2: reverse postorder + dominators ----
 * Lifted in spirit straight from mem2reg's step3, because the wheel is
 * round enough already and a second, subtly different round wheel is how
 * compilers grow haunted corners. */

static void compute_dom(bst_ctx_t *S)
{
    uint32_t nb = S->nb;

    /* RPO via an iterative DFS, no recursion, no stack surprises */
    {
        uint8_t  visited[BST_MAX_BLOCKS];
        uint32_t stack[BST_MAX_BLOCKS];
        uint32_t post[BST_MAX_BLOCKS];
        uint32_t post_count = 0;
        int sp = 0;

        memset(visited, 0, nb);
        stack[sp++] = 0;
        visited[0] = 1;

        while (sp > 0) {
            uint32_t cur = stack[sp - 1];
            int pushed = 0;
            for (uint8_t si = 0; si < S->nsucc[cur] && !pushed; si++) {
                uint32_t s = S->succ[cur][si];
                if (!visited[s] && sp < (int)BST_MAX_BLOCKS) {
                    visited[s] = 1;
                    stack[sp++] = s;
                    pushed = 1;
                }
            }
            if (!pushed) {
                sp--;
                if (post_count < nb) post[post_count++] = cur;
            }
        }

        S->rpo_count = post_count;
        for (uint32_t i = 0; i < post_count; i++) {
            S->rpo[i] = post[post_count - 1 - i];
            S->rpo_num[post[post_count - 1 - i]] = i;
        }
        for (uint32_t i = 0; i < nb; i++)
            if (!visited[i]) S->rpo_num[i] = BST_NONE;   /* unreachable, leave it be */
    }

    /* the iterative dominator fixpoint */
    for (uint32_t i = 0; i < nb; i++) S->idom[i] = BST_NONE;
    S->idom[0] = 0;

    int changed = 1;
    int max_iter = (int)nb * 4 + 16;
    while (changed && max_iter-- > 0) {
        changed = 0;
        for (uint32_t ri = 1; ri < S->rpo_count; ri++) {
            uint32_t b = S->rpo[ri];
            uint32_t new_idom = BST_NONE;
            for (uint8_t pi = 0; pi < S->npreds[b]; pi++) {
                uint32_t p = S->preds[b][pi];
                if (S->idom[p] == BST_NONE) continue;
                if (new_idom == BST_NONE) {
                    new_idom = p;
                } else {
                    uint32_t a = new_idom, bb = p;
                    int guard = (int)nb * 2 + 4;
                    while (a != bb && guard-- > 0) {
                        while (S->rpo_num[a] > S->rpo_num[bb] && guard-- > 0)
                            a = S->idom[a];
                        while (S->rpo_num[bb] > S->rpo_num[a] && guard-- > 0)
                            bb = S->idom[bb];
                    }
                    new_idom = a;
                }
            }
            if (new_idom != BST_NONE && S->idom[b] != new_idom) {
                S->idom[b] = new_idom;
                changed = 1;
            }
        }
    }
}

/* Does block a dominate block b? Walk b's idom chain home to the entry; if
 * a is anywhere along that road then a stands between b and the start and
 * the answer is yes. A block dominates itself, which is the boring base
 * case everyone forgets until it bites. */
static int dominates(const bst_ctx_t *S, uint32_t a, uint32_t b)
{
    uint32_t cur = b;
    int guard = (int)S->nb + 4;
    while (guard-- > 0) {
        if (cur == a) return 1;
        if (cur == 0) return 0;             /* reached the entry without finding a */
        if (S->idom[cur] == BST_NONE) return 0;
        if (S->idom[cur] == cur) return 0;  /* the entry's self-loop, road ends */
        cur = S->idom[cur];
    }
    return 0;
}

/* An edge u -> v is a back edge when its target dominates its source, i.e.
 * the only way to reach u was to come past v first, so going u -> v is
 * going round again. This is the textbook test, and on a reducible graph
 * (which is all that arrives here) it catches exactly the loop latches. */
static int is_back_edge(const bst_ctx_t *S, uint32_t u, uint32_t v)
{
    return dominates(S, v, u);
}

/* Can control get from `from` to `to` by following edges? A plain
 * iterative flood, no recursion. Used to check that a loop's several exit
 * doors all empty into the same hallway before trusting it as the follow. */
static int reaches(const bst_ctx_t *S, uint32_t from, uint32_t to)
{
    if (from == to) return 1;
    uint8_t  vis[BST_MAX_BLOCKS];
    uint32_t stk[BST_MAX_BLOCKS];
    int sp = 0;
    memset(vis, 0, S->nb);
    stk[sp++] = from;
    vis[from] = 1;
    while (sp > 0) {
        uint32_t n = stk[--sp];
        for (uint8_t si = 0; si < S->nsucc[n]; si++) {
            uint32_t s = S->succ[n][si];
            if (s == to) return 1;
            if (!vis[s] && sp < (int)BST_MAX_BLOCKS) { vis[s] = 1; stk[sp++] = s; }
        }
    }
    return 0;
}

/* ---- Step 3: read off the loops and merges ----
 * Mark every back-edge target as a loop header, count each block's forward
 * predecessors to spot the merges, and for each header work out the one
 * block the loop spills into when it is finally done. */

static void analyse_regions(bst_ctx_t *S)
{
    memset(S->is_header, 0, S->nb * sizeof(S->is_header[0]));
    memset(S->is_merge, 0, S->nb * sizeof(S->is_merge[0]));
    for (uint32_t b = 0; b < S->nb; b++) {
        S->follow[b] = BST_NONE;
        S->fwd_preds[b] = 0;
    }

    /* headers and forward-pred tallies in one sweep over the edges */
    for (uint32_t u = 0; u < S->nb; u++) {
        if (S->rpo_num[u] == BST_NONE) continue;   /* skip the unreachable */
        for (uint8_t si = 0; si < S->nsucc[u]; si++) {
            uint32_t v = S->succ[u][si];
            if (is_back_edge(S, u, v))
                S->is_header[v] = 1;
            else
                S->fwd_preds[v]++;
        }
    }
    for (uint32_t b = 0; b < S->nb; b++)
        if (S->fwd_preds[b] >= 2) S->is_merge[b] = 1;

    /* the follow block of each loop, which is to say where you end up once
     * the loop has finally let you go. Walk the natural loop of every back
     * edge into the header, marking membership, then collect every block an
     * inside block can step out to. There can be a few such doors at once:
     * the condition failing is one, and each `break` is another. The catch
     * is that a break block is itself outside the natural loop (it cannot
     * reach the latch, having legged it), so it shows up as its own exit
     * even though it only leads on to the real .end a hop later. So the
     * follow is not "the one exit", it is the exit they all flow back
     * together at, which on a tidy reducible loop is the latest one in RPO.
     * If some exit cannot actually reach that reconvergence, the loop has
     * genuine multi-level break (only goto gets you there), and that is the
     * roadmap's problem, so down tools rather than guess. */
    for (uint32_t h = 0; h < S->nb && !S->bailed; h++) {
        if (!S->is_header[h]) continue;

        uint8_t in_loop[BST_MAX_BLOCKS];
        memset(in_loop, 0, S->nb);
        in_loop[h] = 1;

        /* worklist seeded with the latches: preds of h whose edge is a back edge */
        uint32_t work[BST_MAX_BLOCKS];
        int wp = 0;
        for (uint8_t pi = 0; pi < S->npreds[h]; pi++) {
            uint32_t p = S->preds[h][pi];
            if (is_back_edge(S, p, h) && !in_loop[p]) {
                in_loop[p] = 1;
                work[wp++] = p;
            }
        }
        while (wp > 0) {
            uint32_t n = work[--wp];
            for (uint8_t pi = 0; pi < S->npreds[n]; pi++) {
                uint32_t p = S->preds[n][pi];
                if (!in_loop[p]) {
                    in_loop[p] = 1;
                    if (wp < (int)BST_MAX_BLOCKS) work[wp++] = p;
                }
            }
        }

        /* gather the exit-edge targets, and take the reconvergence (latest
         * in RPO) as the follow */
        uint32_t exits[BST_MAX_PREDS];
        int nexits = 0;
        uint32_t follow = BST_NONE;
        uint32_t follow_rpo = 0;
        for (uint32_t n = 0; n < S->nb; n++) {
            if (!in_loop[n]) continue;
            for (uint8_t si = 0; si < S->nsucc[n]; si++) {
                uint32_t s = S->succ[n][si];
                if (in_loop[s]) continue;
                if (S->rpo_num[s] == BST_NONE) continue;
                /* remember it once */
                int seen = 0;
                for (int k = 0; k < nexits; k++) if (exits[k] == s) { seen = 1; break; }
                if (!seen && nexits < (int)BST_MAX_PREDS) exits[nexits++] = s;
                if (follow == BST_NONE || S->rpo_num[s] > follow_rpo) {
                    follow = s;
                    follow_rpo = S->rpo_num[s];
                }
            }
        }

        /* every other door must lead to that reconvergence, or this is a
         * tangle the single-level break model cannot honestly draw */
        for (int k = 0; k < nexits && !S->bailed; k++) {
            if (exits[k] == follow) continue;
            if (!reaches(S, exits[k], follow))
                bail(S, "loop exits do not reconverge (multi-level break, roadmap)");
        }

        S->follow[h] = follow;   /* NONE for a loop that only ever returns out */
    }
}

/* ---- Step 4: the reducibility check ----
 * A retreating edge is one that points backwards in the RPO numbering. On
 * a reducible graph every retreating edge is a proper back edge whose
 * target dominates its source. Find one that is not, and the graph has a
 * loop with two doors in, which the relooper cannot tidy: that is the
 * irreducible case the roadmap reserves for the switch dispatcher. */

static void check_reducible(bst_ctx_t *S)
{
    for (uint32_t u = 0; u < S->nb && !S->bailed; u++) {
        if (S->rpo_num[u] == BST_NONE) continue;
        for (uint8_t si = 0; si < S->nsucc[u]; si++) {
            uint32_t v = S->succ[u][si];
            if (S->rpo_num[v] == BST_NONE) continue;
            int retreating = S->rpo_num[v] <= S->rpo_num[u];
            if (retreating && !dominates(S, v, u)) {
                bail(S, "irreducible control flow (roadmap: switch dispatcher)");
                return;
            }
        }
    }
}

/* ---- phi destruction ----
 * Structured code has no phi nodes; it has assignments. So for every phi at
 * the head of `succ_block`, find the value the predecessor `pred_block`
 * promised it and record a dest = src copy. The phi operands are laid out
 * as (block, value) pairs, inline when they fit and spilled to
 * extra_operands[] when they do not, the same way bir_print reads them.
 * Returns a COPY node, or BST_NONE when there was nothing owing. */

static uint32_t phi_block(const bir_module_t *M, const bir_inst_t *I,
                          uint32_t pair, uint32_t *out_val)
{
    /* pair is the i-th (block, value) couple; hand back its block and value */
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t start = I->operands[0];
        uint32_t blk = M->extra_operands[start + pair * 2];
        *out_val = M->extra_operands[start + pair * 2 + 1];
        return blk;
    }
    *out_val = I->operands[pair * 2 + 1];
    return I->operands[pair * 2];
}

static uint32_t phi_pair_count(const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW)
        return I->operands[1] / 2;
    return (uint32_t)I->num_operands / 2;
}

static uint32_t mk_edge_copies(bst_ctx_t *S, uint32_t pred_local, uint32_t succ_local)
{
    const bir_module_t *M = S->M;
    bst_tree_t *T = S->T;
    if (pred_local == BST_NONE) return BST_NONE;   /* no predecessor, no debts */

    const bir_block_t *B = &M->blocks[to_abs(S, succ_local)];
    uint32_t pred_abs = to_abs(S, pred_local);
    uint32_t start = T->num_copies;
    uint32_t count = 0;

    for (uint32_t ii = 0; ii < B->num_insts; ii++) {
        uint32_t gi = B->first_inst + ii;
        const bir_inst_t *I = &M->insts[gi];
        if (I->op != BIR_PHI) continue;   /* phis huddle at the top of the block */
        uint32_t pairs = phi_pair_count(I);
        for (uint32_t p = 0; p < pairs; p++) {
            uint32_t val;
            uint32_t blk = phi_block(M, I, p, &val);
            if (blk == pred_abs) {
                if (T->num_copies >= BST_MAX_COPIES) {
                    bail(S, "phi-copy pool overflow");
                    return BST_NONE;
                }
                T->copies[T->num_copies].dest = gi;
                T->copies[T->num_copies].src = val;
                T->num_copies++;
                count++;
                break;
            }
        }
    }

    if (count == 0) return BST_NONE;
    uint32_t id = mk_node(S, BST_COPY);
    S->T->nodes[id].copy_start = start;
    S->T->nodes[id].copy_count = count;
    return id;
}

/* ---- The recursive rebuild ----
 * build_region walks the chain of blocks from `start` up to (but not
 * including) `stop`, returning a single node for the lot. `depth` is how
 * many loops deep the recursion currently sits, indexing ls_header /
 * ls_follow. Two scratch buffers per call collect the children before they
 * are committed to the kids[] pool; the nesting is shallow (it tracks
 * source nesting, not block count) so a modest on-stack buffer is plenty. */

#define KIDBUF 512

static uint32_t build_region(bst_ctx_t *S, uint32_t start, uint32_t stop, int depth);

/* Is `b` a header we are already inside, and at what depth? Returns the
 * loop-stack slot, or -1 for "not an enclosing loop of ours". */
static int enclosing_loop_of_header(const bst_ctx_t *S, uint32_t b, int depth)
{
    for (int d = depth - 1; d >= 0; d--)
        if (S->ls_header[d] == b) return d;
    return -1;
}

static int enclosing_loop_of_follow(const bst_ctx_t *S, uint32_t b, int depth)
{
    for (int d = depth - 1; d >= 0; d--)
        if (S->ls_follow[d] == b) return d;
    return -1;
}

/* Bundle a list of collected child nodes into one node. Nought becomes an
 * empty SEQ (a placeholder that emits nothing), one passes straight
 * through unwrapped, and many become a proper SEQ pointing at a run of the
 * kids[] pool. */
static uint32_t wrap_kids(bst_ctx_t *S, const uint32_t *kids, int n)
{
    if (n <= 0) {
        return mk_node(S, BST_SEQ);   /* empty: nkids stays 0 */
    }
    if (n == 1) return kids[0];

    uint32_t seq = mk_node(S, BST_SEQ);
    bst_tree_t *T = S->T;
    if (T->num_kids + (uint32_t)n > BST_MAX_KIDS) {
        bail(S, "SEQ child pool overflow");
        return seq;
    }
    uint32_t ks = T->num_kids;
    for (uint32_t i = 0; i < (uint32_t)n; i++) T->kids[ks + i] = kids[i];
    T->num_kids += (uint32_t)n;
    T->nodes[seq].kid_start = ks;
    T->nodes[seq].nkids = (uint16_t)n;
    return seq;
}

/* Build one arm of an if. `src` is the conditional block, `tgt` the arm's
 * first block, `stop` where the arm should rejoin the world. The arm might
 * be a back edge (continue), a jump to the loop exit (break), a straight
 * fall to the merge (just the phi copies), or a whole region of its own. */
static uint32_t build_arm(bst_ctx_t *S, uint32_t src, uint32_t tgt,
                          uint32_t stop, int depth)
{
    int dh = enclosing_loop_of_header(S, tgt, depth);
    if (dh >= 0 && is_back_edge(S, src, tgt)) {
        uint32_t buf[2]; int n = 0;
        uint32_t cp = mk_edge_copies(S, src, tgt);
        if (cp != BST_NONE) buf[n++] = cp;
        buf[n++] = mk_node(S, BST_CONTINUE);
        return wrap_kids(S, buf, n);
    }
    int df = enclosing_loop_of_follow(S, tgt, depth);
    if (df >= 0) {
        uint32_t buf[2]; int n = 0;
        uint32_t cp = mk_edge_copies(S, src, tgt);
        if (cp != BST_NONE) buf[n++] = cp;
        buf[n++] = mk_node(S, BST_BREAK);
        return wrap_kids(S, buf, n);
    }
    if (tgt == stop) {
        /* the arm does nothing but reconverge; all it owes is the phi copies */
        uint32_t cp = mk_edge_copies(S, src, tgt);
        if (cp != BST_NONE) return cp;
        return mk_node(S, BST_SEQ);   /* truly empty arm */
    }
    return build_region(S, tgt, stop, depth);
}

/* Find the merge of the conditional at `b`: the nearest block it dominates
 * that two or more forward edges arrive at, sitting inside the current
 * region. That block is where the two arms get back together. NONE means
 * they never do (both arms leave for good), which is a perfectly ordinary
 * thing for an if whose branches both return. */
static uint32_t find_merge(const bst_ctx_t *S, uint32_t b, uint32_t stop)
{
    uint32_t best = BST_NONE;
    uint32_t best_rpo = 0xFFFFFFFFu;
    uint32_t b_rpo = S->rpo_num[b];
    uint32_t stop_rpo = (stop == BST_NONE) ? 0xFFFFFFFFu : S->rpo_num[stop];

    for (uint32_t m = 0; m < S->nb; m++) {
        if (!S->is_merge[m]) continue;
        if (S->idom[m] != b) continue;
        if (S->rpo_num[m] == BST_NONE) continue;
        if (S->rpo_num[m] <= b_rpo) continue;          /* must be ahead of us */
        if (stop != BST_NONE && S->rpo_num[m] >= stop_rpo) continue;  /* and inside the region */
        if (S->rpo_num[m] < best_rpo) { best = m; best_rpo = S->rpo_num[m]; }
    }
    return best;
}

/* The straight-line leaf for a block: a BLOCK node, unless the block opens
 * with phis (it is a merge or a loop header), in which case the phis are
 * not emitted here at all, they are settled by copies on the incoming
 * edges. Either way the terminator is somebody else's job. */
static uint32_t mk_block_leaf(bst_ctx_t *S, uint32_t b)
{
    uint32_t id = mk_node(S, BST_BLOCK);
    S->T->nodes[id].block = to_abs(S, b);
    return id;
}

static uint32_t build_region(bst_ctx_t *S, uint32_t start, uint32_t stop, int depth)
{
    uint32_t kids[KIDBUF];
    int nk = 0;
    uint32_t cur = start;
    uint32_t prev = BST_NONE;
    int guard = (int)S->nb + 8;     /* the chain visits each block at most once */

    /* Cap the recursion. Loop nesting is already bounded by the loop stack,
     * but if-nesting recurses through here without touching it, and a frame
     * carries a 2KB child buffer, so an absurdly nested function could walk
     * the C stack off the edge. Refuse long before that. */
    if (++S->rdepth > BST_MAX_RDEPTH) {
        bail(S, "control flow nested past the structuriser's depth cap");
        S->rdepth--;
        return mk_node(S, BST_SEQ);
    }

    #define PUSH(node) do { \
        if (nk < KIDBUF) kids[nk++] = (node); \
        else bail(S, "region child buffer overflow"); \
    } while (0)

    while (cur != stop && cur != BST_NONE && !S->bailed) {
        if (guard-- <= 0) { bail(S, "region walk did not converge"); break; }

        /* Arriving at a block by an edge (prev set) might mean we have
         * walked onto an enclosing loop's follow, set there as the merge
         * after an if whose arms both broke out. That is a break, not a
         * block to emit: take it and stop. (The copies were already laid
         * down by whoever sent us here.) */
        if (prev != BST_NONE) {
            if (enclosing_loop_of_follow(S, cur, depth) >= 0) {
                PUSH(mk_node(S, BST_BREAK));
                cur = BST_NONE;
                break;
            }
        }

        /* A loop header we are not already inside: open the loop, build its
         * body, then carry on at the follow. The header's phis are settled
         * twice over, once from the block that fell in here (the preheader,
         * which is prev) and once per latch inside the body. */
        if (S->is_header[cur] && enclosing_loop_of_header(S, cur, depth) < 0) {
            uint32_t cp = mk_edge_copies(S, prev, cur);
            if (cp != BST_NONE) PUSH(cp);

            if (depth >= BST_MAX_LOOPS) { bail(S, "loop nesting too deep"); break; }
            uint32_t follow = S->follow[cur];
            S->ls_header[depth] = cur;
            S->ls_follow[depth] = follow;

            uint32_t body = build_region(S, cur, BST_NONE, depth + 1);

            uint32_t loop = mk_node(S, BST_LOOP);
            S->T->nodes[loop].block = to_abs(S, cur);
            S->T->nodes[loop].then_kid = body;
            PUSH(loop);

            prev = cur;
            cur = follow;     /* NONE if the loop only ever returns its way out */
            continue;
        }

        /* an ordinary block: emit its straight-line work, then deal with
         * however it ends */
        PUSH(mk_block_leaf(S, cur));

        uint32_t abs_succ[2];
        (void)term_succs(S, cur, abs_succ);   /* fills abs_succ; bail flag checked next */
        if (S->bailed) break;

        const bir_block_t *B = &S->M->blocks[to_abs(S, cur)];
        const bir_inst_t *term = &S->M->insts[B->first_inst + B->num_insts - 1];

        if (term->op == BIR_RET) {
            uint32_t r = mk_node(S, BST_RET);
            S->T->nodes[r].block = to_abs(S, cur);
            PUSH(r);
            cur = BST_NONE;
            break;
        }
        if (term->op == BIR_UNREACHABLE) {
            PUSH(mk_node(S, BST_UNREACHABLE));
            cur = BST_NONE;
            break;
        }

        if (term->op == BIR_BR) {
            uint32_t t = to_local(S, abs_succ[0]);
            int dh = enclosing_loop_of_header(S, t, depth);
            if (dh >= 0 && is_back_edge(S, cur, t)) {
                uint32_t cp = mk_edge_copies(S, cur, t);
                if (cp != BST_NONE) PUSH(cp);
                PUSH(mk_node(S, BST_CONTINUE));
                cur = BST_NONE;
                break;
            }
            int df = enclosing_loop_of_follow(S, t, depth);
            if (df >= 0) {
                uint32_t cp = mk_edge_copies(S, cur, t);
                if (cp != BST_NONE) PUSH(cp);
                PUSH(mk_node(S, BST_BREAK));
                cur = BST_NONE;
                break;
            }
            if (t == stop) {
                uint32_t cp = mk_edge_copies(S, cur, t);
                if (cp != BST_NONE) PUSH(cp);
                cur = BST_NONE;       /* fall through; the caller emits stop */
                break;
            }
            /* a plain inlined fall to a block we alone reach */
            prev = cur;
            cur = t;
            continue;
        }

        if (term->op == BIR_BR_COND) {
            uint32_t cond = term->operands[0];
            uint32_t tt = to_local(S, abs_succ[0]);
            uint32_t ff = to_local(S, abs_succ[1]);
            uint32_t merge = find_merge(S, cur, stop);
            uint32_t arm_stop = (merge != BST_NONE) ? merge : stop;

            uint32_t then_node = build_arm(S, cur, tt, arm_stop, depth);
            uint32_t else_node = BST_NONE;
            if (!(merge != BST_NONE && ff == merge)) {
                /* there is a real else only when the false target is not
                 * just the merge waiting on the other side of the fork */
                else_node = build_arm(S, cur, ff, arm_stop, depth);
            }

            uint32_t ifn = mk_node(S, BST_IF);
            S->T->nodes[ifn].block = to_abs(S, cur);
            S->T->nodes[ifn].cond = cond;
            S->T->nodes[ifn].then_kid = then_node;
            S->T->nodes[ifn].else_kid = else_node;
            PUSH(ifn);

            prev = cur;
            cur = merge;     /* NONE means both arms left for good; the chain ends */
            continue;
        }

        /* term_succs already bailed on anything else (SWITCH, no terminator) */
        break;
    }

    #undef PUSH
    S->rdepth--;
    return wrap_kids(S, kids, nk);
}

/* ---- Entry point ---- */

int bir_structurize(const bir_module_t *M, uint32_t func_index, bst_tree_t *out)
{
    memset(out, 0, sizeof(*out));
    out->root = BST_NONE;
    out->func = func_index;
    out->ok = 0;

    if (func_index >= M->num_funcs) {
        out->bail_reason = "no such function";
        return BC_OK;
    }
    const bir_func_t *F = &M->funcs[func_index];

    memset(&G, 0, sizeof(G));
    G.M = M;
    G.T = out;
    G.func = func_index;
    G.base_block = F->first_block;
    G.nb = F->num_blocks;
    G.bailed = 0;
    G.bail_reason = NULL;

    if (G.nb == 0) {
        /* a declaration with no body: a clean, empty, honest nothing */
        out->ok = 1;
        out->root = BST_NONE;
        return BC_OK;
    }
    if (G.nb > BST_MAX_BLOCKS) {
        out->bail_reason = "function has more blocks than the structuriser holds";
        return BC_OK;
    }

    build_cfg(&G);
    if (!G.bailed) compute_dom(&G);
    if (!G.bailed) analyse_regions(&G);
    if (!G.bailed) check_reducible(&G);

    if (G.bailed) {
        out->ok = 0;
        out->bail_reason = G.bail_reason;
        return BC_OK;
    }

    out->root = build_region(&G, 0, BST_NONE, 0);

    if (G.bailed) {
        out->ok = 0;
        out->bail_reason = G.bail_reason;
        return BC_OK;
    }

    out->ok = 1;
    return BC_OK;
}
