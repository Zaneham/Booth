/* bir_inline.c -- inline calls to user __device__ functions.
 *
 * BarraCUDA's IR references values by absolute index into one flat,
 * append-only insts[] array, and a function's blocks and instructions are
 * contiguous slices of the global arenas. That makes an in-place insert a
 * non-starter: dropping instructions into the middle would slide every
 * later index and invalidate every value reference in the module. So the
 * trick here is to never insert. Each caller that contains a device call is
 * rebuilt from scratch at the end of the arenas, with the callee bodies
 * spliced in as we go, and the function is then repointed at its fresh
 * blocks. The old blocks are left orphaned, dead but harmless.
 *
 * The one map that keeps this honest is per-opcode operand classification,
 * which slot is a value, which is a block, which is the call's callee index.
 * It is taken from bir_insert.c, which took it from the printer. Keep the
 * three in agreement and a rewrite never corrupts a module.
 */

#include "bir_inline.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define INL_MAX_ROUNDS          64      /* depth guard: recursion never converges */
#define INL_MAX_CALLEE_INSTS    8192    /* a device helper is small */
#define INL_MAX_CALLEE_BLOCKS   256     /* and does not sprawl into many blocks */
#define INL_MAX_ARGS            64

/* spliced[] tags each newly emitted instruction so the remap passes know who
   owns it: 0 = caller-origin, remapped in the caller's deferred pass; 1 =
   callee clone, remapped against the per-splice maps; 2 = synthetic and
   already final (the branches and phi this pass manufactures), never remapped. */
#define INL_CALLER   0
#define INL_CLONE    1
#define INL_SYNTH    2

typedef struct {
    bir_module_t *M;
    uint32_t val_map[BIR_MAX_INSTS];    /* old caller inst index -> new value */
    uint32_t blk_map[BIR_MAX_BLOCKS];   /* old block index -> new block (caller and callee) */
    uint8_t  spliced[BIR_MAX_INSTS];    /* INL_CALLER / INL_CLONE / INL_SYNTH */
    uint32_t local[INL_MAX_CALLEE_INSTS]; /* callee inst index -> new value, per splice */
    uint32_t ret_pred[INL_MAX_CALLEE_BLOCKS]; /* block each callee return sat in */
    uint32_t ret_val[INL_MAX_CALLEE_BLOCKS];  /* value each callee return yielded */
    uint8_t  warned[BIR_MAX_FUNCS];     /* callee already warned about */
} inl_t;

/* ---- Value / block mapping ---- */

/* Map a value operand through a table indexed from `base`. Constants and
 * the null sentinel pass through untouched. */
static uint32_t map_val(const uint32_t *vmap, uint32_t base, uint32_t v)
{
    if (v == BIR_VAL_NONE || BIR_VAL_IS_CONST(v))
        return v;
    return vmap[BIR_VAL_INDEX(v) - base];
}

/* Remap the operands of one already-copied instruction. Value slots go
 * through vmap (based at vbase), block slots through bmap (based at bbase);
 * a null bmap leaves block indices alone, which is the single-block callee
 * case where the only block reference would be a terminator we don't copy.
 * The call's callee index is a function index, not a value, and stays put. */
static void remap_ops(bir_module_t *M, uint32_t ni,
                      const uint32_t *vmap, uint32_t vbase,
                      const uint32_t *bmap, uint32_t bbase)
{
    bir_inst_t *I = &M->insts[ni];
    int      ovf   = (I->num_operands == BIR_OPERANDS_OVERFLOW);
    uint32_t start = I->operands[0];
    uint32_t count = I->operands[1];
    uint32_t *xo   = M->extra_operands;
    uint32_t i;
    uint8_t  k;

#define MV(x) map_val(vmap, vbase, (x))
#define MB(x) (bmap ? bmap[(x) - bbase] : (x))

    switch (I->op) {

    case BIR_BR:                       /* ops[0] = block */
        if (!ovf) I->operands[0] = MB(I->operands[0]);
        break;

    case BIR_BR_COND:                  /* ops[0]=cond value, [1..3]=blocks */
        I->operands[0] = MV(I->operands[0]);
        I->operands[1] = MB(I->operands[1]);
        I->operands[2] = MB(I->operands[2]);
        I->operands[3] = MB(I->operands[3]);
        break;

    case BIR_SWITCH:
        if (ovf) {                     /* extra: val, default-blk, (const,blk)* */
            if (count >= 1) xo[start]     = MV(xo[start]);
            if (count >= 2) xo[start + 1] = MB(xo[start + 1]);
            for (i = 2; i + 1 < count; i += 2)
                xo[start + i + 1] = MB(xo[start + i + 1]);
        } else {                       /* ops[0]=val, ops[1]=default block */
            I->operands[0] = MV(I->operands[0]);
            I->operands[1] = MB(I->operands[1]);
        }
        break;

    case BIR_PHI:                      /* (block, value) pairs */
        if (ovf) {
            for (i = 0; i + 1 < count; i += 2) {
                xo[start + i]     = MB(xo[start + i]);
                xo[start + i + 1] = MV(xo[start + i + 1]);
            }
        } else {
            for (k = 0; (uint32_t)k + 1 < I->num_operands; k += 2) {
                I->operands[k]     = MB(I->operands[k]);
                I->operands[k + 1] = MV(I->operands[k + 1]);
            }
        }
        break;

    case BIR_CALL:                     /* ops[0]=func index (leave), rest=args */
        if (ovf) {
            for (i = 1; i < count; i++)
                xo[start + i] = MV(xo[start + i]);
        } else {
            for (k = 1; k < I->num_operands; k++)
                I->operands[k] = MV(I->operands[k]);
        }
        break;

    default:                           /* every other opcode: all values */
        if (!ovf) {
            uint8_t no = I->num_operands;
            if (no > BIR_OPERANDS_INLINE) no = BIR_OPERANDS_INLINE;
            for (k = 0; k < no; k++)
                I->operands[k] = MV(I->operands[k]);
        } else {
            for (i = 0; i < count; i++)
                xo[start + i] = MV(xo[start + i]);
        }
        break;
    }

#undef MV
#undef MB
}

/* ---- Copying ---- */

/* Append a copy of one instruction to the arenas and return its new index.
 * An overflow-form instruction gets a fresh extra_operands slab so that a
 * callee spliced twice never has two copies aliasing one slab. Operands are
 * copied raw here; the caller remaps them afterwards. */
static uint32_t copy_inst(bir_module_t *M, const bir_inst_t *I, uint32_t line)
{
    uint32_t ni;

    if (M->num_insts >= BIR_MAX_INSTS)
        return BIR_VAL_NONE;

    ni = M->num_insts;
    M->insts[ni]      = *I;
    M->inst_lines[ni] = line;

    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t src = I->operands[0], count = I->operands[1], nstart, k;
        if (M->num_extra_ops + count > BIR_MAX_EXTRA_OPS)
            return BIR_VAL_NONE;
        nstart = M->num_extra_ops;
        for (k = 0; k < count; k++)
            M->extra_operands[nstart + k] = M->extra_operands[src + k];
        M->num_extra_ops += count;
        M->insts[ni].operands[0] = nstart;
        M->insts[ni].operands[1] = count;
    }

    M->num_insts++;
    return ni;
}

/* ---- Call helpers ---- */

static uint32_t call_func_index(const bir_module_t *M, const bir_inst_t *I)
{
    if (I->num_operands == BIR_OPERANDS_OVERFLOW)
        return M->extra_operands[I->operands[0]];
    return I->operands[0];
}

static int read_call_args(const bir_module_t *M, const bir_inst_t *I,
                          uint32_t *out, int max)
{
    int n = 0;
    if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
        uint32_t start = I->operands[0], count = I->operands[1], k;
        for (k = 1; k < count && n < max; k++)
            out[n++] = M->extra_operands[start + k];
    } else {
        uint8_t k;
        for (k = 1; k < I->num_operands && n < max; k++)
            out[n++] = I->operands[k];
    }
    return n;
}

static const char *func_name(const bir_module_t *M, uint32_t fi)
{
    uint32_t off = M->funcs[fi].name;
    return (off < M->string_len) ? &M->strings[off] : "?";
}

static int is_device(const bir_module_t *M, uint32_t fi)
{
    return fi < M->num_funcs && (M->funcs[fi].cuda_flags & CUDA_DEVICE);
}

/* A device callee we can inline: any that fits our bounded working set.
   Single-block ones splice straight in; multi-block ones get the full
   block-cloning splice. Anything larger falls through to a warning. */
static int is_inlinable(const bir_module_t *M, uint32_t fi)
{
    return is_device(M, fi)
        && M->funcs[fi].num_blocks  >= 1
        && M->funcs[fi].num_blocks  <= INL_MAX_CALLEE_BLOCKS
        && M->funcs[fi].total_insts <= INL_MAX_CALLEE_INSTS;
}

/* ---- Block / branch / phi construction ---- */

static uint32_t open_block(inl_t *X, uint32_t name)
{
    bir_module_t *M = X->M;
    uint32_t bn;
    if (M->num_blocks >= BIR_MAX_BLOCKS) return BIR_VAL_NONE;
    bn = M->num_blocks++;
    M->blocks[bn].name       = name;
    M->blocks[bn].first_inst = M->num_insts;
    M->blocks[bn].num_insts  = 0;
    return bn;
}

static void close_block(bir_module_t *M, uint32_t bn)
{
    M->blocks[bn].num_insts = M->num_insts - M->blocks[bn].first_inst;
}

/* Append an unconditional branch to `target` and tag it final. */
static uint32_t append_br(inl_t *X, uint32_t target)
{
    bir_module_t *M = X->M;
    bir_inst_t *I;
    uint32_t ni;
    if (M->num_insts >= BIR_MAX_INSTS) return BIR_VAL_NONE;
    ni = M->num_insts++;
    I = &M->insts[ni];
    I->op           = BIR_BR;
    I->num_operands = 1;
    I->subop        = 0;
    I->type         = bir_type_void(M);
    I->operands[0]  = target;
    M->inst_lines[ni] = 0;
    X->spliced[ni]  = INL_SYNTH;
    return ni;
}

/* Emit a phi merging one (block, value) pair per callee return. Operands are
   final new-space references, so it is tagged final and never remapped. */
static uint32_t emit_phi(inl_t *X, uint32_t type, const uint32_t *preds,
                         const uint32_t *vals, uint32_t n)
{
    bir_module_t *M = X->M;
    bir_inst_t *I;
    uint32_t ni, nops = n * 2, i;

    if (M->num_insts >= BIR_MAX_INSTS) return BIR_VAL_NONE;
    ni = M->num_insts++;
    I = &M->insts[ni];
    I->op    = BIR_PHI;
    I->subop = 0;
    I->type  = type;

    if (nops <= BIR_OPERANDS_INLINE) {
        I->num_operands = (uint8_t)nops;
        for (i = 0; i < n; i++) {
            I->operands[2 * i]     = preds[i];
            I->operands[2 * i + 1] = vals[i];
        }
    } else {
        uint32_t start = M->num_extra_ops;
        if (start + nops > BIR_MAX_EXTRA_OPS) return BIR_VAL_NONE;
        for (i = 0; i < n; i++) {
            M->extra_operands[start + 2 * i]     = preds[i];
            M->extra_operands[start + 2 * i + 1] = vals[i];
        }
        M->num_extra_ops += nops;
        I->num_operands = BIR_OPERANDS_OVERFLOW;
        I->operands[0]  = start;
        I->operands[1]  = nops;
    }
    M->inst_lines[ni] = 0;
    X->spliced[ni]    = INL_SYNTH;
    return ni;
}

/* ---- Splice one call ---- */

/* Splice a single straight-line block callee at `call_oi`. Its parameters
 * bind to the call's arguments (already mapped into the caller's new value
 * space), its body copies into the current block, and its return value
 * becomes the call's result via val_map. Because a straight-line block
 * defines before it uses, the body remaps as it is copied, no second pass. */
static int splice_sb(inl_t *X, uint32_t call_oi, uint32_t cfi)
{
    bir_module_t *M = X->M;
    bir_func_t  *C  = &M->funcs[cfi];
    bir_block_t *CB = &M->blocks[C->first_block];
    uint32_t cb_first = CB->first_inst, cb_num = CB->num_insts, j;
    uint32_t arg_raw[INL_MAX_ARGS], arg_new[INL_MAX_ARGS];
    uint32_t *local = X->local;
    uint32_t retval = BIR_VAL_NONE;
    int nargs, a;

    nargs = read_call_args(M, &M->insts[call_oi], arg_raw, INL_MAX_ARGS);
    for (a = 0; a < nargs; a++)
        arg_new[a] = map_val(X->val_map, 0, arg_raw[a]);

    for (j = 0; j < cb_num; j++) {
        uint32_t ci = cb_first + j;
        bir_inst_t *CI = &M->insts[ci];
        uint32_t ni;

        if (CI->op == BIR_PARAM) {          /* param -> bound argument */
            local[j] = (CI->subop < nargs) ? arg_new[CI->subop] : BIR_VAL_NONE;
            continue;
        }
        if (CI->op == BIR_RET) {            /* return value -> call result */
            retval = (CI->num_operands >= 1)
                   ? map_val(local, cb_first, CI->operands[0]) : BIR_VAL_NONE;
            continue;
        }
        if (CI->op == BIR_UNREACHABLE)      /* falls off, no result */
            continue;

        ni = copy_inst(M, CI, M->inst_lines[ci]);
        if (ni == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
        remap_ops(M, ni, local, cb_first, NULL, 0);
        local[j] = ni;
        X->spliced[ni] = INL_CLONE;
    }

    X->val_map[call_oi] = retval;
    return BC_OK;
}

/* Splice a multi-block device callee at `call_oi`, splitting the current
 * caller block. The block so far is capped with a branch into the cloned
 * callee entry; every callee block is cloned (its branch targets remapped);
 * each return becomes a branch to a fresh continuation block, collecting its
 * value and predecessor. The continuation opens with a phi over those returns
 * (or the lone value if there is only one), which becomes the call's result,
 * and *cur advances to it so the caller keeps emitting there. */
static int splice_mb(inl_t *X, uint32_t call_oi, uint32_t cfi, uint32_t *cur)
{
    bir_module_t *M = X->M;
    bir_func_t  *C  = &M->funcs[cfi];
    uint32_t cb0 = C->first_block, cnb = C->num_blocks;
    uint32_t cbase = M->blocks[cb0].first_inst;   /* callee insts are contiguous */
    uint32_t *local = X->local;
    uint32_t arg_raw[INL_MAX_ARGS], arg_new[INL_MAX_ARGS];
    uint32_t cont, cstart, cend, ni, c, nret = 0, result;
    int nargs, a;

    nargs = read_call_args(M, &M->insts[call_oi], arg_raw, INL_MAX_ARGS);
    for (a = 0; a < nargs; a++)
        arg_new[a] = map_val(X->val_map, 0, arg_raw[a]);

    /* Reserve the callee blocks and one continuation, contiguously after the
       current block, so the whole function stays one contiguous slice. */
    if (M->num_blocks + cnb + 1 > BIR_MAX_BLOCKS) return BC_ERR_OVERFLOW;
    for (c = 0; c < cnb; c++)
        X->blk_map[cb0 + c] = M->num_blocks + c;
    cont = M->num_blocks + cnb;
    M->num_blocks += cnb + 1;

    /* Cap the current block with the jump into the callee entry. */
    if (append_br(X, X->blk_map[cb0]) == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
    close_block(M, *cur);

    /* Clone every callee block. Params bind to arguments, returns turn into
       branches to the continuation while we note the value and its block. */
    cstart = M->num_insts;
    for (c = 0; c < cnb; c++) {
        bir_block_t *OCB = &M->blocks[cb0 + c];
        uint32_t ncb = X->blk_map[cb0 + c], j;
        M->blocks[ncb].name       = OCB->name;
        M->blocks[ncb].first_inst = M->num_insts;

        for (j = 0; j < OCB->num_insts; j++) {
            uint32_t ci = OCB->first_inst + j;
            bir_inst_t *CI = &M->insts[ci];

            if (CI->op == BIR_PARAM) {
                local[ci - cbase] = (CI->subop < nargs)
                                  ? arg_new[CI->subop] : BIR_VAL_NONE;
                continue;
            }
            if (CI->op == BIR_RET) {
                if (nret < INL_MAX_CALLEE_BLOCKS) {
                    X->ret_pred[nret] = ncb;
                    X->ret_val[nret]  = (CI->num_operands >= 1)
                                      ? CI->operands[0] : BIR_VAL_NONE;
                    nret++;
                }
                if (append_br(X, cont) == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
                continue;
            }

            ni = copy_inst(M, CI, M->inst_lines[ci]);
            if (ni == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
            local[ci - cbase] = ni;
            X->spliced[ni] = INL_CLONE;
        }
        close_block(M, ncb);
    }
    cend = M->num_insts;

    /* Now that local is complete, remap the callee clones. Values go through
       local, block targets through blk_map; the synthetic branches skip. */
    for (ni = cstart; ni < cend; ni++)
        if (X->spliced[ni] == INL_CLONE)
            remap_ops(M, ni, local, cbase, X->blk_map, 0);

    /* Open the continuation and settle the call's result. */
    M->blocks[cont].name       = M->blocks[cb0].name;
    M->blocks[cont].first_inst = M->num_insts;

    if (nret == 0) {
        result = BIR_VAL_NONE;
    } else if (nret == 1) {
        result = map_val(local, cbase, X->ret_val[0]);
    } else {
        uint32_t rtype = M->types[C->type].inner, i;
        for (i = 0; i < nret; i++)
            X->ret_val[i] = map_val(local, cbase, X->ret_val[i]);
        result = emit_phi(X, rtype, X->ret_pred, X->ret_val, nret);
        if (result == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
    }

    X->val_map[call_oi] = result;
    *cur = cont;
    return BC_OK;
}

/* ---- Rebuild one caller ---- */

static int rebuild_func(inl_t *X, uint32_t fidx)
{
    bir_module_t *M = X->M;
    bir_func_t  *F  = &M->funcs[fidx];
    uint32_t ob0 = F->first_block, onb = F->num_blocks;
    uint32_t nb_start, new_inst_start, i, ni, b, tot;

    if (onb == 0) return BC_OK;

    nb_start       = M->num_blocks;
    new_inst_start = M->num_insts;

    /* Emit each old block as a fresh block, appended contiguously. A splice
       may grow the block count mid-stream (callee blocks plus a continuation),
       so `cur` is the block we are currently filling rather than a fixed slot,
       and it advances when a multi-block callee splits the block open. */
    for (i = 0; i < onb; i++) {
        bir_block_t *OB = &M->blocks[ob0 + i];
        uint32_t cur = open_block(X, OB->name);
        uint32_t j;

        if (cur == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
        X->blk_map[ob0 + i] = cur;

        for (j = 0; j < OB->num_insts; j++) {
            uint32_t oi = OB->first_inst + j;
            bir_inst_t *I = &M->insts[oi];

            if (I->op == BIR_CALL) {
                uint32_t cfi = call_func_index(M, I);
                if (is_inlinable(M, cfi)) {
                    int rc = (M->funcs[cfi].num_blocks == 1)
                           ? splice_sb(X, oi, cfi)
                           : splice_mb(X, oi, cfi, &cur);
                    if (rc != BC_OK) return rc;
                    continue;
                }
                /* Non-inlinable device or ordinary call: copy it through. */
            }

            ni = copy_inst(M, I, M->inst_lines[oi]);
            if (ni == BIR_VAL_NONE) return BC_ERR_OVERFLOW;
            X->val_map[oi]  = ni;
            X->spliced[ni]  = INL_CALLER;
        }

        close_block(M, cur);
    }

    /* Caller-origin instructions still carry old operand indices (forward
     * references across blocks mean this can't be done during the copy).
     * Remap them now; clones and synthetic instructions are already final. */
    for (ni = new_inst_start; ni < M->num_insts; ni++) {
        if (X->spliced[ni] != INL_CALLER) continue;
        remap_ops(M, ni, X->val_map, 0, X->blk_map, 0);
    }

    F->first_block = nb_start;
    F->num_blocks  = (uint16_t)(M->num_blocks - nb_start);
    tot = 0;
    for (b = nb_start; b < M->num_blocks; b++)
        tot += M->blocks[b].num_insts;
    F->total_insts = tot;

    return BC_OK;
}

/* ---- Driver ---- */

static int func_has_inlinable_call(const bir_module_t *M, uint32_t fidx)
{
    const bir_func_t *F = &M->funcs[fidx];
    uint32_t b;
    for (b = 0; b < F->num_blocks; b++) {
        const bir_block_t *B = &M->blocks[F->first_block + b];
        uint32_t k;
        for (k = 0; k < B->num_insts; k++) {
            const bir_inst_t *I = &M->insts[B->first_inst + k];
            if (I->op == BIR_CALL && is_inlinable(M, call_func_index(M, I)))
                return 1;
        }
    }
    return 0;
}

/* Warn once for each device callee still called after inlining. With the
   splice handling both straight-line and control-flow bodies, the only thing
   left here is a callee too large for the bounded working set. It falls
   through as a plain call the GPU backends can't emit, so say so plainly. */
static void warn_uninlined(inl_t *X)
{
    bir_module_t *M = X->M;
    uint32_t f, b, k;
    for (f = 0; f < M->num_funcs; f++) {
        const bir_func_t *F = &M->funcs[f];
        for (b = 0; b < F->num_blocks; b++) {
            const bir_block_t *B = &M->blocks[F->first_block + b];
            for (k = 0; k < B->num_insts; k++) {
                const bir_inst_t *I = &M->insts[B->first_inst + k];
                uint32_t cfi;
                if (I->op != BIR_CALL) continue;
                cfi = call_func_index(M, I);
                if (is_device(M, cfi) && !X->warned[cfi]) {
                    X->warned[cfi] = 1;
                    fprintf(stderr, "barracuda: warning: __device__ function "
                        "'%s' has control flow and is not inlined yet; calls "
                        "to it will not run correctly on this backend. Inline "
                        "it by hand for now. (issue #101)\n", func_name(M, cfi));
                }
            }
        }
    }
}

/* The pass proper, run against a prepared context. Split from the public
   entry point so the one heap allocation has a single owner and cleanup does
   not have to be threaded through every early exit. */
static int inline_run(inl_t *X)
{
    bir_module_t *M = X->M;
    int round;

    /* Each round inlines every inlinable device call it finds, straight-line
     * and control-flow bodies alike. A callee that itself rings up another
     * device function leaves that inner call behind as a copy, so the loop
     * peels one layer of nesting per pass, onion-style, and settles once none
     * remain. Anything still inlinable after the guard is a __device__ function
     * that calls itself, which would happily inline until the heat death of the
     * universe. We give it 64 rounds and then, regrettably, stop being polite. */
    for (round = 0; round < INL_MAX_ROUNDS; round++) {
        int changed = 0;
        uint32_t nf = M->num_funcs, f;
        for (f = 0; f < nf; f++) {
            if (func_has_inlinable_call(M, f)) {
                int rc = rebuild_func(X, f);
                if (rc != BC_OK) return rc;
                changed = 1;
            }
        }
        if (!changed) break;
    }

    {
        uint32_t f;
        for (f = 0; f < M->num_funcs; f++) {
            if (func_has_inlinable_call(M, f)) {
                fprintf(stderr, "barracuda: device call inlining did not "
                        "converge after %d rounds (recursive __device__ "
                        "function '%s'?)\n", INL_MAX_ROUNDS, func_name(M, f));
                return BC_ERR_VERIFY;
            }
        }
    }

    warn_uninlined(X);
    return BC_OK;
}

int bir_inline_device(bir_module_t *M)
{
    inl_t *X;
    int rc;

    if (!M) return BC_ERR_IO;

    X = (inl_t *)malloc(sizeof(inl_t));
    if (!X) return BC_ERR_IO;
    X->M = M;
    memset(X->spliced, 0, sizeof(X->spliced));
    memset(X->warned, 0, sizeof(X->warned));

    rc = inline_run(X);

    free(X);
    return rc;
}
