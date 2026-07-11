#include "tdf.h"
#include <stdio.h>
#include <string.h>

/*
 * Tensix kernel fission analysis.
 *
 * Reader/compute/writer is not a Tenstorrent invention: it is the same
 * producer/consumer pattern that ran ICL paper-tape pipelines, that CICS named
 * transaction regions after, and that TPF lays out across baby cores. The
 * hardware just makes it physical, three baby RISC-V cores per Tensix talking
 * through L1 circular buffers with a NoC instead of a backplane. Today this pass
 * takes a SOLO Tensix module whose body is a CUDA __global__ in BIR, identifies
 * which pointer parameters are read and which are written, and rewrites the TDF
 * graph as a three-region pipeline with channels for the data flow. It does not
 * yet split the BIR body into three: the CMP region still owns the original BIR,
 * and the RDR and WRT regions are placeholders for bodies a follow-up pass will
 * synthesise, at which point td_lower for Tensix accepts the fissioned shape and
 * the multi-baby-core path comes online.
 */

#define FS_MAX_PARAMS   64
#define FS_TRACE_GUARD  16

/* ---- SSA -> param trace ----
 *
 * Same shape as tensix/datamov.c::trace_to_param. Follows GEPs,
 * bitcasts, and alloca-load chains back to the BIR_PARAM that
 * originally produced the address. Returns the parameter index or
 * -1 if the trail goes cold.
 */
static int trace_to_param(const bir_module_t *bir, uint32_t val)
{
    int guard = FS_TRACE_GUARD;
    while (guard-- > 0) {
        if (val == BIR_VAL_NONE || BIR_VAL_IS_CONST(val)) return -1;
        uint32_t idx = BIR_VAL_INDEX(val);
        if (idx >= bir->num_insts) return -1;
        const bir_inst_t *I = &bir->insts[idx];

        if (I->op == BIR_PARAM)
            return (int)I->subop;

        if (I->op == BIR_GEP    || I->op == BIR_BITCAST ||
            I->op == BIR_INTTOPTR || I->op == BIR_PTRTOINT) {
            val = I->operands[0];
            continue;
        }
        if (I->op == BIR_LOAD) {
            val = I->operands[0];
            continue;
        }
        return -1;
    }
    return -1;
}

/* ---- Kernel locator ----
 *
 * Pick the first CUDA_GLOBAL function in the module. Falls back to
 * the first function if none is marked, which is the common shape
 * after sema has dropped the host code.
 */
static int find_kernel(const bir_module_t *bir, uint32_t *out_func)
{
    for (uint32_t fi = 0; fi < bir->num_funcs; fi++) {
        if (bir->funcs[fi].cuda_flags & CUDA_GLOBAL) {
            *out_func = fi;
            return BC_OK;
        }
    }
    if (bir->num_funcs > 0) {
        *out_func = 0;
        return BC_OK;
    }
    return BC_ERR_TDF;
}

/* ---- Parameter classification ----
 *
 * Walk every instruction in the kernel function. For each load or
 * store, trace the address back to a parameter and tag that
 * parameter as input or output. A parameter that is both loaded and
 * stored is in-out (two channels, one each way).
 *
 * Returns BC_ERR_TDF if a BIR_CALL is encountered, which means we
 * cannot reason about the dataflow without doing inter-procedural
 * analysis, which is a problem for a later sitting.
 */
typedef struct {
    uint8_t  is_ptr;     /* 1 if parameter is a global pointer        */
    uint8_t  loaded;     /* 1 if any load traces back to this param   */
    uint8_t  stored;     /* 1 if any store traces back to this param  */
    uint8_t  _pad;
} fs_param_t;

static int classify_params(const bir_module_t *bir, uint32_t func_idx,
                           fs_param_t *params, uint32_t *nparams_out)
{
    const bir_func_t *F = &bir->funcs[func_idx];
    uint32_t nparams = F->num_params < FS_MAX_PARAMS
                       ? F->num_params : FS_MAX_PARAMS;
    *nparams_out = nparams;
    memset(params, 0, sizeof(fs_param_t) * nparams);

    /* Pass 1: mark which params are global pointers (the only kind
     * that can flow through a Tensix NoC channel). */
    uint32_t first = F->first_block < bir->num_blocks
                     ? bir->blocks[F->first_block].first_inst : 0;
    int guard = 8192;
    for (uint32_t i = first; i < bir->num_insts && guard > 0; i++, guard--) {
        const bir_inst_t *I = &bir->insts[i];
        if (I->op != BIR_PARAM) continue;
        uint32_t pi = I->subop;
        if (pi >= nparams) continue;
        if (I->type < bir->num_types) {
            const bir_type_t *T = &bir->types[I->type];
            if (T->kind == BIR_TYPE_PTR &&
                (T->addrspace == BIR_AS_GLOBAL ||
                 T->addrspace == BIR_AS_GENERIC))
                params[pi].is_ptr = 1;
        }
    }

    /* Pass 2: scan every block, tag loads/stores against their
     * traced-back parameter. Refuse on encountering a BIR_CALL,
     * because that would have us guessing about the called function's
     * effects, and guessing is how CHKDSK loses your dissertation. */
    guard = 262144;
    for (uint32_t bi = 0; bi < F->num_blocks && guard > 0; bi++, guard--) {
        uint32_t bk = F->first_block + bi;
        if (bk >= bir->num_blocks) break;
        const bir_block_t *B = &bir->blocks[bk];
        int bguard = 65536;
        for (uint32_t ii = 0; ii < B->num_insts && bguard > 0; ii++, bguard--) {
            const bir_inst_t *I = &bir->insts[B->first_inst + ii];

            if (I->op == BIR_CALL) {
                fprintf(stderr,
                        "tdf: fission cannot proceed across BIR_CALL "
                        "(device function call). Inline first.\n");
                return BC_ERR_TDF;
            }

            if (I->op == BIR_LOAD) {
                int pi = trace_to_param(bir, I->operands[0]);
                if (pi >= 0 && (uint32_t)pi < nparams)
                    params[(uint32_t)pi].loaded = 1;
            } else if (I->op == BIR_STORE) {
                int pi = trace_to_param(bir, I->operands[1]);
                if (pi >= 0 && (uint32_t)pi < nparams)
                    params[(uint32_t)pi].stored = 1;
            }
        }
    }
    return BC_OK;
}

/* ---- TDF rewrite ----
 *
 * Given the classified parameters, throw the SOLO region out and
 * rebuild M as a three-region pipeline. Tile shape defaults to
 * 32x32 f32 interleaved because that is what Metalium uses by
 * default and we have no shape inference yet; once tile-shape
 * tracking lands (the issue #82 work for Triton) this will read
 * from there instead.
 */
static const uint16_t FS_TILE_R   = 32;
static const uint16_t FS_TILE_C   = 32;
static const uint16_t FS_CB_DEPTH = 2;

static td_tag_t default_tile(void)
{
    td_tag_t t;
    t.rows   = FS_TILE_R;
    t.cols   = FS_TILE_C;
    t.dtype  = BIR_TYPE_FLOAT;
    t.layout = TD_LAY_INTRL;
    t._pad   = 0;
    return t;
}

int td_fission_tensix(td_mod_t *M)
{
    if (M->target != TD_TGT_TENSIX) {
        fprintf(stderr, "tdf: fission only runs on Tensix targets\n");
        return BC_ERR_TDF;
    }
    if (M->nrgn != 1 || M->rgns[0].role != TD_RG_SOLO) {
        fprintf(stderr,
                "tdf: fission expects a SOLO input, got %u regions\n",
                M->nrgn);
        return BC_ERR_TDF;
    }
    if (!M->rgns[0].body) {
        fprintf(stderr, "tdf: fission needs a BIR body to analyse\n");
        return BC_ERR_TDF;
    }

    bir_module_t *body = M->rgns[0].body;

    uint32_t func_idx = 0;
    if (find_kernel(body, &func_idx) != BC_OK) {
        fprintf(stderr, "tdf: no kernel function in module\n");
        return BC_ERR_TDF;
    }

    fs_param_t params[FS_MAX_PARAMS];
    uint32_t nparams = 0;
    if (classify_params(body, func_idx, params, &nparams) != BC_OK)
        return BC_ERR_TDF;

    /* Rebuild the module: three regions, one per baby-core role,
     * BIR body parked on the compute region. The cores follow
     * Metalium convention: DM0 = B, compute = T0, DM1 = NC. */
    td_init(M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(M, TD_RG_CMP);
    uint16_t wrt = td_mkrgn(M, TD_RG_WRT);
    if (rdr == TD_BAD_ID || cmp == TD_BAD_ID || wrt == TD_BAD_ID)
        return BC_ERR_TDF;
    M->rgns[rdr].core = 0;      /* B  (DM0) */
    M->rgns[cmp].core = 1;      /* T0 (compute) */
    M->rgns[wrt].core = 4;      /* NC (DM1) */
    M->rgns[cmp].body = body;

    /* For each pointer parameter that is loaded, build an RDR->CMP
     * channel and the matching PUSH/WAIT arc pair. For each stored
     * pointer parameter, build a CMP->WRT channel. A param that is
     * both loaded and stored gets both, which is the in-place
     * update case (think saxpy with x = a*x + y). */
    td_tag_t tag = default_tile();
    for (uint32_t pi = 0; pi < nparams; pi++) {
        if (!params[pi].is_ptr) continue;

        /* Input parameter: reader pulls a tile from DRAM into the
         * channel's L1 slot, signals via PUSH, compute consumes
         * with WAIT. The NoC RD arc carries the DRAM->L1 transfer;
         * the CB pair carries the producer/consumer handshake. */
        if (params[pi].loaded) {
            uint16_t ch = td_link(M, rdr, cmp, tag, FS_CB_DEPTH);
            if (ch == TD_BAD_ID) return BC_ERR_TDF;
            td_mkarc(M, rdr, TD_AR_RD,   ch, 1, 0);
            td_mkarc(M, rdr, TD_AR_PUSH, ch, 1, 0);
            td_mkarc(M, cmp, TD_AR_WAIT, ch, 1, 0);
        }

        /* Output parameter: compute writes the result tile into a
         * channel slot, signals PUSH, writer waits then drains the
         * tile out to DRAM via NoC. */
        if (params[pi].stored) {
            uint16_t ch = td_link(M, cmp, wrt, tag, FS_CB_DEPTH);
            if (ch == TD_BAD_ID) return BC_ERR_TDF;
            td_mkarc(M, cmp, TD_AR_PUSH, ch, 1, 0);
            td_mkarc(M, wrt, TD_AR_WAIT, ch, 1, 0);
            td_mkarc(M, wrt, TD_AR_WR,   ch, 1, 0);
        }
    }
    return BC_OK;
}
