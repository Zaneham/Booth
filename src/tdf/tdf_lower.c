#include "tdf.h"
#include <stdio.h>
#include <string.h>

/*
 * Tile DataFlow lowering.
 *
 * Target-aware. Two paths today and a third that is still being
 * thought through.
 *
 * AMD and NVIDIA hit the degenerate path: the module is required to
 * contain exactly one region with role SOLO, the region's body is
 * handed back to the caller unchanged, no allocation happens, and
 * the existing isel chains continue exactly as they did before TDF
 * existed. If anything in this path costs more than a memcpy we
 * have done something wrong.
 *
 * Tensix is the real work and lives behind tdf_lower_tensix() which
 * is a stub for now. The first real version will collapse a SOLO
 * module to a single baby-core body for the soft-float Moa path,
 * because that is the smallest thing we can get on hardware. The
 * version after that will accept RDR/CMP/WRT fissioned modules and
 * fan them out across baby cores with channels lowered to L1.
 */

static int lower_solo_passthrough(td_mod_t *M, td_lout_t *out)
{
    if (M->nrgn != 1) {
        fprintf(stderr,
                "tdf: target requires one SOLO region, module has %u\n",
                M->nrgn);
        return BC_ERR_TDF;
    }
    if (M->rgns[0].role != TD_RG_SOLO) {
        fprintf(stderr,
                "tdf: target requires SOLO role, region 0 has %s\n",
                td_role_name(M->rgns[0].role));
        return BC_ERR_TDF;
    }
    if (!M->rgns[0].body) {
        fprintf(stderr, "tdf: SOLO region has no body attached\n");
        return BC_ERR_TDF;
    }
    if (M->ncha != 0 || M->narc != 0) {
        fprintf(stderr,
                "tdf: SOLO module must have no channels or arcs "
                "(have %u channels, %u arcs)\n",
                M->ncha, M->narc);
        return BC_ERR_TDF;
    }
    out->mods[0] = M->rgns[0].body;
    out->owns[0] = 0;
    out->nmods   = 1;
    return BC_OK;
}

static int lower_tensix(td_mod_t *M, td_lout_t *out)
{
    /*
     * Tensix fission lives here. For now we only accept the same
     * SOLO shape as AMD/NVIDIA and pass it through, so we can stand
     * up the rest of the pipeline (frontend hookup, isel binding to
     * a baby core, ELF emission) without needing the channel and
     * arc lowering finished. The day SOLO-on-Tensix stops being
     * enough is the day we add RDR/CMP/WRT handling here, and the
     * shape of that work is laid out in the TDF design notes.
     */
    if (M->nrgn == 1 && M->rgns[0].role == TD_RG_SOLO) {
        return lower_solo_passthrough(M, out);
    }
    fprintf(stderr,
            "tdf: Tensix fission (RDR/CMP/WRT) not implemented yet, "
            "got %u regions\n", M->nrgn);
    return BC_ERR_TDF;
}

int td_lower(td_mod_t *M, td_lout_t *out)
{
    memset(out, 0, sizeof(*out));

    switch (M->target) {
    case TD_TGT_AMD:
    case TD_TGT_NVIDIA:
        return lower_solo_passthrough(M, out);
    case TD_TGT_TENSIX:
        return lower_tensix(M, out);
    default:
        fprintf(stderr, "tdf: unknown target %u\n", M->target);
        return BC_ERR_TDF;
    }
}
