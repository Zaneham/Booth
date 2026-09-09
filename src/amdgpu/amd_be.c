/* amd_be.c -- AMDGPU as a be_desc_t. Full pipeline: isel, sched,
 * regalloc, verify twice, emit. What a mature backend looks like at
 * the descriptor layer; the actual work is in the rest of src/amdgpu/. */

#include "backend.h"
#include "backend_cfg.h"
#include "amdgpu.h"
#include "sched.h"
#include "verify.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int          on;        /* --amdgpu, assembly to stdout or -o */
    int          bin;       /* --amdgpu-bin, .hsaco */
    int          snap;
    int          set;       /* a --gfx* was given, so defaults are out */
    amd_target_t target;
    uint32_t     elfm;
    const char  *chip;
} amd_opts_t;

/* The part list, one row per chip rather than nineteen else-ifs. elfm is
 * EF_AMDGPU_MACH for the exact part; target is the ISA family the rest of
 * the backend switches on, so several chips share one. */
static const struct {
    const char  *flag;
    amd_target_t target;
    uint32_t     elfm;
    const char  *chip;
} amd_parts[] = {
    { "--gfx90a",  AMD_TARGET_GFX90A,  0x3F,  "gfx90a"  },  /* CDNA 2      */
    { "--gfx942",  AMD_TARGET_GFX942,  0x54C, "gfx942"  },  /* CDNA 3      */
    { "--gfx1030", AMD_TARGET_GFX1030, 0x36,  "gfx1030" },  /* RDNA 2      */
    { "--gfx1031", AMD_TARGET_GFX1030, 0x37,  "gfx1031" },
    { "--gfx1032", AMD_TARGET_GFX1030, 0x38,  "gfx1032" },
    { "--gfx1033", AMD_TARGET_GFX1030, 0x39,  "gfx1033" },
    { "--gfx1034", AMD_TARGET_GFX1030, 0x3e,  "gfx1034" },
    { "--gfx1035", AMD_TARGET_GFX1030, 0x3d,  "gfx1035" },
    { "--gfx1036", AMD_TARGET_GFX1030, 0x45,  "gfx1036" },
    { "--gfx1100", AMD_TARGET_GFX1100, 0x41,  "gfx1100" },  /* RDNA 3      */
    { "--gfx1101", AMD_TARGET_GFX1100, 0x46,  "gfx1101" },
    { "--gfx1102", AMD_TARGET_GFX1100, 0x47,  "gfx1102" },
    { "--gfx1103", AMD_TARGET_GFX1100, 0x44,  "gfx1103" },
    { "--gfx1150", AMD_TARGET_GFX1100, 0x43,  "gfx1150" },  /* RDNA 3.5    */
    { "--gfx1151", AMD_TARGET_GFX1100, 0x4a,  "gfx1151" },
    { "--gfx1152", AMD_TARGET_GFX1100, 0x55,  "gfx1152" },
    { "--gfx1153", AMD_TARGET_GFX1100, 0x58,  "gfx1153" },
    { "--gfx1200", AMD_TARGET_GFX1200, 0x48,  "gfx1200" },  /* RDNA 4      */
    { "--gfx1201", AMD_TARGET_GFX1200, 0x4e,  "gfx1201" },
};

#define AMD_NPARTS  (sizeof amd_parts / sizeof amd_parts[0])

static const char *const amd_flags[] = {
    "--amdgpu", "--amdgpu-bin", "--snap",
    "--no-graphcolor", "--ssa-ra", "--max-vgprs",
    "--gfx90a", "--gfx942",
    "--gfx1030", "--gfx1031", "--gfx1032", "--gfx1033",
    "--gfx1034", "--gfx1035", "--gfx1036",
    "--gfx1100", "--gfx1101", "--gfx1102", "--gfx1103",
    "--gfx1150", "--gfx1151", "--gfx1152", "--gfx1153",
    "--gfx1200", "--gfx1201",
    NULL
};

static int amd_parse(const char *arg, const char *next, void *o)
{
    amd_opts_t *p = (amd_opts_t *)o;

    if (strcmp(arg, "--amdgpu")     == 0) { p->on   = 1; return 0; }
    if (strcmp(arg, "--amdgpu-bin") == 0) { p->bin  = 1; return 0; }
    if (strcmp(arg, "--snap")       == 0) { p->snap = 1; return 0; }

    /* Register allocator knobs live as globals in the allocator itself,
     * so this is the backend setting its own state rather than main
     * reaching across for it. */
    if (strcmp(arg, "--no-graphcolor") == 0) { amd_ra_lin = 1; return 0; }
    if (strcmp(arg, "--ssa-ra")        == 0) { amd_ra_ssa = 1; return 0; }
    if (strcmp(arg, "--max-vgprs") == 0) {
        if (next == NULL) {
            fprintf(stderr, "error: --max-vgprs wants a count\n");
            return -1;
        }
        amd_max_vgpr = atoi(next);
        return 1;
    }

    for (uint32_t i = 0; i < AMD_NPARTS; i++) {
        if (strcmp(arg, amd_parts[i].flag) != 0) continue;
        p->target = amd_parts[i].target;
        p->elfm   = amd_parts[i].elfm;
        p->chip   = amd_parts[i].chip;
        p->set    = 1;
        return 0;
    }
    return 0;
}

static int amd_on(const void *o)
{
    const amd_opts_t *p = (const amd_opts_t *)o;
    return p->on || p->bin;
}

static uint32_t amd_warp(const void *o)
{
    const amd_opts_t *p = (const amd_opts_t *)o;
    return (uint32_t)amd_get_wave_size(p->set ? p->target : AMD_TARGET_GFX1100);
}

static int amd_isel(const struct bir_module *M, const be_cfg_t *cfg,
                    const void *o, void **out_mmod)
{
    const amd_opts_t *p = (const amd_opts_t *)o;
    (void)cfg;
    amd_module_t *amd = calloc(1, sizeof(*amd));
    if (amd == NULL) return BE_ENOMEM;

    /* RDNA 3 unless a part was named. */
    amd->target    = p->set ? p->target : AMD_TARGET_GFX1100;
    amd->elf_mach  = p->set ? p->elfm   : 0x41u;
    amd->snap_mode = (uint8_t)p->snap;
    snprintf(amd->chip_name, sizeof(amd->chip_name), "%s",
             p->set && p->chip ? p->chip : "gfx1100");

    if (amdgpu_compile((const bir_module_t *)M, amd) != BC_OK) {
        free(amd);
        return BE_EISEL;
    }
    *out_mmod = amd;
    return BE_OK;
}

static int amd_sched(void *mmod)   { amdgpu_sched(mmod);    return BE_OK; }
static int amd_regalc(void *mmod)  { amdgpu_regalloc(mmod); return BE_OK; }

static int amd_verify(const void *mmod, int phase)
{
    int p = (phase == BE_VFY_ISEL) ? VFY_ISEL : VFY_RA;
    vfy_res_t v = bc_vfy(mmod, p);
    if (v.errs == 0) return BE_OK;
    fprintf(stderr, "verify: %u error(s) after %s\n",
            v.errs, p == VFY_ISEL ? "isel" : "regalloc");
    return BE_EVFY;
}

static int amd_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                    const char *out)
{
    const amd_opts_t *p = (const amd_opts_t *)o;
    (void)cfg;

    /* --amdgpu-bin writes a .hsaco; bare --amdgpu writes assembly, to the
     * file if one was named and stdout otherwise. */
    if (p->bin) {
        int rc = amdgpu_emit_elf((amd_module_t *)mmod, out ? out : "a.hsaco");
        return rc == 0 ? BE_OK : BE_EEMIT;
    }

    if (out != NULL) {
        FILE *af = fopen(out, "w");
        if (af == NULL) {
            fprintf(stderr, "error: cannot open %s for writing\n", out);
            return BE_EIO;
        }
        int rc = amdgpu_emit_asm(mmod, af);
        fclose(af);
        return rc == BC_OK ? BE_OK : BE_EEMIT;
    }

    return amdgpu_emit_asm(mmod, stdout) == BC_OK ? BE_OK : BE_EEMIT;
}

const be_desc_t be_amd = {
    .name      = "amdgpu",
    .triple    = "amdgcn--",
    .feats     = BE_F_SIMT | BE_F_ATOMIC | BE_F_SHARED | BE_F_WARP
               | BE_F_BARRIER | BE_F_DIV | BE_F_SCRATCH | BE_F_TRANSC
               | BE_F_F16 | BE_F_F64 | BE_F_MFMA | BE_F_NOCALL
               | BE_F_BYTES,
    .opts_size = sizeof(amd_opts_t),
    .flags     = amd_flags,
    .parse     = amd_parse,
    .is_on     = amd_on,
    .warp_size = amd_warp,
    .isel      = amd_isel,
    .sched     = amd_sched,
    .regalc    = amd_regalc,
    .verify    = amd_verify,
    .emit      = amd_emit,
    .mfree     = free
};
