/* nv_be.c -- NVIDIA PTX as a be_desc_t.
 * Smallest of the lot: PTX is a text format so we jump isel to emit
 * and skip the fine-grained pipeline. */

#include "backend.h"
#include "backend_cfg.h"
#include "nvidia.h"
#include "barracuda.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    int on;
    int bkhit;
    int sass;
    int cubin;
} nv_opts_t;

static const char *const nv_flags[] = {
    "--nvidia-ptx", "--nvidia-sass", "--nvidia-cubin", "--bkhit", NULL
};

static int nv_parse(const char *arg, const char *next, void *o)
{
    nv_opts_t *p = (nv_opts_t *)o;
    (void)next;
    if (strcmp(arg, "--nvidia-ptx") == 0) { p->on    = 1; return 0; }
    if (strcmp(arg, "--nvidia-sass") == 0) {
        p->on = 1;
        p->sass = 1;
        return 0;
    }
    if (strcmp(arg, "--nvidia-cubin") == 0) {
        p->on = 1;
        p->cubin = 1;
        return 0;
    }
    if (strcmp(arg, "--bkhit")      == 0) { p->bkhit = 1; return 0; }
    return 0;
}

static int nv_on(const void *o) { return ((const nv_opts_t *)o)->on; }

static int nv_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   const void *o, void **out_mmod)
{
    const nv_opts_t *p = (const nv_opts_t *)o;
    (void)cfg;
    nv_module_t *nvm = calloc(1, sizeof(*nvm));
    if (nvm == NULL) return BE_ENOMEM;
    if (nv_compile((const bir_module_t *)M, nvm) != BC_OK
     || nv_vchk(nvm) != BC_OK) {
        free(nvm);
        return BE_EISEL;
    }
    nvm->bkhit = (uint8_t)p->bkhit;
    *out_mmod = nvm;
    return BE_OK;
}

static int nv_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    const nv_opts_t *p = (const nv_opts_t *)o;
    (void)cfg;
    if (p->cubin != 0)
        return nv_sass((nv_module_t *)mmod, out ? out : "a.cubin", 0) == BC_OK
             ? BE_OK : BE_EEMIT;
    if (p->sass != 0)
        return nv_sass((nv_module_t *)mmod, out ? out : "a.sass", 1) == BC_OK
             ? BE_OK : BE_EEMIT;
    return nv_emit_ptx((nv_module_t *)mmod, out ? out : "a.ptx") == BC_OK
         ? BE_OK : BE_EEMIT;
}

const be_desc_t be_ptx = {
    .name      = "nvptx",
    .triple    = "nvptx64--",
    .feats     = BE_F_SIMT | BE_F_ATOMIC | BE_F_SHARED | BE_F_WARP
               | BE_F_BARRIER | BE_F_DIV | BE_F_SCRATCH | BE_F_TRANSC
               | BE_F_F16 | BE_F_F64 | BE_F_BF16 | BE_F_NOCALL
               | BE_F_BYTES,
    .opts_size = sizeof(nv_opts_t),
    .flags     = nv_flags,
    .parse     = nv_parse,
    .is_on     = nv_on,
    .isel      = nv_isel,
    .emit      = nv_emit,
    .mfree     = free
};
