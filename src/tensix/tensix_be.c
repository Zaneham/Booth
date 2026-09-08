/* tensix_be.c -- Tensix Metalium (be_tsx) and RV32 baby-core
 * (be_rvcore) as be_desc_t descriptors.
 *
 * Metalium emits five files off one path stem: compute, reader,
 * writer, host, plus the .bin and .ttinsn drops. It's ceremony but
 * it's the ceremony Tensix wants. Blackhole's in the room and I'm
 * hoping it gets less silly over time. */

#include "backend.h"
#include "backend_cfg.h"
#include "tensix.h"
#include "rv_isel.h"
#include "rv_elf.h"
#include "rv_buf.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

void tterr(bc_eid_t eid, ...)
{
    const char *f = bc_efmt(eid);
    va_list ap;

    if (f == NULL) f = "unspecified Tensix refusal";
    fprintf(stderr, "kath: error[E%03d]: ", (int)eid);
    va_start(ap, eid);
    vfprintf(stderr, f, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* ---- Metalium ----
 * tt_wrap keeps the BIR pointer alive through emit so datamov
 * analysis can run inside isel where BIR is still in scope. */

typedef struct {
    tt_module_t          *tt;
    const bir_module_t   *bir;
} tt_wrap_t;

typedef struct { int on; } tt_opts_t;

static const char *const tt_flags[] = { "--tensix", NULL };

static int tt_parse(const char *arg, const char *next, void *o)
{
    (void)next;
    if (strcmp(arg, "--tensix") == 0) { ((tt_opts_t *)o)->on = 1; }
    return 0;
}

static int tt_on(const void *o) { return ((const tt_opts_t *)o)->on; }

static int tt_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   const void *o, void **out_mmod)
{
    (void)cfg; (void)o;
    tt_wrap_t *w = calloc(1, sizeof(*w));
    if (w == NULL) return BE_ENOMEM;
    w->tt = calloc(1, sizeof(*w->tt));
    if (w->tt == NULL) { free(w); return BE_ENOMEM; }
    w->bir = (const bir_module_t *)M;

    if (tensix_compile(w->bir, w->tt) != BC_OK) {
        fprintf(stderr, "error: Tensix compilation failed\n");
        free(w->tt); free(w);
        return BE_EISEL;
    }
    tensix_coarsen(w->tt);
    tensix_regalloc(w->tt);
    tensix_analyze_datamov(w->bir, w->tt, &w->tt->dmov);
    *out_mmod = w;
    return BE_OK;
}

/* Metalium emits five siblings off a compute-path stem; the file
 * juggling isn't obvious but matches what main.c did before. */
static int tt_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    (void)cfg; (void)o;
    const tt_wrap_t *w = mmod;
    tt_module_t *ttm = w->tt;
    const char *compute = out ? out : "a_compute.cpp";

    tensix_emit_metalium(ttm, compute);

    char path[BC_MAX_PATH];
    const char *st2 = strstr(compute, "_compute");
    int bp = st2 ? (int)(st2 - compute) : (int)strlen(compute);
    snprintf(path, sizeof(path), "%.*s_compute.bin", bp, compute);
    tensix_emit_binary(ttm, path);
    snprintf(path, sizeof(path), "%.*s_compute.ttinsn", bp, compute);
    tensix_emit_ttinsn(ttm, path);

    int pfx;
    if (st2) pfx = (int)(st2 - compute);
    else {
        const char *dot = strrchr(compute, '.');
        pfx = dot ? (int)(dot - compute) : (int)strlen(compute);
    }
    char host[BC_MAX_PATH], rd[BC_MAX_PATH], wr[BC_MAX_PATH];
    snprintf(host, sizeof(host), "%.*s_host.cpp",   pfx, compute);
    snprintf(rd,   sizeof(rd),   "%.*s_reader.cpp", pfx, compute);
    snprintf(wr,   sizeof(wr),   "%.*s_writer.cpp", pfx, compute);
    tensix_emit_reader(ttm, &ttm->dmov, rd);
    tensix_emit_writer(ttm, &ttm->dmov, wr);
    tensix_emit_host_full(ttm, &ttm->dmov, host, rd, compute, wr);

    char stem[BC_MAX_PATH];
    snprintf(stem, sizeof(stem), "%.*s", pfx, compute);
    tensix_emit_kernel_elves(ttm, &ttm->dmov, stem);
    return BE_OK;
}

static void tt_free(void *mmod)
{
    tt_wrap_t *w = mmod;
    if (w == NULL) return;
    free(w->tt);
    free(w);
}

const be_desc_t be_tsx = {
    .name    = "tensix",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_BARRIER | BE_F_MFMA | BE_F_NOCALL
             | BE_F_MULTIOUT | BE_F_F16 | BE_F_BF16,
    .opts_size = sizeof(tt_opts_t),
    .flags   = tt_flags,
    .parse   = tt_parse,
    .is_on   = tt_on,
    .isel    = tt_isel,
    .emit    = tt_emit,
    .mfree   = tt_free
};

/* ---- Baby-core RV32IM ---- */

static const char *const rvc_flags[] = { "--rv-elf", NULL };

static int rvc_parse(const char *arg, const char *next, void *o)
{
    (void)next;
    if (strcmp(arg, "--rv-elf") == 0) { ((tt_opts_t *)o)->on = 1; }
    return 0;
}

static int rvc_on(const void *o) { return ((const tt_opts_t *)o)->on; }

static int rvc_isel(const struct bir_module *M, const be_cfg_t *cfg,
                    const void *o, void **out_mmod)
{
    (void)cfg; (void)o; (void)o;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: BIR module has no functions\n");
        return BE_EINPUT;
    }
    rv_buf_t *code = calloc(1, sizeof(*code));
    if (code == NULL) return BE_ENOMEM;
    rv_buf_init(code);
    if (rv_isel_module(bir, code) != BC_OK) { free(code); return BE_EISEL; }
    *out_mmod = code;
    return BE_OK;
}

static int rvc_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    (void)cfg; (void)o;
    const rv_buf_t *code = mmod;
    const char *p = out ? out : "a.elf";
    if (rv_elf_write(code, p) != BC_OK) return BE_EIO;
    fprintf(stderr, "wrote %s (%u bytes code, %u instructions)\n",
            p, rv_buf_nbytes(code), rv_buf_n_words(code));
    return BE_OK;
}

const be_desc_t be_rvcore = {
    .name    = "tensix-rv32",
    .triple  = "riscv32-unknown-none",
    .feats   = BE_F_SCALAR,
    .opts_size = sizeof(tt_opts_t),
    .flags   = rvc_flags,
    .parse   = rvc_parse,
    .is_on   = rvc_on,
    .isel    = rvc_isel,
    .emit    = rvc_emit,
    .mfree   = free
};
