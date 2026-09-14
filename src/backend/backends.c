/* backends.c -- registration list and dispatch loop.
 * Adding a target is one line in be_list and one <name>_be.c file.
 * Static, ordered, auditable. Boring in the way infra should be. */

#include "backend.h"
#include "backend_cfg.h"
#include "barracuda.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

int be_fail(int eid, ...)
{
    va_list ap;
    fprintf(stderr, "error[E%03u]: ", (unsigned)eid);
    va_start(ap, eid);
    vfprintf(stderr, bc_efmt((bc_eid_t)eid), ap);
    va_end(ap);
    fputc('\n', stderr);
    return BE_UNSUP;
}

/* Bounded so a stray zero terminator can't wander off the list. */
#define BE_MAX  64

extern const be_desc_t be_amd;
extern const be_desc_t be_ptx;
extern const be_desc_t be_tsx;
extern const be_desc_t be_rvcore;
extern const be_desc_t be_x86;
extern const be_desc_t be_rv64;
extern const be_desc_t be_metal;
extern const be_desc_t be_intel;

#ifdef BOOTH_INCLUDE_SKELETON
extern const be_desc_t be_skel;
#endif

const be_desc_t * const be_list[] = {
    &be_amd,
    &be_ptx,
    &be_tsx,
    &be_rvcore,
    &be_metal,
    &be_intel,
    &be_x86,
    &be_rv64,
#ifdef BOOTH_INCLUDE_SKELETON
    &be_skel,
#endif
    NULL
};

/* One slot per registered backend, indexed the same as be_list. Static,
 * so there is no malloc in the driver. */
static be_opts_t be_opts_store[BE_MAX];

void be_reset(void)
{
    memset(be_opts_store, 0, sizeof be_opts_store);
}

int be_parse_flag(const char *arg, const char *next, int *used_next)
{
    if (arg == NULL || used_next == NULL) return 0;
    *used_next = 0;

    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++) {
        const be_desc_t *b = be_list[i];
        if (b->flags == NULL || b->parse == NULL) continue;

        for (uint32_t f = 0; b->flags[f] != NULL; f++) {
            if (strcmp(arg, b->flags[f]) != 0) continue;

            /* A descriptor wanting more room than the slot holds would
             * scribble over its neighbour, so refuse rather than trust it. */
            if (b->opts_size > BE_OPTS_MAX) {
                (void)be_fail(BC_E545, b->name, "the option block",
                              (unsigned)BE_OPTS_MAX);
                return -1;
            }

            int n = b->parse(arg, next, be_opts_store[i].raw);
            if (n < 0) return -1;
            *used_next = n;
            return 1;
        }
    }
    return 0;
}

uint32_t be_num_on(void)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++)
        if (be_list[i]->is_on != NULL && be_list[i]->is_on(be_opts_store[i].raw))
            n++;
    return n;
}

const be_desc_t *be_active(void)
{
    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++)
        if (be_list[i]->is_on != NULL && be_list[i]->is_on(be_opts_store[i].raw))
            return be_list[i];
    return NULL;
}

uint32_t be_warp_size(void)
{
    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++) {
        const be_desc_t *b = be_list[i];
        if (b->is_on == NULL || !b->is_on(be_opts_store[i].raw)) continue;
        if (b->warp_size == NULL) break;
        return b->warp_size(be_opts_store[i].raw);
    }
    return 32u;
}

const be_desc_t *be_find(const char *name)
{
    if (name == NULL) return NULL;
    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++) {
        if (be_list[i]->name != NULL &&
            strcmp(be_list[i]->name, name) == 0) {
            return be_list[i];
        }
    }
    return NULL;
}

int be_run(const struct bir_module *M, const be_cfg_t *cfg)
{
    if (M == NULL || cfg == NULL) return BE_EINPUT;

    /* One target per run. They all emit to cfg->output_file, so asking for
     * several used to write each over the last and leave whichever sorted
     * last in be_list, under the name you picked, with a zero exit. Run
     * kath once per target instead. */
    uint32_t on = be_num_on();

    if (on > 1) {
        fprintf(stderr, "error: %u backends selected, pick one:", on);
        for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++)
            if (be_list[i]->is_on != NULL &&
                be_list[i]->is_on(be_opts_store[i].raw))
                fprintf(stderr, " %s", be_list[i]->name);
        fprintf(stderr, "\n");
        return BE_EINPUT;
    }

    int first = BE_OK;

    for (uint32_t i = 0; i < BE_MAX && be_list[i] != NULL; i++) {
        const be_desc_t *b = be_list[i];
        const void *opts = be_opts_store[i].raw;

        if (b->is_on == NULL || !b->is_on(opts)) continue;

        /* A registered backend without isel or emit is a descriptor
         * bug, not a runtime condition; complain and skip. */
        if (b->isel == NULL || b->emit == NULL) {
            fprintf(stderr, "backend %s: missing required op\n", b->name);
            if (first == BE_OK) first = BE_UNSUP;
            continue;
        }

        void *mmod = NULL;
        int rc = b->isel(M, cfg, opts, &mmod);
        if (rc != BE_OK) {
            if (first == BE_OK) first = rc;
            if (mmod != NULL && b->mfree != NULL) b->mfree(mmod);
            continue;
        }

        if (b->verify != NULL) {
            rc = b->verify(mmod, BE_VFY_ISEL);
            if (rc != BE_OK) goto fail;
        }

        if (b->sched != NULL && !cfg->no_sched) {
            rc = b->sched(mmod);
            if (rc != BE_OK) goto fail;
        }

        if (b->regalc != NULL) {
            rc = b->regalc(mmod);
            if (rc != BE_OK) goto fail;
        }

        if (b->verify != NULL) {
            rc = b->verify(mmod, BE_VFY_RA);
            if (rc != BE_OK) goto fail;
        }

        rc = b->emit(mmod, cfg, opts, cfg->output_file);
        if (rc != BE_OK && first == BE_OK) first = rc;

        if (b->mfree != NULL) b->mfree(mmod);
        continue;

fail:
        if (first == BE_OK) first = rc;
        if (b->mfree != NULL) b->mfree(mmod);
    }

    return first;
}
