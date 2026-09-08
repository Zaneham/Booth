/* backend.h -- the shape every Booth backend implements.
 * Seven function pointers, one struct, no dlopen, no frameworks.
 * See docs/backends.md for the walkthrough. */

#ifndef BOOTH_BE_H
#define BOOTH_BE_H

#include <stdint.h>

struct bir_module;
struct be_cfg;

/* ---- Return codes ---- */

typedef enum {
    BE_OK      =  0,
    BE_UNSUP   = -1,  /* op deliberately not implemented */
    BE_EISEL   = -2,
    BE_ESCHED  = -3,
    BE_ERA     = -4,
    BE_EVFY    = -5,
    BE_EEMIT   = -6,
    BE_EIO     = -7,
    BE_EINPUT  = -8,
    BE_ENOMEM  = -9
} be_ret_t;

/* ---- Verify phase ----
 * Only the two the driver actually calls. Phases for post-sched and
 * post-emit were declared here first, but nothing invoked them and the
 * AMD adapter folds every non-ISEL phase onto its post-regalloc check,
 * so a backend implementing them would have been checking physical
 * registers on a module that still had virtual ones. Add them back with
 * the call sites, not before. */

typedef enum {
    BE_VFY_ISEL,
    BE_VFY_RA
} be_vfy_t;

/* ---- Feature flags ----
 * A backend must not claim a feature it does not implement. The
 * conformance suite (not yet landed) keys tests off these bits. */

#define BE_F_SIMT       (1u <<  0)
#define BE_F_SCALAR     (1u <<  1)
#define BE_F_ATOMIC     (1u <<  2)
#define BE_F_SHARED     (1u <<  3)
#define BE_F_WARP       (1u <<  4)
#define BE_F_MFMA       (1u <<  5)
#define BE_F_BARRIER    (1u <<  6)
#define BE_F_DIV        (1u <<  7)
#define BE_F_SCRATCH    (1u <<  8)
#define BE_F_TRANSC     (1u <<  9)
#define BE_F_F16        (1u << 10)
#define BE_F_F64        (1u << 11)
#define BE_F_BF16       (1u << 12)
#define BE_F_MULTIOUT   (1u << 13)  /* emits more than one file (Tensix) */
/* No real calls in emitted code, so device functions have to be inlined
 * before isel. The vector targets want this; CPU, RV64 and Metal emit
 * calls and are left alone. */
#define BE_F_NOCALL     (1u << 14)
#define BE_F_BYTES      (1u << 15)  /* materialises byte-initialised globals */

/* ---- Options storage ----
 * Each backend gets one fixed slot for whatever its flags set. Sized at
 * compile time and union-aligned, so no malloc and no alignment guessing
 * when a backend casts the slot to its own struct. A descriptor asking
 * for more than BE_OPTS_MAX is refused at startup rather than trusted. */

#define BE_OPTS_MAX  128

typedef union {
    unsigned char raw[BE_OPTS_MAX];
    void         *p;
    long long     l;
    double        d;
} be_opts_t;

/* ---- Backend descriptor ----
 * NULL op means "not applicable, driver skips". isel and emit are
 * required; a registered backend without them is a bug. cfg is opaque
 * here so the contract does not couple to driver internals;
 * implementations cast to be_cfg_t. opts is the backend's own slot,
 * which only it knows the shape of. */

typedef struct be_desc {
    const char *name;
    const char *triple;
    uint32_t    feats;
    uint32_t    opts_size;              /* bytes wanted, <= BE_OPTS_MAX */

    /* Flags this backend owns, NULL-terminated. Declared rather than
     * discovered, so two backends claiming one flag is a static error
     * the suite catches instead of a first-past-the-post race. */
    const char *const *flags;

    /* Called for an argv entry matching one of `flags`. `next` is the
     * following entry or NULL at the end. Returns how many extra entries
     * were consumed (0 or 1), or negative if the value was bad. */
    int  (*parse)  (const char *arg, const char *next, void *opts);

    int  (*is_on)  (const void *opts);

    /* Lanes that move together on this target. The IR needs it while
     * lowering warpSize, long before a backend module exists, and asking
     * for a number here is what keeps AMD types out of the frontend.
     * NULL means the 32 the driver assumes. */
    uint32_t (*warp_size)(const void *opts);
    int  (*isel)   (const struct bir_module *M, const struct be_cfg *cfg,
                    const void *opts, void **out_mmod);
    int  (*sched)  (void *mmod);
    int  (*regalc) (void *mmod);
    int  (*verify) (const void *mmod, int phase);
    int  (*emit)   (const void *mmod, const struct be_cfg *cfg,
                    const void *opts, const char *out_path);
    void (*mfree)  (void *mmod);
} be_desc_t;

/* ---- Registration ---- */

extern const be_desc_t * const be_list[];   /* NULL-terminated */

const be_desc_t *be_find(const char *name);

int be_fail(int eid, ...);

/* Offer one argv entry to the registry. Returns 1 if a backend took it,
 * 0 if no backend owns it, negative on a bad value. On 1, *used_next says
 * whether the following argv entry was consumed as a value. */
int be_parse_flag(const char *arg, const char *next, int *used_next);

/* How many backends the flags turned on, and the first of them. The
 * driver asks rather than keeping its own copy of every mode flag. */
uint32_t         be_num_on(void);
uint32_t         be_warp_size(void);
const be_desc_t *be_active(void);

int be_run(const struct bir_module *M, const struct be_cfg *cfg);

#endif
