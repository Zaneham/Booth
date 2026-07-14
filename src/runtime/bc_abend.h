/* bc_abend.h -- GPU ABEND dump diagnostics for Booth
 *
 * Oh yeah baby, mainframes are IN, chopped error messages are OUT.
 * IBM solved crash diagnostics in the 1960s. We're solving them in the
 * 2020s, sixty years late, on a GPU. Progress is a flat circle.
 *
 * ab_ctx_t is ~37KB, caller heap-allocates it. */

#ifndef BARRACUDA_BC_ABEND_H
#define BARRACUDA_BC_ABEND_H

#include <stdio.h>
#include <stdint.h>

/* Need bc_kernel_t -- include the real header if available,
 * otherwise forward-declare just enough for the pointer. */
#ifndef BC_RUNTIME_H
struct bc_kernel;
typedef struct bc_kernel bc_kernel_t;
#endif

/* ---- Limits ---- */

#define AB_MAX_ALLOC  64    /* tracked GPU allocations */
#define AB_MAX_LABEL  32    /* label length per alloc */
#define AB_MAX_SMAP   4096  /* source map entries */

/* ---- ABEND Codes ----
 * G0Cx = GPU equivalents of IBM S0Cx completion codes.
 * Because if you're going to crash, crash with class. */

#define AB_G0C1  0x0C1  /* illegal instruction */
#define AB_G0C4  0x0C4  /* protection exception */
#define AB_G0C5  0x0C5  /* addressing exception */
#define AB_G0C7  0x0C7  /* data exception */
#define AB_G0CB  0x0CB  /* machine check */
#define AB_G001  0x001  /* dispatch failure */
#define AB_G002  0x002  /* timeout */
#define AB_G003  0x003  /* resource exhaustion */
#define AB_G0FF  0x0FF  /* unknown */

/* ---- Allocation Flags ---- */

#define AB_FL_RW   1    /* read-write */
#define AB_FL_RX   2    /* read-execute */
#define AB_FL_KA   4    /* kernarg buffer */

/* ---- Tracked Allocation ---- */

typedef struct {
    uint64_t base;
    uint64_t size;
    char     label[AB_MAX_LABEL];
    uint8_t  flags;   /* AB_FL_* */
} ab_alloc_t;

/* ---- Dispatch Context (snapshot before launch) ---- */

typedef struct {
    char     kernel[64];
    char     chip[16];
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t sgprs, vgprs;
    uint32_t lds, scratch;
    uint32_t wave_sz;
    uint32_t wg_max;
    uint64_t kobj;       /* kernel object address */
    uint32_t karg_sz;    /* kernarg buffer size */
    uint32_t args_sz;    /* actual args copied */
} ab_dctx_t;

/* ---- Source Map Entry (from .debug_bc) ---- */

typedef struct {
    uint32_t offset;   /* byte offset in .text */
    uint32_t line;     /* source line number */
} ab_smap_t;

/* ---- Master ABEND Context ----
 * Heap-allocated by caller. ~37KB with smap[4096]. */

typedef struct {
    ab_alloc_t allocs[AB_MAX_ALLOC];
    uint32_t   n_alloc;

    ab_dctx_t  dctx;

    uint64_t   tea;          /* fault address -- literally TEA, babe */
    uint32_t   reason;       /* HSA fault reason mask */
    uint16_t   code;         /* AB_G0xx */
    uint8_t    armed;        /* 1 = fault handler registered */
    uint8_t    faulted;      /* 1 = fault occurred */

    ab_smap_t  smap[AB_MAX_SMAP];
    uint32_t   n_smap;
    char       src_file[256];

    uint8_t    args_snap[256]; /* kernarg snapshot */
} ab_ctx_t;

/* ---- API ---- */

/* Initialise ABEND context. Attempts to register HSA fault handler
 * via hsa_lib (dlopen'd libhsa-runtime64.so handle). If the AMD
 * extension is unavailable, armed=0 -- graceful degradation. */
int  ab_init(ab_ctx_t *A, void *hsa_lib);

/* Shutdown -- currently a no-op, but good manners cost nothing. */
void ab_shut(ab_ctx_t *A);

/* Track a GPU memory allocation for fault correlation. */
void ab_trak(ab_ctx_t *A, uint64_t base, uint64_t size,
             const char *label, uint8_t flags);

/* Snapshot dispatch parameters before launch.
 * Copies kernel name, chip, grid/block dims, kernarg data. */
void ab_snag(ab_ctx_t *A, const bc_kernel_t *k,
             const char *name, const char *chip,
             uint32_t gx, uint32_t gy, uint32_t gz,
             uint32_t bx, uint32_t by, uint32_t bz,
             const void *args, uint32_t args_sz);

/* Format and write ABEND dump. Returns 0 on success. */
int  ab_dump(const ab_ctx_t *A, FILE *out);

/* Load .debug_bc source map from an ELF binary.
 * Format: 4B magic "BCDB", 4B count, then 8B entries. */
int  ab_slod(ab_ctx_t *A, const uint8_t *elf, uint32_t elf_sz);

/* ABEND code to human-readable string. */
const char *ab_mstr(uint16_t code);

#endif /* BARRACUDA_BC_ABEND_H */
