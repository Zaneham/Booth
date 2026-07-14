/* bc_abend.c -- GPU ABEND dump diagnostics for Booth
 *
 * When your kernel faults, AMD gives you a cryptic one-liner about
 * "Memory access fault, Reason: Unknown." Cheers for that. We hook
 * the HSA fault callback, correlate the fault address against tracked
 * allocations, and produce a structured dump that would make an IBM
 * systems programmer shed a single tear of pride.
 *
 * Linux-only for the HSA fault handler. The dump formatter and all
 * tracking works everywhere (tested on Windows). */

#ifdef __linux__
#include <dlfcn.h>
#endif

#include "bc_runtime.h"
#include "bc_abend.h"
#include "../fe/bc_err.h"
#include <string.h>
#include <time.h>
#include <inttypes.h>

/* ---- ABEND Code Strings ---- */

static const struct { uint16_t code; const char *name; const char *desc; } ab_codes[] = {
    { AB_G0C1, "G0C1", "ILLEGAL INSTRUCTION"       },
    { AB_G0C4, "G0C4", "MEMORY ACCESS FAULT"        },
    { AB_G0C5, "G0C5", "ADDRESSING EXCEPTION"       },
    { AB_G0C7, "G0C7", "DATA EXCEPTION"             },
    { AB_G0CB, "G0CB", "MACHINE CHECK"              },
    { AB_G001, "G001", "DISPATCH FAILURE"            },
    { AB_G002, "G002", "KERNEL TIMEOUT"              },
    { AB_G003, "G003", "RESOURCE EXHAUSTION"         },
    { AB_G0FF, "G0FF", "UNKNOWN FAULT"               },
    { 0,       NULL,   NULL                          }
};

const char *ab_mstr(uint16_t code)
{
    /* Delegate to bc_err.c's ab_afmt() which checks translations first,
     * then compiled-in defaults. Our local ab_codes[] is the fallback
     * for codes that bc_err.c doesn't know about — belt and braces. */
    const char *xlat = ab_afmt(code);
    if (xlat) return xlat;
    for (int i = 0; ab_codes[i].name; i++)
        if (ab_codes[i].code == code) return ab_codes[i].desc;
    return "UNKNOWN FAULT";
}

/* Look up the 4-char code string (G0C4 etc.) */
static const char *ab_cstr(uint16_t code)
{
    for (int i = 0; ab_codes[i].name; i++)
        if (ab_codes[i].code == code) return ab_codes[i].name;
    return "G0FF";
}

/* ---- HSA Fault Reason to ABEND Code ---- */

/* ---- Fault Callback (Linux/HSA only) ---- */

#ifdef __linux__

/* HSA memory fault reason bits (from hsa_ext_amd.h):
 *   bit 0 = page not present
 *   bit 1 = write to read-only
 *   bit 2 = execute on non-exec
 *   bit 3 = GPU hang / hardware fault */
static uint16_t ab_rmap(uint32_t reason)
{
    if (reason & 8) return AB_G0CB;  /* hardware fault */
    if (reason & 4) return AB_G0C1;  /* execute violation */
    if (reason & 2) return AB_G0C4;  /* write protection */
    if (reason & 1) return AB_G0C5;  /* page not present */
    return AB_G0FF;
}

/* Global pointer so the C callback can find our context.
 * Yes, a global. The HSA callback API gives us no userdata pointer.
 * AMD's API design is a masterclass in making simple things hard. */
static ab_ctx_t *g_actx;

/* HSA event types we care about */
#define HSA_AMD_GPU_MEM_FAULT 0

typedef struct {
    uint64_t agent_handle;
    uint64_t va;
    uint32_t info;
} hsa_amd_event_t;

typedef int (*hsa_amd_sys_ev_fn)(const hsa_amd_event_t *, void *);
typedef int (*hsa_amd_reg_fn)(hsa_amd_sys_ev_fn, void *);

static int ab_fcb(const hsa_amd_event_t *ev, void *data)
{
    (void)data;
    ab_ctx_t *A = g_actx;
    if (!A) return 0;

    A->tea    = ev->va;
    A->reason = ev->info;
    A->code   = ab_rmap(ev->info);
    A->faulted = 1;

    ab_dump(A, stderr);
    return 0;
}

#endif /* __linux__ */

/* ---- Init / Shutdown ---- */

int ab_init(ab_ctx_t *A, void *hsa_lib)
{
    memset(A, 0, sizeof(*A));

#ifdef __linux__
    if (!hsa_lib) return 0;

    /* Try to hook the AMD extension fault handler.
     * If it's not there, we degrade gracefully. Like a gentleman
     * whose parachute didn't open but who still adjusts his cravat. */
    hsa_amd_reg_fn reg = (hsa_amd_reg_fn)dlsym(
        hsa_lib, "hsa_amd_register_system_event_handler");
    if (reg) {
        g_actx = A;
        int st = reg(ab_fcb, NULL);
        A->armed = (st == 0) ? 1 : 0;
    }
#else
    (void)hsa_lib;
#endif

    return 0;
}

void ab_shut(ab_ctx_t *A)
{
    if (!A) return;
#ifdef __linux__
    if (g_actx == A) g_actx = NULL;
#endif
    A->armed = 0;
}

/* ---- Memory Tracking ---- */

void ab_trak(ab_ctx_t *A, uint64_t base, uint64_t size,
             const char *label, uint8_t flags)
{
    if (!A || A->n_alloc >= AB_MAX_ALLOC) return;
    ab_alloc_t *a = &A->allocs[A->n_alloc++];
    a->base  = base;
    a->size  = size;
    a->flags = flags;
    if (label) {
        size_t len = strlen(label);
        if (len >= AB_MAX_LABEL) len = AB_MAX_LABEL - 1;
        memcpy(a->label, label, len);
        a->label[len] = '\0';
    } else {
        a->label[0] = '\0';
    }
}

/* ---- Dispatch Snapshot ---- */

void ab_snag(ab_ctx_t *A, const bc_kernel_t *k,
             const char *name, const char *chip,
             uint32_t gx, uint32_t gy, uint32_t gz,
             uint32_t bx, uint32_t by, uint32_t bz,
             const void *args, uint32_t args_sz)
{
    if (!A) return;
    ab_dctx_t *D = &A->dctx;

    if (name) {
        size_t nl = strlen(name);
        if (nl >= sizeof(D->kernel)) nl = sizeof(D->kernel) - 1;
        memcpy(D->kernel, name, nl);
        D->kernel[nl] = '\0';
    }
    if (chip) {
        size_t cl = strlen(chip);
        if (cl >= sizeof(D->chip)) cl = sizeof(D->chip) - 1;
        memcpy(D->chip, chip, cl);
        D->chip[cl] = '\0';
    }

    D->grid[0] = gx; D->grid[1] = gy; D->grid[2] = gz;
    D->block[0] = bx; D->block[1] = by; D->block[2] = bz;

    if (k) {
        D->kobj    = k->kernel_object;
        D->karg_sz = k->kernarg_size;
        D->sgprs   = 0; /* filled from KD if available */
        D->vgprs   = 0;
        D->lds     = k->group_size;
        D->scratch = k->private_size;
        D->wave_sz = 32; /* default, overridden by caller if known */
    }

    /* Snapshot kernarg contents for the dump */
    if (args && args_sz > 0) {
        uint32_t snap = args_sz;
        if (snap > sizeof(A->args_snap)) snap = (uint32_t)sizeof(A->args_snap);
        memcpy(A->args_snap, args, snap);
        D->args_sz = snap;
    }
}

/* ---- Fault Correlation ---- */

/* Find the nearest tracked allocation to a fault address.
 * Returns a static buffer describing the relationship. */
static const char *ab_near(const ab_ctx_t *A, uint64_t tea,
                           char *buf, int bsz)
{
    if (A->n_alloc == 0) {
        snprintf(buf, (size_t)bsz, "(no tracked allocations)");
        return buf;
    }

    /* Check if TEA falls inside any allocation */
    for (uint32_t i = 0; i < A->n_alloc && i < AB_MAX_ALLOC; i++) {
        const ab_alloc_t *a = &A->allocs[i];
        if (tea >= a->base && tea < a->base + a->size) {
            uint64_t off = tea - a->base;
            snprintf(buf, (size_t)bsz,
                     "inside %s +0x%" PRIX64 " (0x%" PRIX64 ", %" PRIu64 " bytes)",
                     a->label, off, a->base, a->size);
            return buf;
        }
    }

    /* Find nearest allocation */
    uint64_t best_dist = UINT64_MAX;
    int best_idx = -1;
    int best_dir = 0; /* 1 = past end, -1 = before start */

    for (uint32_t i = 0; i < A->n_alloc && i < AB_MAX_ALLOC; i++) {
        const ab_alloc_t *a = &A->allocs[i];
        uint64_t end = a->base + a->size;

        if (tea >= end) {
            uint64_t d = tea - end;
            if (d < best_dist) { best_dist = d; best_idx = (int)i; best_dir = 1; }
        } else if (tea < a->base) {
            uint64_t d = a->base - tea;
            if (d < best_dist) { best_dist = d; best_idx = (int)i; best_dir = -1; }
        }
    }

    if (best_idx < 0) {
        snprintf(buf, (size_t)bsz, "(no nearby allocation found)");
        return buf;
    }

    const ab_alloc_t *a = &A->allocs[best_idx];
    if (best_dir > 0) {
        snprintf(buf, (size_t)bsz,
                 "%s +0x%" PRIX64 " past end (0x%" PRIX64 ", %" PRIu64 " bytes)",
                 a->label, best_dist, a->base, a->size);
    } else {
        snprintf(buf, (size_t)bsz,
                 "0x%" PRIX64 " before %s (0x%" PRIX64 ", %" PRIu64 " bytes)",
                 best_dist, a->label, a->base, a->size);
    }
    return buf;
}

/* ---- Dump Formatter ---- */

/* The flag string for an allocation (RW, RX, etc.) */
static const char *ab_fstr(uint8_t f)
{
    if (f & AB_FL_RX) return "RX";
    if (f & AB_FL_KA) return "RW";  /* kernarg is RW */
    if (f & AB_FL_RW) return "RW";
    return "??";
}

static void ab_line(FILE *out)
{
    fprintf(out, " ============================================================\n");
}

int ab_dump(const ab_ctx_t *A, FILE *out)
{
    if (!A || !out) return -1;

    const char *cstr = ab_cstr(A->code);
    const char *desc = ab_mstr(A->code);
    const ab_dctx_t *D = &A->dctx;

    /* Timestamp */
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char ts[32];
    if (tm) {
        snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d:%02d",
                 tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
                 tm->tm_hour, tm->tm_min, tm->tm_sec);
    } else {
        snprintf(ts, sizeof(ts), "(unknown time)");
    }

    /* ---- Header ---- */
    ab_line(out);
    fprintf(out, "  ABEND %s    %s\n", cstr, desc);
    fprintf(out, "  KERNEL %-16s GPU %-10s %s\n",
            D->kernel[0] ? D->kernel : "(none)",
            D->chip[0]   ? D->chip   : "??",
            ts);
    ab_line(out);
    fprintf(out, "\n");

    /* ---- Completion ---- */
    fprintf(out, " COMPLETION:\n");
    fprintf(out, "   Code   %s (%s)\n", cstr, desc);
    fprintf(out, "   Reason %08X\n", A->reason);
    fprintf(out, "\n");

    /* ---- TEA (Translation Exception Address) ----
     * literally TEA, babe */
    {
        char nbuf[256];
        ab_near(A, A->tea, nbuf, (int)sizeof(nbuf));
        fprintf(out, " FAULT ADDRESS:\n");
        fprintf(out, "   TEA    0x%016" PRIX64 "\n", A->tea);
        fprintf(out, "   Near   %s\n", nbuf);
        fprintf(out, "\n");
    }

    /* ---- Kernel Descriptor ---- */
    if (D->kernel[0]) {
        fprintf(out, " KERNEL DESCRIPTOR:\n");
        fprintf(out, "   Entry  +0x%X", (unsigned)(D->kobj & 0xFFFF));
        fprintf(out, "    SGPRs %u", D->sgprs);
        fprintf(out, "    VGPRs %u\n", D->vgprs);
        fprintf(out, "   LDS    %-10u Scratch %-8u Wave %u\n",
                D->lds, D->scratch, D->wave_sz);
        if (D->wg_max)
            fprintf(out, "   WG Max %u\n", D->wg_max);
        fprintf(out, "\n");
    }

    /* ---- Memory Map ---- */
    if (A->n_alloc > 0) {
        fprintf(out, " MEMORY MAP:\n");
        fprintf(out, "   %-16s %-8s %-6s %s\n", "BASE", "SIZE", "FLAGS", "LABEL");
        for (uint32_t i = 0; i < A->n_alloc && i < AB_MAX_ALLOC; i++) {
            const ab_alloc_t *a = &A->allocs[i];
            fprintf(out, "   %016" PRIX64 " %08" PRIX64 " %-6s %s\n",
                    a->base, a->size, ab_fstr(a->flags), a->label);
        }
        fprintf(out, "\n");
    }

    /* ---- Dispatch ---- */
    if (D->kernel[0]) {
        fprintf(out, " DISPATCH:\n");
        fprintf(out, "   Grid  (%u, %u, %u)    Block (%u, %u, %u)\n",
                D->grid[0], D->grid[1], D->grid[2],
                D->block[0], D->block[1], D->block[2]);
        if (D->args_sz > 0)
            fprintf(out, "   Args  %u bytes\n", D->args_sz);
        fprintf(out, "\n");
    }

    /* ---- Source Map ---- */
    if (A->n_smap > 0 && A->src_file[0]) {
        fprintf(out, " SOURCE MAP (from .debug_bc):\n");
        uint32_t limit = A->n_smap;
        if (limit > 16) limit = 16; /* cap output to keep dumps readable */
        for (uint32_t i = 0; i < limit; i++) {
            fprintf(out, "   0x%04X  %s:%u\n",
                    A->smap[i].offset, A->src_file, A->smap[i].line);
        }
        if (A->n_smap > 16)
            fprintf(out, "   ... (%u more entries)\n", A->n_smap - 16);
        fprintf(out, "\n");
    }

    /* ---- Footer ---- */
    ab_line(out);
    fprintf(out, "  END OF DUMP    %s    %s\n", cstr,
            D->kernel[0] ? D->kernel : "(none)");
    ab_line(out);

    return 0;
}

/* ---- Source Map Loader ---- */

/* Parse .debug_bc section from an ELF binary.
 * We walk the section header table looking for a section named
 * ".debug_bc", then read the BCDB-format entries.
 *
 * Minimal ELF parsing -- just enough to find a section by name.
 * If the ELF is corrupt or missing the section, we quietly return. */

int ab_slod(ab_ctx_t *A, const uint8_t *elf, uint32_t elf_sz)
{
    if (!A || !elf || elf_sz < 64) return -1;

    /* Verify ELF magic */
    if (elf[0] != 0x7F || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
        return -1;
    if (elf[4] != 2) return -1;  /* ELF64 only */

    /* Read header fields (little-endian) */
    uint64_t sh_off;
    uint16_t sh_ent, sh_num, sh_str;
    memcpy(&sh_off, elf + 40, 8);
    memcpy(&sh_ent, elf + 58, 2);
    memcpy(&sh_num, elf + 60, 2);
    memcpy(&sh_str, elf + 62, 2);

    if (sh_ent < 64 || sh_num == 0 || sh_str >= sh_num) return -1;
    if (sh_off + (uint64_t)sh_ent * sh_num > elf_sz) return -1;

    /* Find .shstrtab to resolve section names */
    const uint8_t *str_sh = elf + sh_off + (uint64_t)sh_str * sh_ent;
    uint64_t str_off, str_sz;
    memcpy(&str_off, str_sh + 24, 8);
    memcpy(&str_sz,  str_sh + 32, 8);
    if (str_off + str_sz > elf_sz) return -1;
    const char *strtab = (const char *)(elf + str_off);

    /* Walk sections looking for ".debug_bc" */
    for (uint16_t si = 0; si < sh_num; si++) {
        const uint8_t *sh = elf + sh_off + (uint64_t)si * sh_ent;
        uint32_t name_idx;
        memcpy(&name_idx, sh, 4);
        if (name_idx >= str_sz) continue;

        if (strcmp(strtab + name_idx, ".debug_bc") != 0) continue;

        /* Found it. Read section offset + size. */
        uint64_t sec_off, sec_sz;
        memcpy(&sec_off, sh + 24, 8);
        memcpy(&sec_sz,  sh + 32, 8);
        if (sec_off + sec_sz > elf_sz) return -1;

        const uint8_t *data = elf + sec_off;

        /* BCDB header: 4B magic + 4B count */
        if (sec_sz < 8) return -1;
        if (data[0] != 'B' || data[1] != 'C' ||
            data[2] != 'D' || data[3] != 'B')
            return -1;

        uint32_t count;
        memcpy(&count, data + 4, 4);

        /* Sanity: entries must fit in section */
        if (8 + (uint64_t)count * 8 > sec_sz) return -1;
        if (count > AB_MAX_SMAP) count = AB_MAX_SMAP;

        /* Load entries */
        const uint8_t *ep = data + 8;
        for (uint32_t ei = 0; ei < count; ei++) {
            memcpy(&A->smap[ei].offset, ep + ei * 8,     4);
            memcpy(&A->smap[ei].line,   ep + ei * 8 + 4, 4);
        }
        A->n_smap = count;
        return 0;
    }

    /* No .debug_bc section found -- not an error, just no debug info */
    return -1;
}
