/*
 * verify.c — post-isel verification pass
 *
 * Catches encoding violations before they become "Reason: Unknown"
 * on actual hardware.  The GPU is not your friend; it will eat your
 * instructions and blame you for the indigestion.
 *
 * Five checks, two phases.  Takes about as long as a sneeze.
 */

#include "verify.h"
#include "encode.h"
#include <stdio.h>
#include <stdarg.h>

/* ---- Counters (file-scope, reset per bc_vfy call) ---- */
static uint32_t n_err;
static uint32_t n_wrn;

/* ---- Diagnostics ---- */

static void vfy_err(uint32_t idx, const char *mn,
                    const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "verify: E: [%u] %s: ", idx, mn);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    n_err++;
}

static void vfy_wrn(uint32_t idx, const char *mn,
                    const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "verify: W: [%u] %s: ", idx, mn);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    n_wrn++;
}

/* ---- Operand Queries ---- */

/* true if operand is a vector register (physical or virtual) */
static int is_vreg(uint8_t kind)
{
    return kind == MOP_VGPR || kind == MOP_VREG_V;
}

/* true if operand is a virtual register (not yet allocated) */
static int is_virt(uint8_t kind)
{
    return kind == MOP_VREG_S || kind == MOP_VREG_V;
}

/* ---- Format Checks ---- */

/*
 * Check 1: VGPR in scalar source.
 * SOP1/SOP2/SOPC are scalar-only.  A VGPR operand here encodes as
 * s0 on some targets or simply faults.  Either way, not what you wanted.
 */
static void vfy_sop(const minst_t *mi, uint32_t idx, const char *mn)
{
    uint32_t i;
    for (i = mi->num_defs; i < mi->num_defs + mi->num_uses; i++) {
        uint8_t k = mi->operands[i].kind;
        if (is_vreg(k)) {
            vfy_err(idx, mn, "VGPR v%u in scalar source (op %u)",
                    (unsigned)mi->operands[i].reg_num,
                    i - mi->num_defs);
        }
    }
}

/*
 * Check 2: global memory without SGPR base pointer.
 * FLAT_GBL needs an SGPR pair for the base address.  Without one
 * the encoder fills in s0 and you get a memory fault at address 0,
 * which is technically a valid address but spiritually a war crime.
 */
static void vfy_gbl(const minst_t *mi, uint32_t idx, const char *mn)
{
    uint32_t i;
    int found = 0;
    for (i = mi->num_defs; i < mi->num_defs + mi->num_uses; i++) {
        uint8_t k = mi->operands[i].kind;
        if (k == MOP_SGPR || k == MOP_VREG_S) {
            found = 1;
            break;
        }
    }
    if (!found)
        vfy_wrn(idx, mn, "global mem with no SGPR base (sbase missing?)");
}

/*
 * Check 3: non-VGPR in VSRC1.
 * VOP2/VOPC second source is VGPR-only in hardware.  Immediates or
 * SGPRs in VSRC1 silently encode as v0.  The silicon doesn't even
 * blink — it just reads whatever v0 happens to contain, which is
 * usually not what you had in mind.
 */
static void vfy_vop2(const minst_t *mi, uint32_t idx, const char *mn)
{
    /* VSRC1 is the 2nd USE operand (index num_defs + 1) */
    if (mi->num_uses < 2) return;
    uint32_t si = (uint32_t)(mi->num_defs + 1);
    uint8_t k = mi->operands[si].kind;
    if (!is_vreg(k) && k != MOP_NONE) {
        const char *what = "unknown";
        switch (k) {
        case MOP_IMM:     what = "immediate"; break;
        case MOP_SGPR:    what = "SGPR";      break;
        case MOP_VREG_S:  what = "virtual SGPR"; break;
        case MOP_SPECIAL: what = "special";   break;
        case MOP_LABEL:   what = "label";     break;
        default: break;
        }
        vfy_err(idx, mn, "%s in VSRC1 (will encode as v0)", what);
    }
}

/*
 * Check 4: virtual registers after regalloc.
 * If you're still virtual at this point, something went
 * spectacularly wrong upstream.  Like arriving at the airport
 * and discovering you forgot to book a flight.
 */
static void vfy_phys(const minst_t *mi, uint32_t idx, const char *mn)
{
    uint32_t i;
    uint32_t total = (uint32_t)(mi->num_defs + mi->num_uses);
    for (i = 0; i < total; i++) {
        uint8_t k = mi->operands[i].kind;
        if (is_virt(k)) {
            vfy_err(idx, mn, "virtual reg %%%u still present after RA (op %u)",
                    (unsigned)mi->operands[i].reg_num, i);
        }
    }
}

/*
 * Check 5: physical register out of bounds.
 * SGPR bank is 0-101 (102 regs), VGPR bank is 0-255.
 * Anything beyond that either wraps or faults, depending on
 * the hardware's mood and the phase of the moon.
 */
static void vfy_bnds(const minst_t *mi, uint32_t idx, const char *mn)
{
    uint32_t i;
    uint32_t total = (uint32_t)(mi->num_defs + mi->num_uses);
    for (i = 0; i < total; i++) {
        uint8_t k = mi->operands[i].kind;
        uint16_t r = mi->operands[i].reg_num;
        /* Skip spill-encoded operands (bit 15 = pending resolution) */
        if (r & 0x8000u) continue;
        if (k == MOP_SGPR && r > 101) {
            vfy_err(idx, mn, "s%u out of bounds (max 101)", (unsigned)r);
        }
        if (k == MOP_VGPR && r >= AMD_MAX_VGPRS) {
            vfy_err(idx, mn, "v%u out of bounds (max %u)", (unsigned)r,
                    (unsigned)(AMD_MAX_VGPRS - 1));
        }
    }
}

/* ---- Main ---- */

vfy_res_t bc_vfy(const amd_module_t *A, int phase)
{
    uint32_t i;
    vfy_res_t res;

    n_err = 0;
    n_wrn = 0;

    for (i = 0; i < A->num_minsts; i++) {
        const minst_t *mi = &A->minsts[i];
        const amd_enc_entry_t *ent = amd_enc_ent(A, mi->op);
        uint8_t fmt = ent->fmt;
        const char *mn = ent->mnemonic;

        /* skip pseudos and bare instructions (labels, nops) */
        if (fmt == AMD_FMT_PSEUDO) continue;
        if (mn == NULL) {
            vfy_err(i, "?", "opcode %u has no encoding on this target",
                    (unsigned)mi->op);
            continue;
        }
        if (mi->num_defs == 0 && mi->num_uses == 0) continue;

        /* checks 1-3: encoding format constraints */
        switch (fmt) {
        case AMD_FMT_SOP1:
        case AMD_FMT_SOP2:
        case AMD_FMT_SOPC:
            vfy_sop(mi, i, mn);
            break;
        case AMD_FMT_VOP2:
        case AMD_FMT_VOPC:
            vfy_vop2(mi, i, mn);
            break;
        case AMD_FMT_FLAT_GBL:
            vfy_gbl(mi, i, mn);
            break;
        default:
            break;
        }

        /* checks 4-5: post-regalloc only */
        if (phase == VFY_RA) {
            vfy_phys(mi, i, mn);
            vfy_bnds(mi, i, mn);
        }
    }

    res.errs = n_err;
    res.wrns = n_wrn;
    return res;
}
