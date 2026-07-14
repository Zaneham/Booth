#ifndef BARRACUDA_SOFT_FP_INTERNAL_H
#define BARRACUDA_SOFT_FP_INTERNAL_H

#include <stdint.h>
#include <string.h>

/*
 * On the host the runtime is plain C and __device__ has no meaning;
 * we erase it so the host test compile sees ordinary functions.
 * When Booth compiles the same .c file as part of a kernel
 * compilation unit, __device__ marks each runtime function as
 * callable from kernel code, which is exactly what the float
 * lowering pass needs to be able to BIR_CALL into them.
 */
#ifndef __device__
#define __device__
#endif

/*
 * Internal helpers for the soft-float runtime. The public API is
 * in soft_fp.h; this file is for the arithmetic implementations
 * themselves and is not meant to be included by user code.
 *
 * The pivot point for future-proofing lives here: SFP_STRICT_IEEE
 * is the single compile-time flag that flips the runtime between
 * its v0.5 FTZ/DAZ subset (default, fast, fits the Tensix story)
 * and full IEEE-754 strict mode (slower, branchy, only worth it
 * when we have a workload that depends on subnormals or the
 * rounding modes other than nearest-even).
 *
 * When the flag is 0, subnormal inputs unpack to zero and any
 * arithmetic result that would land in subnormal range packs
 * back to zero. When the flag is 1, the unpacker preserves the
 * subnormal and arithmetic gets a slow path. The pack/unpack
 * helpers are the only places where the flag actually changes
 * behaviour; the arithmetic body is the same code in both modes
 * with the slow path falling through to whatever the helpers
 * report.
 */
#ifndef SFP_STRICT_IEEE
#define SFP_STRICT_IEEE 0
#endif

/* ---- Bit-level fp32 anatomy ---- */
#define SFP_SIGN_BIT     0x80000000u
#define SFP_EXP_MASK     0x7F800000u
#define SFP_EXP_SHIFT    23
#define SFP_EXP_BIAS     127
#define SFP_EXP_MAX      255            /* biased; means Inf or NaN  */
#define SFP_MANT_MASK    0x007FFFFFu
#define SFP_MANT_BITS    23             /* explicit mantissa bits    */
#define SFP_HIDDEN_BIT   0x00800000u    /* implicit leading 1        */

/* The canonical quiet NaN we return from arithmetic that produces
 * a NaN. Matches the value most CUDA and x86 implementations use,
 * which keeps any downstream comparison or print consistent with
 * what a programmer expects to see in the debugger. */
#define SFP_CANONICAL_NAN  0x7FC00000u

/* Bit patterns for +/- infinity, used as constants in fast paths. */
#define SFP_POS_INF        0x7F800000u
#define SFP_NEG_INF        0xFF800000u

/*
 * Decomposed fp32. Unbiased exponent, mantissa with the implicit
 * leading bit attached so normals live in [0x800000, 0xFFFFFF],
 * plus a small set of flags so arithmetic can dispatch on special
 * cases without re-decoding the raw bits.
 *
 * For zero: is_zero set, mant=0, exp irrelevant.
 * For Inf:  is_inf  set, mant=0, exp irrelevant; sign tells +/-.
 * For NaN:  is_nan  set; mant/exp ignored, we'll just return the
 *           canonical NaN bit pattern when packing.
 * For normal: none of the flags; sign/exp/mant carry the value.
 * For subnormal: when SFP_STRICT_IEEE is off, treated as zero on
 *               unpack. When on, preserved (is_sub set, exp = the
 *               minimum unbiased exponent, mant without hidden
 *               bit).
 */
typedef struct {
    int32_t  exp;        /* unbiased exponent */
    uint32_t mant;       /* mantissa with implicit bit added */
    uint8_t  sign;       /* 0 = positive, 1 = negative */
    uint8_t  is_zero;
    uint8_t  is_inf;
    uint8_t  is_nan;
#if SFP_STRICT_IEEE
    uint8_t  is_sub;     /* subnormal; only meaningful in strict mode */
    uint8_t  _pad[3];
#else
    uint8_t  _pad[1];
#endif
} sfp_unpacked_t;

/* ---- Bit reinterpret helpers ----
 *
 * Going between float and its uint32_t bit pattern via memcpy is
 * the C99-portable idiom: it avoids the strict-aliasing pitfalls
 * of unions and the undefined behaviour of pointer-casts. The
 * compiler folds the memcpy into a register move when the types
 * are the same width, so there is no runtime cost.
 */
static inline uint32_t sfp_to_bits(float f)
{
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

static inline float sfp_from_bits(uint32_t b)
{
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* ---- Unpack / pack ---- */
sfp_unpacked_t sfp_unpack(uint32_t bits);
uint32_t       sfp_pack  (sfp_unpacked_t u);

/* ---- Rounding helpers ----
 *
 * sfp_round_normal takes a sign/exp/wide-mantissa triple where
 * the mantissa has SFP_MANT_BITS + extra "guard" bits, and
 * produces a packed fp32 by rounding to nearest-even and packing.
 * The extra bits give the round/sticky information needed to
 * correctly resolve halfway cases.
 *
 * shift_amount is how many bits below the target mantissa width
 * the rounding info lives. For a freshly-computed multiplication
 * that lands in a 48-bit product, shift = 24 (24 wide bits beyond
 * the target 24-bit mantissa).
 */
uint32_t sfp_round_normal(uint8_t sign, int32_t exp,
                          uint64_t wide_mant, int shift_amount);

#endif /* BARRACUDA_SOFT_FP_INTERNAL_H */
