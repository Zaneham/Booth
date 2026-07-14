#ifndef BARRACUDA_SOFT_FP_INTERNAL_H
#define BARRACUDA_SOFT_FP_INTERNAL_H

#include <stdint.h>
#include <string.h>

/* Erase __device__ on the host so tests see plain C; under Booth it marks each
 * function callable from kernel code (the float lowering BIR_CALLs into them). */
#ifndef __device__
#define __device__
#endif

/* Internal helpers; public API is in soft_fp.h, not for user code.
 * SFP_STRICT_IEEE = 0 flushes subnormals to zero (fast default); = 1 preserves
 * them via a slow path. The flag only changes behaviour in pack/unpack. */
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

/* Canonical quiet NaN returned from NaN-producing arithmetic; matches CUDA/x86
 * so downstream compares and prints stay consistent. */
#define SFP_CANONICAL_NAN  0x7FC00000u

/* Bit patterns for +/- infinity, used as constants in fast paths. */
#define SFP_POS_INF        0x7F800000u
#define SFP_NEG_INF        0xFF800000u

/* Decomposed fp32: unbiased exp, mantissa with implicit bit (normals in
 * [0x800000, 0xFFFFFF]), plus flags so arithmetic dispatches without re-decoding. */
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
 * memcpy between float and uint32_t: strict-aliasing-safe and folds to a register
 * move, unlike unions or pointer casts. */
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
 * sfp_round_normal rounds a wide mantissa (24 target bits + shift_amount guard
 * bits below) to nearest-even and packs. shift_amount locates the round/sticky
 * lane, e.g. 24 for a 48-bit multiply product. */
uint32_t sfp_round_normal(uint8_t sign, int32_t exp,
                          uint64_t wide_mant, int shift_amount);

#endif /* BARRACUDA_SOFT_FP_INTERNAL_H */
