#include "soft_fp.h"
#include "soft_fp_internal.h"

/* IEEE-754 fp32 soft-float; libgcc-named entry points for lowered float code.
 * Round-to-nearest-even, FTZ/DAZ on subnormals (SFP_STRICT_IEEE flips it),
 * canonical quiet NaN. Mantissas carry the implicit leading 1 (1.0 = 0x800000). */

/* ---- Unpack / pack ---- */

/* Decompose a 32-bit IEEE pattern into sign/exp/mant; tag Inf/NaN/zero/subnormal.
 * Subnormals flush to zero unless SFP_STRICT_IEEE. */
__device__ sfp_unpacked_t sfp_unpack(uint32_t bits)
{
    sfp_unpacked_t u;
    u.sign     = (uint8_t)((bits >> 31) & 1u);
    u.exp      = 0;
    u.mant     = 0;
    u.is_zero  = 0;
    u.is_inf   = 0;
    u.is_nan   = 0;
#if SFP_STRICT_IEEE
    u.is_sub   = 0;
    u._pad[0] = u._pad[1] = u._pad[2] = 0;
#else
    u._pad[0] = 0;
#endif

    uint32_t biased = (bits >> SFP_EXP_SHIFT) & 0xFFu;
    uint32_t mant   = bits & SFP_MANT_MASK;

    if (biased == SFP_EXP_MAX) {
        if (mant == 0u) u.is_inf = 1;
        else            u.is_nan = 1;
        return u;
    }
    if (biased == 0u) {
        if (mant == 0u) {
            u.is_zero = 1;
            return u;
        }
#if SFP_STRICT_IEEE
        /* Subnormal preserved: unbiased exp is (1 - bias), no hidden bit
         * since subnormals encode 0.frac. */
        u.is_sub = 1;
        u.exp    = 1 - SFP_EXP_BIAS;
        u.mant   = mant;
        return u;
#else
        /* FTZ: flush subnormal input to signed zero. */
        u.is_zero = 1;
        return u;
#endif
    }
    /* Normal number: attach hidden bit, unbias exponent. */
    u.exp  = (int32_t)biased - SFP_EXP_BIAS;
    u.mant = mant | SFP_HIDDEN_BIT;
    return u;
}

/* Pack sign/exp/mant into a 32-bit IEEE pattern; overflow to Inf, underflow to
 * zero (FTZ). Mantissa must arrive normalised (hidden bit at SFP_MANT_BITS). */
__device__ uint32_t sfp_pack(sfp_unpacked_t u)
{
    if (u.is_nan) return SFP_CANONICAL_NAN;
    if (u.is_inf) return u.sign ? SFP_NEG_INF : SFP_POS_INF;
    if (u.is_zero) return u.sign ? SFP_SIGN_BIT : 0u;

    int32_t biased = u.exp + SFP_EXP_BIAS;
    if (biased >= (int32_t)SFP_EXP_MAX) {
        /* Overflow: result is signed infinity. */
        return u.sign ? SFP_NEG_INF : SFP_POS_INF;
    }
    if (biased <= 0) {
#if SFP_STRICT_IEEE
        /* Strict subnormal output not built yet; fall through to FTZ. */
        return u.sign ? SFP_SIGN_BIT : 0u;
#else
        /* FTZ: anything that would be subnormal flushes to zero. */
        return u.sign ? SFP_SIGN_BIT : 0u;
#endif
    }
    uint32_t mant_bits = u.mant & SFP_MANT_MASK;
    return ((uint32_t)u.sign << 31)
         | ((uint32_t)biased << SFP_EXP_SHIFT)
         | mant_bits;
}

/* ---- Rounding ----
 * Round-to-nearest-even from a wide mantissa to 24 bits: round bit is the top
 * lane bit, sticky the OR below, ties to even. Roundup past 24 bits bumps exp. */
__device__ uint32_t sfp_round_normal(uint8_t sign, int32_t exp,
                          uint64_t wide_mant, int shift_amount)
{
    sfp_unpacked_t u;
    u.sign    = sign;
    u.is_zero = 0;
    u.is_inf  = 0;
    u.is_nan  = 0;
#if SFP_STRICT_IEEE
    u.is_sub  = 0;
    u._pad[0] = u._pad[1] = u._pad[2] = 0;
#else
    u._pad[0] = 0;
#endif

    if (wide_mant == 0u) {
        u.exp = exp;
        u.mant = 0;
        u.is_zero = 1;
        return sfp_pack(u);
    }

    /* Normalise the leading 1 to bit (SFP_MANT_BITS + shift_amount); guard
     * counter bounds the shift so corrupted input can't spin forever. */
    int target_top = SFP_MANT_BITS + shift_amount;
    int guard = 64;
    while (((wide_mant >> target_top) == 0u) && guard-- > 0) {
        wide_mant <<= 1;
        exp--;
    }
    /* Now bit target_top is set. Drain any overflow above it
     * into sticky, then mask off above. */
    while (wide_mant >> (target_top + 1)) {
        uint64_t low = wide_mant & 1u;
        wide_mant >>= 1;
        wide_mant |= low;     /* sticky preservation */
        exp++;
    }

    /* Extract round + sticky from the lane below the target. */
    uint64_t round_mask  = 1ull << (shift_amount - 1);
    uint64_t sticky_mask = round_mask - 1ull;
    uint32_t round_bit   = (wide_mant & round_mask) ? 1u : 0u;
    uint32_t sticky_bit  = (wide_mant & sticky_mask) ? 1u : 0u;
    uint32_t mant24      = (uint32_t)(wide_mant >> shift_amount);

    /* Apply round-to-nearest-even. The condition "sticky_bit
     * or LSB is one" is the canonical even-rounding rule. */
    if (round_bit && (sticky_bit || (mant24 & 1u))) {
        mant24++;
        if (mant24 == (1u << (SFP_MANT_BITS + 1))) {
            /* Rounded past the implicit-bit boundary; renormalise and bump exp
             * (pack handles overflow to Inf). */
            mant24 >>= 1;
            exp++;
        }
    }
    u.exp  = exp;
    u.mant = mant24;
    return sfp_pack(u);
}

/* ---- Negation ----
 * Flip the sign bit. No NaN payloads to preserve in this model. */
__device__ float __negsf2(float a)
{
    return sfp_from_bits(sfp_to_bits(a) ^ SFP_SIGN_BIT);
}

/* ---- Addition / Subtraction ----
 * Special cases first (NaN propagates, Inf gives Inf or NaN, zero collapses to the
 * other operand). Else align the smaller mantissa down to the larger exponent in a
 * 64-bit wide value (shift_amount = 24 leaves guard/sticky) and add or subtract. */
__device__ static float sfp_add_signed(float a, float b, int subtract)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    if (subtract) ub.sign ^= 1u;

    /* NaN propagates. */
    if (ua.is_nan || ub.is_nan) return sfp_from_bits(SFP_CANONICAL_NAN);

    /* Inf cases: opposite-sign Inf + Inf is NaN, else Inf collapses to signed Inf. */
    if (ua.is_inf && ub.is_inf) {
        if (ua.sign != ub.sign) return sfp_from_bits(SFP_CANONICAL_NAN);
        return sfp_from_bits(ua.sign ? SFP_NEG_INF : SFP_POS_INF);
    }
    if (ua.is_inf) return sfp_from_bits(ua.sign ? SFP_NEG_INF : SFP_POS_INF);
    if (ub.is_inf) return sfp_from_bits(ub.sign ? SFP_NEG_INF : SFP_POS_INF);

    /* Zero cases. Sign-of-result rule per IEEE: x + (-x) is +0
     * in round-to-nearest. 0 + 0 is +0 unless both are -0. */
    if (ua.is_zero && ub.is_zero) {
        sfp_unpacked_t z;
        z.sign = (uint8_t)(ua.sign & ub.sign);
        z.is_zero = 1; z.is_inf = 0; z.is_nan = 0;
        z.exp = 0; z.mant = 0;
#if SFP_STRICT_IEEE
        z.is_sub = 0; z._pad[0]=z._pad[1]=z._pad[2]=0;
#else
        z._pad[0] = 0;
#endif
        return sfp_from_bits(sfp_pack(z));
    }
    if (ua.is_zero) return b;   /* with sign already accounted for via ub */
    if (ub.is_zero) return a;

    /* Order operands so ua has the larger or equal exponent.
     * After this point ua dominates and ub gets aligned down. */
    if (ub.exp > ua.exp) {
        sfp_unpacked_t t = ua; ua = ub; ub = t;
    }
    int32_t exp_diff = ua.exp - ub.exp;

    /* Widen both mantissas high in a 64-bit register (24 lane bits for
     * guard/sticky); align ub down by exp_diff, keeping any lost bits as sticky. */
    uint64_t ma = (uint64_t)ua.mant << 24;
    uint64_t mb = (uint64_t)ub.mant << 24;
    uint64_t sticky = 0;
    if (exp_diff >= 56) {       /* ub is below precision entirely */
        sticky = (mb != 0u) ? 1u : 0u;
        mb = 0;
    } else if (exp_diff > 0) {
        sticky = mb & ((1ull << exp_diff) - 1ull);
        mb >>= exp_diff;
    }
    if (sticky) mb |= 1u;       /* OR sticky into LSB of aligned mb */

    uint8_t  result_sign = ua.sign;
    int32_t  result_exp  = ua.exp;
    uint64_t result_mant;
    if (ua.sign == ub.sign) {
        result_mant = ma + mb;
    } else {
        if (ma >= mb) {
            result_mant = ma - mb;
        } else {
            result_mant = mb - ma;
            result_sign = ub.sign;
        }
        /* Exact cancellation returns +0 (round-to-nearest sign convention). */
        if (result_mant == 0u) {
            sfp_unpacked_t z;
            z.sign = 0; z.is_zero = 1; z.is_inf = 0; z.is_nan = 0;
            z.exp = 0; z.mant = 0;
#if SFP_STRICT_IEEE
            z.is_sub = 0; z._pad[0]=z._pad[1]=z._pad[2]=0;
#else
            z._pad[0] = 0;
#endif
            return sfp_from_bits(sfp_pack(z));
        }
    }
    return sfp_from_bits(sfp_round_normal(result_sign, result_exp,
                                          result_mant, 24));
}

__device__ float __addsf3(float a, float b) { return sfp_add_signed(a, b, 0); }
__device__ float __subsf3(float a, float b) { return sfp_add_signed(a, b, 1); }

/* ---- Multiplication ----
 * No alignment: XOR the signs, sum the exponents (less one bias), multiply the
 * 24-bit mantissas to a 48-bit product, then normalise and round. Inf*0 = NaN. */
__device__ float __mulsf3(float a, float b)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    if (ua.is_nan || ub.is_nan) return sfp_from_bits(SFP_CANONICAL_NAN);

    uint8_t out_sign = ua.sign ^ ub.sign;

    if ((ua.is_inf && ub.is_zero) || (ua.is_zero && ub.is_inf))
        return sfp_from_bits(SFP_CANONICAL_NAN);
    if (ua.is_inf || ub.is_inf)
        return sfp_from_bits(out_sign ? SFP_NEG_INF : SFP_POS_INF);
    if (ua.is_zero || ub.is_zero) {
        sfp_unpacked_t z;
        z.sign = out_sign; z.is_zero = 1; z.is_inf = 0; z.is_nan = 0;
        z.exp = 0; z.mant = 0;
#if SFP_STRICT_IEEE
        z.is_sub = 0; z._pad[0]=z._pad[1]=z._pad[2]=0;
#else
        z._pad[0] = 0;
#endif
        return sfp_from_bits(sfp_pack(z));
    }

    /* Two 24-bit mantissas give a 48-bit product (leading bit at 46 or 47);
     * round_normal normalises and rounds. */
    uint64_t product = (uint64_t)ua.mant * (uint64_t)ub.mant;
    int32_t result_exp = ua.exp + ub.exp;
    return sfp_from_bits(sfp_round_normal(out_sign, result_exp,
                                          product, 23));
}

/* ---- Division ----
 * Scale the numerator, divide the 24-bit mantissas as a 64-bit quotient with the
 * remainder folded in as sticky; exponent is the difference of input exponents. */
__device__ float __divsf3(float a, float b)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    if (ua.is_nan || ub.is_nan) return sfp_from_bits(SFP_CANONICAL_NAN);

    uint8_t out_sign = ua.sign ^ ub.sign;

    if (ua.is_inf && ub.is_inf) return sfp_from_bits(SFP_CANONICAL_NAN);
    if (ua.is_zero && ub.is_zero) return sfp_from_bits(SFP_CANONICAL_NAN);
    if (ua.is_inf) return sfp_from_bits(out_sign ? SFP_NEG_INF : SFP_POS_INF);
    if (ub.is_zero) return sfp_from_bits(out_sign ? SFP_NEG_INF : SFP_POS_INF);
    if (ua.is_zero || ub.is_inf) {
        sfp_unpacked_t z;
        z.sign = out_sign; z.is_zero = 1; z.is_inf = 0; z.is_nan = 0;
        z.exp = 0; z.mant = 0;
#if SFP_STRICT_IEEE
        z.is_sub = 0; z._pad[0]=z._pad[1]=z._pad[2]=0;
#else
        z._pad[0] = 0;
#endif
        return sfp_from_bits(sfp_pack(z));
    }

    /* Scale numerator by 2^28: quotient leads at bit 28 (Q>=1) or 27 (Q<1, ratio
     * [0.5,2)); 5 lane bits carry round+sticky. On baby cores `/` becomes __udivdi3. */
    uint64_t num = (uint64_t)ua.mant << 28;
    uint64_t den = (uint64_t)ub.mant;
    uint64_t quot = num / den;
    uint64_t rem  = num % den;
    if (rem) quot |= 1;       /* sticky bit at the bottom */

    int32_t result_exp = ua.exp - ub.exp;
    return sfp_from_bits(sfp_round_normal(out_sign, result_exp,
                                          quot, 5));
}

/* ---- Comparisons ----
 * libgcc convention (see public header). NaN is unordered: __unordsf2 is the
 * dedicated check; ordered relations report so "if (a < b)" is false on NaN. */
__device__ int __unordsf2(float a, float b)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    return (ua.is_nan || ub.is_nan) ? 1 : 0;
}

/* Ordered compare: -1 if a < b, 0 if equal, +1 if a > b. NaN returns +1 to match
 * libgcc (caller should screen NaN via __unordsf2 first). */
__device__ static int sfp_cmp(float a, float b)
{
    uint32_t ab = sfp_to_bits(a);
    uint32_t bb = sfp_to_bits(b);
    sfp_unpacked_t ua = sfp_unpack(ab);
    sfp_unpacked_t ub = sfp_unpack(bb);

    if (ua.is_nan || ub.is_nan) return 1;

    /* +0 and -0 are equal per IEEE despite differing bits. */
    if (ua.is_zero && ub.is_zero) return 0;

    /* Different signs: negative is always less. */
    if (ua.sign != ub.sign) return ua.sign ? -1 : 1;

    /* Same sign: magnitude order follows bit-pattern order (IEEE encoding);
     * negate for two negatives, whose order is reversed. */
    uint32_t mag_a = ab & 0x7FFFFFFFu;
    uint32_t mag_b = bb & 0x7FFFFFFFu;
    if (mag_a == mag_b) return 0;
    int order = (mag_a < mag_b) ? -1 : 1;
    return ua.sign ? -order : order;
}

__device__ int __eqsf2(float a, float b)
{
    if (__unordsf2(a, b)) return 1;
    return (sfp_cmp(a, b) != 0) ? 1 : 0;
}

__device__ int __nesf2(float a, float b)
{
    if (__unordsf2(a, b)) return 1;
    return (sfp_cmp(a, b) != 0) ? 1 : 0;
}

__device__ int __ltsf2(float a, float b)
{
    if (__unordsf2(a, b)) return 1;
    return sfp_cmp(a, b);
}

__device__ int __lesf2(float a, float b)
{
    if (__unordsf2(a, b)) return 1;
    return sfp_cmp(a, b);
}

__device__ int __gtsf2(float a, float b)
{
    if (__unordsf2(a, b)) return -1;
    return sfp_cmp(a, b);
}

__device__ int __gesf2(float a, float b)
{
    if (__unordsf2(a, b)) return -1;
    return sfp_cmp(a, b);
}

/* ---- Integer to float ----
 * Leading-bit position gives the unbiased exponent; bits below become the
 * mantissa (rounded past the 24th). Signed path negates by magnitude. */
__device__ static float sfp_uint_to_float(uint32_t v, uint8_t sign)
{
    if (v == 0u) {
        sfp_unpacked_t z;
        z.sign = sign; z.is_zero = 1; z.is_inf = 0; z.is_nan = 0;
        z.exp = 0; z.mant = 0;
#if SFP_STRICT_IEEE
        z.is_sub = 0; z._pad[0]=z._pad[1]=z._pad[2]=0;
#else
        z._pad[0] = 0;
#endif
        return sfp_from_bits(sfp_pack(z));
    }
    /* Leading 1 bit gives the unbiased exponent. */
    int top = 31;
    while (((v >> top) & 1u) == 0u) top--;
    int32_t exp = top;
    /* Put the leading bit at position 47 (23 + shift_amount 24) so rounding
     * matches the multiply path. */
    uint64_t wide;
    if (top >= 23) wide = (uint64_t)v << (47 - top);
    else           wide = (uint64_t)v << (47 - top);
    return sfp_from_bits(sfp_round_normal(sign, exp, wide, 24));
}

__device__ float __floatsisf(int32_t i)
{
    if (i >= 0) return sfp_uint_to_float((uint32_t)i, 0);
    /* INT_MIN has no positive counterpart; negate through uint32_t for the
     * right magnitude. */
    return sfp_uint_to_float((uint32_t)(-(int64_t)i), 1);
}

__device__ float __floatunsisf(uint32_t u)
{
    return sfp_uint_to_float(u, 0);
}

/* ---- Float to integer (truncate toward zero) ----
 * Overflow is implementation-defined per IEEE; follow LLVM/gcc and clamp to
 * INT_MIN/INT_MAX (signed) or 0/UINT_MAX (unsigned). */
__device__ int32_t __fixsfsi(float a)
{
    sfp_unpacked_t u = sfp_unpack(sfp_to_bits(a));
    if (u.is_nan) return 0;
    if (u.is_zero) return 0;
    if (u.is_inf) return u.sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
    if (u.exp < 0) return 0;
    if (u.exp >= 31) {
        return u.sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
    }
    /* Mantissa has leading bit at position 23; shift to place
     * the integer MSB at position exp. */
    uint32_t mag;
    if (u.exp >= 23) mag = u.mant << (u.exp - 23);
    else             mag = u.mant >> (23 - u.exp);
    return u.sign ? -(int32_t)mag : (int32_t)mag;
}

__device__ uint32_t __fixunssfsi(float a)
{
    sfp_unpacked_t u = sfp_unpack(sfp_to_bits(a));
    if (u.is_nan) return 0;
    if (u.is_zero) return 0;
    if (u.sign) return 0;         /* negative values clamp to 0 */
    if (u.is_inf) return 0xFFFFFFFFu;
    if (u.exp < 0) return 0;
    if (u.exp >= 32) return 0xFFFFFFFFu;
    uint32_t mag;
    if (u.exp >= 23) mag = u.mant << (u.exp - 23);
    else             mag = u.mant >> (23 - u.exp);
    return mag;
}
