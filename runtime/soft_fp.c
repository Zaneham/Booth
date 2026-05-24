#include "soft_fp.h"
#include "soft_fp_internal.h"

/*
 * IEEE-754 fp32 soft-float implementation. Each public function is
 * a libgcc-named entry point that lowered float code calls into.
 * The helpers below the public functions handle the pieces that
 * are shared across operations: unpack/pack of the bit layout,
 * round-to-nearest-even with guard and sticky bits, and a couple
 * of small numerical utilities that earn their own functions
 * because they appear in three or four places.
 *
 * Numerical model:
 *   - Round to nearest, ties to even.
 *   - FTZ/DAZ on subnormals (default; SFP_STRICT_IEEE flips it).
 *   - Canonical quiet NaN on all NaN results.
 *   - Inf and zero handled in the special-case dispatch at the
 *     top of each arithmetic function.
 *
 * The internal mantissa convention: when we work on a value we
 * keep the implicit leading 1 attached. So 1.0 has mantissa
 * 0x800000 (bit 23 set, bits 22..0 zero) and unbiased exponent
 * 0. This matches what most reference soft-float code does, and
 * matters because half the operations need to do arithmetic on
 * the full 24-bit value rather than just the 23 explicit bits.
 */

/* ---- Unpack / pack ---- */

/*
 * Decompose a 32-bit IEEE pattern into the unpacked struct.
 * Dispatch order:
 *   1. Exponent all-ones (0xFF biased): Inf if mantissa zero,
 *      NaN otherwise. No further work.
 *   2. Exponent all-zero with nonzero mantissa: subnormal.
 *      In FTZ mode we treat as zero on input; in strict mode
 *      we preserve. The hidden bit is NOT set for subnormals
 *      because the IEEE encoding deliberately drops it (the
 *      true value is 0.f_22f_21...f_0 x 2^-126 rather than
 *      1.something).
 *   3. Exponent all-zero with zero mantissa: signed zero.
 *   4. Anything else: a normal number. Add the implicit bit
 *      and unbias the exponent.
 */
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
        /* Subnormal preserved. The smallest unbiased exponent
         * is (1 - bias) per the IEEE spec; the hidden bit is
         * NOT added because subnormals encode 0.frac. */
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

/*
 * Pack an unpacked value back into a 32-bit IEEE pattern. Handles
 * exponent overflow (-> infinity), exponent underflow (FTZ ->
 * zero; strict mode would produce a subnormal, marked as future
 * work below), and the special-case flags. The caller is
 * responsible for handing us a normalised mantissa (hidden bit
 * at position SFP_MANT_BITS); if the mantissa is over- or
 * under-normalised we mis-pack and the result is wrong.
 */
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
        /* Subnormal output: shift mantissa right by (1 - biased)
         * to denormalise, then pack with biased = 0. Future work;
         * for now we fall through to FTZ behaviour even in strict
         * mode so we do not ship a half-built path. */
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
 *
 * Round-to-nearest-even from a wide intermediate mantissa back
 * down to 24 bits (the implicit-bit width). The wide mantissa
 * has SFP_MANT_BITS + shift_amount bits of precision: the top
 * 24 bits are the target mantissa, and shift_amount bits below
 * that are the round-and-sticky lane.
 *
 * The rule:
 *   - If all the lane bits are zero, no rounding needed.
 *   - Otherwise the round bit is the topmost lane bit (just
 *     below the LSB of the target mantissa). The sticky bit is
 *     the OR of everything strictly below the round bit.
 *   - If round = 0: truncate. The discarded fraction is < 0.5
 *     ULP so we round down.
 *   - If round = 1 and sticky = 0: exactly halfway. Round to
 *     even: bump up only if the target LSB is currently 1.
 *   - If round = 1 and sticky = 1: discarded fraction > 0.5
 *     ULP, round up.
 *
 * When rounding up causes the mantissa to grow past the 24-bit
 * width (e.g. 0xFFFFFF + 1 = 0x1000000), we shift right by one
 * and bump the exponent. Pack then catches any resulting
 * overflow to infinity.
 */
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

    /* Normalise so the leading 1 sits at bit (SFP_MANT_BITS +
     * shift_amount). When the wide mantissa is already
     * normalised this loop runs zero times; when leading bits
     * have cancelled (subtraction) we shift left and decrement
     * the exponent.
     *
     * The upper bound on shift count is chosen to avoid an
     * infinite loop on a corrupted input. */
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
            /* Rounded across the implicit-bit boundary. Bring
             * the mantissa back into range and bump the exponent;
             * pack() handles the overflow-to-Inf check. */
            mant24 >>= 1;
            exp++;
        }
    }
    u.exp  = exp;
    u.mant = mant24;
    return sfp_pack(u);
}

/* ---- Negation ----
 *
 * Trivially flip the sign bit. NaN payload preservation is not
 * a concern in our model; we do not carry payloads. */
__device__ float __negsf2(float a)
{
    return sfp_from_bits(sfp_to_bits(a) ^ SFP_SIGN_BIT);
}

/* ---- Addition / Subtraction ----
 *
 * The fast paths handle special cases first because they avoid
 * needing to align mantissas: NaN inputs propagate, Inf operations
 * resolve to Inf or NaN depending on signs, and zero inputs
 * collapse to the other operand. The slow path is the general
 * normal-plus-normal case where we align the smaller operand
 * mantissa down to the larger operand exponent and then either
 * add (same sign) or subtract (different sign).
 *
 * The alignment uses a 64-bit wide register so we have room for
 * 24 mantissa bits plus enough headroom to capture sticky bits
 * before the small operand vanishes entirely. shift_amount = 24
 * places the mantissa in bits 47..24 of the wide value, leaving
 * 24 lane bits for guard/sticky.
 */
__device__ static float sfp_add_signed(float a, float b, int subtract)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    if (subtract) ub.sign ^= 1u;

    /* NaN propagates. */
    if (ua.is_nan || ub.is_nan) return sfp_from_bits(SFP_CANONICAL_NAN);

    /* Inf cases. Inf - Inf (same magnitude, opposite sign after
     * the subtract-as-add transform) is NaN; everything else
     * with an Inf collapses to a signed Inf. */
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

    /* Widen both mantissas into the upper bits of a 64-bit
     * register, leaving 24 lane bits below for guard/sticky.
     * Shift ub right by exp_diff to align; if the shift would
     * push everything off the bottom we set sticky to 1 since
     * any nonzero remainder rounds up by the smallest amount. */
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
        /* Cancellation can leave the mantissa zero; that is an
         * exact zero, return +0 by convention (round-to-nearest
         * makes the sign positive when the difference is
         * exactly representable). */
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
 *
 * Conceptually simpler than addition because there is no
 * alignment step. Sign of the result is XOR of the input signs;
 * exponent is the sum (we subtract one bias from the result to
 * stay in normal range); mantissa is the product of the two
 * 24-bit values (a 48-bit number), normalised and rounded.
 *
 * Special cases:
 *   NaN propagates.
 *   Inf * 0 = NaN; Inf * anything-else = Inf with XOR sign.
 *   0 * x = signed zero.
 */
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

    /* Product of two 24-bit mantissas is at most 48 bits, with
     * the leading bit at position 46 (if both are exactly 1.0)
     * or 47 (otherwise). round_normal normalises and rounds. */
    uint64_t product = (uint64_t)ua.mant * (uint64_t)ub.mant;
    int32_t result_exp = ua.exp + ub.exp;
    return sfp_from_bits(sfp_round_normal(out_sign, result_exp,
                                          product, 23));
}

/* ---- Division ----
 *
 * Bit-by-bit non-restoring division of the 24-bit mantissas.
 * Shift the dividend left and subtract the divisor, accumulating
 * a quotient bit per shift. We compute one bit more than we need
 * so the round/sticky lane has full precision. The exponent is
 * the difference of input exponents.
 */
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

    /* Scale numerator up by 2^28 so the quotient has the leading
     * bit at position 28 when Q >= 1, or 27 when Q < 1 (since the
     * true mantissa ratio is in [0.5, 2)). round_normal will use
     * normalisation loop handles the Q < 1 case by shifting left
     * and decrementing the exponent. The 5 lane bits below
     * target_top=28 give us round + 4 sticky bits, which is more
     * precision than we actually need but costs nothing in this
     * fixed-shift formulation.
     *
     * On the host this `/` is a native uint64_t division. When
     * the runtime gets compiled FOR the baby cores, this lowers
     * to a __udivdi3 libcall that the soft-int half of the
     * runtime will eventually provide; that work is tracked
     * separately from soft-float and will not block this sitting. */
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
 *
 * The libgcc convention is documented in the public header; we
 * implement straight from there. NaN is unordered: __unordsf2
 * is the dedicated check, and the magnitude comparisons return
 * "not equal, not less, not greater" in a way that makes
 * "if (a < b)" read false when either is NaN.
 */
__device__ int __unordsf2(float a, float b)
{
    sfp_unpacked_t ua = sfp_unpack(sfp_to_bits(a));
    sfp_unpacked_t ub = sfp_unpack(sfp_to_bits(b));
    return (ua.is_nan || ub.is_nan) ? 1 : 0;
}

/*
 * Numerical compare used by all the ordered relations. Returns
 * -1 if a < b, 0 if equal, +1 if a > b. For NaN inputs the
 * caller should already have checked via __unordsf2; here we
 * report +1 by convention (matches what libgcc does, so the
 * downstream branch ends up taking the same arm as it would on
 * hardware FP comparisons).
 */
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

    /* Same sign: compare bit patterns. The IEEE encoding has the
     * convenient property that for positive values, "more
     * positive" means larger bit pattern; for negative values,
     * the order is reversed. So we compare bit patterns with
     * the sign-bit cleared, then negate the result if both were
     * negative. */
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
 *
 * For signed: take absolute value, convert, restore sign. For
 * unsigned: skip the sign handling. Finding the leading bit gives
 * us the unbiased exponent immediately; the bits below it become
 * the mantissa (with rounding from any bits beyond the 24th).
 */
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
    /* Find the position of the leading 1 bit. The mantissa
     * starts with that bit; the unbiased exponent equals that
     * of bit index. */
    int top = 31;
    while (((v >> top) & 1u) == 0u) top--;
    int32_t exp = top;
    /* Place leading bit at position 47 in a wide value so we can
     * round consistently with the multiply path. shift_amount in
     * sfp_round_normal is 24 below the target, so we need the
     * leading bit at position 23+24 = 47. */
    uint64_t wide;
    if (top >= 23) wide = (uint64_t)v << (47 - top);
    else           wide = (uint64_t)v << (47 - top);
    return sfp_from_bits(sfp_round_normal(sign, exp, wide, 24));
}

__device__ float __floatsisf(int32_t i)
{
    if (i >= 0) return sfp_uint_to_float((uint32_t)i, 0);
    /* The most-negative int32 (0x80000000) has no positive
     * counterpart; cast through uint32_t to do the negation
     * unsigned and get the right magnitude. */
    return sfp_uint_to_float((uint32_t)(-(int64_t)i), 1);
}

__device__ float __floatunsisf(uint32_t u)
{
    return sfp_uint_to_float(u, 0);
}

/* ---- Float to integer (truncate toward zero) ----
 *
 * IEEE-754 says conversions that overflow the integer range
 * return an implementation-defined value. We follow the LLVM /
 * gcc convention: clamp to INT_MIN or INT_MAX for signed, 0 or
 * UINT_MAX for unsigned.
 */
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
