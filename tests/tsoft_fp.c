/* tsoft_fp.c -- Host-compiled tests for the IEEE-754 soft-float
 * runtime. Validates the math against bit-exact expected values
 * for IEEE corner cases (NaN, Inf, signed zero, etc.) and against
 * the host FPU for general arithmetic so any drift from real
 * hardware behaviour shows up immediately.
 *
 * The runtime is FTZ/DAZ by default, so tests that exercise
 * subnormals are guarded with SFP_STRICT_IEEE and currently mark
 * SKIP. When the strict-mode work lands, flipping the flag and
 * removing the SKIPs is the regression-proof migration path. */

#include "tharns.h"
#include "soft_fp.h"
#include "soft_fp_internal.h"
#include <math.h>
#include <string.h>

/* Bit-pattern helpers so tests can talk in canonical hex. */
static float fbits(uint32_t b)
{
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}
static uint32_t bbits(float f)
{
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

/* ---- Bit-pattern constants ---- */

#define F_POS_ZERO   0x00000000u
#define F_NEG_ZERO   0x80000000u
#define F_POS_ONE    0x3F800000u   /* +1.0     */
#define F_NEG_ONE    0xBF800000u   /* -1.0     */
#define F_POS_TWO    0x40000000u   /* +2.0     */
#define F_NEG_TWO    0xC0000000u   /* -2.0     */
#define F_POS_INF    0x7F800000u
#define F_NEG_INF    0xFF800000u
#define F_QNAN       0x7FC00000u
#define F_SMALLEST_NORMAL  0x00800000u  /* 2^-126   */
#define F_LARGEST_NORMAL   0x7F7FFFFFu  /* close to 2^128 */

/* ---- Negation ---- */

static void sfp_neg_one(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_ONE))), F_NEG_ONE);
    CHEQ(bbits(__negsf2(fbits(F_NEG_ONE))), F_POS_ONE);
    PASS();
}
TH_REG("soft_fp", sfp_neg_one);

static void sfp_neg_zero(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_ZERO))), F_NEG_ZERO);
    CHEQ(bbits(__negsf2(fbits(F_NEG_ZERO))), F_POS_ZERO);
    PASS();
}
TH_REG("soft_fp", sfp_neg_zero);

static void sfp_neg_inf(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_INF))), F_NEG_INF);
    CHEQ(bbits(__negsf2(fbits(F_NEG_INF))), F_POS_INF);
    PASS();
}
TH_REG("soft_fp", sfp_neg_inf);

/* ---- Addition: canonical values ---- */

static void sfp_add_one_plus_one(void)
{
    /* 1.0 + 1.0 = 2.0 */
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_POS_ONE))), F_POS_TWO);
    PASS();
}
TH_REG("soft_fp", sfp_add_one_plus_one);

static void sfp_add_one_minus_one(void)
{
    /* 1.0 + (-1.0) = +0.0 per round-to-nearest. */
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_NEG_ONE))), F_POS_ZERO);
    PASS();
}
TH_REG("soft_fp", sfp_add_one_minus_one);

static void sfp_add_zero_zero(void)
{
    /* +0 + +0 = +0; -0 + -0 = -0; +0 + -0 = +0. */
    CHEQ(bbits(__addsf3(fbits(F_POS_ZERO), fbits(F_POS_ZERO))), F_POS_ZERO);
    CHEQ(bbits(__addsf3(fbits(F_NEG_ZERO), fbits(F_NEG_ZERO))), F_NEG_ZERO);
    CHEQ(bbits(__addsf3(fbits(F_POS_ZERO), fbits(F_NEG_ZERO))), F_POS_ZERO);
    PASS();
}
TH_REG("soft_fp", sfp_add_zero_zero);

static void sfp_add_inf(void)
{
    /* Inf + Inf = Inf; Inf - Inf = NaN. */
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_POS_INF))), F_POS_INF);
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_NEG_INF))), F_QNAN);
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_POS_ONE))), F_POS_INF);
    PASS();
}
TH_REG("soft_fp", sfp_add_inf);

static void sfp_add_nan(void)
{
    /* NaN + anything = NaN. */
    CHEQ(bbits(__addsf3(fbits(F_QNAN), fbits(F_POS_ONE))), F_QNAN);
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_QNAN))), F_QNAN);
    PASS();
}
TH_REG("soft_fp", sfp_add_nan);

/* ---- Addition vs host FPU on random normal values ---- */

static uint32_t prng_state = 1u;
static uint32_t prng_u32(void)
{
    /* xorshift32, plenty good for spreading test inputs around. */
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

/* Generate a random fp32 that is finite and normal (no NaN, Inf,
 * or subnormal). */
static float random_normal_float(void)
{
    uint32_t bits = prng_u32();
    /* Force exponent into normal range [1, 254]. */
    uint32_t exp = 1u + (prng_u32() % 254u);
    bits = (bits & 0x807FFFFFu) | (exp << 23);
    return fbits(bits);
}

static void sfp_add_vs_host(void)
{
    prng_state = 0xC0FFEEu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a + b;
        float got  = __addsf3(a, b);
        /* If host result is in subnormal range we expect FTZ to
         * differ, so skip those. Same if NaN or Inf appears for
         * some reason in the host result; those cases are covered
         * by the canonical tests above. */
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  ADD mismatch: 0x%08X + 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("soft_fp", sfp_add_vs_host);

/* ---- Multiplication ---- */

static void sfp_mul_one_times_two(void)
{
    CHEQ(bbits(__mulsf3(fbits(F_POS_ONE), fbits(F_POS_TWO))), F_POS_TWO);
    PASS();
}
TH_REG("soft_fp", sfp_mul_one_times_two);

static void sfp_mul_sign(void)
{
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ONE), fbits(F_POS_TWO))), F_NEG_TWO);
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ONE), fbits(F_NEG_ONE))), F_POS_ONE);
    PASS();
}
TH_REG("soft_fp", sfp_mul_sign);

static void sfp_mul_inf_zero(void)
{
    /* Inf * 0 is NaN per IEEE. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_POS_ZERO))), F_QNAN);
    CHEQ(bbits(__mulsf3(fbits(F_POS_ZERO), fbits(F_POS_INF))), F_QNAN);
    PASS();
}
TH_REG("soft_fp", sfp_mul_inf_zero);

static void sfp_mul_inf_finite(void)
{
    /* Inf * finite_positive = Inf; Inf * finite_negative = -Inf. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_POS_ONE))), F_POS_INF);
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_NEG_ONE))), F_NEG_INF);
    PASS();
}
TH_REG("soft_fp", sfp_mul_inf_finite);

static void sfp_mul_zero_sign(void)
{
    /* Sign-of-zero rule: +0 * -1 = -0. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_ZERO), fbits(F_NEG_ONE))), F_NEG_ZERO);
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ZERO), fbits(F_NEG_ONE))), F_POS_ZERO);
    PASS();
}
TH_REG("soft_fp", sfp_mul_zero_sign);

static void sfp_mul_vs_host(void)
{
    prng_state = 0xBADBEEFu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a * b;
        float got  = __mulsf3(a, b);
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  MUL mismatch: 0x%08X * 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("soft_fp", sfp_mul_vs_host);

/* ---- Division ---- */

static void sfp_div_one_by_two(void)
{
    /* 1.0 / 2.0 = 0.5 */
    CHEQ(bbits(__divsf3(fbits(F_POS_ONE), fbits(F_POS_TWO))), 0x3F000000u);
    PASS();
}
TH_REG("soft_fp", sfp_div_one_by_two);

static void sfp_div_by_zero(void)
{
    /* x / 0 = signed Inf for finite x; 0/0 = NaN. */
    CHEQ(bbits(__divsf3(fbits(F_POS_ONE), fbits(F_POS_ZERO))), F_POS_INF);
    CHEQ(bbits(__divsf3(fbits(F_NEG_ONE), fbits(F_POS_ZERO))), F_NEG_INF);
    CHEQ(bbits(__divsf3(fbits(F_POS_ZERO), fbits(F_POS_ZERO))), F_QNAN);
    PASS();
}
TH_REG("soft_fp", sfp_div_by_zero);

static void sfp_div_inf_inf(void)
{
    /* Inf / Inf = NaN. */
    CHEQ(bbits(__divsf3(fbits(F_POS_INF), fbits(F_POS_INF))), F_QNAN);
    PASS();
}
TH_REG("soft_fp", sfp_div_inf_inf);

static void sfp_div_vs_host(void)
{
    prng_state = 0xDEADBABEu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a / b;
        float got  = __divsf3(a, b);
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  DIV mismatch: 0x%08X / 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("soft_fp", sfp_div_vs_host);

/* ---- Comparisons ---- */

static void sfp_cmp_equal(void)
{
    CHEQ(__eqsf2(fbits(F_POS_ONE), fbits(F_POS_ONE)), 0);
    CHEQ(__eqsf2(fbits(F_POS_ONE), fbits(F_POS_TWO)), 1);
    /* +0 == -0 by IEEE rule. */
    CHEQ(__eqsf2(fbits(F_POS_ZERO), fbits(F_NEG_ZERO)), 0);
    PASS();
}
TH_REG("soft_fp", sfp_cmp_equal);

static void sfp_cmp_less(void)
{
    CHECK(__ltsf2(fbits(F_NEG_ONE), fbits(F_POS_ONE)) < 0);
    CHECK(__ltsf2(fbits(F_POS_ONE), fbits(F_NEG_ONE)) > 0);
    CHEQ(__ltsf2(fbits(F_POS_ONE), fbits(F_POS_ONE)), 0);
    PASS();
}
TH_REG("soft_fp", sfp_cmp_less);

static void sfp_cmp_nan(void)
{
    /* NaN comparisons are unordered; the magnitude comparisons
     * return a positive value per libgcc convention. */
    CHECK(__unordsf2(fbits(F_QNAN), fbits(F_POS_ONE)));
    CHECK(__ltsf2  (fbits(F_QNAN), fbits(F_POS_ONE)) > 0);
    CHECK(__lesf2  (fbits(F_QNAN), fbits(F_POS_ONE)) > 0);
    PASS();
}
TH_REG("soft_fp", sfp_cmp_nan);

/* ---- Integer to float ---- */

static void sfp_int_to_float(void)
{
    CHEQ(bbits(__floatsisf(0)),  F_POS_ZERO);
    CHEQ(bbits(__floatsisf(1)),  F_POS_ONE);
    CHEQ(bbits(__floatsisf(-1)), F_NEG_ONE);
    CHEQ(bbits(__floatsisf(2)),  F_POS_TWO);
    PASS();
}
TH_REG("soft_fp", sfp_int_to_float);

static void sfp_int_to_float_large(void)
{
    /* 16777216 = 2^24 is exactly representable. */
    CHEQ(bbits(__floatsisf(16777216)),  0x4B800000u);
    CHEQ(bbits(__floatsisf(-16777216)), 0xCB800000u);
    PASS();
}
TH_REG("soft_fp", sfp_int_to_float_large);

/* ---- Float to integer ---- */

static void sfp_float_to_int(void)
{
    CHEQ(__fixsfsi(fbits(F_POS_ZERO)),  0);
    CHEQ(__fixsfsi(fbits(F_POS_ONE)),   1);
    CHEQ(__fixsfsi(fbits(F_NEG_ONE)),  -1);
    CHEQ(__fixsfsi(fbits(F_POS_TWO)),   2);
    PASS();
}
TH_REG("soft_fp", sfp_float_to_int);

static void sfp_float_to_int_truncates(void)
{
    /* 1.5 = 0x3FC00000 should truncate to 1. */
    CHEQ(__fixsfsi(fbits(0x3FC00000u)),  1);
    /* -1.5 truncates to -1. */
    CHEQ(__fixsfsi(fbits(0xBFC00000u)), -1);
    PASS();
}
TH_REG("soft_fp", sfp_float_to_int_truncates);

static void sfp_float_to_int_overflow(void)
{
    /* Inf clamps to INT_MAX / INT_MIN. */
    CHEQ(__fixsfsi(fbits(F_POS_INF)), (int32_t)0x7FFFFFFF);
    CHEQ(__fixsfsi(fbits(F_NEG_INF)), (int32_t)0x80000000);
    /* NaN returns 0 per convention. */
    CHEQ(__fixsfsi(fbits(F_QNAN)), 0);
    PASS();
}
TH_REG("soft_fp", sfp_float_to_int_overflow);

/* ---- Subnormal handling (strict-only; SKIP by default) ----
 *
 * In FTZ/DAZ mode (the default for v0.5) subnormal inputs flush
 * to zero on the way in and subnormal outputs flush on the way
 * out. These tests would only pass when SFP_STRICT_IEEE is on;
 * we keep them visible so the strict-mode work has a known test
 * surface to revive. */

static void sfp_subnormal_unpacks_to_zero(void)
{
#if SFP_STRICT_IEEE
    SKIP("strict-mode subnormal handling not yet implemented");
#else
    /* Smallest positive subnormal is 0x00000001. Under FTZ, any
     * arithmetic involving it should treat it as +0. */
    float sub  = fbits(0x00000001u);
    float zero = fbits(F_POS_ZERO);
    CHEQ(bbits(__addsf3(sub, zero)), F_POS_ZERO);
    PASS();
#endif
}
TH_REG("soft_fp", sfp_subnormal_unpacks_to_zero);
