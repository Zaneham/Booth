#ifndef BARRACUDA_SOFT_FP_H
#define BARRACUDA_SOFT_FP_H

#include <stdint.h>

/*
 * IEEE-754 fp32 soft-float runtime for the Tenstorrent baby
 * RISC-V cores. These functions implement floating-point arithmetic
 * using only RV32IM integer operations. They are called by code
 * the Booth compiler emits whenever a BIR_FADD, BIR_FMUL,
 * BIR_FCMP, etc. needs to lower on a target without hardware
 * floating point.
 *
 * The function names follow the libgcc convention so any other
 * RV32 toolchain that does the same lowering interoperates with
 * us, and so future work that swaps in a different runtime (a
 * vendor library, an SFPU-accelerated path) only has to match
 * these signatures.
 *
 * Numerical model (subset chosen for the Tensix bring-up):
 *
 *   - Round-to-nearest, ties-to-even (CUDA default).
 *   - Flush-to-zero on subnormal inputs and outputs (FTZ/DAZ;
 *     matches CUDA --use_fast_math and Intel SSE FTZ/DAZ modes).
 *   - NaNs propagate; we return a canonical quiet NaN
 *     (0x7FC00000) rather than preserving payloads.
 *   - Inf arithmetic follows the standard cases: Inf+Inf=Inf,
 *     Inf-Inf=NaN, Inf*0=NaN, 0/0=NaN, x/0=Inf for finite nonzero x.
 *   - No exception flags. No signalling NaN distinction. No
 *     rounding-mode register; the rounding mode is hard-coded
 *     to nearest-even in every arithmetic path.
 *
 * A SFP_STRICT_IEEE compile-time flag in soft_fp_internal.h adds
 * subnormal arithmetic and (eventually) the other rounding modes;
 * the flag defaults to 0 (FTZ) for the v0.5 release and the strict
 * path is marked as future work in each affected function.
 *
 * The libgcc-numbered names like "sf3" mean "single-precision
 * float, 3 operands" (two inputs + return); "sf2" means two
 * operands (one input + return, or a comparison returning int).
 * sf3 is for binary arithmetic; sf2 is for unary ops and
 * comparisons.
 */

/* ---- Arithmetic (binary, sf3) ---- */
float __addsf3(float a, float b);   /* a + b */
float __subsf3(float a, float b);   /* a - b */
float __mulsf3(float a, float b);   /* a * b */
float __divsf3(float a, float b);   /* a / b */

/* ---- Negation (unary, sf2) ---- */
float __negsf2(float a);            /* -a */

/* ---- Comparisons (sf2, return int) ----
 *
 * The libgcc comparison convention is unusual but universal:
 *
 *   __eqsf2(a, b) -> 0 if equal, nonzero otherwise.
 *   __nesf2(a, b) -> 0 if NOT equal, nonzero otherwise.
 *   __ltsf2(a, b) -> negative if a <  b, zero or positive otherwise.
 *   __lesf2(a, b) -> non-positive if a <= b, positive otherwise.
 *   __gtsf2(a, b) -> positive if a >  b, non-positive otherwise.
 *   __gesf2(a, b) -> non-negative if a >= b, negative otherwise.
 *   __unordsf2(a, b) -> nonzero if either a or b is NaN.
 *
 * In practice every implementation returns the sign of (a-b) for
 * the magnitude comparisons (clamped to -1, 0, +1), and 0/1 for
 * equality. We follow that convention so the codegen pattern of
 * "call libgcc compare, then branch on sign" works as expected.
 *
 * NaN handling: per IEEE-754, any comparison involving NaN should
 * report "unordered". The convention for the libgcc comparisons:
 * __ltsf2/__lesf2/__gtsf2/__gesf2 all return +1 for NaN comparisons
 * (signalling "not less", "not equal", etc. equivalently to "greater
 * than" so that "if (a < b)" reads false when a or b is NaN, which
 * is what IEEE-754 wants).
 */
int __eqsf2   (float a, float b);
int __nesf2   (float a, float b);
int __ltsf2   (float a, float b);
int __lesf2   (float a, float b);
int __gtsf2   (float a, float b);
int __gesf2   (float a, float b);
int __unordsf2(float a, float b);

/* ---- Conversions ---- */
float    __floatsisf  (int32_t  i);  /* signed   int -> float (round to nearest) */
float    __floatunsisf(uint32_t u);  /* unsigned int -> float (round to nearest) */
int32_t  __fixsfsi    (float a);     /* float -> signed int (truncate toward zero) */
uint32_t __fixunssfsi (float a);     /* float -> unsigned int (truncate toward zero) */

#endif /* BARRACUDA_SOFT_FP_H */
