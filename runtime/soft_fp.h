#ifndef BARRACUDA_SOFT_FP_H
#define BARRACUDA_SOFT_FP_H

#include <stdint.h>

/* IEEE-754 fp32 soft-float for the Tenstorrent baby RV32IM cores; libgcc-named
 * so other RV32 toolchains (or a future vendor/SFPU path) interoperate. Round-
 * nearest-even, FTZ/DAZ, canonical quiet NaN (0x7FC00000), no exception flags or
 * rounding-mode register. SFP_STRICT_IEEE (default 0) adds subnormals. */

/* ---- Arithmetic (binary, sf3) ---- */
float __addsf3(float a, float b);   /* a + b */
float __subsf3(float a, float b);   /* a - b */
float __mulsf3(float a, float b);   /* a * b */
float __divsf3(float a, float b);   /* a / b */

/* ---- Negation (unary, sf2) ---- */
float __negsf2(float a);            /* -a */

/* ---- Comparisons (sf2, return int) ----
 * libgcc convention: eq/ne return 0/nonzero; lt/le/gt/ge return sign of (a-b)
 * clamped to -1/0/+1; unord is nonzero on NaN. On NaN the ordered compares return
 * the value that makes their predicate read false. */
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
