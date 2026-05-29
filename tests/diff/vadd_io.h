/* vadd_io.h -- shared problem setup for the vector_add differential test.
 *
 * Every backend has to run the exact same numbers or the diff is
 * meaningless, so the input generation lives here, once. It is pure float
 * arithmetic with no libc, which means the freestanding rv64 runner can
 * include it just as happily as the x86 host comparator. The values pick
 * up signs and fractions on purpose, clean integers would let a broken
 * float path slip through looking fine. */
#ifndef VADD_IO_H
#define VADD_IO_H

#define VADD_N      64
#define VADD_BLOCK  256

static void vadd_inputs(float *x, float *y)
{
    const int mid = VADD_N / 2;
    for (int i = 0; i < VADD_N; i++) {
        x[i] = (float)(i - mid) * 0.5f;
        y[i] = (float)(VADD_N - i) * -0.25f;
    }
}

#endif /* VADD_IO_H */
