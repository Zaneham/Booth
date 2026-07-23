/* rv64_host.c -- host side of the RV64 QEMU job. Not part of trunner; it is
 * cross-compiled and linked against the object --rv64 emits for rv64_vadd.cu.
 *
 * The kernel takes its declared parameters in a0.. and then one hidden
 * trailing argument giving the thread count, which it loops over internally.
 */

#include <stdio.h>

extern void vadd(int *out, int *a, int *b, long nthreads);

int main(void)
{
    enum { N = 64 };
    int a[N], b[N], out[N];

    for (int i = 0; i < N; i++) {
        a[i]   = i;
        b[i]   = i * 3;
        out[i] = -1;
    }

    vadd(out, a, b, N);

    for (int i = 0; i < N; i++) {
        int want = a[i] + b[i];
        if (out[i] != want) {
            printf("vadd[%d] = %d, want %d\n", i, out[i], want);
            return 1;
        }
    }
    printf("vadd correct across %d elements under QEMU\n", N);
    return 0;
}
