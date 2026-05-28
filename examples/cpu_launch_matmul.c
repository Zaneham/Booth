/* cpu_launch_matmul.c -- run a Triton matmul kernel on the CPU.
 *
 * The headline 0.5 demo: take the matmul kernel from tests/tri_matmul.py,
 * compile it through BarraCUDA's --cpu backend, link it here, and run it
 * natively against a host reference. No GPU, no LLVM anywhere.
 *
 * The kernel is one 4x4 output tile (BLOCK_M = BLOCK_N = BLOCK_K = 4).
 * The constexpr values fold into the body during lowering (the tile is
 * materialised at compile time), but the params themselves stay in the
 * runtime signature, so they have to be passed even though the kernel
 * does not read them. The rank-2 tile path runs as a single logical
 * thread, so nthreads = 1.
 *
 * Build (Linux / WSL):
 *   ./barracuda --triton --cpu -o matmul.o tests/tri_matmul.py
 *   gcc -no-pie examples/cpu_launch_matmul.c matmul.o -o cpu_matmul
 *   ./cpu_matmul
 */
#include <stdio.h>

/* a, b, c, six strides, the three (unused at runtime) constexpr block
 * sizes, then the hidden nthreads. */
extern void matmul(float *a, float *b, float *c,
                   int stride_am, int stride_ak,
                   int stride_bk, int stride_bn,
                   int stride_cm, int stride_cn,
                   int block_m, int block_n, int block_k,
                   int nthreads);

#define M 4
#define N 4
#define K 4

int main(void)
{
    float a[M * K], b[K * N], c[M * N], ref[M * N];

    /* Some signs and fractions, so the float path is exercised properly
     * and a wrong shift or stride would jump out. */
    for (int i = 0; i < M * K; i++) a[i] = (float)((i * 7) % 11) * 0.25f - 1.0f;
    for (int i = 0; i < K * N; i++) b[i] = (float)((i * 5) % 13) * 0.5f  - 2.0f;
    for (int i = 0; i < M * N; i++) c[i] = -123.0f;

    /* Row-major strides for all three matrices. Block sizes echo the
     * constexpr defaults; nthreads = 1 because the tile is materialised. */
    matmul(a, b, c, K, 1, N, 1, N, 1, M, N, K, /* nthreads */ 1);

    /* Host reference: plain triple loop. */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += a[m * K + k] * b[k * N + n];
            ref[m * N + n] = acc;
        }
    }

    int ok = 1;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != ref[i]) { ok = 0; break; }
    }

    printf("C =\n");
    for (int m = 0; m < M; m++) {
        printf(" ");
        for (int n = 0; n < N; n++) printf(" %8.3f", c[m * N + n]);
        printf("\n");
    }
    printf(ok ? "OK: Triton matmul ran on the CPU, matches host reference.\n"
              : "MISMATCH: kernel output differs from host reference.\n");
    return ok ? 0 : 1;
}
