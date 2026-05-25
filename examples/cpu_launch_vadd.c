/* cpu_launch_vadd.c — Run a BarraCUDA --cpu kernel natively, no GPU.
 *
 * Demonstrates the CPU backend's SIMT-on-CPU calling convention: the
 * kernel's own parameters come first, then a hidden trailing `nthreads`
 * argument. The kernel body runs once per thread_id in [0, nthreads),
 * with block_id=0, so a single call covers a full 1-D launch when
 * nthreads == element count.
 *
 * Build (Triton source):
 *   ./barracuda --triton --cpu -o vadd.o tests/tri_vadd.py
 *   gcc -no-pie examples/cpu_launch_vadd.c vadd.o -o cpu_vadd
 *   ./cpu_vadd
 *
 * Or from CUDA C with a matching add(float*,float*,float*) kernel:
 *   ./barracuda --cpu -o vadd.o vadd.cu      (then link as above)
 *
 * tri_vadd.py signature: vector_add(x_ptr, y_ptr, out_ptr, n_elements,
 *                                   BLOCK_SIZE) + hidden nthreads.
 */

#include <stdio.h>

/* x, y, out, n_elements, BLOCK_SIZE, then the hidden nthreads. */
extern void vector_add(float *x, float *y, float *out,
                       int n_elements, int block_size, int nthreads);

#define N      8
#define BLOCK  256

int main(void)
{
    float x[N], y[N], out[N];
    for (int i = 0; i < N; i++) { x[i] = (float)i; y[i] = (float)(10 * i); out[i] = -1.0f; }

    /* One block, nthreads = N: thread_id sweeps every element. */
    vector_add(x, y, out, N, BLOCK, N);

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float want = (float)i + (float)(10 * i);
        printf("out[%d] = %g (expect %g)\n", i, out[i], want);
        if (out[i] != want) ok = 0;
    }
    printf(ok ? "OK: Triton vector_add ran on the CPU.\n"
              : "MISMATCH\n");
    return ok ? 0 : 1;
}
