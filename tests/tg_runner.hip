/* tg_runner.hip — load Booth .hsaco and run tg_smoke on MI300X
 *
 * Compile on target: hipcc tg_runner.hip -o tg_runner
 * Run: ./tg_runner tg_compat.hsaco
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 64
#define BLOCK 64

#define HIP_CHECK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.hsaco>\n", argv[0]);
        return 1;
    }

    /* Load the .hsaco module */
    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, argv[1]));
    printf("[ok] loaded %s\n", argv[1]);

    /* Get kernel function */
    hipFunction_t kern;
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "tg_smoke"));
    printf("[ok] found tg_smoke\n");

    /* Allocate + init */
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_in[i] = 0.5f + (float)i * 0.1f;
        h_out[i] = -1.0f;
    }

    float *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_in,  N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));

    /* Launch: tg_smoke(float *out, float *in, int n) */
    int n = N;
    struct { float *out; float *in; int n; } args = { d_out, d_in, n };
    size_t arg_size = sizeof(args);
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };

    HIP_CHECK(hipModuleLaunchKernel(kern,
        1, 1, 1,       /* grid */
        BLOCK, 1, 1,   /* block */
        0, 0,          /* shared, stream */
        NULL, config));
    HIP_CHECK(hipDeviceSynchronize());
    printf("[ok] kernel launched and completed\n");

    /* Read back */
    HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

    /* Verify against CPU reference */
    int pass = 0, fail = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        /* reproduce tg_smoke logic */
        float a = exp2f(x);
        float b = log2f(a);
        float c = sqrtf(fabsf(b));
        float d = sinf(c) + cosf(c);
        float e = floorf(d);
        float f = ceilf(d);
        float expected = fmaxf(e, fminf(f, x));

        float diff = fabsf(h_out[i] - expected);
        if (diff < 0.01f) {
            pass++;
        } else {
            if (fail < 5)
                printf("[FAIL] i=%d: got %.6f, expected %.6f (diff=%.6f)\n",
                       i, h_out[i], expected, diff);
            fail++;
        }
    }

    printf("\n=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");

    /* Print first few values for inspection */
    printf("\nFirst 8 results:\n");
    for (int i = 0; i < 8 && i < N; i++)
        printf("  [%d] in=%.4f out=%.6f\n", i, h_in[i], h_out[i]);

    hipFree(d_out);
    hipFree(d_in);
    hipModuleUnload(mod);

    return fail ? 1 : 0;
}
