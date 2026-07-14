/* tiny_runner.hip — load Booth .hsaco and run 'tiny' kernel on MI300X
 *
 * Compile on target: hipcc tiny_runner.hip -o tiny_runner
 * Run: ./tiny_runner tiny.hsaco
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "tiny"));
    printf("[ok] found tiny\n");

    /* Allocate + init */
    float h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1.0f;

    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));

    /* Launch: tiny(float *out) */
    struct { float *out; } args = { d_out };
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

    /* Verify: every element should be 1.0f */
    int pass = 0, fail = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] == 1.0f) {
            pass++;
        } else {
            if (fail < 5)
                printf("[FAIL] i=%d: got %.6f, expected 1.000000\n", i, h_out[i]);
            fail++;
        }
    }

    printf("\n=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");

    printf("\nFirst 8 results:\n");
    for (int i = 0; i < 8 && i < N; i++)
        printf("  [%d] out=%.6f\n", i, h_out[i]);

    hipFree(d_out);
    hipModuleUnload(mod);

    return fail ? 1 : 0;
}
