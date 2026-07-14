/* launch_saxpy.c — Run a Booth-compiled SAXPY kernel on real hardware.
 *
 * Build:  gcc -std=c99 -O2 -I src/runtime
 *             examples/launch_saxpy.c src/runtime/bc_runtime.c
 *             -ldl -lm -o launch_saxpy
 * Run:    ./kath --amdgpu-bin -o test.hsaco tests/canonical.cu
 *         ./launch_saxpy test.hsaco
 */

#include "bc_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N       1024
#define BLOCK   256
#define A_VAL   2.5f
#define EPSILON 0.001f

int main(int argc, char *argv[])
{
    const char *hsaco = (argc > 1) ? argv[1] : "test.hsaco";

    printf("Booth SAXPY launcher\n");
    printf("  .hsaco:  %s\n", hsaco);
    printf("  N:       %d\n", N);
    printf("  a:       %.1f\n", A_VAL);

    bc_device_t dev;
    int rc = bc_device_init(&dev);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "Device init failed (%d)\n", rc);
        return 1;
    }

    /* kernarg layout: {float *y, float *x, float a, int n} = 24 bytes */
    bc_kernel_t kern;
    rc = bc_load_kernel(&dev, hsaco, "saxpy", &kern);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "Kernel load failed (%d)\n", rc);
        bc_device_shutdown(&dev);
        return 1;
    }

    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    float *h_expected = (float *)malloc(N * sizeof(float));
    if (!h_x || !h_y || !h_expected) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)(N - i);
        h_expected[i] = A_VAL * h_x[i] + h_y[i];
    }

    void *d_x = bc_alloc(&dev, N * sizeof(float));
    void *d_y = bc_alloc(&dev, N * sizeof(float));
    if (!d_x || !d_y) {
        fprintf(stderr, "Device alloc failed\n");
        bc_device_shutdown(&dev);
        return 1;
    }

    bc_copy_h2d(&dev, d_x, h_x, N * sizeof(float));
    bc_copy_h2d(&dev, d_y, h_y, N * sizeof(float));

    struct {
        void    *y;
        void    *x;
        float    a;
        uint32_t n;
    } args;

    args.y = d_y;
    args.x = d_x;
    args.a = A_VAL;
    args.n = N;

    uint32_t num_blocks = (N + BLOCK - 1) / BLOCK;
    printf("  dispatch: %u blocks x %d threads\n", num_blocks, BLOCK);

    rc = bc_dispatch(&dev, &kern, num_blocks, 1, 1, BLOCK, 1, 1,
                     &args, sizeof(args));
    if (rc != BC_RT_OK) {
        fprintf(stderr, "Dispatch failed (%d)\n", rc);
        bc_free(&dev, d_x);
        bc_free(&dev, d_y);
        bc_device_shutdown(&dev);
        return 1;
    }

    bc_copy_d2h(&dev, h_y, d_y, N * sizeof(float));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_y[i] - h_expected[i]);
        if (diff > EPSILON) {
            if (errors < 5) {
                printf("  MISMATCH [%d]: got %.4f, expected %.4f\n",
                       i, h_y[i], h_expected[i]);
            }
            errors++;
        }
    }

    if (errors == 0)
        printf("  PASS: all %d elements correct\n", N);
    else
        printf("  FAIL: %d/%d mismatches\n", errors, N);

    bc_free(&dev, d_x);
    bc_free(&dev, d_y);
    bc_unload_kernel(&dev, &kern);
    bc_device_shutdown(&dev);
    free(h_x);
    free(h_y);
    free(h_expected);

    return errors > 0 ? 1 : 0;
}
