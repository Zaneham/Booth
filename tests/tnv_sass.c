/* tnv_sass.c -- run cubins Booth built with no NVIDIA tool in the loop.
 *
 * Three kernels, because between them they cover the parts of the SASS back
 * end that can go quietly wrong: a bounds check that leaves a warp divergent,
 * a loop whose back edge has to reconverge, shared memory across a barrier,
 * and a spread of integer work where a 64-bit value must not be shifted or
 * added as a 32-bit one.
 *
 * Usage:
 *   ./kath --nvidia-cubin examples/cmake/vadd.cu -o build/vadd.cubin
 *   ./kath --nvidia-cubin tests/sass_loop.cu    -o build/sass_loop.cubin
 *   ./kath --nvidia-cubin tests/sass_shr.cu     -o build/sass_shr.cubin
 *   ./kath --nvidia-cubin tests/sass_bits.cu    -o build/sass_bits.cubin
 *   gcc -O2 -Iruntime/include tests/tnv_sass.c runtime/host/cuda/nv_rt.c \
 *       -o tnv_sass && ./tnv_sass
 *
 * Exits 77 when there is no CUDA driver, the way gpu_mma does. */

#include "booth/nv_rt.h"
#include <stdio.h>
#include <math.h>

#define NF  (1 << 16)
#define NI  4096
#define BLK 256

static float fa[NF], fb[NF], fc[NF];
static int   ia[NI], io[NI];

static int fail(const char *k, int i, double got, double want)
{
    fprintf(stderr, "tnv_sass: %s o[%d] = %g, want %g\n", k, i, got, want);
    return 1;
}

static int runf(nv_dev_t *d, const char *cub)
{
    nv_kern_t k;
    CUdevptr da, db, dc;
    size_t by = sizeof fa;
    int n = NF - 37, errs = 0, rc;
    void *args[4];

    if (nv_rt_load(d, cub, "vadd", &k) != NV_RT_OK) return 1;
    for (int i = 0; i < NF; i++) {
        fa[i] = (float)i;
        fb[i] = (float)(i * 2);
        fc[i] = -1.0f;
    }
    da = nv_rt_alloc(d, by);
    db = nv_rt_alloc(d, by);
    dc = nv_rt_alloc(d, by);
    rc  = nv_rt_h2d(d, da, fa, by);
    rc |= nv_rt_h2d(d, db, fb, by);
    rc |= nv_rt_h2d(d, dc, fc, by);
    args[0] = &da; args[1] = &db; args[2] = &dc; args[3] = &n;
    if (rc == 0) rc = nv_rt_launch(d, &k, NF / BLK, 1, 1, BLK, 1, 1, 0, args);
    if (rc == 0) rc = nv_rt_sync(d);
    if (rc == 0) rc = nv_rt_d2h(d, fc, dc, by);
    if (rc != 0) { fprintf(stderr, "tnv_sass: vadd run failed\n"); errs = 1; }
    for (int i = 0; i < NF && errs < 5; i++) {
        float want = (i < n) ? fa[i] + fb[i] : -1.0f;

        if (fabsf(fc[i] - want) > 1e-5f)
            errs += fail("vadd", i, (double)fc[i], (double)want);
    }
    if (errs == 0)
        printf("tnv_sass: vadd  %d in bounds, %d past it, all correct "
               "(c[1]=%g c[%d]=%g c[%d]=%g)\n", n, NF - n, (double)fc[1],
               n - 1, (double)fc[n - 1], n, (double)fc[n]);
    nv_rt_free(d, da);
    nv_rt_free(d, db);
    nv_rt_free(d, dc);
    nv_rt_unload(d, &k);
    return errs != 0;
}

static int want_i(int i, int n, int mode)
{
    int s = 0;

    if (i >= n) return -12345;
    if (mode == 0) {
        for (int q = 0; q <= (i & 15); q++) s += ia[q] * (q + 1);
        return s;
    }
    if (mode == 1) {
        int t = i % BLK, j = (i - t) + ((t + 1) & 255);

        return ia[i] + ((j < n) ? ia[j] : 0);
    }
    {
        unsigned x = (unsigned)ia[i];
        unsigned y = (x << 3) ^ (x >> 2);
        long long q = (long long)ia[i] * 1103515245LL + 12345LL;
        int sg = (ia[i] < 0) ? -1 : 1;

        y = (y & 0xF0F0F0F0u) | (~y & 0x0F0F0F0Fu);
        return (int)(y + (unsigned)(q >> 7)) * sg - (ia[i] >> 3);
    }
}

static int runi(nv_dev_t *d, const char *cub, const char *kn, int mode)
{
    nv_kern_t k;
    CUdevptr da, dob;
    size_t by = sizeof ia;
    int n = NI - 13, errs = 0, rc;
    void *args[3];

    if (nv_rt_load(d, cub, kn, &k) != NV_RT_OK) return 1;
    for (int i = 0; i < NI; i++) {
        ia[i] = (mode == 2) ? (int)((unsigned)i * 2654435761u) ^ (i << 5)
                            : i * 3 - 7;
        io[i] = -12345;
    }
    da  = nv_rt_alloc(d, by);
    dob = nv_rt_alloc(d, by);
    rc  = nv_rt_h2d(d, da, ia, by);
    rc |= nv_rt_h2d(d, dob, io, by);
    args[0] = &da; args[1] = &dob; args[2] = &n;
    if (rc == 0) rc = nv_rt_launch(d, &k, NI / BLK, 1, 1, BLK, 1, 1, 0, args);
    if (rc == 0) rc = nv_rt_sync(d);
    if (rc == 0) rc = nv_rt_d2h(d, io, dob, by);
    if (rc != 0) { fprintf(stderr, "tnv_sass: %s run failed\n", kn); errs = 1; }
    for (int i = 0; i < NI && errs < 5; i++) {
        int want = want_i(i, n, mode);

        if (io[i] != want) errs += fail(kn, i, (double)io[i], (double)want);
    }
    if (errs == 0)
        printf("tnv_sass: %-5s %d in bounds, %d past it, all correct "
               "(o[0]=%d o[17]=%d o[%d]=%d o[%d]=%d)\n", kn, n, NI - n,
               io[0], io[17], n - 1, io[n - 1], n, io[n]);
    nv_rt_free(d, da);
    nv_rt_free(d, dob);
    nv_rt_unload(d, &k);
    return errs != 0;
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "build";
    char p[512];
    nv_dev_t dev;
    int bad = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        printf("tnv_sass: no CUDA driver, skipping\n");
        return 77;
    }
    printf("tnv_sass: %s, sm_%d%d\n", dev.dev_name, dev.sm_major,
           dev.sm_minor);
    if (snprintf(p, sizeof p, "%s/vadd.cubin", dir) < 0) return 1;
    bad |= runf(&dev, p);
    if (snprintf(p, sizeof p, "%s/sass_loop.cubin", dir) < 0) return 1;
    bad |= runi(&dev, p, "loopk", 0);
    if (snprintf(p, sizeof p, "%s/sass_shr.cubin", dir) < 0) return 1;
    bad |= runi(&dev, p, "shrk", 1);
    if (snprintf(p, sizeof p, "%s/sass_bits.cubin", dir) < 0) return 1;
    bad |= runi(&dev, p, "bitk", 2);
    nv_rt_shut(&dev);
    return bad;
}
