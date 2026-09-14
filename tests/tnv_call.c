#include "booth/nv_rt.h"
#include <stdio.h>

#define NTHR  8
#define NIF   130

static int host[NTHR];

static int brf(int n, float f)
{
    int a = n, j;
    for (j = 0; j < NIF; j++) {
        if (a > j) a += j + 2;
        else       a -= 1;
    }
    return a + (int)f;
}

static int want(int t, int m)
{
    int r = 0, i;
    for (i = 0; i < m; i++) {
        if (i & 1) r += brf(t + i, 1.5f);
        else       r -= i;
    }
    return r;
}

static int check(nv_dev_t *dev, nv_kern_t *kern, int m)
{
    CUdevptr d_out;
    void *args[2];
    int errs = 0, i;

    for (i = 0; i < NTHR; i++) host[i] = -1;

    d_out = nv_rt_alloc(dev, sizeof host);
    if (!d_out) { fprintf(stderr, "  alloc failed\n"); return 1; }
    if (nv_rt_h2d(dev, d_out, host, sizeof host) != NV_RT_OK) {
        fprintf(stderr, "  H2D failed\n");
        nv_rt_free(dev, d_out);
        return 1;
    }

    args[0] = &d_out;
    args[1] = &m;
    if (nv_rt_launch(dev, kern, 1, 1, 1, NTHR, 1, 1, 0, args) != NV_RT_OK) {
        fprintf(stderr, "  launch failed (m=%d)\n", m);
        nv_rt_free(dev, d_out);
        return 1;
    }
    if (nv_rt_sync(dev) != NV_RT_OK) {
        fprintf(stderr, "  sync failed (m=%d)\n", m);
        nv_rt_free(dev, d_out);
        return 1;
    }
    if (nv_rt_d2h(dev, host, d_out, sizeof host) != NV_RT_OK) {
        fprintf(stderr, "  D2H failed\n");
        nv_rt_free(dev, d_out);
        return 1;
    }
    nv_rt_free(dev, d_out);

    for (i = 0; i < NTHR; i++) {
        int w = want(i, m);
        if (host[i] != w) {
            printf("  m=%d thread %d: got %d, want %d\n", m, i, host[i], w);
            errs++;
        }
    }
    if (errs == 0) printf("  m=%d: %d threads agree\n", m, NTHR);
    return errs;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "callbr.ptx";
    nv_dev_t dev;
    nv_kern_t kern;
    int errs = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "tnv_call: no device\n");
        return 2;
    }
    printf("tnv_call: %s, sm_%d%d\n", dev.dev_name, dev.sm_major,
           dev.sm_minor);

    if (nv_rt_load(&dev, ptx, "ker", &kern) != NV_RT_OK) {
        fprintf(stderr, "tnv_call: JIT refused %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }
    printf("tnv_call: %s loaded\n", ptx);

    errs += check(&dev, &kern, 1);
    errs += check(&dev, &kern, 5);
    errs += check(&dev, &kern, 12);

    printf("tnv_call: %s (%d mismatches)\n", errs ? "FAIL" : "PASS", errs);

    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
