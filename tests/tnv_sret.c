/* tnv_sret.c -- a struct returned by value, checked on a real card
 *
 *   ./kath --nvidia-ptx tests/sret.cu -o sret.ptx
 *   ./tnv_sret sret.ptx */

#include "booth/nv_rt.h"
#include <stdio.h>

#define NSLOT 14

static int host[NSLOT];

static void want(int n, int *w)
{
    w[0]  = n + 1;
    w[1]  = n + n + 1;
    w[2]  = n + 1;
    w[3]  = n + 2;
    w[4]  = n * 3;
    w[5]  = n + 23;
    w[6]  = n;
    w[7]  = n + 1;
    w[8]  = n + 2;
    w[9]  = n + 2;
    w[10] = n;
    w[11] = n + 1;
    w[12] = n + n + 1;
    w[13] = n + 100;
}

static int check(nv_dev_t *dev, nv_kern_t *kern, int n)
{
    int w[NSLOT], errs = 0, i;
    CUdevptr d_out, d_in;
    void *args[2];

    want(n, w);
    for (i = 0; i < NSLOT; i++) host[i] = -1;

    d_out = nv_rt_alloc(dev, sizeof host);
    d_in  = nv_rt_alloc(dev, sizeof(int));
    if (!d_out || !d_in) { fprintf(stderr, "  alloc failed\n"); return 1; }

    if (nv_rt_h2d(dev, d_out, host, sizeof host) != NV_RT_OK
     || nv_rt_h2d(dev, d_in, &n, sizeof n) != NV_RT_OK) {
        fprintf(stderr, "  H2D failed\n");
        nv_rt_free(dev, d_out);
        nv_rt_free(dev, d_in);
        return 1;
    }

    args[0] = &d_out;
    args[1] = &d_in;

    if (nv_rt_launch(dev, kern, 1, 1, 1, 1, 1, 1, 0, args) != NV_RT_OK
     || nv_rt_sync(dev) != NV_RT_OK
     || nv_rt_d2h(dev, host, d_out, sizeof host) != NV_RT_OK) {
        fprintf(stderr, "  launch or readback failed\n");
        nv_rt_free(dev, d_out);
        nv_rt_free(dev, d_in);
        return 1;
    }
    nv_rt_free(dev, d_out);
    nv_rt_free(dev, d_in);

    printf("  n=%d ->", n);
    for (i = 0; i < NSLOT; i++) printf(" %d", host[i]);
    printf("\n");

    for (i = 0; i < NSLOT; i++)
        if (host[i] != w[i]) {
            fprintf(stderr, "    slot %d: got %d, wanted %d\n",
                    i, host[i], w[i]);
            errs++;
        }
    return errs;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "sret.ptx";
    nv_dev_t dev;
    nv_kern_t kern;
    int errs = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "tnv_sret: no device\n");
        return 2;
    }
    printf("tnv_sret: %s, sm_%d%d\n", dev.dev_name, dev.sm_major,
           dev.sm_minor);

    if (nv_rt_load(&dev, ptx, "sret_probe", &kern) != NV_RT_OK) {
        fprintf(stderr, "tnv_sret: JIT refused %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }
    printf("tnv_sret: %s loaded\n", ptx);

    errs += check(&dev, &kern, 7);
    errs += check(&dev, &kern, 0);
    errs += check(&dev, &kern, 41);

    printf("tnv_sret: %s (%d mismatches)\n", errs ? "FAIL" : "PASS", errs);

    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
