#include "booth/nv_rt.h"
#include <stdio.h>
#include <stdlib.h>

#define CU_ATTR_SMS   16
#define CU_ATTR_COOP  95

#define BLK   256
#define RND     8
#define MAXN  (1 << 20)

static int host[MAXN];
static int want[MAXN];

static void model(int n, int rnd)
{
    static int cur[MAXN];
    static int nxt[MAXN];

    for (int r = 0; r < n; r++) cur[r] = r + 1;
    for (int i = 0; i < rnd; i++) {
        for (int r = 0; r < n; r++) nxt[r] = cur[(r + n / 2) % n] + 1;
        for (int r = 0; r < n; r++) cur[r] = nxt[r];
    }
    for (int r = 0; r < n; r++) want[r] = cur[r];
}

static int trial(nv_dev_t *dev, nv_kern_t *kern, int blocks)
{
    int n = blocks * BLK;
    int rnd = RND;
    size_t bytes = (size_t)n * sizeof(int);
    CUdevptr d_d, d_s;
    void *args[4];
    int errs = 0;

    if (n > MAXN) return -1;

    d_d = nv_rt_alloc(dev, bytes);
    d_s = nv_rt_alloc(dev, bytes);
    if (!d_d || !d_s) return -1;

    for (int i = 0; i < n; i++) host[i] = -1;
    if (nv_rt_h2d(dev, d_d, host, bytes) != NV_RT_OK
     || nv_rt_h2d(dev, d_s, host, bytes) != NV_RT_OK) {
        nv_rt_free(dev, d_d); nv_rt_free(dev, d_s);
        return -1;
    }

    args[0] = &d_d; args[1] = &d_s; args[2] = &n; args[3] = &rnd;

    if (nv_rt_colaunch(dev, kern, (uint32_t)blocks, 1, 1, BLK, 1, 1, 0, args)
        != NV_RT_OK) {
        nv_rt_free(dev, d_d); nv_rt_free(dev, d_s);
        return -1;
    }
    if (nv_rt_sync(dev) != NV_RT_OK
     || nv_rt_d2h(dev, host, d_s, bytes) != NV_RT_OK) {
        nv_rt_free(dev, d_d); nv_rt_free(dev, d_s);
        return -1;
    }
    nv_rt_free(dev, d_d);
    nv_rt_free(dev, d_s);

    model(n, rnd);
    for (int i = 0; i < n; i++)
        if (host[i] != want[i]) {
            if (errs < 4)
                fprintf(stderr, "    t%d: got %d, wanted %d\n",
                        i, host[i], want[i]);
            errs++;
        }
    printf("  blocks=%d threads=%d rounds=%d mismatches=%d\n",
           blocks, n, rnd, errs);
    return errs;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "grid_xb.ptx";
    nv_dev_t dev;
    nv_kern_t kern;
    int sms = 0, coop = 0, top, errs = 0, ran = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "tnv_grid: no device\n");
        return 2;
    }
    printf("tnv_grid: %s, sm_%d%d\n",
           dev.dev_name, dev.sm_major, dev.sm_minor);

    dev.cuDevAttr(&sms, CU_ATTR_SMS, dev.dev);
    dev.cuDevAttr(&coop, CU_ATTR_COOP, dev.dev);
    printf("tnv_grid: %d SMs, cooperative launch %d\n", sms, coop);
    if (!coop) { nv_rt_shut(&dev); return 2; }

    if (nv_rt_load(&dev, ptx, "gridxb", &kern) != NV_RT_OK) {
        fprintf(stderr, "tnv_grid: JIT refused %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }

    top = sms * 8;
    if (top > MAXN / BLK) top = MAXN / BLK;

    for (int b = top; b >= 2; b /= 2) {
        int r = trial(&dev, &kern, b);
        if (r < 0) continue;
        ran++;
        errs += r;
    }

    if (!ran) {
        fprintf(stderr, "tnv_grid: no grid size launched\n");
        nv_rt_unload(&dev, &kern);
        nv_rt_shut(&dev);
        return 1;
    }
    printf("tnv_grid: %s (%d grid sizes, %d mismatches)\n",
           errs ? "FAIL" : "PASS", ran, errs);

    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
