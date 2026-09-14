#include "booth/nv_rt.h"

#include <stdio.h>

#define BASE 1000

static int trial(nv_dev_t *dev, const char *ptx, const char *kern, int want)
{
    nv_kern_t kn;
    CUdevptr d;
    int got = -1, base = BASE;
    void *args[2];

    if (nv_rt_load(dev, ptx, kern, &kn) != NV_RT_OK) {
        fprintf(stderr, "gpu_tid: load %s failed\n", kern);
        return 1;
    }
    d = nv_rt_alloc(dev, sizeof got);
    if (!d) { nv_rt_unload(dev, &kn); return 1; }
    nv_rt_h2d(dev, d, &got, sizeof got);
    args[0] = &d;
    args[1] = &base;
    if (nv_rt_launch(dev, &kn, 1, 1, 1, 1, 1, 1, 0, args) != NV_RT_OK
     || nv_rt_sync(dev) != NV_RT_OK
     || nv_rt_d2h(dev, &got, d, sizeof got) != NV_RT_OK) {
        fprintf(stderr, "gpu_tid: launch %s failed\n", kern);
        nv_rt_free(dev, d);
        nv_rt_unload(dev, &kn);
        return 1;
    }
    nv_rt_free(dev, d);
    nv_rt_unload(dev, &kn);
    printf("  %-7s got %d want %d  %s\n", kern, got, want,
           got == want ? "ok" : "WRONG");
    return got != want;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "build/tid_spec.ptx";
    nv_dev_t dev;
    int bad;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "gpu_tid: no CUDA driver, skipping\n");
        return 77;
    }
    printf("gpu_tid: %s sm_%d%d\n", dev.dev_name, dev.sm_major, dev.sm_minor);
    bad  = trial(&dev, ptx, "tsmall", BASE + 3);
    bad += trial(&dev, ptx, "tbig", BASE + 44850);
    nv_rt_shut(&dev);
    printf("gpu_tid: %s\n", bad ? "FAIL" : "PASS - each specialisation folded its own bound");
    return bad ? 1 : 0;
}
