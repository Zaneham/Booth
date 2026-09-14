/* gpu_hcvt.c -- packed half2/bfloat162 conversions on a real card
 *
 *   kath --nvidia-ptx tests/hcvt.cu -o hcvt.ptx
 *   gcc tests/gpu_hcvt.c runtime/host/cuda/nv_rt.c -Iruntime/include -o gpu_hcvt
 *   ./gpu_hcvt hcvt.ptx
 *
 * Every input is an exact tie, so anything but round-to-even misses. */
#include "booth/nv_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NT 16

typedef unsigned short u16;
typedef unsigned int   u32;

static u16 rn16(float f)
{
    union { float f; u32 u; } p;
    u32 u, s, m, r, g, st;
    int e, he, sh;

    p.f = f;
    u = p.u;
    s = (u >> 16) & 0x8000u;
    e = (int)((u >> 23) & 0xFFu);
    m = u & 0x7FFFFFu;
    if (e == 0xFF) return (u16)(s | 0x7C00u | (m ? 0x200u : 0u));
    if (e == 0) return (u16)s;
    he = e - 127 + 15;
    if (he >= 31) return (u16)(s | 0x7C00u);
    if (he <= 0) {
        m |= 0x800000u;
        sh = 14 - he;
        if (sh > 24) return (u16)s;
        r  = m >> sh;
        g  = (m >> (sh - 1)) & 1u;
        st = (m & ((1u << (sh - 1)) - 1u)) != 0;
        if (g && (st || (r & 1u))) r++;
        return (u16)(s | r);
    }
    r  = ((u32)he << 10) | (m >> 13);
    g  = (m >> 12) & 1u;
    st = (m & 0xFFFu) != 0;
    if (g && (st || (r & 1u))) r++;
    return (u16)(s | r);
}

static u16 rnbf(float f)
{
    union { float f; u32 u; } p;
    u32 u, r, g, st;

    p.f = f;
    u = p.u;
    r = u >> 16;
    if (((u >> 23) & 0xFFu) == 0xFFu)
        return (u16)(r | ((u & 0x7FFFFFu) ? 0x40u : 0u));
    g  = (u >> 15) & 1u;
    st = (u & 0x7FFFu) != 0;
    if (g && (st || (r & 1u))) r++;
    return (u16)r;
}

static float h2f(u16 h)
{
    union { float f; u32 u; } p;
    u32 s = (u32)(h & 0x8000u) << 16;
    int e = (h >> 10) & 0x1F;
    u32 m = h & 0x3FFu;

    if (e == 0) {
        if (!m) { p.u = s; return p.f; }
        e = 1;
        while (!(m & 0x400u)) { m <<= 1; e--; }
        m &= 0x3FFu;
    } else if (e == 31) {
        p.u = s | 0x7F800000u | (m << 13);
        return p.f;
    }
    p.u = s | ((u32)(e - 15 + 127) << 23) | (m << 13);
    return p.f;
}

static float b2f(u16 b)
{
    union { float f; u32 u; } p;

    p.u = (u32)b << 16;
    return p.f;
}

static float hin[2 * NT], bin[2 * NT];
static u16   hsrc[4 * NT], bsrc[4 * NT];
static u16   hgot[2 * NT * 10], hwant[2 * NT * 10];
static u16   bgot[2 * NT * 9],  bwant[2 * NT * 9];
static float fgot[2 * NT], fwant[2 * NT];
static u16   sgot[2 * NT], swant[2 * NT];
static int   fails;

static void chk16(const char *nm, int i, u16 got, u16 want)
{
    if (got == want) return;
    if (fails < 8)
        fprintf(stderr, "  %-20s [%d] got %04x want %04x\n",
                nm, i, got, want);
    fails++;
}

static void chkf(const char *nm, int i, float got, float want)
{
    if (got == want) return;
    if (fails < 8)
        fprintf(stderr, "  %-20s [%d] got %.9g want %.9g\n",
                nm, i, (double)got, (double)want);
    fails++;
}

static void mkin(void)
{
    for (int i = 0; i < NT; i++) {
        float t = (float)(2 * i + 1) * 0.5f;

        hin[2*i]     =  1.0f + t / 1024.0f;
        hin[2*i + 1] = -1.0f - t / 1024.0f;
        bin[2*i]     =  1.0f + t / 128.0f;
        bin[2*i + 1] = -1.0f - t / 128.0f;
    }
    for (int j = 0; j < 4 * NT; j++) {
        hsrc[j] = rn16((float)(100 + j));
        bsrc[j] = rnbf((float)(100 + j));
    }
}

static void href(void)
{
    for (int i = 0; i < NT; i++) {
        u16 *o = hwant + i * 20;
        u16 a = rn16(hin[2*i]), b = rn16(hin[2*i + 1]);

        o[0]  = a;             o[1]  = b;
        o[2]  = a;             o[3]  = b;
        o[4]  = a;             o[5]  = a;
        o[6]  = a;             o[7]  = b;
        o[8]  = hsrc[4*i];     o[9]  = hsrc[4*i + 2];
        o[10] = hsrc[4*i + 1]; o[11] = hsrc[4*i + 3];
        o[12] = hsrc[4*i];     o[13] = hsrc[4*i];
        o[14] = hsrc[4*i + 1]; o[15] = hsrc[4*i + 1];
        o[16] = hsrc[4*i + 1]; o[17] = hsrc[4*i];
        o[18] = hsrc[4*i];     o[19] = hsrc[4*i + 3];
        fwant[2*i]     = h2f(hsrc[4*i]);
        fwant[2*i + 1] = h2f(hsrc[4*i + 1]);
    }
}

static void bref(void)
{
    for (int i = 0; i < NT; i++) {
        u16 *o = bwant + i * 18;
        u16 a = rnbf(bin[2*i]), b = rnbf(bin[2*i + 1]);

        o[0]  = a;             o[1]  = b;
        o[2]  = a;             o[3]  = b;
        o[4]  = a;             o[5]  = a;
        o[6]  = a;             o[7]  = b;
        o[8]  = a;             o[9]  = a;
        o[10] = bsrc[4*i];     o[11] = bsrc[4*i + 2];
        o[12] = bsrc[4*i + 1]; o[13] = bsrc[4*i + 3];
        o[14] = bsrc[4*i];     o[15] = bsrc[4*i];
        o[16] = bsrc[4*i + 1]; o[17] = bsrc[4*i];
        fwant[2*i]     = b2f(bsrc[4*i]);
        fwant[2*i + 1] = b2f(bsrc[4*i + 1]);
        swant[2*i]     = bsrc[4*i];
        swant[2*i + 1] = bsrc[4*i + 1];
    }
}

static int runh(nv_dev_t *dev, const char *ptx)
{
    nv_kern_t kn;
    CUdevptr df, dh, doo, dg;
    void *args[4];
    int rc, base = fails;

    if (nv_rt_load(dev, ptx, "hpack", &kn) != NV_RT_OK) {
        fprintf(stderr, "gpu_hcvt: load hpack failed\n");
        return 1;
    }
    df  = nv_rt_alloc(dev, sizeof hin);
    dh  = nv_rt_alloc(dev, sizeof hsrc);
    doo = nv_rt_alloc(dev, sizeof hgot);
    dg  = nv_rt_alloc(dev, sizeof fgot);
    nv_rt_h2d(dev, df, hin, sizeof hin);
    nv_rt_h2d(dev, dh, hsrc, sizeof hsrc);
    args[0] = &df; args[1] = &dh; args[2] = &doo; args[3] = &dg;
    rc = nv_rt_launch(dev, &kn, 1, 1, 1, NT, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, hgot, doo, sizeof hgot);
    nv_rt_d2h(dev, fgot, dg, sizeof fgot);
    nv_rt_free(dev, df); nv_rt_free(dev, dh);
    nv_rt_free(dev, doo); nv_rt_free(dev, dg);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gpu_hcvt: launch hpack failed\n");
        return 1;
    }
    href();
    for (int j = 0; j < 2 * NT * 10; j++)
        chk16("half2", j, hgot[j], hwant[j]);
    for (int j = 0; j < 2 * NT; j++)
        chkf("__half22float2", j, fgot[j], fwant[j]);
    printf("  hpack  %d lanes  %s\n", 2 * NT * 10,
           fails == base ? "PASS" : "FAIL");
    return fails != base;
}

static int runb(nv_dev_t *dev, const char *ptx)
{
    nv_kern_t kn;
    CUdevptr df, db, doo, dg, ds;
    void *args[5];
    int rc, base = fails;

    if (nv_rt_load(dev, ptx, "bpack", &kn) != NV_RT_OK) {
        fprintf(stderr, "gpu_hcvt: load bpack failed\n");
        return 1;
    }
    df  = nv_rt_alloc(dev, sizeof bin);
    db  = nv_rt_alloc(dev, sizeof bsrc);
    doo = nv_rt_alloc(dev, sizeof bgot);
    dg  = nv_rt_alloc(dev, sizeof fgot);
    ds  = nv_rt_alloc(dev, sizeof sgot);
    nv_rt_h2d(dev, df, bin, sizeof bin);
    nv_rt_h2d(dev, db, bsrc, sizeof bsrc);
    args[0] = &df; args[1] = &db; args[2] = &doo;
    args[3] = &dg; args[4] = &ds;
    rc = nv_rt_launch(dev, &kn, 1, 1, 1, NT, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, bgot, doo, sizeof bgot);
    nv_rt_d2h(dev, fgot, dg, sizeof fgot);
    nv_rt_d2h(dev, sgot, ds, sizeof sgot);
    nv_rt_free(dev, df); nv_rt_free(dev, db); nv_rt_free(dev, doo);
    nv_rt_free(dev, dg); nv_rt_free(dev, ds);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gpu_hcvt: launch bpack failed\n");
        return 1;
    }
    bref();
    for (int j = 0; j < 2 * NT * 9; j++)
        chk16("bfloat162", j, bgot[j], bwant[j]);
    for (int j = 0; j < 2 * NT; j++)
        chkf("__bfloat1622float2", j, fgot[j], fwant[j]);
    for (int j = 0; j < 2 * NT; j++)
        chk16("low/high2bfloat16", j, sgot[j], swant[j]);
    printf("  bpack  %d lanes  %s\n", 2 * NT * 9,
           fails == base ? "PASS" : "FAIL");
    return fails != base;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "hcvt.ptx";
    nv_dev_t dev;
    int bad = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "gpu_hcvt: no CUDA driver, skipping\n");
        return 77;
    }
    printf("gpu_hcvt: %s sm_%d%d\n", dev.dev_name, dev.sm_major, dev.sm_minor);
    mkin();
    bad += runh(&dev, ptx);
    bad += runb(&dev, ptx);
    nv_rt_shut(&dev);
    printf("gpu_hcvt: %s (%d mismatches)\n", bad ? "FAIL" : "PASS", fails);
    return bad ? 1 : 0;
}
