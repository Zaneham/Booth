/* gpu_wmma.c -- run the WMMA kernels on a real NVIDIA GPU.
 * PTX is JITed by the driver, so no CUDA SDK is needed.
 *
 *   kath --nvidia-ptx tests/wmma16.cu -o wmma16.ptx
 *   gcc tests/gpu_wmma.c runtime/host/cuda/nv_rt.c -Iruntime/include -o gpu_wmma
 *   ./gpu_wmma wmma16.ptx
 */
#include "booth/nv_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARP 32
#define MAXD 32
#define LD   32

/* Asymmetric in both indices: a layout that transposed a fragment, or
 * swapped two halves of a fragment, would not survive this. */
#define AV(i, k) ((float)((((i) * 3 + (k) * 5) % 7) - 3))
#define BV(k, j) ((float)((((k) * 2 + (j) * 3) % 5) - 2))

static unsigned short f2b(float f)
{
    union { float f; unsigned int u; } p;
    p.f = f;
    return (unsigned short)(p.u >> 16);
}

/* Only small integers go through here, so the exponent path is the whole
 * story and there is nothing to round. */
static unsigned short f2h(float f)
{
    union { float f; unsigned int u; } p;
    p.f = f;
    unsigned int s = (p.u >> 16) & 0x8000u;
    int e = (int)((p.u >> 23) & 0xFFu) - 127 + 15;
    unsigned int m = p.u & 0x7FFFFFu;
    if (p.f == 0.0f) return (unsigned short)s;
    if (e <= 0 || e >= 31) return (unsigned short)(s | 0x7C00u);
    return (unsigned short)(s | ((unsigned int)e << 10) | (m >> 13));
}

struct kcase {
    const char *kern;
    int m, n, k;
    int arow, brow;   /* 1 = row-major fragment, 0 = column-major */
    int bf;           /* multiplicands are bf16 */
    int zero;         /* the kernel fills C with zero instead of loading it */
};

static unsigned short ha[MAXD * LD], hb[MAXD * LD];
static float hd[MAXD * LD], ref[MAXD * LD];

static void mkmat(const struct kcase *c)
{
    memset(ha, 0, sizeof ha);
    memset(hb, 0, sizeof hb);
    for (int i = 0; i < c->m; i++)
        for (int q = 0; q < c->k; q++) {
            float v = AV(i, q);
            int o = c->arow ? i * LD + q : q * LD + i;
            ha[o] = c->bf ? f2b(v) : f2h(v);
        }
    for (int q = 0; q < c->k; q++)
        for (int j = 0; j < c->n; j++) {
            float v = BV(q, j);
            int o = c->brow ? q * LD + j : j * LD + q;
            hb[o] = c->bf ? f2b(v) : f2h(v);
        }
    for (int i = 0; i < MAXD * LD; i++) hd[i] = (float)(i * 2 + 1);
    memcpy(ref, hd, sizeof ref);
    for (int i = 0; i < c->m; i++)
        for (int j = 0; j < c->n; j++) {
            float acc = c->zero ? 0.0f : hd[i * LD + j];
            for (int q = 0; q < c->k; q++)
                acc += AV(i, q) * BV(q, j);
            ref[i * LD + j] = acc;
        }
}

static int onecase(nv_dev_t *dev, const char *ptx, const struct kcase *c)
{
    nv_kern_t kn;
    if (nv_rt_load(dev, ptx, c->kern, &kn) != NV_RT_OK) {
        fprintf(stderr, "gpu_wmma: load %s failed\n", c->kern);
        return 1;
    }
    mkmat(c);

    CUdevptr da = nv_rt_alloc(dev, sizeof ha);
    CUdevptr db = nv_rt_alloc(dev, sizeof hb);
    CUdevptr dd = nv_rt_alloc(dev, sizeof hd);
    nv_rt_h2d(dev, da, ha, sizeof ha);
    nv_rt_h2d(dev, db, hb, sizeof hb);
    nv_rt_h2d(dev, dd, hd, sizeof hd);
    int lda = LD, ldb = LD, ldd = LD;
    void *args[6] = { &da, &db, &dd, &lda, &ldb, &ldd };
    int rc = nv_rt_launch(dev, &kn, 1, 1, 1, WARP, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, hd, dd, sizeof hd);
    nv_rt_free(dev, da); nv_rt_free(dev, db); nv_rt_free(dev, dd);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "gpu_wmma: launch %s failed\n", c->kern);
        return 1;
    }

    int bad = 0;
    for (int i = 0; i < MAXD; i++)
        for (int j = 0; j < LD; j++) {
            int o = i * LD + j;
            if (hd[o] == ref[o]) continue;
            if (bad < 4)
                fprintf(stderr, "  %s [%d,%d] got %.1f want %.1f\n",
                        c->kern, i, j, (double)hd[o], (double)ref[o]);
            bad++;
        }
    printf("  %-6s %2dx%2dx%2d %s%s %-4s  %s\n", c->kern, c->m, c->n, c->k,
           c->arow ? "row" : "col", c->brow ? "row" : "col",
           c->bf ? "bf16" : "f16", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}


/* Integer, tf32 and f64 want their own element types, so each gets its own
 * reference rather than bending the half-precision harness around them. */
static signed char ia[MAXD * LD], ib[MAXD * LD];
static int id[MAXD * LD], iref[MAXD * LD];

static int ints32(nv_dev_t *dev, const char *ptx)
{
    nv_kern_t kn;
    if (nv_rt_load(dev, ptx, "ws8", &kn) != NV_RT_OK) return 1;
    memset(ia, 0, sizeof ia);
    memset(ib, 0, sizeof ib);
    for (int i = 0; i < 16; i++)
        for (int q = 0; q < 16; q++)
            ia[i * LD + q] = (signed char)((i * 3 + q * 5) % 7 - 3);
    for (int q = 0; q < 16; q++)
        for (int j = 0; j < 16; j++)
            ib[j * LD + q] = (signed char)((q * 2 + j * 3) % 5 - 2);
    for (int i = 0; i < MAXD * LD; i++) id[i] = i * 2 + 1;
    memcpy(iref, id, sizeof iref);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            int acc = id[i * LD + j];
            for (int q = 0; q < 16; q++)
                acc += (int)ia[i * LD + q] * (int)ib[j * LD + q];
            iref[i * LD + j] = acc;
        }

    CUdevptr da = nv_rt_alloc(dev, sizeof ia);
    CUdevptr db = nv_rt_alloc(dev, sizeof ib);
    CUdevptr dd = nv_rt_alloc(dev, sizeof id);
    nv_rt_h2d(dev, da, ia, sizeof ia);
    nv_rt_h2d(dev, db, ib, sizeof ib);
    nv_rt_h2d(dev, dd, id, sizeof id);
    int lda = LD, ldb = LD, ldd = LD;
    void *args[6] = { &da, &db, &dd, &lda, &ldb, &ldd };
    int rc = nv_rt_launch(dev, &kn, 1, 1, 1, WARP, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, id, dd, sizeof id);
    nv_rt_free(dev, da); nv_rt_free(dev, db); nv_rt_free(dev, dd);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) return 1;

    int bad = 0;
    for (int i = 0; i < MAXD * LD; i++) {
        if (id[i] == iref[i]) continue;
        if (bad < 4)
            fprintf(stderr, "  ws8 [%d,%d] got %d want %d\n",
                    i / LD, i % LD, id[i], iref[i]);
        bad++;
    }
    printf("  %-6s 16x16x16 rowcol s8    %s\n", "ws8", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static float fa[MAXD * LD], fb[MAXD * LD], fd[MAXD * LD], fref[MAXD * LD];

static int tf32c(nv_dev_t *dev, const char *ptx)
{
    nv_kern_t kn;
    if (nv_rt_load(dev, ptx, "wtf", &kn) != NV_RT_OK) return 1;
    memset(fa, 0, sizeof fa);
    memset(fb, 0, sizeof fb);
    for (int i = 0; i < 16; i++)
        for (int q = 0; q < 8; q++) fa[i * LD + q] = AV(i, q);
    for (int q = 0; q < 8; q++)
        for (int j = 0; j < 16; j++) fb[j * LD + q] = BV(q, j);
    for (int i = 0; i < MAXD * LD; i++) fd[i] = (float)(i * 2 + 1);
    memcpy(fref, fd, sizeof fref);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float acc = fd[i * LD + j];
            for (int q = 0; q < 8; q++) acc += AV(i, q) * BV(q, j);
            fref[i * LD + j] = acc;
        }

    CUdevptr da = nv_rt_alloc(dev, sizeof fa);
    CUdevptr db = nv_rt_alloc(dev, sizeof fb);
    CUdevptr dd = nv_rt_alloc(dev, sizeof fd);
    nv_rt_h2d(dev, da, fa, sizeof fa);
    nv_rt_h2d(dev, db, fb, sizeof fb);
    nv_rt_h2d(dev, dd, fd, sizeof fd);
    int lda = LD, ldb = LD, ldd = LD;
    void *args[6] = { &da, &db, &dd, &lda, &ldb, &ldd };
    int rc = nv_rt_launch(dev, &kn, 1, 1, 1, WARP, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, fd, dd, sizeof fd);
    nv_rt_free(dev, da); nv_rt_free(dev, db); nv_rt_free(dev, dd);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) return 1;

    int bad = 0;
    for (int i = 0; i < MAXD * LD; i++) {
        if (fd[i] == fref[i]) continue;
        if (bad < 4)
            fprintf(stderr, "  wtf [%d,%d] got %.1f want %.1f\n",
                    i / LD, i % LD, (double)fd[i], (double)fref[i]);
        bad++;
    }
    printf("  %-6s 16x16x 8 rowcol tf32  %s\n", "wtf", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static double da8[MAXD * LD], db8[MAXD * LD], dd8[MAXD * LD], dr8[MAXD * LD];

static int f64c(nv_dev_t *dev, const char *ptx)
{
    nv_kern_t kn;
    if (nv_rt_load(dev, ptx, "wf64", &kn) != NV_RT_OK) return 1;
    memset(da8, 0, sizeof da8);
    memset(db8, 0, sizeof db8);
    for (int i = 0; i < 8; i++)
        for (int q = 0; q < 4; q++) da8[i * LD + q] = (double)AV(i, q);
    for (int q = 0; q < 4; q++)
        for (int j = 0; j < 8; j++) db8[j * LD + q] = (double)BV(q, j);
    for (int i = 0; i < MAXD * LD; i++) dd8[i] = (double)(i * 2 + 1);
    memcpy(dr8, dd8, sizeof dr8);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double acc = dd8[i * LD + j];
            for (int q = 0; q < 4; q++)
                acc += (double)AV(i, q) * (double)BV(q, j);
            dr8[i * LD + j] = acc;
        }

    CUdevptr da = nv_rt_alloc(dev, sizeof da8);
    CUdevptr db = nv_rt_alloc(dev, sizeof db8);
    CUdevptr dd = nv_rt_alloc(dev, sizeof dd8);
    nv_rt_h2d(dev, da, da8, sizeof da8);
    nv_rt_h2d(dev, db, db8, sizeof db8);
    nv_rt_h2d(dev, dd, dd8, sizeof dd8);
    int lda = LD, ldb = LD, ldd = LD;
    void *args[6] = { &da, &db, &dd, &lda, &ldb, &ldd };
    int rc = nv_rt_launch(dev, &kn, 1, 1, 1, WARP, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, dd8, dd, sizeof dd8);
    nv_rt_free(dev, da); nv_rt_free(dev, db); nv_rt_free(dev, dd);
    nv_rt_unload(dev, &kn);
    if (rc != NV_RT_OK) return 1;

    int bad = 0;
    for (int i = 0; i < MAXD * LD; i++) {
        if (dd8[i] == dr8[i]) continue;
        if (bad < 4)
            fprintf(stderr, "  wf64 [%d,%d] got %.1f want %.1f\n",
                    i / LD, i % LD, dd8[i], dr8[i]);
        bad++;
    }
    printf("  %-6s  8x 8x 4 rowcol f64   %s\n", "wf64", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "wmma16.ptx";
    static const struct kcase cs[] = {
        { "wrc",   16, 16, 16, 1, 0, 0, 0 },
        { "wrr",   16, 16, 16, 1, 1, 0, 0 },
        { "wcr",   16, 16, 16, 0, 1, 0, 0 },
        { "wzero", 16, 16, 16, 1, 0, 0, 1 },
        { "wbf",   16, 16, 16, 1, 0, 1, 0 },
        { "w8n32",  8, 32, 16, 1, 0, 0, 0 },
        { "w32n8", 32,  8, 16, 1, 0, 0, 0 },
    };
    nv_dev_t dev;
    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "gpu_wmma: no CUDA driver, skipping\n");
        return 77;
    }
    printf("gpu_wmma: %s sm_%d%d\n", dev.dev_name, dev.sm_major, dev.sm_minor);
    int bad = 0;
    for (unsigned i = 0; i < sizeof cs / sizeof cs[0]; i++)
        bad += onecase(&dev, ptx, &cs[i]);
    bad += ints32(&dev, ptx);
    bad += tf32c(&dev, ptx);
    bad += f64c(&dev, ptx);
    nv_rt_shut(&dev);
    printf("gpu_wmma: %s\n",
           bad ? "FAIL" : "PASS - every case matches the host reference");
    return bad ? 1 : 0;
}
