/* booth_run.c -- stage a kernel's arguments and run it on whatever is plugged in */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "exec.h"
#include "booth_run.h"

#define MAX_ARGS   32
#define MAX_PATH   260
#define MAX_BYTES  (256u * 1024u * 1024u)

enum { A_IN, A_OUT, A_IO, A_VAL };

typedef struct {
    int      kind;
    char     path[MAX_PATH];
    size_t   bytes;
    rt_ptr   dptr;
    void    *host;
    uint8_t  klass;
    union { uint32_t u; int32_t i; float f; } val;
} slot_t;

static slot_t   slots[MAX_ARGS];
static rt_arg_t args[MAX_ARGS];
static uint32_t nslot;

/* "16" or "16,4" or "16,4,2". Missing dimensions are 1, as everywhere else. */
int brun_dims(const char *s, uint32_t out[3])
{
    out[0] = out[1] = out[2] = 1u;
    for (int i = 0; i < 3 && *s != '\0'; i++) {
        char *end = NULL;
        unsigned long v = strtoul(s, &end, 10);
        if (end == s || v == 0ul) return 0;
        out[i] = (uint32_t)v;
        s = (*end == ',') ? end + 1 : end;
    }
    return 1;
}

static void *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    rewind(f);
    if (n < 0 || (unsigned long)n > MAX_BYTES) { fclose(f); return NULL; }

    void *p = malloc((size_t)n ? (size_t)n : 1u);
    if (p == NULL) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f);
    *len = (size_t)n;
    return p;
}

static int spill(const char *path, const void *p, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    size_t w = fwrite(p, 1, n, f);
    int closed = (fclose(f) == 0);
    return w == n && closed;
}

void brun_reset(void)
{
    for (uint32_t i = 0; i < nslot; i++) {
        free(slots[i].host);
        slots[i].host = NULL;
    }
    memset(slots, 0, sizeof slots);
    nslot = 0;
}

/* One argument spec into the next slot. Buffers are sized here, not staged. */
int brun_addarg(const char *s)
{
    if (nslot >= MAX_ARGS) return 0;
    slot_t *sl = &slots[nslot];
    const char *c = strchr(s, ':');
    if (c == NULL) return 0;
    size_t klen = (size_t)(c - s);
    const char *rest = c + 1;

    if (klen == 3 && !strncmp(s, "u32", 3)) {
        sl->kind = A_VAL; sl->klass = RT_K_INT;
        sl->val.u = (uint32_t)strtoul(rest, NULL, 0);
        nslot++; return 1;
    }
    if (klen == 3 && !strncmp(s, "i32", 3)) {
        sl->kind = A_VAL; sl->klass = RT_K_INT;
        sl->val.i = (int32_t)strtol(rest, NULL, 0);
        nslot++; return 1;
    }
    if (klen == 3 && !strncmp(s, "f32", 3)) {
        sl->kind = A_VAL; sl->klass = RT_K_FLT;
        sl->val.f = strtof(rest, NULL);
        nslot++; return 1;
    }

    if (klen == 2 && !strncmp(s, "in", 2)) sl->kind = A_IN;
    else if (klen == 2 && !strncmp(s, "io", 2)) sl->kind = A_IO;
    else if (klen == 3 && !strncmp(s, "out", 3)) sl->kind = A_OUT;
    else return 0;
    sl->klass = RT_K_INT;

    if (sl->kind == A_OUT) {
        /* out:PATH:BYTES, and the path may itself hold a colon on Windows. */
        const char *last = strrchr(rest, ':');
        if (last == NULL) return 0;
        size_t plen = (size_t)(last - rest);
        if (plen == 0 || plen >= MAX_PATH) return 0;
        memcpy(sl->path, rest, plen);
        sl->path[plen] = '\0';
        unsigned long b = strtoul(last + 1, NULL, 0);
        if (b == 0ul || b > MAX_BYTES) return 0;
        sl->bytes = (size_t)b;
        nslot++; return 1;
    }

    if (strlen(rest) >= MAX_PATH) return 0;
    strcpy(sl->path, rest);
    nslot++; return 1;
}

static void release(rt_dev_t *dev)
{
    for (uint32_t i = 0; i < nslot; i++) {
        if (slots[i].dptr != 0) rt_free(dev, slots[i].dptr);
        free(slots[i].host);
        slots[i].host = NULL;
        slots[i].dptr = 0;
    }
}

static int stage(rt_dev_t *dev)
{
    for (uint32_t i = 0; i < nslot; i++) {
        slot_t *sl = &slots[i];
        if (sl->kind == A_VAL) continue;

        if (sl->kind == A_IN || sl->kind == A_IO) {
            sl->host = slurp(sl->path, &sl->bytes);
            if (sl->host == NULL) {
                fprintf(stderr, "kath run: cannot read %s\n", sl->path);
                return RT_EIO;
            }
        } else {
            sl->host = calloc(1, sl->bytes);
            if (sl->host == NULL) return RT_EINPUT;
        }

        sl->dptr = rt_alloc(dev, sl->bytes);
        if (sl->dptr == 0) {
            fprintf(stderr, "kath run: out of device memory for %s\n", sl->path);
            return RT_EDRV;
        }

        if (sl->kind == A_IN || sl->kind == A_IO) {
            int rc = rt_h2d(dev, sl->dptr, sl->host, sl->bytes);
            if (rc != RT_OK) {
                fprintf(stderr, "kath run: upload %s: %s\n",
                        sl->path, rt_errs(dev, rc));
                return rc;
            }
        }
    }
    return RT_OK;
}

/* A buffer argument is its device address, so the address is what gets passed
 * and rt_arg_t.size is the width of that address. */
static void build_args(void)
{
    for (uint32_t i = 0; i < nslot; i++) {
        args[i].klass = slots[i].klass;
        if (slots[i].kind == A_VAL) {
            args[i].p = &slots[i].val;
            args[i].size = (uint32_t)sizeof(uint32_t);
        } else {
            args[i].p = &slots[i].dptr;
            args[i].size = (uint32_t)sizeof(rt_ptr);
        }
    }
}

static int fetch(rt_dev_t *dev)
{
    for (uint32_t i = 0; i < nslot; i++) {
        slot_t *sl = &slots[i];
        if (sl->kind != A_OUT && sl->kind != A_IO) continue;

        int rc = rt_d2h(dev, sl->host, sl->dptr, sl->bytes);
        if (rc != RT_OK) {
            fprintf(stderr, "kath run: download %s: %s\n",
                    sl->path, rt_errs(dev, rc));
            return rc;
        }
        if (!spill(sl->path, sl->host, sl->bytes)) {
            fprintf(stderr, "kath run: cannot write %s\n", sl->path);
            return RT_EIO;
        }
        printf("wrote %s (%lu bytes)\n", sl->path, (unsigned long)sl->bytes);
    }
    return RT_OK;
}

int brun_launch(const char *artefact, const char *kernel,
                const char *target, const rt_dim_t *dim)
{
    if (target != NULL && rt_find(target) == NULL) {
        fprintf(stderr, "kath run: no target named '%s'.", target);
        if (rt_list[0] == NULL) {
            fputs(" None are registered.\n", stderr);
        } else {
            fputs(" Known:", stderr);
            for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++)
                fprintf(stderr, " %s", rt_list[i]->name);
            fputc('\n', stderr);
        }
        return 2;
    }

    rt_dev_t dev;
    int rc = rt_open(&dev, target);
    if (rc != RT_OK) {
        fprintf(stderr, "kath run: %s\n", rt_errs(&dev, rc));
        return 1;
    }

    rt_kern_t kern;
    rc = rt_load(&dev, artefact, kernel, &kern);
    if (rc != RT_OK) {
        fprintf(stderr, "kath run: %s: %s\n", artefact, rt_errs(&dev, rc));
        rt_shut(&dev);
        return 1;
    }

    rc = stage(&dev);
    if (rc == RT_OK) {
        build_args();
        rc = rt_run(&dev, &kern, dim, args, nslot);
        if (rc != RT_OK)
            fprintf(stderr, "kath run: launch: %s\n", rt_errs(&dev, rc));
    }
    if (rc == RT_OK) {
        rc = rt_sync(&dev);
        if (rc != RT_OK)
            fprintf(stderr, "kath run: %s\n", rt_errs(&dev, rc));
    }
    if (rc == RT_OK) rc = fetch(&dev);

    release(&dev);
    rt_unload(&dev, &kern);
    rt_shut(&dev);
    return (rc == RT_OK) ? 0 : 1;
}
