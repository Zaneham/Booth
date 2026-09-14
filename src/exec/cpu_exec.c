/* cpu_exec.c -- run a --cpu object in this process, no linker, no GPU.
 * The kernel's .text is mapped executable and called through a sysv_abi
 * pointer, with nthreads as the trailing hidden argument. */

#include "exec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <dlfcn.h>
#endif

#define CE_MAX_FILE  (16u * 1024u * 1024u)

typedef struct { int open; } ce_dev_t;

typedef struct {
    unsigned char *exec;
    size_t         size;
    size_t         entry;
} ce_kern_t;

/* ---- little-endian field reads, bounds-guarded ---- */

static int rd16(const unsigned char *b, size_t n, size_t o, uint16_t *v)
{
    if (o + 2 > n) return 0;
    uint16_t t; memcpy(&t, b + o, 2); *v = t; return 1;
}
static int rd32(const unsigned char *b, size_t n, size_t o, uint32_t *v)
{
    if (o + 4 > n) return 0;
    uint32_t t; memcpy(&t, b + o, 4); *v = t; return 1;
}
static int rd64(const unsigned char *b, size_t n, size_t o, uint64_t *v)
{
    if (o + 8 > n) return 0;
    uint64_t t; memcpy(&t, b + o, 8); *v = t; return 1;
}

/* ---- executable pages ---- */

static unsigned char *ce_map(const unsigned char *code, size_t n)
{
    if (n == 0) return NULL;
#ifdef _WIN32
    void *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (p == NULL) return NULL;
    memcpy(p, code, n);
    return (unsigned char *)p;
#else
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    memcpy(p, code, n);
    return (unsigned char *)p;
#endif
}

static int ce_protect(unsigned char *p, size_t n)
{
#ifdef _WIN32
    DWORD old;
    if (!VirtualProtect(p, n, PAGE_EXECUTE_READ, &old)) return 0;
    FlushInstructionCache(GetCurrentProcess(), p, n);
    return 1;
#else
    if (mprotect(p, n, PROT_READ | PROT_EXEC) != 0) return 0;
    __builtin___clear_cache((char *)p, (char *)p + n);
    return 1;
#endif
}

static void ce_unmap(unsigned char *p, size_t n)
{
    if (p == NULL) return;
#ifdef _WIN32
    (void)n; VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, n);
#endif
}

static int ce_resolve(const char *name, uint64_t *addr)
{
#ifdef _WIN32
    static const char *mods[] = { "ucrtbase.dll", "msvcrt.dll", "api-ms-win-crt-math-l1-1-0.dll" };
    for (unsigned i = 0; i < sizeof mods / sizeof mods[0]; i++) {
        HMODULE h = GetModuleHandleA(mods[i]);
        if (h == NULL) h = LoadLibraryA(mods[i]);
        if (h == NULL) continue;
        FARPROC f = GetProcAddress(h, name);
        if (f != NULL) { void *v; memcpy(&v, &f, sizeof v); *addr = (uint64_t)(uintptr_t)v; return 1; }
    }
    return 0;
#else
    void *v = dlsym(RTLD_DEFAULT, name);
    if (v == NULL) return 0;
    *addr = (uint64_t)(uintptr_t)v;
    return 1;
#endif
}

/* ---- descriptor ops ---- */

static int ce_open(void *dev) { ((ce_dev_t *)dev)->open = 1; return RT_OK; }
static void ce_shut(void *dev) { ((ce_dev_t *)dev)->open = 0; }

static rt_ptr ce_alloc(void *dev, size_t n)
{
    (void)dev;
    void *p = malloc(n ? n : 1u);
    return (rt_ptr)(uintptr_t)p;
}
static void ce_dfre(void *dev, rt_ptr p) { (void)dev; free((void *)(uintptr_t)p); }

static int ce_h2d(void *dev, rt_ptr dst, const void *src, size_t n)
{
    (void)dev; memcpy((void *)(uintptr_t)dst, src, n); return RT_OK;
}
static int ce_d2h(void *dev, void *dst, rt_ptr src, size_t n)
{
    (void)dev; memcpy(dst, (const void *)(uintptr_t)src, n); return RT_OK;
}
static int ce_sync(void *dev) { (void)dev; return RT_OK; }

static int ce_load(void *dev, const char *path, const char *kern, void *out)
{
    (void)dev;
    ce_kern_t *K = (ce_kern_t *)out;
    memset(K, 0, sizeof *K);

    FILE *f = fopen(path, "rb");
    if (f == NULL) return RT_EIO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return RT_EIO; }
    long fl = ftell(f);
    rewind(f);
    if (fl < 64 || (unsigned long)fl > CE_MAX_FILE) { fclose(f); return RT_EIO; }
    size_t n = (size_t)fl;
    unsigned char *b = (unsigned char *)malloc(n);
    if (b == NULL) { fclose(f); return RT_EINPUT; }
    if (fread(b, 1, n, f) != n) { free(b); fclose(f); return RT_EIO; }
    fclose(f);

    if (!(b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F')
        || b[4] != 2 || b[5] != 1) { free(b); return RT_EINPUT; }
    uint16_t machine = 0; rd16(b, n, 18, &machine);
    if (machine != 62) { free(b); return RT_EUNSUP; }   /* x86-64 only */

    uint64_t shoff = 0; uint16_t shentsz = 0, shnum = 0, shstrndx = 0;
    if (!rd64(b, n, 0x28, &shoff) || !rd16(b, n, 0x3a, &shentsz)
        || !rd16(b, n, 0x3c, &shnum) || !rd16(b, n, 0x3e, &shstrndx)
        || shentsz < 64 || shstrndx >= shnum) { free(b); return RT_EINPUT; }

    uint64_t shstr_off = 0;
    if (!rd64(b, n, (size_t)shoff + (size_t)shstrndx * shentsz + 0x18, &shstr_off)) {
        free(b); return RT_EINPUT;
    }

    size_t text_off = 0, text_sz = 0;
    size_t sym_off = 0, sym_sz = 0, sym_link = 0;
    size_t str_off = 0, str_sz = 0;
    size_t rela_off = 0, rela_sz = 0;
    uint16_t text_idx = 0;

    for (uint16_t s = 0; s < shnum; s++) {
        size_t sh = (size_t)shoff + (size_t)s * shentsz;
        uint32_t nm = 0, ty = 0, lk = 0;
        uint64_t of = 0, sz = 0;
        if (!rd32(b, n, sh + 0, &nm) || !rd32(b, n, sh + 4, &ty)
            || !rd64(b, n, sh + 0x18, &of) || !rd64(b, n, sh + 0x20, &sz)
            || !rd32(b, n, sh + 0x28, &lk)) { free(b); return RT_EINPUT; }
        const char *name = (shstr_off + nm < n) ? (const char *)(b + shstr_off + nm) : "";
        if (strcmp(name, ".text") == 0) { text_off = (size_t)of; text_sz = (size_t)sz; text_idx = s; }
        else if (ty == 2) { sym_off = (size_t)of; sym_sz = (size_t)sz; sym_link = lk; }   /* SHT_SYMTAB */
        else if (strcmp(name, ".rela.text") == 0) { rela_off = (size_t)of; rela_sz = (size_t)sz; }
    }
    if (text_sz == 0 || text_off + text_sz > n || sym_off == 0) { free(b); return RT_EINPUT; }

    {
        size_t sh = (size_t)shoff + sym_link * shentsz;
        uint64_t of = 0, sz = 0;
        if (!rd64(b, n, sh + 0x18, &of) || !rd64(b, n, sh + 0x20, &sz)) { free(b); return RT_EINPUT; }
        str_off = (size_t)of; str_sz = (size_t)sz;
    }

    uint64_t entry = 0; int found = 0;
    for (size_t so = sym_off; so + 24 <= sym_off + sym_sz && so + 24 <= n; so += 24) {
        uint32_t stn = 0; uint16_t shndx = 0; uint64_t val = 0;
        rd32(b, n, so + 0, &stn); rd16(b, n, so + 6, &shndx); rd64(b, n, so + 8, &val);
        unsigned char info = b[so + 4];
        const char *snm = (str_off + stn < str_off + str_sz && str_off + stn < n)
                          ? (const char *)(b + str_off + stn) : "";
        if (shndx != text_idx) continue;
        if ((info & 0x0f) != 2) continue;   /* STT_FUNC */
        if (kern != NULL && kern[0] != '\0') {
            if (strcmp(snm, kern) == 0) { entry = val; found = 1; break; }
        } else if (!found) { entry = val; found = 1; }
    }
    if (!found) { free(b); return RT_EKERN; }

    unsigned char *exec = ce_map(b + text_off, text_sz);
    if (exec == NULL) { free(b); return RT_EDRV; }

    for (size_t ro = rela_off; rela_sz && ro + 24 <= rela_off + rela_sz && ro + 24 <= n; ro += 24) {
        uint64_t roff = 0, rinfo = 0, add = 0;
        rd64(b, n, ro + 0, &roff); rd64(b, n, ro + 8, &rinfo); rd64(b, n, ro + 16, &add);
        uint32_t rtype = (uint32_t)(rinfo & 0xffffffffu);
        uint32_t rsym  = (uint32_t)(rinfo >> 32);
        if (rtype != 4 && rtype != 2) { ce_unmap(exec, text_sz); free(b); return RT_EUNSUP; }
        size_t sso = sym_off + (size_t)rsym * 24;
        uint32_t stn = 0; rd32(b, n, sso + 0, &stn);
        const char *snm = (str_off + stn < n) ? (const char *)(b + str_off + stn) : "";
        uint64_t saddr = 0;
        if (!ce_resolve(snm, &saddr)) { ce_unmap(exec, text_sz); free(b); return RT_ESYM; }
        uint64_t site = (uint64_t)(uintptr_t)(exec + roff);
        int64_t rel = (int64_t)(saddr + (uint64_t)(int64_t)add) - (int64_t)site;
        if (rel < INT32_MIN || rel > INT32_MAX || roff + 4 > text_sz) {
            ce_unmap(exec, text_sz); free(b); return RT_EUNSUP;
        }
        int32_t r32 = (int32_t)rel;
        memcpy(exec + roff, &r32, 4);
    }

    free(b);
    if (!ce_protect(exec, text_sz)) { ce_unmap(exec, text_sz); return RT_EDRV; }

    K->exec = exec;
    K->size = text_sz;
    K->entry = (size_t)entry;
    return RT_OK;
}

static void ce_unload(void *dev, void *kern)
{
    (void)dev;
    ce_kern_t *K = (ce_kern_t *)kern;
    ce_unmap(K->exec, K->size);
    K->exec = NULL;
}

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
typedef void (*ce_fn_t)(long long, long long, long long, long long, long long, long long,
                        float, float, float, float, float, float, float, float)
    __attribute__((sysv_abi));
#endif

static int ce_run(void *dev, const void *kern, const rt_dim_t *dim,
                  const rt_arg_t *args, uint32_t nargs)
{
    (void)dev;
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
    const ce_kern_t *K = (const ce_kern_t *)kern;
    long long iv[6]; float fv[8];
    for (int i = 0; i < 6; i++) iv[i] = 0;
    for (int i = 0; i < 8; i++) fv[i] = 0.0f;
    int ni = 0, nf = 0;

    for (uint32_t i = 0; i < nargs; i++) {
        if (args[i].klass == RT_K_FLT) {
            if (nf >= 8) return RT_EUNSUP;
            float f = 0.0f; memcpy(&f, args[i].p, 4); fv[nf++] = f;
        } else if (args[i].size >= 8) {
            if (ni >= 6) return RT_EUNSUP;
            uint64_t v = 0; memcpy(&v, args[i].p, 8); iv[ni++] = (long long)v;
        } else {
            if (ni >= 6) return RT_EUNSUP;
            uint32_t v = 0; memcpy(&v, args[i].p, 4); iv[ni++] = (long long)(uint64_t)v;
        }
    }

    uint64_t nthreads = (uint64_t)dim->grid[0] * dim->grid[1] * dim->grid[2]
                      * dim->block[0] * dim->block[1] * dim->block[2];
    if (ni >= 6) return RT_EUNSUP;
    iv[ni++] = (long long)nthreads;

    ce_fn_t fn;
    void *ep = K->exec + K->entry;
    memcpy(&fn, &ep, sizeof fn);
    fn(iv[0], iv[1], iv[2], iv[3], iv[4], iv[5],
       fv[0], fv[1], fv[2], fv[3], fv[4], fv[5], fv[6], fv[7]);
    return RT_OK;
#else
    (void)kern; (void)dim; (void)args; (void)nargs;
    return RT_EUNSUP;
#endif
}

static const char *ce_errs(int rc)
{
    switch (rc) {
    case RT_EUNSUP: return "the in-process CPU launcher cannot run this object "
                           "(needs external calls or an unsupported ABI)";
    case RT_ESYM:   return "an external symbol the kernel calls could not be resolved";
    default:        return NULL;
    }
}

const rt_desc_t cpu_rt_desc = {
    .name      = "cpu",
    .artefact  = "o",
    .dev_size  = (uint32_t)sizeof(ce_dev_t),
    .kern_size = (uint32_t)sizeof(ce_kern_t),
    .open   = ce_open,
    .shut   = ce_shut,
    .load   = ce_load,
    .unload = ce_unload,
    .alloc  = ce_alloc,
    .dfre   = ce_dfre,
    .h2d    = ce_h2d,
    .d2h    = ce_d2h,
    .run    = ce_run,
    .sync   = ce_sync,
    .hmap   = NULL,
    .hfre   = NULL,
    .trak   = NULL,
    .errs   = ce_errs
};
