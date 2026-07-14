/* nv_rt.c — Booth NVIDIA Runtime Launcher
 *
 * Loads nvcuda.dll (Windows) or libcuda.so (Linux) at runtime.
 * No CUDA SDK needed at compile time. Just a working NVIDIA driver.
 *
 * The CUDA Driver API is refreshingly sane compared to HSA. No AQL
 * packets, no doorbell signals, no kernarg pools, no IOMMU permission
 * dances. You just... call a function. Revolutionary concept, really.
 * Almost makes you forget they won't show you the source code. */

#include "nv_rt.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---- Platform-specific dynamic loading ---- */

#ifdef _WIN32
#include <windows.h>
#define NV_LIB_NAME   "nvcuda.dll"
#define nv_dlopen(n)  ((void *)LoadLibraryA(n))
/* ISO C says function→object pointer cast is undefined, but the
 * Windows ABI guarantees it works. We go through memcpy to keep
 * -Wpedantic happy. The things we do for correctness. */
static void *nv_dlsym_w(void *h, const char *s) {
    FARPROC fp = GetProcAddress((HMODULE)h, s);
    void *p;
    memcpy(&p, &fp, sizeof(p));
    return p;
}
#define nv_dlsym(h,s) nv_dlsym_w((h), (s))
#define nv_dlclose(h) FreeLibrary((HMODULE)(h))
#else
#include <dlfcn.h>
#define NV_LIB_NAME   "libcuda.so.1"
#define nv_dlopen(n)  dlopen((n), RTLD_LAZY)
#define nv_dlsym(h,s) dlsym((h), (s))
#define nv_dlclose(h) dlclose(h)
#endif

/* ---- CUDA Device Attribute Constants ---- */

#define CU_ATTR_SM_MAJOR   75   /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
#define CU_ATTR_SM_MINOR   76   /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */

/* ---- Helpers ---- */

static const char *nv_errstr(nv_dev_t *D, CUresult rc)
{
    const char *str = NULL;
    if (D->cuGetErr && D->cuGetErr(rc, &str) == CUDA_SUCCESS && str)
        return str;
    return "unknown CUDA error";
}

/* Load a single symbol, return 0 on success */
static int nv_sym(void *lib, const char *name, void **out)
{
    *out = nv_dlsym(lib, name);
    if (!*out) {
        fprintf(stderr, "nv_rt: missing symbol: %s\n", name);
        return -1;
    }
    return 0;
}

/* ---- Init ---- */

int nv_rt_init(nv_dev_t *D)
{
    memset(D, 0, sizeof(*D));

    /* Load the CUDA driver library */
    D->lib = nv_dlopen(NV_LIB_NAME);
    if (!D->lib) {
        fprintf(stderr, "nv_rt: failed to load %s\n", NV_LIB_NAME);
        return NV_RT_ERR_DL;
    }

    /* Resolve every function pointer. The _v2 suffix is the 64-bit
     * ABI — NVIDIA kept the old names for 32-bit compat but the
     * actual symbols in nvcuda.dll are versioned. Charming. */
    int bad = 0;
    bad += nv_sym(D->lib, "cuInit",                     (void **)&D->cuInit);
    bad += nv_sym(D->lib, "cuDeviceGet",                (void **)&D->cuDevGet);
    bad += nv_sym(D->lib, "cuDeviceGetName",            (void **)&D->cuDevName);
    bad += nv_sym(D->lib, "cuDeviceGetAttribute",       (void **)&D->cuDevAttr);
    bad += nv_sym(D->lib, "cuCtxCreate_v2",             (void **)&D->cuCtxCreate);
    bad += nv_sym(D->lib, "cuCtxDestroy_v2",            (void **)&D->cuCtxDestroy);
    bad += nv_sym(D->lib, "cuCtxSynchronize",           (void **)&D->cuCtxSync);
    bad += nv_sym(D->lib, "cuModuleLoadData",           (void **)&D->cuModLoad);
    bad += nv_sym(D->lib, "cuModuleLoadDataEx",         (void **)&D->cuModLoadEx);
    bad += nv_sym(D->lib, "cuModuleUnload",             (void **)&D->cuModUnload);
    bad += nv_sym(D->lib, "cuModuleGetFunction",        (void **)&D->cuModGetFn);
    bad += nv_sym(D->lib, "cuMemAlloc_v2",              (void **)&D->cuMemAlloc);
    bad += nv_sym(D->lib, "cuMemFree_v2",               (void **)&D->cuMemFree);
    bad += nv_sym(D->lib, "cuMemcpyHtoD_v2",            (void **)&D->cuMemH2D);
    bad += nv_sym(D->lib, "cuMemcpyDtoH_v2",            (void **)&D->cuMemD2H);
    bad += nv_sym(D->lib, "cuLaunchKernel",             (void **)&D->cuLaunch);
    bad += nv_sym(D->lib, "cuGetErrorString",           (void **)&D->cuGetErr);

    /* Optional: mapped host memory for ABEND breadcrumbs.
     * Non-fatal if missing — old drivers can limp along without. */
    D->cuMemHostAlloc  = NULL;
    D->cuMemFreeHost   = NULL;
    D->cuMemHostGetDev = NULL;
    {
        void *p1 = nv_dlsym(D->lib, "cuMemHostAlloc");
        void *p2 = nv_dlsym(D->lib, "cuMemFreeHost");
        void *p3 = nv_dlsym(D->lib, "cuMemHostGetDevicePointer_v2");
        memcpy(&D->cuMemHostAlloc,  &p1, sizeof(p1));
        memcpy(&D->cuMemFreeHost,   &p2, sizeof(p2));
        memcpy(&D->cuMemHostGetDev, &p3, sizeof(p3));
    }

    if (bad) {
        nv_dlclose(D->lib);
        D->lib = NULL;
        return NV_RT_ERR_SYM;
    }

    /* Initialize CUDA */
    CUresult rc = D->cuInit(0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuInit failed: %s\n", nv_errstr(D, rc));
        nv_dlclose(D->lib);
        D->lib = NULL;
        return NV_RT_ERR_CUDA;
    }

    /* Get first GPU */
    rc = D->cuDevGet(&D->dev, 0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: no CUDA GPU found: %s\n", nv_errstr(D, rc));
        nv_dlclose(D->lib);
        D->lib = NULL;
        return NV_RT_ERR_NO_GPU;
    }

    /* Query device info */
    D->cuDevName(D->dev_name, (int)sizeof(D->dev_name), D->dev);
    D->cuDevAttr(&D->sm_major, CU_ATTR_SM_MAJOR, D->dev);
    D->cuDevAttr(&D->sm_minor, CU_ATTR_SM_MINOR, D->dev);

    /* Create context */
    rc = D->cuCtxCreate(&D->ctx, 0, D->dev);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuCtxCreate failed: %s\n", nv_errstr(D, rc));
        nv_dlclose(D->lib);
        D->lib = NULL;
        return NV_RT_ERR_CUDA;
    }

    printf("nv_rt: %s (sm_%d%d)\n", D->dev_name, D->sm_major, D->sm_minor);
    return NV_RT_OK;
}

/* ---- Shutdown ---- */

void nv_rt_shut(nv_dev_t *D)
{
    if (!D || !D->lib) return;
    if (D->ctx) D->cuCtxDestroy(D->ctx);
    nv_dlclose(D->lib);
    memset(D, 0, sizeof(*D));
}

/* ---- Load PTX Kernel ---- */

#define NV_PTX_MAX  (4 * 1024 * 1024)  /* 4 MB max PTX file */

int nv_rt_load(nv_dev_t *D, const char *ptx_path,
               const char *kern_name, nv_kern_t *out)
{
    memset(out, 0, sizeof(*out));

    /* Read PTX file */
    FILE *fp = fopen(ptx_path, "rb");
    if (!fp) {
        fprintf(stderr, "nv_rt: cannot open '%s'\n", ptx_path);
        return NV_RT_ERR_IO;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz >= NV_PTX_MAX) {
        fprintf(stderr, "nv_rt: PTX file too large (%ld bytes)\n", sz);
        fclose(fp);
        return NV_RT_ERR_IO;
    }

    char *ptx_buf = (char *)malloc((size_t)sz + 1);
    if (!ptx_buf) {
        fclose(fp);
        return NV_RT_ERR_IO;
    }
    size_t rd = fread(ptx_buf, 1, (size_t)sz, fp);
    ptx_buf[rd] = '\0';
    fclose(fp);

    /* JIT compile PTX -> SASS via cuModuleLoadDataEx.
     * The driver does all the hard work — register allocation,
     * instruction scheduling, the whole nine yards. We just
     * hand over the text and trust the black box. But at least
     * we asked for the error log, which is more than NVCC does
     * when it silently generates wrong code. */
    #define CU_JIT_ERROR_LOG_BUFFER      5
    #define CU_JIT_ERROR_LOG_BUFFER_SIZE 6
    #define CU_JIT_INFO_LOG_BUFFER       3
    #define CU_JIT_INFO_LOG_BUFFER_SIZE  4

    char jit_err[4096]; jit_err[0] = '\0';
    char jit_inf[4096]; jit_inf[0] = '\0';
    int opts[] = {
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE,
        CU_JIT_INFO_LOG_BUFFER,  CU_JIT_INFO_LOG_BUFFER_SIZE
    };
    void *vals[] = {
        jit_err, (void *)(size_t)sizeof(jit_err),
        jit_inf, (void *)(size_t)sizeof(jit_inf)
    };
    CUresult rc = D->cuModLoadEx(&out->mod, ptx_buf, 4, opts, vals);
    free(ptx_buf);

    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuModuleLoadDataEx failed: %s\n",
                nv_errstr(D, rc));
        if (jit_err[0])
            fprintf(stderr, "nv_rt: JIT error log:\n%s\n", jit_err);
        if (jit_inf[0])
            fprintf(stderr, "nv_rt: JIT info log:\n%s\n", jit_inf);
        return NV_RT_ERR_CUDA;
    }

    /* Extract kernel function */
    rc = D->cuModGetFn(&out->func, out->mod, kern_name);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: kernel '%s' not found: %s\n",
                kern_name, nv_errstr(D, rc));
        D->cuModUnload(out->mod);
        out->mod = NULL;
        return NV_RT_ERR_KERN;
    }

    snprintf(out->name, sizeof(out->name), "%s", kern_name);
    printf("nv_rt: loaded kernel '%s' from %s\n", kern_name, ptx_path);
    return NV_RT_OK;
}

/* ---- Unload ---- */

void nv_rt_unload(nv_dev_t *D, nv_kern_t *kern)
{
    if (!D || !D->lib || !kern->mod) return;
    D->cuModUnload(kern->mod);
    memset(kern, 0, sizeof(*kern));
}

/* ---- Memory ---- */

CUdevptr nv_rt_alloc(nv_dev_t *D, size_t size)
{
    CUdevptr ptr = 0;
    CUresult rc = D->cuMemAlloc(&ptr, size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuMemAlloc(%llu) failed: %s\n",
                (unsigned long long)size, nv_errstr(D, rc));
        return 0;
    }
    return ptr;
}

void nv_rt_free(nv_dev_t *D, CUdevptr ptr)
{
    if (ptr) D->cuMemFree(ptr);
}

/* ---- Copies ---- */

int nv_rt_h2d(nv_dev_t *D, CUdevptr dst,
              const void *src, size_t size)
{
    CUresult rc = D->cuMemH2D(dst, src, size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: H2D copy failed: %s\n", nv_errstr(D, rc));
        return NV_RT_ERR_CUDA;
    }
    return NV_RT_OK;
}

int nv_rt_d2h(nv_dev_t *D, void *dst,
              CUdevptr src, size_t size)
{
    CUresult rc = D->cuMemD2H(dst, src, size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: D2H copy failed: %s\n", nv_errstr(D, rc));
        return NV_RT_ERR_CUDA;
    }
    return NV_RT_OK;
}

/* ---- Launch ---- */

int nv_rt_launch(nv_dev_t *D, nv_kern_t *kern,
                 uint32_t gx, uint32_t gy, uint32_t gz,
                 uint32_t bx, uint32_t by, uint32_t bz,
                 uint32_t shmem, void **args)
{
    CUresult rc = D->cuLaunch(
        kern->func,
        gx, gy, gz,       /* grid dimensions (blocks) */
        bx, by, bz,       /* block dimensions (threads) */
        shmem,             /* shared memory bytes */
        NULL,              /* stream (NULL = default) */
        args,              /* kernel parameters */
        NULL               /* extra (unused) */
    );

    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuLaunchKernel '%s' failed: %s\n",
                kern->name, nv_errstr(D, rc));
        return NV_RT_ERR_CUDA;
    }
    return NV_RT_OK;
}

/* ---- Sync ---- */

int nv_rt_sync(nv_dev_t *D)
{
    CUresult rc = D->cuCtxSync();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: sync failed: %s\n", nv_errstr(D, rc));
        return NV_RT_ERR_CUDA;
    }
    return NV_RT_OK;
}

/* ---- Mapped Host Memory ----
 * Page-locked host memory mapped into GPU address space.
 * GPU writes directly to host RAM — the data survives context
 * corruption from illegal memory accesses. Like writing your
 * last words on the parachute before jumping: if you don't make
 * it, at least someone can read the note. */

#define CU_MEMHOSTALLOC_DEVICEMAP  2u

int nv_rt_mmap(nv_dev_t *D, void **hpp, CUdevptr *dpp, size_t sz)
{
    if (!D->cuMemHostAlloc || !D->cuMemHostGetDev) {
        fprintf(stderr, "nv_rt: mapped memory not available\n");
        return NV_RT_ERR_CUDA;
    }

    *hpp = NULL;
    *dpp = 0;

    CUresult rc = D->cuMemHostAlloc(hpp, sz, CU_MEMHOSTALLOC_DEVICEMAP);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuMemHostAlloc(%llu) failed: %s\n",
                (unsigned long long)sz, nv_errstr(D, rc));
        return NV_RT_ERR_CUDA;
    }

    rc = D->cuMemHostGetDev(dpp, *hpp, 0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "nv_rt: cuMemHostGetDevicePointer failed: %s\n",
                nv_errstr(D, rc));
        D->cuMemFreeHost(*hpp);
        *hpp = NULL;
        return NV_RT_ERR_CUDA;
    }

    return NV_RT_OK;
}

void nv_rt_mfre(nv_dev_t *D, void *hp)
{
    if (hp && D->cuMemFreeHost)
        D->cuMemFreeHost(hp);
}
