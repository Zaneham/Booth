/* nv_rt.h — Booth NVIDIA Runtime Launcher
 *
 * Thin wrapper around the CUDA Driver API for dispatching PTX kernels
 * on NVIDIA GPUs. Loads nvcuda.dll / libcuda.so at runtime via
 * LoadLibrary / dlopen — no compile-time CUDA SDK dependency.
 *
 * The irony of an open-source CUDA compiler using NVIDIA's closed-source
 * driver to actually run code is not lost on us. We're auditing the
 * compilation, not the silicon. Baby steps. */

#ifndef NV_RT_H
#define NV_RT_H

#include <stdint.h>
#include <stddef.h>

/* ---- Error Codes ---- */

#define NV_RT_OK           0
#define NV_RT_ERR_DL      -1   /* failed to load nvcuda.dll / libcuda.so */
#define NV_RT_ERR_SYM     -2   /* missing CUDA function symbol */
#define NV_RT_ERR_CUDA    -3   /* CUDA Driver API call returned error */
#define NV_RT_ERR_NO_GPU  -4   /* no CUDA-capable GPU found */
#define NV_RT_ERR_IO      -5   /* file I/O error */
#define NV_RT_ERR_KERN    -6   /* kernel symbol not found in module */

/* ---- CUDA Driver API Types ---- */
/* Defined here so we don't need cuda.h at compile time.
 * These must match the NVIDIA ABI exactly. */

typedef int           CUresult;
typedef int           CUdevice;
typedef void         *CUctx;
typedef void         *CUmod;
typedef void         *CUfunc;
typedef void         *CUstream;
typedef unsigned long long CUdevptr;

#define CUDA_SUCCESS  0

/* ---- Kernel Handle ---- */

typedef struct {
    CUmod   mod;
    CUfunc  func;
    char    name[128];
} nv_kern_t;

/* ---- Device Context ---- */
/* Unlike the HSA backend, CUDA is clean enough that we don't need
 * to hide behind an opaque blob. Transparency: the novel concept. */

typedef struct {
    void     *lib;       /* LoadLibrary / dlopen handle */
    CUctx     ctx;
    CUdevice  dev;
    char      dev_name[128];
    int       sm_major;
    int       sm_minor;

    /* Function pointers — loaded at runtime */
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDevGet)(CUdevice *, int);
    CUresult (*cuDevName)(char *, int, CUdevice);
    CUresult (*cuDevAttr)(int *, int, CUdevice);
    CUresult (*cuCtxCreate)(CUctx *, unsigned, CUdevice);
    CUresult (*cuCtxDestroy)(CUctx);
    CUresult (*cuCtxSync)(void);
    CUresult (*cuModLoad)(CUmod *, const void *);
    CUresult (*cuModLoadEx)(CUmod *, const void *, unsigned,
                            int *, void **);
    CUresult (*cuModUnload)(CUmod);
    CUresult (*cuModGetFn)(CUfunc *, CUmod, const char *);
    CUresult (*cuMemAlloc)(CUdevptr *, size_t);
    CUresult (*cuMemFree)(CUdevptr);
    CUresult (*cuMemH2D)(CUdevptr, const void *, size_t);
    CUresult (*cuMemD2H)(void *, CUdevptr, size_t);
    CUresult (*cuLaunch)(CUfunc,
                         unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned,
                         unsigned, CUstream,
                         void **, void **);
    CUresult (*cuGetErr)(CUresult, const char **);

    /* Mapped host memory — for ABEND breadcrumbs that survive crashes.
     * Optional: loaded but non-fatal if missing (old drivers). */
    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuMemHostGetDev)(CUdevptr *, void *, unsigned);
} nv_dev_t;

/* ---- Public API ---- */

int  nv_rt_init(nv_dev_t *dev);
void nv_rt_shut(nv_dev_t *dev);

/* Load a PTX file and extract a kernel by name */
int  nv_rt_load(nv_dev_t *dev, const char *ptx_path,
                const char *kern_name, nv_kern_t *out);
void nv_rt_unload(nv_dev_t *dev, nv_kern_t *kern);

/* Device memory */
CUdevptr nv_rt_alloc(nv_dev_t *dev, size_t size);
void     nv_rt_free(nv_dev_t *dev, CUdevptr ptr);

/* Host <-> Device copies */
int  nv_rt_h2d(nv_dev_t *dev, CUdevptr dst,
               const void *src, size_t size);
int  nv_rt_d2h(nv_dev_t *dev, void *dst,
               CUdevptr src, size_t size);

/* Synchronous kernel launch. grid/block in CUDA convention. */
int  nv_rt_launch(nv_dev_t *dev, nv_kern_t *kern,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  uint32_t bx, uint32_t by, uint32_t bz,
                  uint32_t shmem, void **args);

/* Synchronize (wait for all pending work) */
int  nv_rt_sync(nv_dev_t *dev);

/* Mapped host memory — GPU writes to host RAM, readable after crash.
 * Returns host pointer in *hpp, device pointer in *dpp.
 * Both must be used: hpp for CPU reads, dpp for kernel args. */
int  nv_rt_mmap(nv_dev_t *dev, void **hpp, CUdevptr *dpp, size_t sz);
void nv_rt_mfre(nv_dev_t *dev, void *hp);

#endif /* NV_RT_H */
