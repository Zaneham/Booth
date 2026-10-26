#ifndef CUDA_H
#define CUDA_H

#include <stddef.h>

#if defined(__cplusplus)
#define CUDRV_C extern "C"
#else
#define CUDRV_C extern
#endif

typedef int                CUresult;
typedef int                CUdevice;
typedef void              *CUcontext;
typedef void              *CUmodule;
typedef void              *CUfunction;
typedef void              *CUstream;
typedef void              *CUevent;
typedef unsigned long long CUdeviceptr;

typedef struct CUuuid { char bytes[16]; } CUuuid;

typedef unsigned long long CUmemGenericAllocationHandle;

typedef enum CUmemAllocationType {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED  = 1,
    CU_MEM_ALLOCATION_TYPE_MAX     = 2
} CUmemAllocationType;

typedef enum CUmemLocationType {
    CU_MEM_LOCATION_TYPE_INVALID           = 0,
    CU_MEM_LOCATION_TYPE_DEVICE            = 1,
    CU_MEM_LOCATION_TYPE_HOST              = 2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA         = 3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
    CU_MEM_LOCATION_TYPE_MAX               = 5
} CUmemLocationType;

typedef enum CUmemAllocationHandleType {
    CU_MEM_HANDLE_TYPE_NONE                  = 0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
    CU_MEM_HANDLE_TYPE_WIN32                 = 2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT             = 4,
    CU_MEM_HANDLE_TYPE_FABRIC                = 8
} CUmemAllocationHandleType;

typedef enum CUmemAccess_flags {
    CU_MEM_ACCESS_FLAGS_PROT_NONE      = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READ      = 1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
    CU_MEM_ACCESS_FLAGS_PROT_MAX       = 4
} CUmemAccess_flags;

typedef enum CUmemAllocationGranularity_flags {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM     = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
} CUmemAllocationGranularity_flags;

typedef struct CUmemLocation {
    CUmemLocationType type;
    int               id;
} CUmemLocation;

typedef struct CUmemAllocationProp {
    CUmemAllocationType       type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation             location;
    void                     *win32HandleMetaData;
    struct {
        unsigned char  compressionType;
        unsigned char  gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char  reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc {
    CUmemLocation     location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

enum {
    CUDA_SUCCESS                  = 0,
    CUDA_ERROR_INVALID_VALUE      = 1,
    CUDA_ERROR_OUT_OF_MEMORY      = 2,
    CUDA_ERROR_NOT_INITIALIZED    = 3,
    CUDA_ERROR_DEINITIALIZED      = 4,
    CUDA_ERROR_NO_DEVICE          = 100,
    CUDA_ERROR_INVALID_DEVICE     = 101,
    CUDA_ERROR_INVALID_IMAGE      = 200,
    CUDA_ERROR_INVALID_CONTEXT    = 201,
    CUDA_ERROR_INVALID_HANDLE     = 400,
    CUDA_ERROR_NOT_FOUND          = 500,
    CUDA_ERROR_NOT_READY          = 600,
    CUDA_ERROR_LAUNCH_FAILED      = 719,
    CUDA_ERROR_NOT_SUPPORTED      = 801,
    CUDA_ERROR_UNKNOWN            = 999
};

enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X       = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y       = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z       = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X        = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y        = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z        = 7,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE             = 10,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT  = 16,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID            = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID         = 34,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID         = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH       = 95,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
};

enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS              = 4
};

enum {
    CU_CTX_SCHED_AUTO      = 0,
    CU_STREAM_DEFAULT      = 0,
    CU_STREAM_NON_BLOCKING = 1,
    CU_EVENT_DEFAULT       = 0
};

CUDRV_C CUresult cuInit(unsigned flags);
CUDRV_C CUresult cuDriverGetVersion(int *v);
CUDRV_C CUresult cuGetErrorName(CUresult r, const char **s);
CUDRV_C CUresult cuGetErrorString(CUresult r, const char **s);

CUDRV_C CUresult cuDeviceGet(CUdevice *d, int ordinal);
CUDRV_C CUresult cuDeviceGetCount(int *n);
CUDRV_C CUresult cuDeviceGetName(char *buf, int len, CUdevice d);
CUDRV_C CUresult cuDeviceGetUuid(CUuuid *u, CUdevice d);
CUDRV_C CUresult cuDeviceGetAttribute(int *out, int attr, CUdevice d);
CUDRV_C CUresult cuDeviceTotalMem_v2(size_t *n, CUdevice d);
CUDRV_C CUresult cuDeviceComputeCapability(int *maj, int *min, CUdevice d);

CUDRV_C CUresult cuCtxCreate_v2(CUcontext *c, unsigned flags, CUdevice d);
CUDRV_C CUresult cuCtxDestroy_v2(CUcontext c);
CUDRV_C CUresult cuCtxSetCurrent(CUcontext c);
CUDRV_C CUresult cuCtxGetCurrent(CUcontext *c);
CUDRV_C CUresult cuCtxGetDevice(CUdevice *d);
CUDRV_C CUresult cuCtxSynchronize(void);
CUDRV_C CUresult cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice d);
CUDRV_C CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d);

CUDRV_C CUresult cuModuleLoadData(CUmodule *m, const void *image);
CUDRV_C CUresult cuModuleLoadDataEx(CUmodule *m, const void *image,
                                    unsigned n, int *opts, void **vals);
CUDRV_C CUresult cuModuleUnload(CUmodule m);
CUDRV_C CUresult cuModuleGetFunction(CUfunction *f, CUmodule m,
                                     const char *name);
CUDRV_C CUresult cuFuncGetAttribute(int *out, int attr, CUfunction f);

CUDRV_C CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n);
CUDRV_C CUresult cuMemFree_v2(CUdeviceptr p);
CUDRV_C CUresult cuMemHostAlloc(void **p, size_t n, unsigned flags);
CUDRV_C CUresult cuMemFreeHost(void *p);
CUDRV_C CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *d, void *p,
                                              unsigned flags);
CUDRV_C CUresult cuMemGetInfo_v2(size_t *freeb, size_t *total);

CUDRV_C CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                                     size_t alignment, CUdeviceptr addr,
                                     unsigned long long flags);
CUDRV_C CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUDRV_C CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                             const CUmemAllocationProp *prop,
                             unsigned long long flags);
CUDRV_C CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUDRV_C CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags);
CUDRV_C CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUDRV_C CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc *desc, size_t count);
CUDRV_C CUresult cuMemGetAccess(unsigned long long *flags,
                                const CUmemLocation *location, CUdeviceptr ptr);
CUDRV_C CUresult cuMemGetAllocationGranularity(
                     size_t *granularity, const CUmemAllocationProp *prop,
                     CUmemAllocationGranularity_flags option);

CUDRV_C CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n);
CUDRV_C CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n,
                               CUstream s);
CUDRV_C CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n);
CUDRV_C CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n);
CUDRV_C CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n);
CUDRV_C CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src,
                                      size_t n, CUstream s);
CUDRV_C CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n,
                                      CUstream s);
CUDRV_C CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                                      size_t n, CUstream s);
CUDRV_C CUresult cuMemsetD8_v2(CUdeviceptr p, unsigned char v, size_t n);
CUDRV_C CUresult cuMemsetD8Async(CUdeviceptr p, unsigned char v, size_t n,
                                 CUstream s);
CUDRV_C CUresult cuMemsetD32_v2(CUdeviceptr p, unsigned v, size_t n);
CUDRV_C CUresult cuMemsetD32Async(CUdeviceptr p, unsigned v, size_t n,
                                  CUstream s);

CUDRV_C CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy,
                                unsigned gz, unsigned bx, unsigned by,
                                unsigned bz, unsigned sh, CUstream s,
                                void **args, void **extra);

CUDRV_C CUresult cuStreamCreate(CUstream *s, unsigned flags);
CUDRV_C CUresult cuStreamDestroy_v2(CUstream s);
CUDRV_C CUresult cuStreamSynchronize(CUstream s);
CUDRV_C CUresult cuStreamQuery(CUstream s);

CUDRV_C CUresult cuEventCreate(CUevent *e, unsigned flags);
CUDRV_C CUresult cuEventDestroy_v2(CUevent e);
CUDRV_C CUresult cuEventRecord(CUevent e, CUstream s);
CUDRV_C CUresult cuEventSynchronize(CUevent e);
CUDRV_C CUresult cuEventQuery(CUevent e);
CUDRV_C CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b);

#endif
