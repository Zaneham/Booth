#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

#include <stddef.h>

#define CUDART_VERSION 11020

#if defined(__cplusplus)
#define CUDART_C extern "C"
#else
#define CUDART_C extern
#endif

typedef int   cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;

enum {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorInvalidConfiguration  = 9,
    cudaErrorInvalidSymbol         = 13,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice              = 100,
    cudaErrorInvalidDevice         = 101,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorInvalidPtx            = 218,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorNotReady              = 600,
    cudaErrorIllegalAddress        = 700,
    cudaErrorLaunchFailure         = 719,
    cudaErrorNotSupported          = 801,
    cudaErrorUnknown               = 999
};

enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

enum {
    cudaStreamDefault     = 0,
    cudaStreamNonBlocking = 1
};

#define cudaStreamLegacy    ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)

typedef enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
} cudaStreamCaptureMode;

enum {
    cudaDeviceScheduleAuto         = 0x00,
    cudaDeviceScheduleSpin         = 0x01,
    cudaDeviceScheduleYield        = 0x02,
    cudaDeviceScheduleBlockingSync = 0x04,
    cudaDeviceMapHost              = 0x08,
    cudaDeviceLmemResizeToMax      = 0x10
};

enum {
    cudaHostRegisterDefault  = 0x00,
    cudaHostRegisterPortable = 0x01,
    cudaHostRegisterMapped   = 0x02,
    cudaHostRegisterIoMemory = 0x04,
    cudaHostRegisterReadOnly = 0x08
};

enum {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrWarpSize           = 10,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrCooperativeLaunch  = 95
};

enum {
    cudaEventDefault       = 0,
    cudaEventBlockingSync  = 1,
    cudaEventDisableTiming = 2
};

enum {
    cudaStreamCaptureStatusNone = 0
};

enum {
    cudaHostAllocDefault       = 0x00,
    cudaHostAllocPortable      = 0x01,
    cudaHostAllocMapped        = 0x02,
    cudaHostAllocWriteCombined = 0x04
};

enum {
    cudaMemAttachGlobal = 0x01,
    cudaMemAttachHost   = 0x02,
    cudaMemAttachSingle = 0x04
};

enum {
    cudaFuncAttributeMaxDynamicSharedMemorySize     = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout  = 9
};

typedef enum {
    CUDA_R_32F  = 0,
    CUDA_R_64F  = 1,
    CUDA_R_16F  = 2,
    CUDA_R_8I   = 3,
    CUDA_C_32F  = 4,
    CUDA_C_64F  = 5,
    CUDA_C_16F  = 6,
    CUDA_C_8I   = 7,
    CUDA_R_8U   = 8,
    CUDA_C_8U   = 9,
    CUDA_R_32I  = 10,
    CUDA_C_32I  = 11,
    CUDA_R_32U  = 12,
    CUDA_C_32U  = 13,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15
} cudaDataType_t;

typedef cudaDataType_t cudaDataType;

typedef struct char2 { signed char x, y; } char2;
typedef struct char4 { signed char x, y, z, w; } char4;
typedef struct uchar2 { unsigned char x, y; } uchar2;
typedef struct uchar4 { unsigned char x, y, z, w; } uchar4;
typedef struct short2 { short x, y; } short2;
typedef struct ushort2 { unsigned short x, y; } ushort2;
typedef struct int2 { int x, y; } int2;
typedef struct int4 { int x, y, z, w; } int4;
typedef struct uint2 { unsigned x, y; } uint2;
typedef struct uint3 { unsigned x, y, z; } uint3;
typedef struct uint4 { unsigned x, y, z, w; } uint4;
typedef struct float2 { float x, y; } float2;
typedef struct float4 { float x, y, z, w; } float4;
typedef struct double2 { double x, y; } double2;

#define BC_MKV2(T, E) \
    static T make_##T(E x, E y) { T r; r.x = x; r.y = y; return r; }
#define BC_MKV3(T, E) \
    static T make_##T(E x, E y, E z) { T r; r.x = x; r.y = y; r.z = z; \
                                       return r; }
#define BC_MKV4(T, E) \
    static T make_##T(E x, E y, E z, E w) { T r; r.x = x; r.y = y; r.z = z; \
                                            r.w = w; return r; }

BC_MKV2(char2, signed char)
BC_MKV4(char4, signed char)
BC_MKV2(uchar2, unsigned char)
BC_MKV4(uchar4, unsigned char)
BC_MKV2(short2, short)
BC_MKV2(ushort2, unsigned short)
BC_MKV2(int2, int)
BC_MKV4(int4, int)
BC_MKV2(uint2, unsigned)
BC_MKV3(uint3, unsigned)
BC_MKV4(uint4, unsigned)
BC_MKV2(float2, float)
BC_MKV4(float4, float)
BC_MKV2(double2, double)

typedef struct dim3 {
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int a = 1u, unsigned int b = 1u, unsigned int c = 1u)
        : x(a), y(b), z(c) {}
#endif
} dim3;

typedef struct cudaUUID_t { char bytes[16]; } cudaUUID_t;

typedef struct cudaDeviceProp {
    char       name[256];
    cudaUUID_t uuid;
    char       luid[8];
    unsigned   luidDeviceNodeMask;
    unsigned   bcrsv0;
    size_t     totalGlobalMem;
    size_t     sharedMemPerBlock;
    int        regsPerBlock;
    int        warpSize;
    size_t     memPitch;
    int        maxThreadsPerBlock;
    int        maxThreadsDim[3];
    int        maxGridSize[3];
    int        clockRate;
    size_t     totalConstMem;
    int        major;
    int        minor;
    size_t     textureAlignment;
    size_t     texturePitchAlignment;
    int        deviceOverlap;
    int        multiProcessorCount;
    int        kernelExecTimeoutEnabled;
    int        integrated;
    char       bcrsv1[208];
    int        memoryClockRate;
    int        memoryBusWidth;
    int        l2CacheSize;
    char       bcrsv2[4];
    int        maxThreadsPerMultiProcessor;
    char       bcrsv3[12];
    size_t     sharedMemPerMultiprocessor;
    char       bcrsv4[48];
    size_t     sharedMemPerBlockOptin;
    char       bcrsv5[8];
    size_t     reservedSharedMemPerBlock;
    char       bcrsv6[64];
} cudaDeviceProp;

CUDART_C const char *cudaGetErrorString(cudaError_t e);
CUDART_C cudaError_t cudaGetLastError(void);
CUDART_C cudaError_t cudaPeekAtLastError(void);

CUDART_C void **__cudaRegisterFatBinary(void *wrap);
CUDART_C void   __cudaRegisterFatBinaryEnd(void **h);
CUDART_C void   __cudaUnregisterFatBinary(void **h);
CUDART_C void   __cudaRegisterFunction(void **h, const char *stub, char *dfn,
                                       const char *name, int lim, void *tid,
                                       void *bid, void *bdim, void *gdim,
                                       int *wsz);
CUDART_C void   __cudaRegisterVar(void **h, char *host, char *dev,
                                  const char *name, int ext, size_t size,
                                  int cons, int global);

CUDART_C unsigned    __cudaPushCallConfiguration(dim3 g, dim3 b, size_t sh,
                                                 void *stream);
CUDART_C cudaError_t __cudaPopCallConfiguration(dim3 *g, dim3 *b, size_t *sh,
                                                void *stream);
CUDART_C cudaError_t cudaLaunchKernel(const void *fn, dim3 g, dim3 b,
                                      void **args, size_t sh, cudaStream_t s);

CUDART_C cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                         int *out, const void *fn, int bsz, size_t sh,
                         unsigned flags);

CUDART_C cudaError_t cudaFuncSetAttribute(const void *fn, int attr, int val);
CUDART_C cudaError_t cudaLaunchCooperativeKernel(const void *fn, dim3 g,
                                                 dim3 b, void **args,
                                                 size_t sh, cudaStream_t s);

CUDART_C cudaError_t cudaMalloc(void **out, size_t n);
CUDART_C cudaError_t cudaFree(void *p);
CUDART_C cudaError_t cudaMallocHost(void **out, size_t n);
CUDART_C cudaError_t cudaFreeHost(void *p);
CUDART_C cudaError_t cudaHostAlloc(void **out, size_t n, unsigned flags);
CUDART_C cudaError_t cudaMallocManaged(void **out, size_t n, unsigned flags);
CUDART_C cudaError_t cudaHostRegister(void *p, size_t n, unsigned flags);
CUDART_C cudaError_t cudaHostUnregister(void *p);
CUDART_C cudaError_t cudaHostGetDevicePointer(void **out, void *p,
                                              unsigned flags);
CUDART_C cudaError_t cudaMemGetInfo(size_t *freeb, size_t *total);

CUDART_C cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n,
                                     int kind, cudaStream_t s);
CUDART_C cudaError_t cudaMemcpy2DAsync(void *dst, size_t dp, const void *src,
                                       size_t sp, size_t w, size_t h,
                                       int kind, cudaStream_t s);
CUDART_C cudaError_t cudaMemcpyPeerAsync(void *dst, int dd, const void *src,
                                         int sd, size_t n, cudaStream_t s);
CUDART_C cudaError_t cudaMemset(void *p, int v, size_t n);
CUDART_C cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s);

CUDART_C cudaError_t cudaStreamCreateWithFlags(cudaStream_t *out,
                                               unsigned flags);
CUDART_C cudaError_t cudaStreamDestroy(cudaStream_t s);
CUDART_C cudaError_t cudaStreamSynchronize(cudaStream_t s);
CUDART_C cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e,
                                         unsigned flags);
CUDART_C cudaError_t cudaStreamIsCapturing(cudaStream_t s, int *out);
CUDART_C cudaError_t cudaStreamBeginCapture(cudaStream_t s,
                                            cudaStreamCaptureMode mode);
CUDART_C cudaError_t cudaStreamEndCapture(cudaStream_t s, void **graph);

CUDART_C cudaError_t cudaEventCreateWithFlags(cudaEvent_t *out,
                                              unsigned flags);
CUDART_C cudaError_t cudaEventDestroy(cudaEvent_t e);
CUDART_C cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s);
CUDART_C cudaError_t cudaEventSynchronize(cudaEvent_t e);

CUDART_C cudaError_t cudaGetDevice(int *out);
CUDART_C cudaError_t cudaSetDevice(int dev);
CUDART_C cudaError_t cudaGetDeviceCount(int *out);
CUDART_C cudaError_t cudaDeviceGetAttribute(int *out, int attr, int dev);
CUDART_C cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *prop, int dev);
CUDART_C cudaError_t cudaDeviceGetPCIBusId(char *buf, int len, int dev);
CUDART_C cudaError_t cudaDeviceSynchronize(void);
CUDART_C cudaError_t cudaSetDeviceFlags(unsigned flags);
CUDART_C cudaError_t cudaDeviceCanAccessPeer(int *out, int dev, int peer);
CUDART_C cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned flags);

#if !defined(__CUDACC__)

extern const uint3 threadIdx;
extern const uint3 blockIdx;
extern const dim3  blockDim;
extern const dim3  gridDim;
extern const int   warpSize;

static cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, int kind)
{
    cudaError_t e = cudaMemcpyAsync(dst, src, n, kind, (cudaStream_t)0);

    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize((cudaStream_t)0);
}

static cudaError_t cudaMemcpy2D(void *dst, size_t dp, const void *src,
                                size_t sp, size_t w, size_t h, int kind)
{
    cudaError_t e = cudaMemcpy2DAsync(dst, dp, src, sp, w, h, kind,
                                      (cudaStream_t)0);

    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize((cudaStream_t)0);
}

static cudaError_t cudaMemcpyPeer(void *dst, int dd, const void *src, int sd,
                                  size_t n)
{
    cudaError_t e = cudaMemcpyPeerAsync(dst, dd, src, sd, n, (cudaStream_t)0);

    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize((cudaStream_t)0);
}

static cudaError_t cudaStreamCreate(cudaStream_t *out)
{
    return cudaStreamCreateWithFlags(out, (unsigned)cudaStreamDefault);
}

static cudaError_t cudaEventCreate(cudaEvent_t *out)
{
    return cudaEventCreateWithFlags(out, (unsigned)cudaEventDefault);
}

static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int dev)
{
    return cudaGetDeviceProperties_v2(prop, dev);
}

#if defined(__cplusplus)

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e,
                                           unsigned flags = 0u);
extern "C" cudaError_t cudaMallocManaged(void **out, size_t n,
                                         unsigned flags = cudaMemAttachGlobal);

template <class T>
static cudaError_t cudaFuncSetAttribute(T *fn, int attr, int val)
{
    return cudaFuncSetAttribute((const void *)fn, attr, val);
}

template <class T>
static cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       int *out, T fn, int bsz, size_t sh)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
               out, (const void *)fn, bsz, sh, 0u);
}

#endif

#endif

#endif
