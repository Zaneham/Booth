#ifndef CUBLAS_V2_H
#define CUBLAS_V2_H

#include <stddef.h>
#include "cuda_runtime.h"

#if defined(__cplusplus)
#define CUBLAS_C extern "C"
#else
#define CUBLAS_C extern
#endif

typedef enum {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_NOT_INITIALIZED  = 1,
    CUBLAS_STATUS_ALLOC_FAILED     = 3,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_ARCH_MISMATCH    = 8,
    CUBLAS_STATUS_MAPPING_ERROR    = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR   = 14,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15
} cublasStatus_t;

typedef enum {
    CUBLAS_POINTER_MODE_HOST   = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_DEFAULT_MATH        = 0,
    CUBLAS_TENSOR_OP_MATH      = 1,
    CUBLAS_PEDANTIC_MATH       = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
} cublasMath_t;

typedef enum {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL   = 2
} libraryPropertyType;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
} cublasOperation_t;

typedef enum {
    CUBLAS_SIDE_LEFT  = 0,
    CUBLAS_SIDE_RIGHT = 1
} cublasSideMode_t;

typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL  = 2
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT     = 1
} cublasDiagType_t;

typedef enum {
    CUBLAS_COMPUTE_16F           = 64,
    CUBLAS_COMPUTE_32F           = 68,
    CUBLAS_COMPUTE_64F           = 70,
    CUBLAS_COMPUTE_32I           = 72,
    CUBLAS_COMPUTE_32F_FAST_16F  = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77
} cublasComputeType_t;

typedef enum {
    CUBLAS_GEMM_DEFAULT            = -1,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  = 99
} cublasGemmAlgo_t;

typedef void *cublasHandle_t;

CUBLAS_C cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
CUBLAS_C cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
CUBLAS_C cublasStatus_t cublasSetStream_v2(cublasHandle_t handle,
                                           cudaStream_t sid);
CUBLAS_C cublasStatus_t cublasGetStream_v2(cublasHandle_t handle,
                                           cudaStream_t *sid);
CUBLAS_C cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle,
                                              void *wksp, size_t wsz);
CUBLAS_C cublasStatus_t cublasSetMathMode(cublasHandle_t handle,
                                          cublasMath_t mode);
CUBLAS_C cublasStatus_t cublasGetMathMode(cublasHandle_t handle,
                                          cublasMath_t *mode);
CUBLAS_C cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle,
                                            int *version);
CUBLAS_C cublasStatus_t cublasGetProperty(libraryPropertyType type,
                                          int *value);
CUBLAS_C cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                                cublasPointerMode_t mode);
CUBLAS_C cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle,
                                                cublasPointerMode_t *mode);

CUBLAS_C cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
                                       const float *alpha, const float *x,
                                       int incx, float *y, int incy);
CUBLAS_C cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n,
                                       const float *alpha, float *x,
                                       int incx);
CUBLAS_C cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n,
                                       const float *x, int incx, float *y,
                                       int incy);
CUBLAS_C cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n,
                                       float *x, int incx, float *y,
                                       int incy);
CUBLAS_C cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n,
                                      float *x, int incx, float *y, int incy,
                                      const float *c, const float *s);
CUBLAS_C cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
                                      const float *x, int incx,
                                      const float *y, int incy,
                                      float *result);
CUBLAS_C cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n,
                                       const float *x, int incx,
                                       float *result);
CUBLAS_C cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
                                       const float *x, int incx,
                                       float *result);
CUBLAS_C cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n,
                                        const float *x, int incx,
                                        int *result);

CUBLAS_C cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const float *alpha, const float *A,
                                       int lda, const float *B, int ldb,
                                       const float *beta, float *C, int ldc);
CUBLAS_C cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m, int n, int k,
                                                  const float *alpha,
                                                  const float *A, int lda,
                                                  long long strideA,
                                                  const float *B, int ldb,
                                                  long long strideB,
                                                  const float *beta,
                                                  float *C, int ldc,
                                                  long long strideC,
                                                  int batchCount);
CUBLAS_C cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const float *alpha,
                                           const float *const Aarray[],
                                           int lda,
                                           const float *const Barray[],
                                           int ldb, const float *beta,
                                           float *const Carray[], int ldc,
                                           int batchCount);

CUBLAS_C cublasStatus_t cublasStrsm_v2(cublasHandle_t handle,
                                       cublasSideMode_t side,
                                       cublasFillMode_t uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t diag, int m, int n,
                                       const float *alpha, const float *A,
                                       int lda, float *B, int ldb);
CUBLAS_C cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
                                           cublasSideMode_t side,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           cublasDiagType_t diag,
                                           int m, int n, const float *alpha,
                                           const float *const A[], int lda,
                                           float *const B[], int ldb,
                                           int batchCount);

CUBLAS_C cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb,
                                     int m, int n, int k, const void *alpha,
                                     const void *A, cudaDataType_t Atype,
                                     int lda, const void *B,
                                     cudaDataType_t Btype, int ldb,
                                     const void *beta, void *C,
                                     cudaDataType_t Ctype, int ldc,
                                     cublasComputeType_t computeType,
                                     cublasGemmAlgo_t algo);
CUBLAS_C cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb,
                                     int m, int n, int k, const void *alpha,
                                     const void *A, cudaDataType_t Atype,
                                     int lda, long long strideA,
                                     const void *B, cudaDataType_t Btype,
                                     int ldb, long long strideB,
                                     const void *beta, void *C,
                                     cudaDataType_t Ctype, int ldc,
                                     long long strideC, int batchCount,
                                     cublasComputeType_t computeType,
                                     cublasGemmAlgo_t algo);
CUBLAS_C cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb,
                                     int m, int n, int k, const void *alpha,
                                     const void *const Aarray[],
                                     cudaDataType_t Atype, int lda,
                                     const void *const Barray[],
                                     cudaDataType_t Btype, int ldb,
                                     const void *beta, void *const Carray[],
                                     cudaDataType_t Ctype, int ldc,
                                     int batchCount,
                                     cublasComputeType_t computeType,
                                     cublasGemmAlgo_t algo);

#define cublasStrsm           cublasStrsm_v2
#define cublasCreate          cublasCreate_v2
#define cublasDestroy         cublasDestroy_v2
#define cublasSetStream       cublasSetStream_v2
#define cublasGetStream       cublasGetStream_v2
#define cublasSetWorkspace    cublasSetWorkspace_v2
#define cublasGetVersion      cublasGetVersion_v2
#define cublasSetPointerMode  cublasSetPointerMode_v2
#define cublasGetPointerMode  cublasGetPointerMode_v2
#define cublasSaxpy           cublasSaxpy_v2
#define cublasSscal           cublasSscal_v2
#define cublasScopy           cublasScopy_v2
#define cublasSswap           cublasSswap_v2
#define cublasSrot            cublasSrot_v2
#define cublasSdot            cublasSdot_v2
#define cublasSasum           cublasSasum_v2
#define cublasSnrm2           cublasSnrm2_v2
#define cublasIsamax          cublasIsamax_v2
#define cublasSgemm           cublasSgemm_v2

#endif
