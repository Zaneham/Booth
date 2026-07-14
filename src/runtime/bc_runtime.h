/* bc_runtime.h — Booth HSA Runtime Launcher
 * Thin wrapper around libhsa-runtime64.so for dispatching .hsaco kernels
 * on AMD GPUs. Linux/ROCm only. Loads libhsa at runtime via dlopen. */

#ifndef BC_RUNTIME_H
#define BC_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

/* ---- Error Codes ---- */

#define BC_RT_OK            0
#define BC_RT_ERR_DLOPEN   -1   /* failed to load libhsa-runtime64.so */
#define BC_RT_ERR_DLSYM    -2   /* missing HSA function symbol */
#define BC_RT_ERR_HSA      -3   /* HSA API call returned error */
#define BC_RT_ERR_NO_GPU   -4   /* no GPU agent found */
#define BC_RT_ERR_NO_MEM   -5   /* required memory region not found */
#define BC_RT_ERR_IO       -6   /* file I/O error */
#define BC_RT_ERR_SYMBOL   -7   /* kernel symbol not found in code object */

/* ---- Kernel Handle ---- */

typedef struct bc_kernel {
    uint64_t kernel_object;   /* GPU address of kernel descriptor */
    uint32_t kernarg_size;    /* kernel argument buffer size (bytes) */
    uint32_t group_size;      /* LDS / shared memory per workgroup */
    uint32_t private_size;    /* scratch memory per work-item */
    /* internal handles — do not touch */
    uint64_t _exec;
    uint64_t _reader;
} bc_kernel_t;

/* ---- Device Context (opaque, stack-allocatable) ---- */

#define BC_DEVICE_OPAQUE_SIZE 384
typedef struct bc_device {
    uint8_t _data[BC_DEVICE_OPAQUE_SIZE];
} bc_device_t;

/* ---- API ---- */

int bc_device_init(bc_device_t *dev);
void bc_device_shutdown(bc_device_t *dev);

/* kernel_name is the bare name (e.g. "saxpy"), not "saxpy.kd" */
int bc_load_kernel(bc_device_t *dev, const char *hsaco_path,
                   const char *kernel_name, bc_kernel_t *out);
void bc_unload_kernel(bc_device_t *dev, bc_kernel_t *kern);

void *bc_alloc(bc_device_t *dev, size_t size);
void bc_free(bc_device_t *dev, void *ptr);

/* Track a GPU allocation for ABEND fault correlation.
 * label = short name (e.g. "parts", "xs_data"), flags = AB_FL_* from bc_abend.h.
 * Optional — but without it the dump can't tell you what you hit. */
void bc_trak(bc_device_t *dev, void *ptr, size_t size,
             const char *label, int flags);

int bc_copy_h2d(bc_device_t *dev, void *dst, const void *src, size_t size);
int bc_copy_d2h(bc_device_t *dev, void *dst, const void *src, size_t size);

/* Synchronous dispatch. grid = workgroups, block = work-items per group. */
int bc_dispatch(bc_device_t *dev, const bc_kernel_t *kern,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                const void *args, uint32_t args_size);

#endif /* BC_RUNTIME_H */
