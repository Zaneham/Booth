/* sysprint_device.h -- device-side SYSPRINT emit.
 *
 * Kernels include this header to emit class-tagged records into
 * a host-visible buffer. The buffer is allocated and managed by
 * the host through sysprint.h; the device side appends records
 * via atomicAdd on the head.
 *
 * The emit is a macro rather than a function so the kernel does
 * not depend on device-call support. Same .cu file compiles
 * through barracuda (for the GPU) and host gcc (for development
 * and the runtime test path). */

#ifndef BARRACUDA_SYSPRINT_DEVICE_H
#define BARRACUDA_SYSPRINT_DEVICE_H

#include <stdint.h>

/* Host-build fallback for atomicAdd. The barracuda frontend
 * recognises atomicAdd as an intrinsic; under host gcc we route
 * it through gcc's __atomic builtin so a .cu file is portable
 * to the CPU for the runtime test path. */
#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__) && !defined(__BARRACUDA__)
static inline uint32_t bc_sp_dev_atomic_add(uint32_t *p, uint32_t v)
{
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}
#define atomicAdd bc_sp_dev_atomic_add
#endif

/* Layout must match the host-side bc_sp_buf_t in sysprint.h.
 * Field for field. The include guard below stops double-typedef
 * when both headers are included in the same translation unit
 * (which the host launcher does). */
#ifndef BC_SP_BUF_T_DEFINED
#define BC_SP_BUF_T_DEFINED
typedef struct {
    uint8_t *data;
    uint32_t size;
    uint32_t head;
    uint32_t dropped;
} bc_sp_buf_t;
#endif

/* Emit a record into the buffer. The class id should match what
 * the host interned (see examples/sysprint_classes.h for the
 * convention used by the demo). Payload is copied byte by byte
 * into the buffer; the host decodes by class. */
#define BC_SYSPRINT(buf, class_id, payload, len)                            \
    do {                                                                    \
        uint32_t _sp_bytes = 8u + (((uint32_t)(len) + 3u) & ~3u);           \
        uint32_t _sp_head  = atomicAdd(&(buf)->head, _sp_bytes);            \
        if (_sp_head + _sp_bytes > (buf)->size) {                           \
            atomicAdd(&(buf)->dropped, 1u);                                 \
        } else {                                                            \
            uint8_t *_sp_dst = (buf)->data + _sp_head;                      \
            uint32_t *_sp_hdr = (uint32_t *)_sp_dst;                        \
            _sp_hdr[0] = (uint32_t)(class_id);                              \
            _sp_hdr[1] = (uint32_t)(len);                                   \
            for (uint32_t _sp_i = 0; _sp_i < (uint32_t)(len); _sp_i++) {    \
                _sp_dst[8u + _sp_i] = ((const uint8_t *)(payload))[_sp_i];  \
            }                                                               \
        }                                                                   \
    } while (0)

#endif /* BARRACUDA_SYSPRINT_DEVICE_H */
