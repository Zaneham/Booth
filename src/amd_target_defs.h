#ifndef BARRACUDA_AMD_TARGET_DEFS_H
#define BARRACUDA_AMD_TARGET_DEFS_H

/*
 * AMD Target definitions shared between frontend, IR, and backend.
 */

typedef enum {
    AMD_TARGET_GFX90A,    /* CDNA 2 (MI250, Wave64) */
    AMD_TARGET_GFX942,    /* CDNA 3 (MI300X, Wave64) */
    AMD_TARGET_GFX1030,   /* RDNA 2 */
    AMD_TARGET_GFX1100,   /* RDNA 3 */
    AMD_TARGET_GFX1200,   /* RDNA 4 */
} amd_target_t;

#define AMD_WAVE_SIZE       32
#define AMD_WAVE64          64

static inline int amd_get_wave_size(amd_target_t target) {
    switch (target) {
        case AMD_TARGET_GFX90A:
        case AMD_TARGET_GFX942:
            return AMD_WAVE64;
        case AMD_TARGET_GFX1030:
        case AMD_TARGET_GFX1100:
        case AMD_TARGET_GFX1200:
        default:
            return AMD_WAVE_SIZE;
    }
}

#endif /* BARRACUDA_AMD_TARGET_DEFS_H */
