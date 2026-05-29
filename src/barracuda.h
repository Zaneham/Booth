#ifndef BARRACUDA_H
#define BARRACUDA_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ---- Version ---- */
#define BC_VERSION_MAJOR    0
#define BC_VERSION_MINOR    5
#define BC_VERSION_PATCH    0
#define BC_VERSION_STRING   "0.5.0"

/* The universe has limits. So do our buffers. No malloc, no madness. */
#define BC_MAX_SOURCE       (4 * 1024 * 1024)
#define BC_MAX_TOKENS       (1 << 20)
#define BC_MAX_IDENT        256
#define BC_MAX_ERRORS       64
#define BC_MAX_PATH         512
#define BC_MAX_DEPTH        256
#define CUDA_GLOBAL         0x0001
#define CUDA_DEVICE         0x0002
#define CUDA_HOST           0x0004
#define CUDA_SHARED         0x0008
#define CUDA_CONSTANT       0x0010
#define CUDA_MANAGED        0x0020
#define CUDA_GRID_CONSTANT  0x0040
#define CUDA_RESTRICT       0x0080
#define CUDA_FORCEINLINE    0x0100
#define CUDA_NOINLINE       0x0200
#define CUDA_LAUNCH_BOUNDS  0x0400

#define BC_OK               0
#define BC_ERR_IO          -1
#define BC_ERR_LEX         -2
#define BC_ERR_PARSE       -3
#define BC_ERR_OVERFLOW    -4
#define BC_ERR_VERIFY      -5

typedef struct {
    uint32_t line;
    uint16_t col;
    uint32_t offset;
} bc_loc_t;

typedef struct {
    bc_loc_t loc;
    char     msg[256];
    int      code;
    uint16_t eid;      /* bc_eid_t — language-neutral error ID */
} bc_error_t;

#include "fe/bc_err.h"

#endif /* BARRACUDA_H */
