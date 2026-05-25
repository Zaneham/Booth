/* cpu.h -- x86-64 CPU backend for BarraCUDA.
 *
 * Compiles a kernel to a host-runnable x86-64 function so Triton
 * and CUDA kernels can be developed on a laptop with no GPU. The
 * SIMT model is one logical thread per call for now; the
 * thread-to-loop transform that lets one call cover a whole block
 * lands in a later sitting. Stack-everything codegen, no register
 * allocator: correct first, fast later. */

#ifndef BARRACUDA_CPU_H
#define BARRACUDA_CPU_H

#include "bir.h"

/* ---- GPR indices (ModRM) ---- */
#define X_RAX 0
#define X_RCX 1
#define X_RDX 2
#define X_RBX 3
#define X_RSP 4
#define X_RBP 5
#define X_RSI 6
#define X_RDI 7

/* ---- XMM ---- */
#define X_XMM0 0
#define X_XMM1 1

/* ---- Condition codes (Jcc / SETcc low nibble) ---- */
#define XCC_E  0x04
#define XCC_NE 0x05
#define XCC_L  0x0C
#define XCC_GE 0x0D
#define XCC_LE 0x0E
#define XCC_G  0x0F
#define XCC_B  0x02
#define XCC_AE 0x03
#define XCC_BE 0x06
#define XCC_A  0x07

#define CPU_CODE_MAX  (256 * 1024)
#define CPU_FIX_MAX   8192
#define CPU_ALLOCA_MAX 4096

typedef struct {
    const bir_module_t *M;

    uint8_t   code[CPU_CODE_MAX];
    uint32_t  codelen;

    int32_t   slots[BIR_MAX_INSTS];     /* inst -> RBP offset, 0 = none */
    uint32_t  blk_off[BIR_MAX_BLOCKS];  /* block -> code offset */

    /* per-alloca backing-store RBP offsets, in inst-walk order: the
     * pre-pass that sizes the frame fills these, the body reads them
     * back in the same order. */
    int32_t   alloca_off[CPU_ALLOCA_MAX];

    struct { uint32_t off; uint32_t blk; } fix[CPU_FIX_MAX];
    int       n_fix;

    int       n_errs;
} cpu_mod_t;

void cpu_init(cpu_mod_t *X, const bir_module_t *M);
int  cpu_emit(cpu_mod_t *X);
int  cpu_elf(const cpu_mod_t *X, const char *path);

#endif /* BARRACUDA_CPU_H */
