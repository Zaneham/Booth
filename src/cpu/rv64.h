/* rv64.h -- RISC-V 64 (RV64IMFD) CPU backend for BarraCUDA.
 *
 * Same idea as the x86 --cpu backend, different chip: turn a kernel into
 * a host-runnable RV64 object so you can run CUDA and Triton on a RISC-V
 * machine (or qemu-riscv64) with no GPU. Stack-everything, no register
 * allocator: correct first, fast later.
 *
 * The instruction encoder and ELF writer are lifted from the RV64 backend
 * I wrote for Karearea (same furniture, BIR instead of JIR). The SIMT-on-
 * CPU model is identical to cpu_emit.c: a __global__ kernel's body runs in
 * a loop over thread_id, with nthreads a hidden trailing arg. */

#ifndef BARRACUDA_RV64_H
#define BARRACUDA_RV64_H

#include "bir.h"

/* ---- GPR indices (x0-x31, ABI names) ---- */
#define V_ZERO  0
#define V_RA    1     /* return address */
#define V_SP    2     /* stack pointer */
#define V_S0    8     /* frame pointer (our rbp) */
#define V_T0    5     /* scratch */
#define V_T1    6
#define V_T2    7
#define V_A0   10     /* int/ptr arg + return registers a0..a7 */
#define V_A7   17

/* ---- FPR indices (f0-f31, ABI names) ---- */
#define V_FT0   0     /* float scratch */
#define V_FT1   1
#define V_FT2   2
#define V_FA0  10     /* float arg registers fa0..fa7 */

#define RV_CODE_MAX   (256 * 1024)
#define RV_FIX_MAX    8192
#define RV_ALLOCA_MAX 4096

typedef struct {
    const bir_module_t *M;

    uint8_t   code[RV_CODE_MAX];
    uint32_t  codelen;

    int32_t   slots[BIR_MAX_INSTS];     /* inst -> s0 offset, 0 = none */
    uint32_t  blk_off[BIR_MAX_BLOCKS];  /* block -> code offset */

    /* alloca backing-store offsets, in inst-walk order (see cpu.h) */
    int32_t   alloca_off[RV_ALLOCA_MAX];

    /* branch fixups: a 32-bit instruction at code[off] targets block blk */
    struct { uint32_t off; uint32_t blk; uint8_t kind; } fix[RV_FIX_MAX];
    int       n_fix;

    int       n_errs;
} rv64_mod_t;

void rv64_init(rv64_mod_t *V, const bir_module_t *M);
int  rv64_emit(rv64_mod_t *V);
int  rv64_elf(const rv64_mod_t *V, const char *path);

#endif /* BARRACUDA_RV64_H */
