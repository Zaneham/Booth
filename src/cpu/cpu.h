/* cpu.h -- x86-64 CPU backend for BarraCUDA.
 *
 * Turns a kernel into a host-runnable x86-64 function, so you can hack
 * on Triton and CUDA kernels on a laptop with no GPU in it. Codegen is
 * stack-everything, no register allocator: correct first, fast later.
 * How one call ends up running a whole block of threads is cpu_emit.c's
 * story to tell. */

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
#define X_XMM2 2

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
#define CPU_RELOC_MAX  4096
#define CPU_EXTSYM_MAX 64
#define CPU_EXTSYM_LEN 24

typedef struct {
    const bir_module_t *M;

    uint8_t   code[CPU_CODE_MAX];
    uint32_t  codelen;

    int32_t   slots[BIR_MAX_INSTS];     /* inst -> RBP offset, 0 = none */
    uint32_t  blk_off[BIR_MAX_BLOCKS];  /* block -> code offset */

    /* An alloca needs somewhere to actually live, past the slot that
     * just holds its pointer. The frame-sizing pre-pass parks each
     * one's offset here and the body reads them back in the same walk
     * order, so the two never have to talk to each other directly. */
    int32_t   alloca_off[CPU_ALLOCA_MAX];

    struct { uint32_t off; uint32_t blk; } fix[CPU_FIX_MAX];
    int       n_fix;

    /* External calls (libm and friends). Each call to an outside symbol
     * leaves a rel32 hole in .text and a note here saying which symbol
     * fills it; the ELF writer turns the notes into .rela.text entries and
     * undefined symbols so the linker can wire them up. */
    struct { uint32_t off; uint32_t sym; } reloc[CPU_RELOC_MAX];
    int       n_reloc;
    char      extsym[CPU_EXTSYM_MAX][CPU_EXTSYM_LEN];
    int       n_extsym;

    /* Device function calls. Every function lands in the same .text, so a
     * call to one is a plain rel32 to where it starts -- no relocation, no
     * symbol. func_off remembers where each landed; a forward call does not
     * know its target's offset yet, so it leaves a hole noted here and a
     * final pass fills every hole once all the offsets are known. */
    uint32_t  func_off[BIR_MAX_FUNCS];
    struct { uint32_t off; uint32_t func; } callfix[CPU_FIX_MAX];
    int       n_callfix;

    int       n_errs;
} cpu_mod_t;

void cpu_init(cpu_mod_t *X, const bir_module_t *M);
int  cpu_emit(cpu_mod_t *X);
int  cpu_elf(const cpu_mod_t *X, const char *path);

#endif /* BARRACUDA_CPU_H */
