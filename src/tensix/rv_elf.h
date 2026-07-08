#ifndef BARRACUDA_RV_ELF_H
#define BARRACUDA_RV_ELF_H

#include "barracuda.h"
#include "rv_buf.h"

/*
 * RV32IM ELF emitter for the Tenstorrent Wormhole baby RISC-V cores. The file
 * it produces is an ELF32 header, a PT_LOAD program header, 4-byte-aligned code,
 * the .shstrtab contents, and a three-entry section header table. The constants
 * below come from the System V ABI ELF32 spec and sfpi-binutils common.h
 * (ELFCLASS32=1, ELFDATA2LSB=1, ET_EXEC=2, EM_RISCV=243, EV_CURRENT=1); the L1
 * base of 0x00000000 comes from the tt-isa-documentation BabyRISCV README, and
 * the tt-metal SDK loads via PT_LOAD with sections as optional metadata any
 * reasonable disassembler will read.
 *
 * It's a soft-float ABI with no compressed instructions and no F/D extension, so
 * e_flags = 0; EF_RISCV_FLOAT_ABI_SOFT happens to also be 0, so this is correct
 * rather than accidentally correct. What it does NOT do today is relocations,
 * symbol tables, debug info, or BSS: the current path expects a self-contained
 * kernel with no external symbol references, which is fine for the first
 * integer-only bring-up.
 */

/* L1 base address where code is loaded. Spec source above. */
#define RV_ELF_LOAD_ADDR  0x00000000u

/* Entry point at the start of code. The spec is silent on whether the tt-metal
 * loader honours e_entry or assumes a known address, so we set both to
 * RV_ELF_LOAD_ADDR and hope they line up. The PC register the host writes on
 * launch lives in Tensix MMIO and is undocumented in the public ISA reference. */
#define RV_ELF_ENTRY      RV_ELF_LOAD_ADDR

/*
 * Write a code buffer to an RV32IM ELF file at the given path. Returns BC_OK on
 * success or a BC_ERR_* code on failure. The caller handles any post-write steps
 * like chmod or staging onto the deploy host.
 */
int rv_elf_write(const rv_buf_t *code, const char *path);

#endif /* BARRACUDA_RV_ELF_H */
