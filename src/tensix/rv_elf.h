#ifndef BARRACUDA_RV_ELF_H
#define BARRACUDA_RV_ELF_H

#include "barracuda.h"
#include "rv_buf.h"

/*
 * RV32IM ELF emitter for the Tenstorrent Wormhole baby RISC-V cores.
 *
 * Layout of the produced file:
 *   [ELF32 header, 52 bytes]
 *   [PT_LOAD program header, 32 bytes]
 *   [code bytes, 4-byte aligned]
 *   [.shstrtab contents]
 *   [Section header table, 3 entries * 40 bytes]
 *
 * Spec sources:
 *   System V ABI ELF32 specification (the constants below are from
 *   the standard elf.h on any Linux box).
 *   sfpi-binutils/include/elf/common.h, ELFCLASS32 = 1, ELFDATA2LSB = 1,
 *   ET_EXEC = 2, EM_RISCV = 243, EV_CURRENT = 1.
 *   tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/README.md
 *   for the L1 base address being 0x00000000 and code living there.
 *   tt-metal SDK loads via PT_LOAD; sections are optional metadata
 *   that any reasonable disassembler will read.
 *
 * Soft-float ABI, no compressed instructions, no F/D extension:
 *   e_flags = 0. EF_RISCV_FLOAT_ABI_SOFT happens to also be 0 so this
 *   is correct rather than accidentally correct.
 *
 * What this emitter does NOT do today: relocations, symbol tables,
 * debug info, BSS. The current path expects the kernel to be
 * self-contained code with no external symbol references, which is
 * fine for the first integer-only bring-up.
 */

/* L1 base address where code is loaded. Spec source above. */
#define RV_ELF_LOAD_ADDR  0x00000000u

/* Entry point at the start of code; spec is silent on whether the
 * tt-metal loader honours e_entry or assumes a known address, so
 * we set both to RV_ELF_LOAD_ADDR and hope they line up. The PC
 * register that gets written by the host on launch lives in
 * Tensix MMIO and is undocumented in the public ISA reference. */
#define RV_ELF_ENTRY      RV_ELF_LOAD_ADDR

/*
 * Write the contents of a code buffer to an RV32IM ELF file at the
 * given path. Returns BC_OK on success or a BC_ERR_* code on
 * failure. The caller is responsible for any post-write actions
 * such as chmod or staging onto the deploy host.
 */
int rv_elf_write(const rv_buf_t *code, const char *path);

#endif /* BARRACUDA_RV_ELF_H */
