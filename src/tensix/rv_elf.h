#ifndef BARRACUDA_RV_ELF_H
#define BARRACUDA_RV_ELF_H

#include "barracuda.h"
#include "rv_buf.h"
#include "tdf.h"     /* td_txtmax, td_chip */

/*
 * RV32IM ELF emitter for the Tenstorrent baby RISC-V cores. The shape is
 * dictated by tt-metal's loader in tt_metal/llrt/tt_elffile.cpp, which is
 * stricter than the System V ABI alone: sections are mandatory, a .segments
 * metadata section drives trimming and size checks, and a kernel must carry at
 * least one relocation section or ReadImage throws outright.
 *
 * We emit no relocation entries because every branch and call is patched
 * PC-relative at selection time, and XIP relocates the text segment as a unit.
 * The empty .rela.text exists purely to satisfy the loader's count.
 */

/* Text VMA. Three fields must agree on it: e_entry, the first PT_LOAD's
 * p_vaddr, and the .segments vma. TrimSegments runs during ReadImage and
 * matches on the link-time address, whereas XIPify only rezeros it afterwards,
 * so moving this off zero means moving all three together. */
#define RV_ELF_LOAD_ADDR  0x00000000u
#define RV_ELF_ENTRY      RV_ELF_LOAD_ADDR

/* The .segments size limit is the target chip's text budget, td_txtmax(). The
 * compiler's own ceiling is RV_BUF_MAX_WORDS and is reported separately, so a
 * kernel that outgrows the buffer says so rather than blaming the hardware. */

/* tt-metal links with -Wl,-z,max-page-size=16. */
#define RV_ELF_SEG_ALIGN  16u

/*
 * Write a code buffer to an RV32IM ELF file at the given path. Returns BC_OK on
 * success or a BC_ERR_* code on failure.
 */
int rv_elf_write(const rv_buf_t *code, const char *path);

#endif /* BARRACUDA_RV_ELF_H */
