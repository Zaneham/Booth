#ifndef BARRACUDA_TENSIX_NOC_H
#define BARRACUDA_TENSIX_NOC_H

#include "rv_buf.h"

/* NoC transfer primitive for the baby RISC-V data-movement cores.
 *
 * A transfer is programmed by storing to the NIU request-initiator registers
 * (NIU_BASE 0xFFB20000, initiator 0) and writing 1 to NOC_CMD_CTRL to trigger.
 * Each call appends the RISC-V `sw` sequence to `code`. Addresses are 36-bit:
 * _lo is the low 32 bits, _mid packs the high 4 bits with the remote tile's
 * NoC X/Y coordinates (see NoC/Coordinates.md). Register layout from the
 * Wormhole B0 ISA docs, NoC/MemoryMap.md.
 *
 * read:  remote(target) -> local L1(return).
 * write: local L1(target) -> remote(return).  (per NOC_CMD_WR semantics.) */
int tt_noc_read (rv_buf_t *code, uint32_t targ_lo, uint32_t targ_mid,
                 uint32_t ret_lo, uint32_t ret_mid, uint32_t len);
int tt_noc_write(rv_buf_t *code, uint32_t targ_lo, uint32_t targ_mid,
                 uint32_t ret_lo, uint32_t ret_mid, uint32_t len);

#endif /* BARRACUDA_TENSIX_NOC_H */
