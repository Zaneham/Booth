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

/* CB synchronisation. The signal (tt_sem_inc) is a NoC atomic increment of a
 * counter in a remote core's L1; the wait (tt_sem_wait_ge) is a local spin.
 * The four cb_* ops are these two under their producer/consumer names:
 *   producer: reserve_back (wait free) ... push_back (signal recv)
 *   consumer: wait_front  (wait recv) ... pop_front  (signal free) */
int tt_sem_inc    (rv_buf_t *code, uint32_t sem_lo, uint32_t sem_mid,
                   uint32_t incr);
int tt_sem_wait_ge(rv_buf_t *code, uint32_t sem_addr, uint32_t threshold);

int tt_cb_reserve_back(rv_buf_t *code, uint32_t free_addr, uint32_t credits);
int tt_cb_push_back   (rv_buf_t *code, uint32_t recv_lo, uint32_t recv_mid,
                       uint32_t n);
int tt_cb_wait_front  (rv_buf_t *code, uint32_t recv_addr, uint32_t n);
int tt_cb_pop_front   (rv_buf_t *code, uint32_t free_lo, uint32_t free_mid,
                       uint32_t n);

/* Materialise a 32-bit value into a register (lui+addi). */
void tt_li32(rv_buf_t *code, uint8_t reg, uint32_t v);

/* Register-sourced transfer (low addresses come from treg/rreg so a loop can
 * advance them) and a completion barrier that spins until issued transfers
 * have landed. is_write picks the write path (marked, ack-tracked). */
int tt_noc_xfer_reg(rv_buf_t *code, int is_write, uint8_t treg, uint8_t rreg,
                    uint32_t targ_mid, uint32_t ret_mid, uint32_t len);
int tt_noc_barrier (rv_buf_t *code, int is_write);

#endif /* BARRACUDA_TENSIX_NOC_H */
