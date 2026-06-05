/* bir_insert.h -- insert instructions into a function body, safely.
 *
 * BIR keeps every instruction in one flat array; blocks own contiguous
 * [first_inst, num_insts) ranges, and operands reference instructions
 * by absolute index. So inserting one instruction means shifting every
 * instruction after it AND remapping every value reference that pointed
 * past the insertion point. Get the remap wrong and an instruction
 * quietly references the wrong value, which only shows up as garbage
 * three passes downstream. This is the one tested primitive that does
 * it correctly. Passes that insert should come through here rather than
 * re-derive the block-vs-value operand classifier by hand. */
#ifndef BARRACUDA_BIR_INSERT_H
#define BARRACUDA_BIR_INSERT_H

#include "bir.h"

/* Insert `n` instructions (from `src`) into block `block_idx` at local
 * position `pos` (0 = block start, block's num_insts = block end).
 *
 * Existing value references are remapped so the module stays
 * well-formed; block references, function indices (the CALL callee),
 * and constants are left untouched. The inserted instructions are
 * copied verbatim, so any value operands they carry must already
 * reference post-insert indices. Instructions with no value operands
 * (ALLOCA, THREAD_ID, ...) need no patching, which is the common case.
 *
 * Returns the absolute index of the first inserted instruction (its
 * result value is BIR_MAKE_VAL of that index), or BIR_VAL_NONE on
 * overflow or bad arguments. */
uint32_t bir_insert(bir_module_t *M, uint32_t block_idx, uint32_t pos,
                    const bir_inst_t *src, uint32_t n);

#endif /* BARRACUDA_BIR_INSERT_H */
